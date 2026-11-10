import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { API } from './api'
import {
  type BoneParentMap, type JointAdjustments, type JointDragMode, type JointMove, type JointUndoEntry,
  adjustedJointCount, clearJointAdjustments, descendantJointNames, fingerJointNames, jointDelta, popJointUndo,
  pushJointUndo, pushJointUndoGroup, readMoveWithChildrenPreference, recordJointAdjustment, reparentOnRemove,
  resolveJointDragMode, resolveRemovableJoints, resolveSingleRemoval, spurJointNames, translateJointPositions,
  writeMoveWithChildrenPreference,
} from './jointAdjustments'

export type { JointAdjustments } from './jointAdjustments'
export type JointPositionCollector = () => { positions: JointAdjustments; removeJoints: string[] }

type DragState = {
  name: string
  mode: JointDragMode
  plane: THREE.Plane
  offset: THREE.Vector3
  startPosition: THREE.Vector3
  // World position of the dragged handle when the gesture began — the anchor
  // the carried subtree is offset from.
  startWorld: THREE.Vector3
  // Subtree mode: every carried descendant's world position at drag start, so
  // its handle can be re-placed by a pure translation on each move.
  carriedStartWorld: JointAdjustments
  // Single-joint mode: the direct children pinned in world space, with the
  // local positions to restore on undo.
  pinnedChildren: Array<{ bone: THREE.Bone; world: THREE.Vector3; startLocal: THREE.Vector3 }>
}

function sessionPreferenceStorage(): Storage | null {
  try {
    return typeof window === 'undefined' ? null : window.sessionStorage
  } catch {
    return null
  }
}

const HANDLE_COLOR = 0xe08a3c
const HANDLE_COLOR_REMOVED = 0x565c6a
const HANDLE_SIZE = 0.006

/** Convert a glTF-world (Y-up) position to Blender-space (Z-up) joints. */
function blenderFromGltf(world: THREE.Vector3): [number, number, number] {
  return [world.x, -world.z, world.y]
}

function isEditableFocus(): boolean {
  const tag = (document.activeElement?.tagName || '').toLowerCase()
  return tag === 'input' || tag === 'textarea'
}

/** Review viewport with draggable joint handles. The visible mesh is baked
 *  static at load time (frozen in its predicted rest pose) — dragging or
 *  removing a joint only moves handles and bone-helper lines; the mesh
 *  itself only updates once Save & rebind actually re-skins it on the
 *  backend. The parent reads final Blender-space joint positions and the
 *  removed-joint list through `collectRef` on save. */
export default function RigAdjustViewport({ artifactId, collectRef, onAdjusted, controlsHost }: {
  artifactId: string
  collectRef: React.MutableRefObject<JointPositionCollector | null>
  onAdjusted?: (adjustedCount: number) => void
  // Where the adjust controls render. `undefined` keeps them inline (bare
  // component use / tests); an element portals them into the workspace's
  // right-hand control bar; `null` means the workspace has not mounted that
  // element yet, so render nothing rather than flashing them over the canvas.
  controlsHost?: HTMLElement | null
}) {
  const host = useRef<HTMLDivElement>(null)
  const [jointCount, setJointCount] = useState(0)
  const [jointNames, setJointNames] = useState<string[]>([])
  const [selected, setSelected] = useState('')
  const [selectedDescendants, setSelectedDescendants] = useState(0)
  const [adjustedCount, setAdjustedCount] = useState(0)
  const [removedCount, setRemovedCount] = useState(0)
  const [removedNames, setRemovedNames] = useState<string[]>([])
  const [canUndo, setCanUndo] = useState(false)
  const [moveWithChildren, setMoveWithChildren] = useState(() => readMoveWithChildrenPreference(sessionPreferenceStorage()))

  // Pointer handlers live inside the scene effect and must not re-subscribe
  // when the toggle flips, so they read the live value through a ref.
  const moveWithChildrenRef = useRef(moveWithChildren)
  useEffect(() => {
    moveWithChildrenRef.current = moveWithChildren
    writeMoveWithChildrenPreference(sessionPreferenceStorage(), moveWithChildren)
  }, [moveWithChildren])

  const bonesRef = useRef<Map<string, THREE.Bone>>(new Map())
  const originalsRef = useRef<Map<string, THREE.Vector3>>(new Map())
  const handlesRef = useRef<Map<string, THREE.Mesh>>(new Map())
  // Original hierarchy as loaded — never mutated, used to know a joint's
  // true parent/children when restoring a removal.
  const parentOfRef = useRef<BoneParentMap>({})
  // Live hierarchy reflecting every removal applied so far (splices close
  // the gap left by a single-joint removal). Kept in sync with the actual
  // THREE.Bone parent/child structure via .attach()/.removeFromParent().
  const currentParentRef = useRef<BoneParentMap>({})
  const adjustmentsRef = useRef<JointAdjustments>(clearJointAdjustments())
  const undoRef = useRef<JointUndoEntry[]>([])
  const removedRef = useRef<Set<string>>(new Set())
  // Order joints were removed in, so "Reset all" can restore the hierarchy
  // by undoing removals last-first.
  const removalOrderRef = useRef<string[]>([])
  const selectedRef = useRef('')
  const sceneRootRef = useRef<THREE.Object3D | null>(null)
  const helperRef = useRef<THREE.SkeletonHelper | null>(null)

  // Buttons in the render tree call into the live Three.js scene through
  // this ref (the scene itself only exists inside the effect below).
  const actionsRef = useRef<{
    removeJoints: (names: string[]) => void
    removeJointSingle: (name: string) => void
    restoreJoint: (name: string) => void
    resetAll: () => void
    undo: () => void
  } | null>(null)

  useEffect(() => {
    const element = host.current
    if (!element) return

    // This effect re-runs whenever `artifactId` changes (switching the
    // journey selector to a different rig). Every ref below is scoped to
    // the previous artifact's THREE scene, which the cleanup below has
    // already torn down — reset them all before wiring up the new one, or
    // removed/adjusted bookkeeping (and stale bone/handle references) leaks
    // from the old rig into the new one.
    bonesRef.current = new Map()
    originalsRef.current = new Map()
    handlesRef.current = new Map()
    parentOfRef.current = {}
    currentParentRef.current = {}
    adjustmentsRef.current = clearJointAdjustments()
    undoRef.current = []
    removedRef.current = new Set()
    removalOrderRef.current = []
    helperRef.current = null
    sceneRootRef.current = null
    selectedRef.current = ''
    setSelected('')
    setSelectedDescendants(0)
    setRemovedNames([])
    setRemovedCount(0)
    setAdjustedCount(0)
    setCanUndo(false)
    setJointCount(0)
    setJointNames([])

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0b0d11)
    const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 1000)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    element.appendChild(renderer.domElement)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    scene.add(new THREE.HemisphereLight(0xffffff, 0x20243a, 2.2))
    const key = new THREE.DirectionalLight(0xffffff, 2)
    key.position.set(3, 4, 5)
    scene.add(key)

    let frame = 0
    let disposed = false
    let drag: DragState | null = null
    const pointer = new THREE.Vector2()
    const raycaster = new THREE.Raycaster()

    const resize = () => {
      const { width, height } = element.getBoundingClientRect()
      if (!width || !height) return
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height)
    }
    const observer = new ResizeObserver(resize)
    observer.observe(element)
    resize()

    const toLocal = (bone: THREE.Bone, world: THREE.Vector3): THREE.Vector3 => {
      const parentMatrix = bone.parent ? bone.parent.matrixWorld : new THREE.Matrix4()
      return world.clone().applyMatrix4(parentMatrix.clone().invert())
    }

    const reportAdjusted = () => {
      const count = adjustedJointCount(adjustmentsRef.current)
      setAdjustedCount(count)
      onAdjusted?.(count)
    }
    const reportRemoved = () => {
      setRemovedCount(removedRef.current.size)
      setRemovedNames([...removedRef.current])
    }
    const reportUndo = () => setCanUndo(undoRef.current.length > 0)

    const settlePosition = (name: string) => {
      const bone = bonesRef.current.get(name)
      const original = originalsRef.current.get(name)
      if (!bone || !original) return
      adjustmentsRef.current = recordJointAdjustment(
        adjustmentsRef.current,
        name,
        [bone.position.x, bone.position.y, bone.position.z],
        [original.x, original.y, original.z],
      )
      reportAdjusted()
    }

    const syncHandleVisual = (name: string) => {
      const handle = handlesRef.current.get(name)
      if (!handle) return
      const material = handle.material as THREE.MeshBasicMaterial
      const removed = removedRef.current.has(name)
      material.color.setHex(removed ? HANDLE_COLOR_REMOVED : HANDLE_COLOR)
      material.opacity = removed ? 0.45 : 0.95
    }

    const syncHandles = () => {
      for (const [name, handle] of handlesRef.current) {
        const bone = bonesRef.current.get(name)
        if (bone) handle.position.setFromMatrixPosition(bone.matrixWorld)
      }
    }

    const pointerNdc = (event: PointerEvent | MouseEvent): THREE.Vector2 => {
      const rect = renderer.domElement.getBoundingClientRect()
      return new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      )
    }

    const pickHandle = (event: PointerEvent | MouseEvent): string | null => {
      raycaster.setFromCamera(pointerNdc(event), camera)
      const hits = raycaster.intersectObjects(Array.from(handlesRef.current.values()), false)
      const hit = hits[0]
      return hit ? String(hit.object.userData.jointName) : null
    }

    // Distance (in original-hierarchy hops) from the root — used to remove a
    // requested subtree leaf-first so each removeJointSingle call always
    // sees a bone whose remaining live children are exactly the ones that
    // still need to be re-parented onto it.
    const depthOf = (name: string): number => {
      let depth = 0
      let current = parentOfRef.current[name]
      const seen = new Set<string>([name])
      while (current) {
        if (seen.has(current)) break
        seen.add(current)
        depth += 1
        current = parentOfRef.current[current]
      }
      return depth
    }

    const rebuildSkeletonHelper = () => {
      const root = sceneRootRef.current
      if (!root) return
      const old = helperRef.current
      if (old) {
        old.parent?.remove(old)
        old.geometry.dispose()
      }
      const helper = new THREE.SkeletonHelper(root)
      const helperMaterial = helper.material as THREE.LineBasicMaterial
      helperMaterial.color.set(0xa98cff)
      helperMaterial.depthTest = false
      helperMaterial.transparent = true
      helperMaterial.opacity = 0.9
      helper.renderOrder = 10
      root.add(helper)
      helperRef.current = helper
    }

    // Splices a single joint out of the hierarchy: its current children
    // re-attach (preserving world transform) onto its current parent, then
    // the joint itself detaches. `rebuild: false` defers the skeleton-helper
    // rebuild + handle sync + removed-count report to the caller, so a
    // batch (subtree) removal only pays for one rebuild instead of one per
    // joint.
    const removeJointSingle = (name: string, options?: { rebuild?: boolean }): boolean => {
      const rebuild = options?.rebuild ?? true
      const check = resolveSingleRemoval(name, currentParentRef.current)
      if (!check.removable) return false
      const bone = bonesRef.current.get(name)
      if (!bone) return false
      const grandparentName = currentParentRef.current[name] ?? null
      const grandparentBone = grandparentName ? bonesRef.current.get(grandparentName) : sceneRootRef.current
      if (!grandparentBone) return false
      const children = bone.children.filter((child): child is THREE.Bone => child instanceof THREE.Bone)
      for (const child of children) grandparentBone.attach(child)
      bone.removeFromParent()
      removedRef.current.add(name)
      removalOrderRef.current.push(name)
      currentParentRef.current = reparentOnRemove(currentParentRef.current, name)
      syncHandleVisual(name)
      if (rebuild) {
        sceneRootRef.current?.updateMatrixWorld(true)
        rebuildSkeletonHelper()
        syncHandles()
        reportRemoved()
      }
      return true
    }

    // Subtree removal (secondary "Remove joint + children" button, and the
    // finger/spur shortcuts): expand to the full descendant set, then remove
    // leaf-first through removeJointSingle so the hierarchy bookkeeping
    // (currentParentRef, the live THREE bone graph, the skeleton helper)
    // stays consistent the same way a single removal does.
    const removeJointsByName = (names: string[]) => {
      const { removable } = resolveRemovableJoints(names, parentOfRef.current)
      if (!removable.length) return
      const ordered = [...removable].sort((a, b) => depthOf(b) - depthOf(a))
      let removedAny = false
      for (const name of ordered) {
        if (removeJointSingle(name, { rebuild: false })) removedAny = true
      }
      if (removedAny) {
        sceneRootRef.current?.updateMatrixWorld(true)
        rebuildSkeletonHelper()
        syncHandles()
        reportRemoved()
      }
    }

    // Recomputes currentParentRef straight from the live THREE.Bone graph —
    // the source of truth after a restore's .attach() calls — rather than
    // trying to hand-splice the map back together.
    const recomputeCurrentParents = () => {
      const map: BoneParentMap = {}
      for (const [name, bone] of bonesRef.current) {
        if (removedRef.current.has(name)) continue
        map[name] = bone.parent instanceof THREE.Bone ? bone.parent.name : null
      }
      currentParentRef.current = map
    }

    const restoreJoint = (name: string) => {
      if (!removedRef.current.has(name)) return
      const bone = bonesRef.current.get(name)
      if (!bone) return
      const originalParentName = parentOfRef.current[name] ?? null
      const originalParentBone = originalParentName ? bonesRef.current.get(originalParentName) : null
      const attachTarget = originalParentBone || sceneRootRef.current
      if (attachTarget) attachTarget.attach(bone)
      // Reclaim every bone whose ORIGINAL parent was this joint and which
      // isn't itself still removed — it was re-parented onto our own
      // grandparent when we were removed; bring it back home.
      for (const [childName, childParentName] of Object.entries(parentOfRef.current)) {
        if (childParentName !== name || childName === name) continue
        if (removedRef.current.has(childName)) continue
        const childBone = bonesRef.current.get(childName)
        if (childBone) bone.attach(childBone)
      }
      removedRef.current.delete(name)
      removalOrderRef.current = removalOrderRef.current.filter((entry) => entry !== name)
      recomputeCurrentParents()
      syncHandleVisual(name)
      sceneRootRef.current?.updateMatrixWorld(true)
      rebuildSkeletonHelper()
      syncHandles()
      reportRemoved()
    }

    const resetAll = () => {
      for (const name of [...removalOrderRef.current].reverse()) restoreJoint(name)
      removalOrderRef.current = []
      removedRef.current = new Set()
      currentParentRef.current = { ...parentOfRef.current }
      for (const [name, bone] of bonesRef.current) {
        const original = originalsRef.current.get(name)
        if (original) bone.position.copy(original)
        bone.updateMatrixWorld(true)
      }
      adjustmentsRef.current = clearJointAdjustments()
      undoRef.current = []
      for (const name of handlesRef.current.keys()) syncHandleVisual(name)
      syncHandles()
      reportAdjusted()
      reportRemoved()
      reportUndo()
    }

    // One Z undoes one gesture — including a move-with-children drag, whose
    // group holds every joint whose local position that single gesture wrote.
    const undo = () => {
      const { entry, stack } = popJointUndo(undoRef.current)
      undoRef.current = stack
      reportUndo()
      if (!entry) return
      for (const move of entry.moves) {
        const bone = bonesRef.current.get(move.name)
        if (bone) bone.position.set(...move.position)
      }
      sceneRootRef.current?.updateMatrixWorld(true)
      for (const move of entry.moves) settlePosition(move.name)
      syncHandles()
    }

    actionsRef.current = {
      removeJoints: removeJointsByName,
      removeJointSingle: (name: string) => { removeJointSingle(name) },
      restoreJoint,
      resetAll,
      undo,
    }

    new GLTFLoader().load(`${API}/assets/${artifactId}/download`, (gltf) => {
      if (disposed) return
      const bones: THREE.Bone[] = []
      gltf.scene.traverse((node) => {
        if (node instanceof THREE.SkinnedMesh) {
          node.skeleton.bones.forEach((bone) => { if (!bones.includes(bone)) bones.push(bone) })
        }
      })
      gltf.scene.updateMatrixWorld(true)
      const handleGeometry = new THREE.SphereGeometry(HANDLE_SIZE, 12, 8)
      const parentOf: BoneParentMap = {}
      for (const bone of bones) {
        bonesRef.current.set(bone.name, bone)
        originalsRef.current.set(bone.name, bone.position.clone())
        parentOf[bone.name] = bone.parent instanceof THREE.Bone ? bone.parent.name : null
        const material = new THREE.MeshBasicMaterial({ color: HANDLE_COLOR, depthTest: false, transparent: true, opacity: 0.95 })
        const handle = new THREE.Mesh(handleGeometry, material)
        handle.renderOrder = 20
        handle.position.setFromMatrixPosition(bone.matrixWorld)
        handle.userData = { jointName: bone.name }
        scene.add(handle)
        handlesRef.current.set(bone.name, handle)
      }
      parentOfRef.current = parentOf
      currentParentRef.current = { ...parentOf }
      sceneRootRef.current = gltf.scene
      setJointCount(bones.length)
      setJointNames(bones.map((bone) => bone.name))

      // Freeze the visible mesh in its predicted rest pose: bake the current
      // (pre-edit) skinned vertex positions once into a plain, static
      // geometry, then hide the live SkinnedMesh. Joint handles and the
      // skeleton helper below keep following the editable bones — only the
      // rendered surface stops deforming while joints are dragged.
      const vertex = new THREE.Vector3()
      gltf.scene.traverse((node) => {
        if (!(node instanceof THREE.SkinnedMesh)) return
        const positionAttr = node.geometry.attributes.position as THREE.BufferAttribute | undefined
        if (!positionAttr) return
        const baked = new Float32Array(positionAttr.count * 3)
        for (let i = 0; i < positionAttr.count; i += 1) {
          vertex.fromBufferAttribute(positionAttr, i)
          node.applyBoneTransform(i, vertex)
          baked[i * 3] = vertex.x
          baked[i * 3 + 1] = vertex.y
          baked[i * 3 + 2] = vertex.z
        }
        const staticGeometry = node.geometry.clone()
        staticGeometry.setAttribute('position', new THREE.BufferAttribute(baked, 3))
        staticGeometry.deleteAttribute('skinIndex')
        staticGeometry.deleteAttribute('skinWeight')
        staticGeometry.computeVertexNormals()
        const staticMesh = new THREE.Mesh(staticGeometry, node.material)
        staticMesh.name = `${node.name}-frozen-rest-pose`
        staticMesh.position.copy(node.position)
        staticMesh.quaternion.copy(node.quaternion)
        staticMesh.scale.copy(node.scale)
        node.visible = false
        node.parent?.add(staticMesh)
      })

      rebuildSkeletonHelper()

      const bounds = new THREE.Box3().setFromObject(gltf.scene)
      const center = bounds.getCenter(new THREE.Vector3())
      const size = bounds.getSize(new THREE.Vector3())
      const radius = Math.max(size.length() * 0.65, 0.25)
      controls.target.copy(center)
      camera.position.set(center.x + radius * 1.25, center.y + radius * 0.4, center.z + radius * 1.7)
      camera.near = Math.max(radius / 100, 0.001)
      camera.far = radius * 100
      camera.updateProjectionMatrix()
      controls.update()
      scene.add(gltf.scene)
    })

    // Descendants of a joint in the LIVE hierarchy, skipping anything already
    // removed (a removed bone is detached from the scene graph and cannot be
    // carried along by its former parent).
    const liveDescendants = (name: string): string[] =>
      descendantJointNames(name, currentParentRef.current).filter((entry) => !removedRef.current.has(entry))

    const worldOf = (object: THREE.Object3D): THREE.Vector3 =>
      new THREE.Vector3().setFromMatrixPosition(object.matrixWorld)

    const onPointerDown = (event: PointerEvent) => {
      const name = pickHandle(event)
      if (!name) return
      const handle = handlesRef.current.get(name)
      const bone = bonesRef.current.get(name)
      if (!handle || !bone) return
      selectedRef.current = name
      setSelected(name)
      const descendants = liveDescendants(name)
      setSelectedDescendants(descendants.length)
      controls.enabled = false
      const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(
        camera.getWorldDirection(new THREE.Vector3()).negate(),
        handle.position.clone(),
      )
      raycaster.setFromCamera(pointerNdc(event), camera)
      const intersection = new THREE.Vector3()
      raycaster.ray.intersectPlane(plane, intersection)
      // Mode is fixed for the whole gesture: whatever the toggle says, with
      // Shift held at press time flipping it for this drag only.
      const mode = resolveJointDragMode({
        moveWithChildren: moveWithChildrenRef.current,
        shiftKey: event.shiftKey,
        hasDescendants: descendants.length > 0,
      })
      // Subtree mode carries the descendants: snapshot their world positions
      // so each move re-places their handles by one rigid translation.
      const carriedStartWorld: JointAdjustments = {}
      if (mode === 'subtree') {
        for (const descendant of descendants) {
          const descendantBone = bonesRef.current.get(descendant)
          if (!descendantBone) continue
          const world = worldOf(descendantBone)
          carriedStartWorld[descendant] = [world.x, world.y, world.z]
        }
      }
      // Single-joint mode pins the direct children in world space — everything
      // deeper follows them, so only one generation needs rewriting.
      const pinnedChildren = mode === 'single'
        ? bone.children
          .filter((child): child is THREE.Bone => child instanceof THREE.Bone)
          .map((child) => ({ bone: child, world: worldOf(child), startLocal: child.position.clone() }))
        : []
      // Capture the pre-drag position so a completed drag pushes exactly one
      // undo entry — this is the entry the old code never created.
      drag = {
        name,
        mode,
        plane,
        offset: handle.position.clone().sub(intersection),
        startPosition: bone.position.clone(),
        startWorld: worldOf(bone),
        carriedStartWorld,
        pinnedChildren,
      }
      renderer.domElement.setPointerCapture(event.pointerId)
    }

    const onPointerMove = (event: PointerEvent) => {
      if (!drag) {
        renderer.domElement.style.cursor = pickHandle(event) ? 'grab' : 'default'
        return
      }
      raycaster.setFromCamera(pointerNdc(event), camera)
      const intersection = new THREE.Vector3()
      if (!raycaster.ray.intersectPlane(drag.plane, intersection)) return
      const world = intersection.add(drag.offset)
      const bone = bonesRef.current.get(drag.name)
      const handle = handlesRef.current.get(drag.name)
      if (!bone || !handle) return
      bone.position.copy(toLocal(bone, world))
      bone.updateMatrixWorld(true)
      handle.position.copy(world)
      if (drag.mode === 'single') {
        // Rewrite each direct child's local position so its world position is
        // exactly where it was before the parent moved.
        for (const child of drag.pinnedChildren) child.bone.position.copy(toLocal(child.bone, child.world))
        bone.updateMatrixWorld(true)
      } else {
        // The bones already follow through the parent transform; the handles
        // are separate scene objects, so translate them by the same delta or
        // the fingers appear to stay behind.
        const delta = jointDelta([drag.startWorld.x, drag.startWorld.y, drag.startWorld.z], [world.x, world.y, world.z])
        const carried = translateJointPositions(drag.carriedStartWorld, delta)
        for (const [carriedName, position] of Object.entries(carried)) {
          handlesRef.current.get(carriedName)?.position.set(...position)
        }
      }
    }

    const onPointerUp = (event: PointerEvent) => {
      if (drag) {
        const bone = bonesRef.current.get(drag.name)
        const moved = bone ? bone.position.distanceTo(drag.startPosition) > 1e-6 : false
        // Every joint this one gesture wrote a local position for, undone as
        // a single step.
        const moves: JointMove[] = [
          { name: drag.name, position: [drag.startPosition.x, drag.startPosition.y, drag.startPosition.z] },
          ...drag.pinnedChildren.map((child) => ({
            name: child.bone.name,
            position: [child.startLocal.x, child.startLocal.y, child.startLocal.z] as [number, number, number],
          })),
        ]
        if (moved) {
          undoRef.current = pushJointUndoGroup(undoRef.current, moves)
          reportUndo()
        }
        for (const move of moves) settlePosition(move.name)
        syncHandles()
        controls.enabled = true
        drag = null
      }
      if (renderer.domElement.hasPointerCapture(event.pointerId)) {
        renderer.domElement.releasePointerCapture(event.pointerId)
      }
    }

    const onDoubleClick = (event: MouseEvent) => {
      const name = pickHandle(event)
      if (!name) return
      const bone = bonesRef.current.get(name)
      const original = originalsRef.current.get(name)
      if (!bone || !original) return
      undoRef.current = pushJointUndo(undoRef.current, name, [bone.position.x, bone.position.y, bone.position.z])
      reportUndo()
      bone.position.copy(original)
      bone.updateMatrixWorld(true)
      settlePosition(name)
      syncHandles()
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (isEditableFocus()) return
      if (event.key.toLowerCase() === 'z') {
        // Bare "z" (matching the on-screen hint) undoes the last completed
        // gesture — a finished drag or a reset — regardless of modifiers.
        event.preventDefault()
        undo()
        return
      }
      if (event.key.toLowerCase() === 'c') {
        // Same toggle as the control-bar switch, for hands already on the rig.
        event.preventDefault()
        setMoveWithChildren((current) => !current)
        return
      }
      if ((event.key === 'Delete' || event.key === 'Backspace') && selectedRef.current) {
        event.preventDefault()
        removeJointSingle(selectedRef.current)
      }
    }

    renderer.domElement.addEventListener('pointerdown', onPointerDown)
    renderer.domElement.addEventListener('pointermove', onPointerMove)
    renderer.domElement.addEventListener('pointerup', onPointerUp)
    renderer.domElement.addEventListener('dblclick', onDoubleClick)
    window.addEventListener('keydown', onKeyDown)

    const renderFrame = () => {
      controls.update()
      renderer.render(scene, camera)
      frame = requestAnimationFrame(renderFrame)
    }
    renderFrame()

    return () => {
      disposed = true
      cancelAnimationFrame(frame)
      observer.disconnect()
      controls.dispose()
      renderer.dispose()
      renderer.domElement.remove()
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.domElement.removeEventListener('pointermove', onPointerMove)
      renderer.domElement.removeEventListener('pointerup', onPointerUp)
      renderer.domElement.removeEventListener('dblclick', onDoubleClick)
      window.removeEventListener('keydown', onKeyDown)
      actionsRef.current = null
    }
  }, [artifactId, onAdjusted])

  // expose position + removal collection for the parent's Save action
  useEffect(() => {
    collectRef.current = () => {
      const positions: JointAdjustments = {}
      for (const [name, bone] of bonesRef.current) {
        if (removedRef.current.has(name)) continue
        const world = new THREE.Vector3().setFromMatrixPosition(bone.matrixWorld)
        positions[name] = blenderFromGltf(world)
      }
      return { positions, removeJoints: [...removedRef.current] }
    }
    return () => { collectRef.current = null }
  }, [collectRef])

  const removed = removedNames.includes(selected)
  const fingerCandidates = fingerJointNames(jointNames).filter((name) => !removedRef.current.has(name))
  const spurCandidates = spurJointNames(jointNames).filter((name) => !removedRef.current.has(name))

  const controls = <div className="rig-adjust-controls">
    <div className="rig-adjust-status">
      <strong>{jointCount - removedCount} joints</strong>
      <span>{adjustedCount} adjusted · {removedCount} removed</span>
      {selected
        ? <code>{selected}{selectedDescendants > 0 ? ` · ${selectedDescendants} children` : ' · leaf'}</code>
        : <span>No joint selected</span>}
    </div>
    <div className="rig-adjust-mode" role="group" aria-label="Joint drag mode">
      <button
        type="button"
        className={moveWithChildren ? 'selected' : ''}
        aria-pressed={moveWithChildren}
        onClick={() => setMoveWithChildren(true)}
      >Move with children</button>
      <button
        type="button"
        className={moveWithChildren ? '' : 'selected'}
        aria-pressed={!moveWithChildren}
        onClick={() => setMoveWithChildren(false)}
      >Move single joint</button>
    </div>
    <small className="rig-adjust-shortcuts">
      Hold Shift while dragging for the other mode · C toggles · Z undo · Delete removes the selected joint
    </small>
    <div className="rig-adjust-toolbar">
      <button type="button" className="secondary-button" onClick={() => actionsRef.current?.undo()} disabled={!canUndo}>Undo</button>
      <button type="button" className="secondary-button" onClick={() => actionsRef.current?.resetAll()} disabled={adjustedCount === 0 && removedCount === 0}>Reset all</button>
      {selected && (removed
        ? <button type="button" className="secondary-button" onClick={() => actionsRef.current?.restoreJoint(selected)}>Restore joint</button>
        : <>
            <button type="button" className="secondary-button" onClick={() => actionsRef.current?.removeJointSingle(selected)}>Remove joint</button>
            <button type="button" className="secondary-button" onClick={() => actionsRef.current?.removeJoints([selected])}>Remove joint + children</button>
          </>)}
      {fingerCandidates.length > 0 && <button type="button" className="secondary-button" onClick={() => actionsRef.current?.removeJoints(fingerCandidates)}>Remove all finger joints</button>}
      {spurCandidates.length > 0 && <button type="button" className="secondary-button" onClick={() => actionsRef.current?.removeJoints(spurCandidates)}>Remove spur joints</button>}
    </div>
  </div>

  return <section className="rig-adjust-viewport">
    <div className="rig-skeleton-canvas" ref={host} />
    <div className="rig-adjust-hint">
      {moveWithChildren ? 'Drag moves the joint + children · Shift = single joint' : 'Drag moves one joint · Shift = with children'}
      {' · mesh updates after Save & rebind'}
    </div>
    {controlsHost === undefined ? controls : controlsHost ? createPortal(controls, controlsHost) : null}
  </section>
}
