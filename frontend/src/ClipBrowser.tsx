import { useEffect, useMemo, useRef, useState } from 'react'
import { ChevronDown, ChevronUp, Search } from 'lucide-react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { curatedDonorActionName, type CuratedMotionClip } from './animations'
import { batchClipIds, pendingClipIds, poseTimeForClip, THUMBNAIL_RENDER_SIZE } from './clipGrid'

// The curated donor library (vendor/mesh2motion-app/static/animations/human-base-animations.glb)
// is Vite's publicDir (see frontend/vite.config.ts) so it is reachable same-origin, unauthenticated,
// at this path — verified with `curl http://127.0.0.1:5174/animations/human-base-animations.glb`.
// It carries both the template character mesh AND all 87 curated clips as its own embedded
// gltf.animations, so a single load here previews every clip with no backend round trip.
const DONOR_GLB_URL = '/animations/human-base-animations.glb'

// Cached per-clip thumbnail dataURLs, keyed by CuratedMotionClip.id, at
// MODULE scope (not component state) — ClipBrowser unmounts/remounts every
// time the "Attach motion" step leaves and re-enters view (e.g. an
// already-embedded clip becomes selected and the Attach section swaps out),
// so a component-scoped cache would re-render all 87 thumbnails on every
// reopen. An empty string marks a clip whose donor action could not be
// resolved (should not happen for a validated curated clip, but keeps a
// missing action from being retried every batch pass).
const thumbnailCache = new Map<string, string>()

const supportsIdleCallback = typeof requestIdleCallback === 'function'
function scheduleIdle(callback: () => void): number | ReturnType<typeof setTimeout> {
  return supportsIdleCallback ? requestIdleCallback(callback) : setTimeout(callback, 32)
}
function cancelIdle(handle: number | ReturnType<typeof setTimeout>) {
  if (supportsIdleCallback) cancelIdleCallback(handle as number)
  else clearTimeout(handle as ReturnType<typeof setTimeout>)
}

type Props = {
  clips: CuratedMotionClip[]
  selectedClipId: string
  onSelect: (clipId: string) => void
}

// A visual grid picker for the Animate panel's "Attach motion" section
// (operator: "I liked mesh2motion's unified grid of clips (visual, not just
// names) that makes it easier to pick an appropriate animation" — this
// replaces the old select + plain name list). Filter box on top, then a
// scrollable grid of per-clip pose thumbnails; hovering or focusing a cell
// swaps in a single live three.js preview absolutely positioned over that
// cell so there is exactly one WebGL context for the whole grid, never one
// per cell. Camera framing, lighting, and background mirror
// RigAdjustViewport.tsx's proven-working setup (rather than
// StudioViewport.tsx's WireframePreview, whose LineBasicMaterial wireframes
// are unlit and so were never a validated reference for a lit, textured
// mesh).
export function ClipBrowser({ clips, selectedClipId, onSelect }: Props) {
  const [collapsed, setCollapsed] = useState(false)
  const [query, setQuery] = useState('')
  const [hoveredClipId, setHoveredClipId] = useState<string | null>(null)
  const [donorReady, setDonorReady] = useState(false)
  const [donorError, setDonorError] = useState(false)
  // Bumped whenever a thumbnail batch finishes so cells re-read the module
  // -scope cache — the cache itself is a plain Map, not React state.
  const [, setThumbVersion] = useState(0)
  const gridWrap = useRef<HTMLDivElement>(null)
  const overlay = useRef<HTMLDivElement>(null)
  const thumbRefs = useRef<Map<string, HTMLElement>>(new Map())
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const mixerRef = useRef<THREE.AnimationMixer | null>(null)
  const actionsByNameRef = useRef<Map<string, THREE.AnimationAction>>(new Map())
  const collapsedRef = useRef(collapsed)
  collapsedRef.current = collapsed
  const hoveredClipIdRef = useRef<string | null>(null)
  hoveredClipIdRef.current = hoveredClipId
  // Set by the mount effect below so hover positioning can force a resize
  // immediately instead of trusting ResizeObserver to fire promptly on a
  // 0x0 -> sized transition.
  const resizeRef = useRef<(() => void) | null>(null)

  const filteredClips = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return clips
    return clips.filter((clip) => clip.label.toLowerCase().includes(needle))
  }, [clips, query])

  // Loads the donor GLB and builds the three.js scene/mixer exactly once —
  // the host div stays in the DOM (hidden via CSS, not unmounted) while
  // collapsed so re-expanding never re-fetches the 5.6MB donor file. Loading
  // /error status is rendered as an absolutely-positioned sibling instead of
  // JSX children of the canvas host, same as WireframePreview and
  // RigAdjustViewport's host divs, so React never has to reconcile children
  // around a canvas it doesn't know about.
  useEffect(() => {
    const element = overlay.current
    if (!element) return
    let disposed = false
    let frameHandle = 0
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0b0d11)
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    element.appendChild(renderer.domElement)
    sceneRef.current = scene
    cameraRef.current = camera
    rendererRef.current = renderer
    scene.add(new THREE.HemisphereLight(0xffffff, 0x20243a, 2.2))
    const key = new THREE.DirectionalLight(0xffffff, 2)
    key.position.set(3, 4, 5)
    scene.add(key)

    const resize = () => {
      const { width, height } = element.getBoundingClientRect()
      if (!width || !height) return
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height, false)
    }
    resizeRef.current = resize
    const observer = new ResizeObserver(resize)
    observer.observe(element)
    resize()

    new GLTFLoader().load(DONOR_GLB_URL, (gltf) => {
      if (disposed) return
      gltf.scene.updateMatrixWorld(true)
      const bounds = new THREE.Box3().setFromObject(gltf.scene)
      const size = bounds.getSize(new THREE.Vector3())
      const center = bounds.getCenter(new THREE.Vector3())
      // Stand the character at the origin, feet on the floor — bounds are
      // captured before this move, so size/center still describe the
      // pre-move geometry (matches the order RigAdjustViewport uses).
      gltf.scene.position.x -= center.x
      gltf.scene.position.z -= center.z
      gltf.scene.position.y -= bounds.min.y
      scene.add(gltf.scene)

      const radius = Math.max(size.length() * 0.65, 0.25)
      const targetHeight = size.y * 0.5
      camera.position.set(radius * 1.25, targetHeight + radius * 0.4, radius * 1.7)
      camera.near = Math.max(radius / 100, 0.001)
      camera.far = radius * 100
      camera.updateProjectionMatrix()
      camera.lookAt(0, targetHeight, 0)

      const mixer = new THREE.AnimationMixer(gltf.scene)
      mixerRef.current = mixer
      const actionsByName = new Map<string, THREE.AnimationAction>()
      for (const clip of gltf.animations) actionsByName.set(clip.name, mixer.clipAction(clip))
      actionsByNameRef.current = actionsByName
      setDonorReady(true)
    }, undefined, () => { if (!disposed) setDonorError(true) })

    const clock = new THREE.Clock()
    const render = () => {
      frameHandle = requestAnimationFrame(render)
      // Only the hovered/focused cell's clip actually plays — everything
      // else is a static cached thumbnail, so there is no reason to touch
      // the mixer or pay a render() cost while nothing is highlighted.
      // Deliberately NOT gated on document.visibilityState (see
      // StudioViewport.tsx's RetargetPreview postmortem for why) — this
      // panel is short-lived enough per hover that a hidden window simply
      // missing a few preview frames is an acceptable tradeoff, and it is
      // never the sole render loop this app depends on.
      if (collapsedRef.current || !hoveredClipIdRef.current) return
      mixerRef.current?.update(clock.getDelta())
      renderer.render(scene, camera)
    }
    render()

    return () => {
      disposed = true
      resizeRef.current = null
      rendererRef.current = null
      sceneRef.current = null
      cameraRef.current = null
      cancelAnimationFrame(frameHandle)
      observer.disconnect()
      mixerRef.current?.stopAllAction()
      mixerRef.current = null
      actionsByNameRef.current.clear()
      scene.traverse((child) => {
        if (!(child instanceof THREE.Mesh)) return
        child.geometry.dispose()
        const material = child.material
        if (Array.isArray(material)) material.forEach((entry) => entry.dispose())
        else material?.dispose()
      })
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [])

  // Renders each clip's characteristic pose into the module-scope thumbnail
  // cache, once the donor is loaded — sequenced in small batches via
  // requestIdleCallback (falling back to setTimeout) so 87 renders don't
  // jank the UI thread in one synchronous burst. Reuses the SAME
  // renderer/scene/camera as the live hover preview below (one WebGL
  // context total) rather than spinning up a second one.
  useEffect(() => {
    if (!donorReady) return
    const renderer = rendererRef.current
    const scene = sceneRef.current
    const camera = cameraRef.current
    const mixer = mixerRef.current
    if (!renderer || !scene || !camera || !mixer) return
    const pending = pendingClipIds(clips.map((clip) => clip.id), thumbnailCache)
    if (pending.length === 0) return
    const batches = batchClipIds(pending)
    const savedAspect = camera.aspect
    let cancelled = false
    let handle: number | ReturnType<typeof setTimeout> = 0
    let index = 0

    const runBatch = (batch: string[]) => {
      renderer.setSize(THUMBNAIL_RENDER_SIZE, THUMBNAIL_RENDER_SIZE, false)
      camera.aspect = 1
      camera.updateProjectionMatrix()
      for (const id of batch) {
        const clip = clips.find((candidate) => candidate.id === id)
        const action = clip ? actionsByNameRef.current.get(curatedDonorActionName(clip.animation)) : undefined
        mixer.stopAllAction()
        if (!clip || !action) { thumbnailCache.set(id, ''); continue }
        action.reset().play()
        action.time = poseTimeForClip(action.getClip().duration)
        mixer.update(0)
        renderer.render(scene, camera)
        thumbnailCache.set(id, renderer.domElement.toDataURL('image/png'))
      }
      mixer.stopAllAction()
    }

    const step = () => {
      if (cancelled) return
      const batch = batches[index]
      index += 1
      if (batch) runBatch(batch)
      setThumbVersion((value) => value + 1)
      if (index < batches.length) handle = scheduleIdle(step)
      else {
        camera.aspect = savedAspect
        camera.updateProjectionMatrix()
        resizeRef.current?.()
      }
    }
    handle = scheduleIdle(step)
    return () => { cancelled = true; cancelIdle(handle) }
  }, [donorReady, clips])

  // Forces an immediate resize whenever the live overlay is repositioned
  // over a newly hovered cell, rather than relying solely on ResizeObserver
  // to notice a 0x0 -> sized transition promptly.
  useEffect(() => {
    if (!collapsed) resizeRef.current?.()
  }, [collapsed])

  // Swaps the single live renderer to whichever clip is hovered/focused —
  // instant swap, no crossfade needed for a scrub-through-clips preview.
  // Nothing plays (mixer idle) while no cell is highlighted.
  useEffect(() => {
    const mixer = mixerRef.current
    if (!donorReady || !mixer) return
    mixer.stopAllAction()
    if (!hoveredClipId) return
    const clip = clips.find((candidate) => candidate.id === hoveredClipId)
    if (!clip) return
    const action = actionsByNameRef.current.get(curatedDonorActionName(clip.animation))
    if (!action) return
    action.reset().setLoop(THREE.LoopRepeat, Infinity).play()
  }, [donorReady, hoveredClipId, clips])

  const highlight = (clipId: string) => {
    setHoveredClipId(clipId)
    const overlayEl = overlay.current
    const thumbEl = thumbRefs.current.get(clipId)
    if (!overlayEl || !thumbEl) return
    overlayEl.style.left = `${thumbEl.offsetLeft}px`
    overlayEl.style.top = `${thumbEl.offsetTop}px`
    overlayEl.style.width = `${thumbEl.offsetWidth}px`
    overlayEl.style.height = `${thumbEl.offsetHeight}px`
    overlayEl.style.display = 'block'
    resizeRef.current?.()
  }
  const unhighlight = () => {
    setHoveredClipId(null)
    if (overlay.current) overlay.current.style.display = 'none'
  }

  return <div className="clip-browser">
    <button type="button" className="clip-browser-toggle" onClick={() => setCollapsed((value) => !value)}>
      {collapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
      {collapsed ? 'Show clip grid' : 'Hide clip grid'}
    </button>
    <div className="clip-browser-body" hidden={collapsed}>
      <label className="clip-browser-filter">
        <Search size={13} />
        <input type="text" placeholder="Filter clips…" value={query} onChange={(event) => setQuery(event.target.value)} />
      </label>
      {!donorReady && !donorError && <div className="clip-browser-status">Loading previews…</div>}
      {donorError && <div className="clip-browser-status">Could not load clip previews — pick by name below.</div>}
      <div className="clip-grid-wrap" ref={gridWrap}>
        <div className="clip-grid">
          {filteredClips.map((clip) => {
            const dataUrl = thumbnailCache.get(clip.id)
            const isSelected = clip.id === selectedClipId
            return <button
              key={clip.id}
              type="button"
              className={isSelected ? 'clip-grid-cell selected' : 'clip-grid-cell'}
              onMouseEnter={() => highlight(clip.id)}
              onMouseLeave={unhighlight}
              onFocus={() => highlight(clip.id)}
              onBlur={unhighlight}
              onClick={() => onSelect(clip.id)}
            >
              <span className="clip-grid-thumb" ref={(element) => {
                if (element) thumbRefs.current.set(clip.id, element)
                else thumbRefs.current.delete(clip.id)
              }}>
                {dataUrl ? <img src={dataUrl} alt="" /> : <span className="clip-grid-skeleton" />}
              </span>
              <span className="clip-grid-label">{clip.label}</span>
            </button>
          })}
        </div>
        <div className="clip-grid-hover-overlay" ref={overlay} />
        {filteredClips.length === 0 && <div className="clip-grid-empty">No clips match "{query}".</div>}
      </div>
    </div>
  </div>
}
