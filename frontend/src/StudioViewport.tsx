import { useEffect, useRef, useState } from 'react'
import { Box, Focus, Layers3, Pause, Play, Repeat2, ScanLine, Upload } from 'lucide-react'
import '@google/model-viewer'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { ModelViewerElement } from '@google/model-viewer'
import { readableAnimationName } from './animation'
import { curatedDonorActionName } from './animations'
import { API } from './api'
import { findSkinnedMesh, loadDonorLibrary, retargetDonorClip } from './retargetPreview'

type ViewMode = 'scene' | 'solid' | 'wireframe'
// The instant client-side clip preview (no server job): `animation` is the
// donor's exact action name (ClipBrowser.tsx/animations.ts's
// CuratedMotionClip.animation), `label` is only for the on-screen hint.
export type PreviewClip = { animation: string; label: string }
type Props = { artifactId?: string; artifactKind?: string; artifactName?: string; onImportModel?: () => void; onModelLoaded?: () => void; showAnimationControls?: boolean; previewClip?: PreviewClip | null }
function WireframePreview({ src, onReady, onError }: { src: string; onReady: () => void; onError: () => void }) {
  const host = useRef<HTMLDivElement>(null)
  const onReadyRef = useRef(onReady)
  const onErrorRef = useRef(onError)
  useEffect(() => { onReadyRef.current = onReady; onErrorRef.current = onError }, [onReady, onError])

  useEffect(() => {
    const element = host.current
    if (!element) return
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1000)
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x0b0d11, 1)
    element.appendChild(renderer.domElement)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.07
    controls.target.set(0, 0, 0)
    const key = new THREE.DirectionalLight(0xffffff, 1.5)
    key.position.set(3, 4, 5)
    scene.add(key, new THREE.HemisphereLight(0xd7ddff, 0x151925, 1.6))
    const grid = new THREE.GridHelper(8, 16, 0x293047, 0x1d2231)
    grid.position.y = -1.2
    scene.add(grid)
    let frame = 0
    let disposed = false
    const resize = () => {
      const { width, height } = element.getBoundingClientRect()
      if (!width || !height) return
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height, false)
    }
    const observer = new ResizeObserver(resize)
    observer.observe(element)
    resize()
    new GLTFLoader().load(src, (gltf) => {
      if (disposed) return
      const lines = new THREE.Group()
      gltf.scene.updateMatrixWorld(true)
      gltf.scene.traverse((node) => {
        if (!(node instanceof THREE.Mesh) || !node.geometry) return
        const wire = new THREE.WireframeGeometry(node.geometry)
        const material = new THREE.LineBasicMaterial({ color: 0xa98cff, transparent: true, opacity: 0.94 })
        const segment = new THREE.LineSegments(wire, material)
        node.updateWorldMatrix(true, false)
        segment.applyMatrix4(node.matrixWorld)
        lines.add(segment)
      })
      if (!lines.children.length) { onErrorRef.current(); return }
      const bounds = new THREE.Box3().setFromObject(lines)
      const size = bounds.getSize(new THREE.Vector3())
      const center = bounds.getCenter(new THREE.Vector3())
      lines.position.sub(center)
      const radius = Math.max(size.length() * 0.58, 0.25)
      camera.position.set(radius * 1.25, radius * 0.9, radius * 1.65)
      camera.near = Math.max(radius / 100, 0.001)
      camera.far = radius * 100
      camera.updateProjectionMatrix()
      controls.target.set(0, 0, 0)
      controls.update()
      scene.add(lines)
      onReadyRef.current()
    }, undefined, () => onErrorRef.current())
    const render = () => { controls.update(); renderer.render(scene, camera); frame = requestAnimationFrame(render) }
    render()
    return () => {
      disposed = true
      cancelAnimationFrame(frame)
      observer.disconnect()
      controls.dispose()
      scene.traverse((child) => {
        if (child instanceof THREE.LineSegments) { child.geometry.dispose(); (child.material as THREE.Material).dispose() }
      })
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [src])

  return <div className="wireframe-preview" ref={host} />
}

// requestAnimationFrame is spec-guaranteed to never fire at all while
// document.hidden is true (there is no compositor frame to synchronize
// with) -- not merely throttled, genuinely zero callbacks, for as long as
// the page stays hidden. A window can read as hidden here even while it is
// the one on screen (see ClipBrowser.tsx's black-canvas postmortem: this
// box's remote/automated sessions have shown document.hidden === true with
// the panel visibly on screen the whole time), so a RetargetPreview mounted
// during one of those windows would call requestAnimationFrame exactly
// once, get scheduled, and then simply never be invoked again -- the mixer
// stops advancing and the canvas is stuck on whatever pose it last drew,
// which is exactly the reported symptom (poses the character once, then
// frozen forever, three captures pixel-identical). Falling back to a timer
// while hidden keeps the mixer clock moving; it re-checks visibility on
// every tick so it switches back to real requestAnimationFrame the moment
// the page is visible again instead of timer-driving forever.
function scheduleFrame(callback: () => void): number | ReturnType<typeof setTimeout> {
  if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return setTimeout(callback, 16)
  return requestAnimationFrame(callback)
}
function cancelScheduledFrame(handle: number | ReturnType<typeof setTimeout>) {
  if (typeof handle === 'number') cancelAnimationFrame(handle)
  else clearTimeout(handle)
}

// Instant client-side clip preview: loads this artifact's own GLB (raw
// three.js, not <model-viewer> -- model-viewer has no public API to play an
// AnimationClip that was never part of the original file), retargets the
// requested donor clip onto its skeleton with SkeletonUtils (see
// retargetPreview.ts for the verified bone-name map and why the donor
// library is loaded through a shared cache), and plays it immediately. No
// server job, no rest-pose alignment -- rougher than a baked "Attach
// motion" clip by design, see the "Quick preview" hint rendered by the
// parent. Deliberately does NOT gate its render loop on
// document.visibilityState (see ClipBrowser.tsx's black-canvas postmortem:
// visibilityState goes "hidden" for a merely-unfocused window, which would
// permanently freeze this on a blank canvas) -- and uses scheduleFrame
// above instead of a bare requestAnimationFrame so a hidden window still
// advances the mixer instead of freezing on its first pose forever.
function RetargetPreview({ src, clipName, onReady, onError }: { src: string; clipName: string; onReady: () => void; onError: () => void }) {
  const host = useRef<HTMLDivElement>(null)
  const onReadyRef = useRef(onReady)
  const onErrorRef = useRef(onError)
  useEffect(() => { onReadyRef.current = onReady; onErrorRef.current = onError }, [onReady, onError])

  useEffect(() => {
    const element = host.current
    if (!element) return
    let disposed = false
    let frame: number | ReturnType<typeof setTimeout> = 0
    let mixer: THREE.AnimationMixer | null = null
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0b0d11)
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1000)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    element.appendChild(renderer.domElement)
    // Operator-reported bug: this canvas never wired up orbit/zoom at all
    // (unlike WireframePreview above, which has always had its own
    // OrbitControls) -- dragging/scrolling over a playing clip preview did
    // nothing because nothing was listening. Mirrors WireframePreview's
    // controls setup exactly (damping, target at the origin, `.update()`
    // every frame, disposed on teardown).
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.07
    controls.target.set(0, 0, 0)
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
    const observer = new ResizeObserver(resize)
    observer.observe(element)
    resize()

    Promise.all([new GLTFLoader().loadAsync(src), loadDonorLibrary()]).then(([targetGltf, donor]) => {
      if (disposed) return
      const targetSkinnedMesh = findSkinnedMesh(targetGltf.scene)
      // `clipName` (PreviewClip.animation) is the exact Blender-import
      // action name (the "_Armature"-suffixed form the server-side bake
      // uses) -- this donor library was loaded directly via GLTFLoader, so
      // its own gltf.animations names never carry that suffix (see
      // curatedDonorActionName's doc comment in animations.ts / the
      // backend/app/curated_motion.py module docstring for how that was
      // confirmed).
      const donorClip = donor.clipsByName.get(curatedDonorActionName(clipName))
      if (!targetSkinnedMesh || !donorClip) {
        console.error('[RetargetPreview] cannot preview: ', targetSkinnedMesh ? `no donor clip named ${clipName}` : 'target GLB has no SkinnedMesh')
        onErrorRef.current()
        return
      }
      const retargeted = retargetDonorClip(targetSkinnedMesh, donor.scene, donorClip)

      targetGltf.scene.updateMatrixWorld(true)
      const bounds = new THREE.Box3().setFromObject(targetGltf.scene)
      const size = bounds.getSize(new THREE.Vector3())
      const center = bounds.getCenter(new THREE.Vector3())
      const radius = Math.max(size.length() * 0.6, 0.25)
      camera.position.set(center.x + radius * 1.25, center.y + radius * 0.4, center.z + radius * 1.7)
      camera.near = Math.max(radius / 100, 0.001)
      camera.far = radius * 100
      camera.updateProjectionMatrix()
      camera.lookAt(center)
      controls.target.copy(center)
      controls.update()

      scene.add(targetGltf.scene)
      // Root the mixer on the SkinnedMesh itself, not the GLTF scene group.
      // retargetDonorClip's tracks use SkeletonUtils.retargetClip's own
      // `.bones[boneName].quaternion` naming (empty node name, bracketed
      // bone lookup) -- PropertyBinding only resolves that form against a
      // root with `.isSkinnedMesh && .skeleton`, so a Group/Scene root
      // can't bind it at all. Confirmed live: rooted on the scene, every
      // mapped bone's binding silently resolved to the scene root itself
      // instead of erroring, so mixer.update() kept advancing (mixer.time
      // and action.time both ticked correctly) while every bone's world
      // transform stayed pixel-identical forever -- the exact "poses once,
      // then frozen" symptom. Rooting on the SkinnedMesh (which owns
      // `.skeleton`) lets PropertyBinding find each named bone, and the
      // pose visibly advances every frame.
      mixer = new THREE.AnimationMixer(targetSkinnedMesh)
      mixer.clipAction(retargeted).setLoop(THREE.LoopRepeat, Infinity).play()
      onReadyRef.current()
    }, (error) => { console.error('[RetargetPreview] failed to load the donor library or target GLB:', error); if (!disposed) onErrorRef.current() })
      .catch((error) => { console.error('[RetargetPreview] failed to retarget the clip:', error); if (!disposed) onErrorRef.current() })

    const clock = new THREE.Clock()
    const render = () => {
      frame = scheduleFrame(render)
      controls.update()
      mixer?.update(clock.getDelta())
      renderer.render(scene, camera)
    }
    render()

    return () => {
      disposed = true
      cancelScheduledFrame(frame)
      observer.disconnect()
      controls.dispose()
      mixer?.stopAllAction()
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
  }, [src, clipName])

  return <div className="retarget-preview" ref={host} />
}

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return '0:00'
  const whole = Math.max(0, Math.floor(seconds))
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`
}

export default function StudioViewport({ artifactId, artifactKind, artifactName, onImportModel, onModelLoaded, showAnimationControls = false, previewClip = null }: Props) {
  const viewerRef = useRef<ModelViewerElement>(null)
  const [status, setStatus] = useState<'empty' | 'reference' | 'loading' | 'ready' | 'error'>(artifactKind === 'image' ? 'reference' : artifactId ? 'loading' : 'empty')
  const [mode, setMode] = useState<ViewMode>('scene')
  const [animations, setAnimations] = useState<string[]>([])
  const [selectedAnimation, setSelectedAnimation] = useState('')
  const [playing, setPlaying] = useState(false)
  const [loop, setLoop] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const playbackRef = useRef({ selectedAnimation, playing, currentTime, speed, loop })
  playbackRef.current = { selectedAnimation, playing, currentTime, speed, loop }
  const isModel = artifactKind === 'glb'
  const src = artifactId ? `${API}/assets/${artifactId}/download` : ''

  useEffect(() => {
    setMode('scene')
    setStatus(artifactKind === 'image' ? 'reference' : isModel ? 'loading' : 'empty')
    setAnimations([])
    setSelectedAnimation('')
    setPlaying(false)
    setCurrentTime(0)
    setDuration(0)
  }, [artifactId, artifactKind, isModel])

  // Starting (or clearing back to) a quick clip preview swaps which
  // component owns the canvas (RetargetPreview vs. <model-viewer>) --
  // either way that component's own load/error handling will resolve this
  // back to 'ready'/'error' shortly, same as a normal artifact switch.
  useEffect(() => {
    if (isModel) setStatus('loading')
  }, [previewClip, isModel])

  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || !isModel || mode === 'wireframe' || previewClip) return
    const ready = () => {
      if (mode === 'solid') {
        viewer.model?.materials.forEach((material) => {
          const pbr = material.pbrMetallicRoughness
          pbr.baseColorTexture?.setTexture(null)
          pbr.setBaseColorFactor('#8f78ff')
          pbr.setMetallicFactor(0.05)
          pbr.setRoughnessFactor(0.64)
        })
      }
      const available = [...viewer.availableAnimations]
      setAnimations(available)
      if (showAnimationControls && available.length) {
        const saved = playbackRef.current
        const selected = available.includes(saved.selectedAnimation) ? saved.selectedAnimation : available[0]
        const shouldPlay = !saved.selectedAnimation || saved.playing
        viewer.animationName = selected
        viewer.timeScale = saved.speed
        viewer.currentTime = saved.selectedAnimation ? saved.currentTime : 0
        if (shouldPlay) viewer.play({ repetitions: saved.loop ? Infinity : 1, pingpong: false })
        setSelectedAnimation(selected)
        setPlaying(shouldPlay)
        setCurrentTime(viewer.currentTime)
        setDuration(viewer.duration)
      }
      setStatus('ready')
      onModelLoaded?.()
    }
    const failed = () => setStatus('error')
    viewer.addEventListener('load', ready)
    viewer.addEventListener('error', failed)
    return () => { viewer.removeEventListener('load', ready); viewer.removeEventListener('error', failed) }
  }, [artifactId, isModel, mode, onModelLoaded, showAnimationControls])

  useEffect(() => {
    if (!playing || !showAnimationControls || status !== 'ready') return
    const timer = window.setInterval(() => {
      const viewer = viewerRef.current
      if (!viewer) return
      setCurrentTime(viewer.currentTime)
      setDuration(viewer.duration)
      if (viewer.paused) setPlaying(false)
    }, 100)
    return () => window.clearInterval(timer)
  }, [playing, showAnimationControls, status])

  const selectAnimation = (name: string) => {
    const viewer = viewerRef.current
    if (!viewer) return
    viewer.pause()
    viewer.animationName = name
    viewer.currentTime = 0
    viewer.timeScale = speed
    viewer.play({ repetitions: loop ? Infinity : 1, pingpong: false })
    setSelectedAnimation(name)
    setCurrentTime(0)
    setDuration(viewer.duration)
    setPlaying(true)
  }
  const togglePlayback = () => {
    const viewer = viewerRef.current
    if (!viewer || !selectedAnimation) return
    if (viewer.paused) {
      viewer.play({ repetitions: loop ? Infinity : 1, pingpong: false })
      setPlaying(true)
    } else {
      viewer.pause()
      setPlaying(false)
      setCurrentTime(viewer.currentTime)
    }
  }
  const toggleLoop = () => {
    const viewer = viewerRef.current
    const next = !loop
    setLoop(next)
    if (viewer && playing) viewer.play({ repetitions: next ? Infinity : 1, pingpong: false })
  }
  const changeSpeed = (next: number) => {
    setSpeed(next)
    if (viewerRef.current) viewerRef.current.timeScale = next
  }

  const changeMode = (next: ViewMode) => {
    if (next === mode) return
    setStatus('loading')
    setMode(next)
  }
  const resetCamera = () => {
    const viewer = viewerRef.current
    if (!viewer) return
    viewer.cameraOrbit = '35deg 75deg auto'
    viewer.fieldOfView = '30deg'
    viewer.jumpCameraToGoal()
  }
  const label = artifactName || (isModel ? 'Untitled model.glb' : 'Reference image')

  return <div className="viewport">
    {isModel && artifactId && previewClip && <RetargetPreview
      key={`${artifactId}-preview-${previewClip.animation}`}
      src={src}
      clipName={previewClip.animation}
      onReady={() => { setStatus('ready'); onModelLoaded?.() }}
      onError={() => setStatus('error')}
    />}
    {isModel && artifactId && !previewClip && mode !== 'wireframe' && <model-viewer
      key={`${artifactId}-${mode}`}
      ref={viewerRef}
      src={src}
      alt={label}
      camera-controls
      touch-action="pan-y"
      interaction-prompt="none"
      camera-orbit="35deg 75deg auto"
      field-of-view="30deg"
      min-camera-orbit="auto auto 60%"
      max-camera-orbit="auto auto 300%"
      environment-image="neutral"
      exposure="1.15"
      loading="eager"
      className="model-preview"
    />}
    {isModel && artifactId && !previewClip && mode === 'wireframe' && <WireframePreview src={src} onReady={() => { setStatus('ready'); onModelLoaded?.() }} onError={() => setStatus('error')} />}

    {isModel && !previewClip && <div className="viewer-toolbar" aria-label="Viewer controls">
      <button className={mode === 'scene' ? 'active' : ''} onClick={() => changeMode('scene')} title="Textured scene"><Layers3 size={15} /> Scene</button>
      <button className={mode === 'solid' ? 'active' : ''} onClick={() => changeMode('solid')} title="Single-material solid view"><Box size={15} /> Solid</button>
      <button className={mode === 'wireframe' ? 'active' : ''} onClick={() => changeMode('wireframe')} title="Mesh wireframe"><ScanLine size={15} /> Wireframe</button>
      <span className="viewer-divider" />
      <button onClick={resetCamera} title="Reset camera"><Focus size={15} /> Reset view</button>
      {onImportModel && <button onClick={onImportModel} title="Import a GLB from your computer"><Upload size={15} /> Import GLB</button>}
    </div>}
    {showAnimationControls && isModel && !previewClip && mode !== 'wireframe' && status === 'ready' && animations.length > 0 && <div className="animation-controls" aria-label="Animation controls">
      <div className="animation-controls-row">
        <button onClick={togglePlayback} aria-label={playing ? 'Pause animation' : 'Play animation'}>{playing ? <Pause size={16} /> : <Play size={16} fill="currentColor" />}</button>
        <select aria-label="Animation" value={selectedAnimation} onChange={(event) => selectAnimation(event.target.value)}>{animations.map((name) => <option value={name} key={name}>{readableAnimationName(name)}</option>)}</select>
        <button className={loop ? 'active' : ''} onClick={toggleLoop} aria-pressed={loop} title="Loop animation"><Repeat2 size={16} /> Loop</button>
        <label>Speed <select aria-label="Playback speed" value={speed} onChange={(event) => changeSpeed(Number(event.target.value))}><option value="0.5">0.5×</option><option value="1">1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></label>
      </div>
      <div className="animation-timeline"><span>{formatTime(currentTime)}</span><input aria-label="Animation timeline" type="range" min="0" max={Math.max(duration, 0.01)} step="0.01" value={Math.min(currentTime, Math.max(duration, 0.01))} onChange={(event) => { const next = Number(event.target.value); if (viewerRef.current) viewerRef.current.currentTime = next; setCurrentTime(next) }} /><span>{formatTime(duration)}</span></div>
    </div>}
    {showAnimationControls && isModel && !previewClip && mode !== 'wireframe' && status === 'ready' && animations.length === 0 && <div className="animation-empty">No embedded animations in this GLB.</div>}
    {!isModel && onImportModel && <button className="viewport-import" onClick={onImportModel}><Upload size={15} /> Import GLB</button>}
    {isModel && <div className="asset-chip"><Box size={14} /> <span>{previewClip ? `Quick preview: ${previewClip.label}` : label}</span>{status === 'loading' && <em>Loading…</em>}</div>}
    {/* Operator follow-up: the disclaimer used to be bottom fine print
        (easy to miss); it now overlays the viewport as a loud badge right
        under the asset chip whenever a clip picker's instant client-side
        preview owns the canvas, so it can't be mistaken for the final baked
        "Attach motion" result. */}
    {isModel && previewClip && <div className="preview-rough-badge">ROUGH PREVIEW — bake for final quality</div>}
    {status === 'reference' && artifactId && <img className="reference-preview" src={src} alt="Uploaded reference" />}
    {status === 'empty' && <div className="viewport-empty">Generate or import a GLB to preview it here.</div>}
    {status === 'error' && previewClip && <div className="viewport-empty">Could not preview this clip on the current rig.</div>}
    {status === 'error' && !previewClip && <div className="viewport-empty">Could not load this GLB. Try exporting it again or import another file.</div>}
    {status === 'ready' && !previewClip && <div className="viewport-hint">Drag to orbit · right-drag to pan · scroll to zoom</div>}
  </div>
}
