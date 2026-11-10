import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { API } from './api'

export default function RigSkeletonViewport({ artifactId }: { artifactId: string }) {
  const host = useRef<HTMLDivElement>(null)
  const [jointCount, setJointCount] = useState(0)
  useEffect(() => {
    const element = host.current
    if (!element) return
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
    const resize = () => { const { width, height } = element.getBoundingClientRect(); if (!width || !height) return; camera.aspect = width / height; camera.updateProjectionMatrix(); renderer.setSize(width, height, false) }
    const observer = new ResizeObserver(resize)
    observer.observe(element)
    resize()
    new GLTFLoader().load(`${API}/assets/${artifactId}/download`, (gltf) => {
      if (disposed) return
      const bones = new Set<THREE.Bone>()
      gltf.scene.traverse((node) => { if (node instanceof THREE.SkinnedMesh) node.skeleton.bones.forEach((bone) => bones.add(bone)) })
      const helper = new THREE.SkeletonHelper(gltf.scene)
      const material = helper.material as THREE.LineBasicMaterial
      material.color.set(0xa98cff)
      material.depthTest = false
      material.transparent = true
      material.opacity = 0.95
      helper.renderOrder = 10
      gltf.scene.add(helper)
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
      setJointCount(bones.size)
    })
    const render = () => { controls.update(); renderer.render(scene, camera); frame = requestAnimationFrame(render) }
    render()
    return () => { disposed = true; cancelAnimationFrame(frame); observer.disconnect(); controls.dispose(); renderer.dispose(); renderer.domElement.remove() }
  }, [artifactId])
  return <section className="rig-skeleton-inspector"><div className="mesh2motion-label">Applied skeleton · {jointCount || 'loading'} joints · drag to orbit</div><div className="rig-skeleton-canvas" ref={host} /></section>
}
