import { useFrame, useThree } from '@react-three/fiber'
import { useMemo } from 'react'
import { AvatarCore } from './AvatarCore'
import { AudioVisualizer } from './AudioVisualizer'
import { ParticleField } from './ParticleField'
import { StatusRings } from './StatusRings'
import { Scanlines } from './Scanlines'
import { HUDGrid } from './HUDGrid'
import { CornerBrackets } from './CornerBrackets'
import { useVegaStore } from '@stores/vegaStore'
import { useSettingsStore } from '@stores/settingsStore'

export function HolographicHUD() {
  const { hudMode, hudVisible } = useVegaStore()
  const { settings } = useSettingsStore()
  const { scene, camera, gl } = useThree()

  const hudElements = useMemo(() => {
    if (!hudVisible) return []

    const elements = []

    // Always show core elements
    elements.push(<AvatarCore key="avatar" />)
    elements.push(<AudioVisualizer key="audio" />)
    elements.push(<StatusRings key="status-rings" />)

    if (hudMode === 'full') {
      elements.push(<ParticleField key="particles" count={500} />)
      elements.push(<HUDGrid key="grid" />)
      elements.push(<CornerBrackets key="corners" />)
      elements.push(<Scanlines key="scanlines" />)
    }

    return elements
  }, [hudVisible, hudMode])

  // Animation loop for HUD elements
  useFrame((state, delta) => {
    // Global time for shaders
    if (state.clock) {
      // Global time uniform would be set here
    }
  })

  if (!hudVisible) return null

  return (
    <>
      {hudElements}
    </>
  )
}

// HUD Camera rig for orthographic overlay
export function HUDCamera() {
  const { camera } = useThree()

  const hudCamera = useMemo(() => {
    const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 1000)
    cam.position.z = 10
    return cam
  }, [])

  useFrame(() => {
    // HUD camera always faces forward
  }, 1)

  return null // Camera is used by HUD components via useThree
}

// HUD Layer for 2D overlay elements
export function HUDLayer({ children }: { children: React.ReactNode }) {
  return (
    <group>
      {children}
    </group>
  )
}

// Utility: World to screen coordinates
export function worldToScreen(vector: THREE.Vector3, camera: THREE.Camera, renderer: THREE.WebGLRenderer) {
  const vector2 = new THREE.Vector2()
  vector.project(camera)
  vector2.x = (vector.x * 0.5 + 0.5) * renderer.domElement.width
  vector2.y = (-(vector.y - 1) * 0.5) * renderer.domElement.height
  return vector2
}

// Utility: Screen to world coordinates
export function screenToWorld(x: number, y: number, camera: THREE.Camera, renderer: THREE.WebGLRenderer) {
  const mouse = new THREE.Vector2(
    (x / renderer.domElement.width) * 2 - 1,
    -(y / renderer.domElement.height) * 2 + 1
  )
  const ray = new THREE.Raycaster()
  ray.setFromCamera(new THREE.Vector2(x, y), camera)
  return ray
}