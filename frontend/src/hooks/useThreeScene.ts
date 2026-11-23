import { useRef, useEffect, useCallback, useState } from 'react'
import * as THREE from 'three'
import { useVegaStore } from '@stores/vegaStore'
import { useSettingsStore } from '@stores/settingsStore'

interface UseThreeSceneReturn {
  sceneRef: React.RefObject<THREE.Scene>
  cameraRef: React.RefObject<THREE.PerspectiveCamera>
  rendererRef: React.RefObject<THREE.WebGLRenderer>
  composerRef: React.RefObject<any>
  initScene: (container: HTMLDivElement) => void
  resize: (width: number, height: number) => void
  dispose: () => void
  getCameraPosition: () => THREE.Vector3
  setCameraPosition: (position: THREE.Vector3) => void
  raycast: (x: number, y: number) => THREE.Intersection[]
}

export function useThreeScene(): UseThreeSceneReturn {
  const { settings } = useSettingsStore()
  const { hudMode, hudVisible } = useVegaStore()
  const [isInitialized, setIsInitialized] = useState(false)

  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const composerRef = useRef<any>(null)
  const animationFrameRef = useRef<number | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)

  const sceneConfig = {
    backgroundColor: 0x0a0a0f,
    fogColor: 0x0a0a0f,
    fogNear: 1,
    fogFar: 100,
    cameraFov: 60,
    cameraNear: 0.1,
    cameraFar: 1000,
  }

  const initScene = useCallback((container: HTMLDivElement) => {
    if (isInitialized) return

    containerRef.current = container
    const width = container.clientWidth
    const height = container.clientHeight

    // Create scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(sceneConfig.backgroundColor)
    scene.fog = new THREE.Fog(sceneConfig.fogColor, sceneConfig.fogNear, sceneConfig.fogFar)
    sceneRef.current = scene

    // Create camera
    const camera = new THREE.PerspectiveCamera(
      sceneConfig.cameraFov,
      width / height,
      sceneConfig.cameraNear,
      sceneConfig.cameraFar
    )
    camera.position.set(0, 0, 5)
    camera.lookAt(0, 0, 0)
    cameraRef.current = camera

    // Create renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(sceneConfig.backgroundColor, 0)
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    rendererRef.current = renderer

    container.appendChild(renderer.domElement)

    // Post-processing
    import('@react-three/postprocessing').then(({ EffectComposer, RenderPass, BloomEffect, ChromaticAberrationEffect, VignetteEffect, NoiseEffect }) => {
      const composer = new EffectComposer(renderer)
      composer.addPass(new RenderPass(scene, camera))

      // Bloom
      const bloomEffect = new BloomEffect({
        intensity: 0.8,
        mipmapBlur: true,
        luminanceThreshold: 0.8,
        luminanceSmoothing: 0.02,
      })
      composer.addPass(bloomEffect)

      // Chromatic aberration (subtle)
      const chromaticEffect = new ChromaticAberrationEffect({ offset: new THREE.Vector2(0.001, 0.001) })
      composer.addPass(chromaticEffect)

      // Vignette
      const vignetteEffect = new VignetteEffect({
        eskil: 0.3,
        darkness: 0.4,
      })
      composer.addPass(vignetteEffect)

      // Noise
      const noiseEffect = new NoiseEffect({
        premultiply: true,
        opacity: 0.02,
      })
      composer.addPass(noiseEffect)

      composerRef.current = composer
    }).catch(() => {
      console.warn('[Three] Post-processing not available')
    })

    // Add basic lighting
    const ambientLight = new THREE.AmbientLight(0x404060, 0.5)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0x00d4ff, 1)
    directionalLight.position.set(5, 10, 5)
    directionalLight.castShadow = true
    directionalLight.shadow.mapSize.width = 2048
    directionalLight.shadow.mapSize.height = 2048
    scene.add(directionalLight)

    const pointLight = new THREE.PointLight(0xffd700, 1, 50)
    pointLight.position.set(-5, 5, -5)
    scene.add(pointLight)

    // Start render loop
    let lastTime = 0
    const animate = (time: number) => {
      animationFrameRef.current = requestAnimationFrame(animate)

      const delta = (time - lastTime) / 1000
      lastTime = time

      // Update scene animations here
      // This is where you'd update HUD elements, particles, etc.

      // Render
      if (composerRef.current) {
        composerRef.current.render(delta)
      } else {
        renderer.render(scene, camera)
      }
    }

    animationFrameRef.current = requestAnimationFrame(animate)
    setIsInitialized(true)

    console.log('[Three] Scene initialized')
  }, [])

  const resize = useCallback((width: number, height: number) => {
    if (!cameraRef.current || !rendererRef.current) return

    cameraRef.current.aspect = width / height
    cameraRef.current.updateProjectionMatrix()
    rendererRef.current.setSize(width, height)
    rendererRef.current.setPixelRatio(Math.min(window.devicePixelRatio, 2))

    if (composerRef.current) {
      composerRef.current.setSize(width, height)
    }
  }, [])

  const dispose = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }

    if (rendererRef.current) {
      rendererRef.current.dispose()
      rendererRef.current.forceContextLoss()
      rendererRef.current.domElement.remove()
      rendererRef.current = null
    }

    if (composerRef.current) {
      composerRef.current.dispose()
      composerRef.current = null
    }

    if (sceneRef.current) {
      sceneRef.current.traverse((object) => {
        if (object.geometry) object.geometry.dispose()
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach(m => m.dispose())
          } else {
            object.material.dispose()
          }
        }
      })
      sceneRef.current = null
    }

    cameraRef.current = null
    containerRef.current = null
    setIsInitialized(false)
  }, [])

  const getCameraPosition = useCallback(() => {
    return cameraRef.current?.position.clone() || new THREE.Vector3()
  }, [])

  const setCameraPosition = useCallback((position: THREE.Vector3) => {
    if (cameraRef.current) {
      cameraRef.current.position.copy(position)
      cameraRef.current.lookAt(0, 0, 0)
    }
  }, [])

  const raycast = useCallback((x: number, y: number) => {
    if (!cameraRef.current || !sceneRef.current || !rendererRef.current) return []

    const rect = rendererRef.current.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((x - rect.left) / rect.width) * 2 - 1,
      -((y - rect.top) / rect.height) * 2 + 1
    )

    const raycaster = new THREE.Raycaster()
    raycaster.setFromCamera(mouse, cameraRef.current)

    return raycaster.intersectObjects(sceneRef.current.children, true)
  }, [])

  return {
    sceneRef: sceneRef as any,
    cameraRef: cameraRef as any,
    rendererRef: rendererRef as any,
    composerRef: composerRef as any,
    initScene,
    resize,
    dispose,
    getCameraPosition,
    setCameraPosition,
    raycast,
  }
}