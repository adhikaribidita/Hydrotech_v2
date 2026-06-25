import React, { Suspense, useRef, useEffect, useMemo, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { useTexture, Html } from '@react-three/drei'
import * as THREE from 'three'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import { useStore } from '../store/useStore'
import gsap from 'gsap'
import { Globe } from 'lucide-react'
import Water from './Water'
import Particles from './Particles'

type Props = { onClick?: () => void }

// Custom Fresnel atmosphere shader to simulate realistic earth atmospheric glow
const AtmosphereShader = {
  vertexShader: `
    varying vec3 vNormal;
    varying vec3 vViewPosition;
    void main() {
      vNormal = normalize(normalMatrix * normal);
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      vViewPosition = -mvPosition.xyz;
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  fragmentShader: `
    varying vec3 vNormal;
    varying vec3 vViewPosition;
    uniform vec3 uColor;
    uniform float uIntensityMultiplier;
    void main() {
      // Glow is brightest at the outer edges of the sphere silhouette relative to view vector
      vec3 normal = normalize(vNormal);
      vec3 viewDir = normalize(vViewPosition);
      float intensity = pow(max(0.0, 0.75 - dot(normal, viewDir)), 3.5);
      gl_FragColor = vec4(uColor, 1.0) * intensity * uIntensityMultiplier;
    }
  `
}

// Custom Satellite Component (Central Metallic Body + Blue Solar Panels)
function Satellite({ scale = 0.05, bodyColor = 0xffd700, panelColor = 0x00aaff, ...props }) {
  const groupRef = useRef<THREE.Group>(null)

  useFrame(({ clock }) => {
    if (groupRef.current) {
      // Rotate the satellite body gently
      groupRef.current.rotation.y = clock.getElapsedTime() * 0.8
    }
  })

  return (
    <group ref={groupRef} scale={[scale, scale, scale]} {...props}>
      {/* Central satellite core body */}
      <mesh>
        <boxGeometry args={[1, 1, 1.5]} />
        <meshStandardMaterial metalness={0.9} roughness={0.1} color={bodyColor} />
      </mesh>
      
      {/* Gold foil dishes or sensors */}
      <mesh position={[0, 0, 0.9]}>
        <cylinderGeometry args={[0.3, 0.3, 0.2, 16]} rotation={[Math.PI / 2, 0, 0]} />
        <meshStandardMaterial color={0xffaa00} metalness={0.9} />
      </mesh>

      {/* Solar Panel Array Left */}
      <mesh position={[-1.6, 0, 0]}>
        <boxGeometry args={[2, 0.05, 0.8]} />
        <meshStandardMaterial color={panelColor} roughness={0.2} metalness={0.8} emissive={panelColor} emissiveIntensity={0.2} />
      </mesh>
      {/* Solar Panel Connectors Left */}
      <mesh position={[-0.6, 0, 0]}>
        <cylinderGeometry args={[0.06, 0.06, 0.4, 8]} rotation={[0, 0, Math.PI / 2]} />
        <meshStandardMaterial color={0x888888} metalness={0.9} />
      </mesh>

      {/* Solar Panel Array Right */}
      <mesh position={[1.6, 0, 0]}>
        <boxGeometry args={[2, 0.05, 0.8]} />
        <meshStandardMaterial color={panelColor} roughness={0.2} metalness={0.8} emissive={panelColor} emissiveIntensity={0.2} />
      </mesh>
      {/* Solar Panel Connectors Right */}
      <mesh position={[0.6, 0, 0]}>
        <cylinderGeometry args={[0.06, 0.06, 0.4, 8]} rotation={[0, 0, Math.PI / 2]} />
        <meshStandardMaterial color={0x888888} metalness={0.9} />
      </mesh>
    </group>
  )
}

// Starfield Component for Twinkling Background
function Starfield() {
  const pointsRef = useRef<THREE.Points>(null!)
  const count = 300

  const [positions, sizes] = useMemo(() => {
    const pos = new Float32Array(count * 3)
    const szs = new Float32Array(count)
    for (let i = 0; i < count; i++) {
      // Position stars in a large sphere shell far away
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos((Math.random() * 2) - 1)
      const r = 25.0 + Math.random() * 15.0
      pos[i * 3 + 0] = r * Math.sin(phi) * Math.cos(theta)
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      pos[i * 3 + 2] = r * Math.cos(phi)

      szs[i] = Math.random() * 0.15 + 0.05
    }
    return [pos, szs]
  }, [])

  useFrame(({ clock }) => {
    if (pointsRef.current) {
      const time = clock.getElapsedTime()
      // Twinkle rotation
      pointsRef.current.rotation.y = time * 0.005
      pointsRef.current.rotation.x = time * 0.002
    }
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          array={positions}
          itemSize={3}
          count={positions.length / 3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.15}
        color={0xffffff}
        transparent
        opacity={0.65}
        depthWrite={false}
        sizeAttenuation={true}
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}

// Cinematic Space Dust Particles orbiting the Earth
function DustParticles() {
  const pointsRef = useRef<THREE.Points>(null!)
  const earthHovered = useStore((s) => s.earthHovered)
  const earthClicked = useStore((s) => s.earthClicked)
  const count = 800

  const { positions, velocities } = useMemo(() => {
    const pos = new Float32Array(count * 3)
    const vels = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      // Dust rings or orbits surrounding Earth
      const theta = Math.random() * Math.PI * 2
      const radius = 2.2 + Math.random() * 3.5
      pos[i * 3 + 0] = Math.cos(theta) * radius
      pos[i * 3 + 1] = (Math.random() - 0.5) * 1.5
      pos[i * 3 + 2] = Math.sin(theta) * radius

      // Orbital velocities (perpendicular to radius)
      vels[i * 3 + 0] = -Math.sin(theta) * (0.05 + Math.random() * 0.05)
      vels[i * 3 + 1] = (Math.random() - 0.5) * 0.02
      vels[i * 3 + 2] = Math.cos(theta) * (0.05 + Math.random() * 0.05)
    }
    return { positions: pos, velocities: vels }
  }, [])

  const explosionScale = useRef({ value: 1.0 })

  useEffect(() => {
    if (earthClicked) {
      gsap.to(explosionScale.current, {
        value: 15.0,
        duration: 2.2,
        ease: 'power3.out',
      })
    } else {
      explosionScale.current.value = 1.0
    }
  }, [earthClicked])

  useFrame((state, delta) => {
    if (!pointsRef.current) return
    const time = state.clock.getElapsedTime()
    const rotSpeed = earthHovered ? 0.08 : 0.025
    
    // Ambient spin
    pointsRef.current.rotation.y += rotSpeed * delta

    const posAttr = pointsRef.current.geometry.attributes.position
    const posArr = posAttr.array as Float32Array

    for (let i = 0; i < count; i++) {
      const idx = i * 3
      
      // Floating motion
      const floatVal = Math.sin(time * 0.5 + i) * 0.001
      
      // Apply explosion scaling factor
      posArr[idx + 0] = (positions[idx + 0] + floatVal) * explosionScale.current.value
      posArr[idx + 1] = (positions[idx + 1] + floatVal) * explosionScale.current.value
      posArr[idx + 2] = (positions[idx + 2] + floatVal) * explosionScale.current.value
    }
    posAttr.needsUpdate = true
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          array={positions}
          itemSize={3}
          count={positions.length / 3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.035}
        color={earthHovered ? 0x65f3ff : 0x00c8ff}
        transparent
        opacity={0.75}
        depthWrite={false}
        sizeAttenuation={true}
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}

// Cinematic Camera Controller Component
function CinematicCamera() {
  const { camera } = useThree()
  const earthHovered = useStore((s) => s.earthHovered)
  const earthClicked = useStore((s) => s.earthClicked)

  useFrame((state) => {
    const time = state.clock.getElapsedTime()
    
    if (earthClicked) {
      // Fly-by zoom animation towards the center globe
      camera.position.x = THREE.MathUtils.lerp(camera.position.x, 0.0, 0.06)
      camera.position.y = THREE.MathUtils.lerp(camera.position.y, 0.0, 0.06)
      camera.position.z = THREE.MathUtils.lerp(camera.position.z, 1.25, 0.06)
      camera.lookAt(0, 0, 0)
    } else {
      // Gentle floating orbital path (never static)
      // Radius of the camera orbit
      const orbitRadius = 4.2
      // Passive slow angle change
      const angle = time * 0.045
      
      // Target camera position combines orbital drift + mouse parallax
      const targetX = Math.sin(angle) * orbitRadius + state.pointer.x * 0.55
      const targetZ = Math.cos(angle) * orbitRadius + state.pointer.x * 0.55
      const targetY = Math.sin(time * 0.03) * 0.65 + state.pointer.y * 0.45
      
      // Interpolate for movie-like camera stability
      camera.position.x = THREE.MathUtils.lerp(camera.position.x, targetX, 0.04)
      camera.position.y = THREE.MathUtils.lerp(camera.position.y, targetY, 0.04)
      camera.position.z = THREE.MathUtils.lerp(camera.position.z, targetZ, 0.04)
      camera.lookAt(0, 0, 0)
    }
  })

  return null
}

function EarthMesh({ onClick }: Props) {
  const earthRef = useRef<THREE.Mesh>(null!)
  const cloudsRef = useRef<THREE.Mesh>(null!)
  const atmosphereRef = useRef<THREE.ShaderMaterial>(null!)
  const energyRingRef1 = useRef<THREE.Mesh>(null!)
  const energyRingRef2 = useRef<THREE.Mesh>(null!)
  const energyRingRef3 = useRef<THREE.Mesh>(null!)
  const sat1Ref = useRef<THREE.Group>(null!)
  const sat2Ref = useRef<THREE.Group>(null!)
  const floatingGroupRef = useRef<THREE.Group>(null!)

  const earthHovered = useStore((s) => s.earthHovered)
  const earthClicked = useStore((s) => s.earthClicked)
  const setEarthHovered = useStore((s) => s.setEarthHovered)

  // Drag interaction states (using refs for performance, avoiding Canvas re-renders)
  const isDragging = useRef(false)
  const previousPointerPosition = useRef({ x: 0, y: 0 })
  const rotationVelocity = useRef({ x: 0.0, y: 0.004 }) // slow start rotation

  // Click-to-Inundate Shockwave ripples state
  interface Shockwave {
    id: number
    position: THREE.Vector3
    quaternion: THREE.Quaternion
    scale: number
    opacity: number
  }
  const [shockwaves, setShockwaves] = useState<Shockwave[]>([])

  // Load high-resolution earth textures using ThreeJS texture loaders
  const [colorMap, normalMap, specMap, cloudMap, lightsMap] = useTexture([
    '/earthmap1k.jpg',
    '/earth_normal_map.png',
    '/earthspec1k.jpg',
    '/earthcloudmap.jpg',
    '/earth_lights_2048.png'
  ])

  // GSAP animation triggers for hover responses
  useEffect(() => {
    if (atmosphereRef.current) {
      gsap.to(atmosphereRef.current.uniforms.uIntensityMultiplier, {
        value: earthHovered ? 2.6 : 1.4,
        duration: 0.6,
        ease: 'power2.out'
      })
    }
  }, [earthHovered])

  // Pointer drag event handlers
  const handlePointerDown = (e: any) => {
    e.stopPropagation()
    isDragging.current = true
    previousPointerPosition.current = { x: e.clientX, y: e.clientY }
    if (e.target && 'setPointerCapture' in e.target) {
      try {
        e.target.setPointerCapture(e.pointerId)
      } catch (err) {}
    }
  }

  const handlePointerUp = (e: any) => {
    e.stopPropagation()
    isDragging.current = false
    if (e.target && 'releasePointerCapture' in e.target) {
      try {
        e.target.releasePointerCapture(e.pointerId)
      } catch (err) {}
    }
  }

  const handlePointerMove = (e: any) => {
    if (!isDragging.current) return
    e.stopPropagation()
    const deltaX = e.clientX - previousPointerPosition.current.x
    const deltaY = e.clientY - previousPointerPosition.current.y
    
    // Accumulate velocity based on delta movement (Y drag -> rotates Y, X drag -> rotates X)
    rotationVelocity.current.y += deltaX * 0.002
    rotationVelocity.current.x += deltaY * 0.002
    
    previousPointerPosition.current = { x: e.clientX, y: e.clientY }
  }

  const handleEarthClick = (e: any) => {
    e.stopPropagation()
    if (e.point && earthRef.current) {
      const id = Math.random()
      
      // Convert clicked vector to the Earth mesh local coordinate space so the ripple rotates WITH the Earth
      const localPoint = e.point.clone()
      earthRef.current.worldToLocal(localPoint)
      
      // Calculate normal vector at intersection point (on unit sphere shape)
      const normal = localPoint.clone().normalize()
      
      // Align 2D ring facing vector (0, 0, 1) with surface normal
      const quaternion = new THREE.Quaternion()
      quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal)
      
      setShockwaves((prev) => [...prev, { id, position: localPoint, quaternion, scale: 0.1, opacity: 0.9 }])
    }
    
    if (onClick) onClick()
  }

  useFrame((state, delta) => {
    if (!earthRef.current || !cloudsRef.current) return
    const time = state.clock.getElapsedTime()

    // 1. Floating weightless space flight motion
    if (floatingGroupRef.current) {
      floatingGroupRef.current.position.y = Math.sin(time * 0.45) * 0.12
      floatingGroupRef.current.position.x = Math.cos(time * 0.35) * 0.08
      floatingGroupRef.current.position.z = Math.sin(time * 0.25) * 0.05
    }

    // 2. Continuous rotation & dragging physics
    if (isDragging.current) {
      earthRef.current.rotation.y += rotationVelocity.current.y
      earthRef.current.rotation.x += rotationVelocity.current.x
      // Damp rotation velocity quickly during dragging to follow cursor
      rotationVelocity.current.y *= 0.5
      rotationVelocity.current.x *= 0.5
    } else {
      // Natural slow rotation drift (spin speed is increased on hover)
      const baseSpeed = earthHovered ? 0.045 : 0.02
      earthRef.current.rotation.y += baseSpeed * delta + rotationVelocity.current.y
      earthRef.current.rotation.x += rotationVelocity.current.x
      
      // Momentum decay friction (simulates heavy inertial spin)
      rotationVelocity.current.y *= 0.94
      rotationVelocity.current.x *= 0.94
    }

    // Rotate clouds at a slightly offset volumetric rate
    cloudsRef.current.rotation.y = earthRef.current.rotation.y * 1.05 + time * 0.008
    cloudsRef.current.rotation.x = earthRef.current.rotation.x

    // 3. Multi-axis energy ring scanner rotation
    const rotSpeedFactor = earthHovered ? 1.6 : 1.0
    if (energyRingRef1.current) {
      energyRingRef1.current.rotation.z = time * 0.25 * rotSpeedFactor
      energyRingRef1.current.rotation.y = Math.sin(time * 0.1) * 0.2
    }
    if (energyRingRef2.current) {
      energyRingRef2.current.rotation.z = -time * 0.18 * rotSpeedFactor
      energyRingRef2.current.rotation.x = Math.cos(time * 0.1) * 0.15
    }
    if (energyRingRef3.current) {
      energyRingRef3.current.rotation.y = time * 0.12 * rotSpeedFactor
      energyRingRef3.current.rotation.z = Math.sin(time * 0.15) * 0.25
    }

    // 4. Orbiting satellite telemetry trackers
    const orbitSpeed = earthHovered ? 0.52 : 0.28
    if (sat1Ref.current) {
      const angle = time * orbitSpeed
      sat1Ref.current.position.set(Math.cos(angle) * 2.3, Math.sin(angle) * 0.45, Math.sin(angle) * 2.3)
      sat1Ref.current.lookAt(0, 0, 0)
    }
    if (sat2Ref.current) {
      const angle = time * orbitSpeed * 0.8 + 3.14
      sat2Ref.current.position.set(Math.sin(angle) * 2.7, Math.cos(angle) * 2.7 * 0.35, Math.cos(angle) * 2.7)
      sat2Ref.current.lookAt(0, 0, 0)
    }

    // 5. Click ripple shockwaves animation
    if (shockwaves.length > 0) {
      setShockwaves((prev) =>
        prev
          .map((wave) => ({
            ...wave,
            scale: wave.scale + 0.08,
            opacity: wave.opacity - 0.03
          }))
          .filter((wave) => wave.opacity > 0)
      )
    }
  })

  return (
    <group ref={floatingGroupRef}>
      {/* 3D Earth mesh with maps & realistic lighting parameters */}
      <mesh
        ref={earthRef}
        onClick={handleEarthClick}
        onPointerOver={() => setEarthHovered(true)}
        onPointerOut={() => setEarthHovered(false)}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onPointerMove={handlePointerMove}
      >
        <sphereGeometry args={[1.5, 64, 64]} />
        <meshStandardMaterial
          map={colorMap}
          normalMap={normalMap}
          normalScale={new THREE.Vector2(1.2, 1.2)}
          roughnessMap={specMap}
          metalness={0.12}
          roughness={0.65}
          emissiveMap={lightsMap}
          emissive={new THREE.Color('#ffde9e')}
          emissiveIntensity={earthHovered ? 2.5 : 1.4}
        />

        {/* Dynamic click shockwave ripples */}
        {shockwaves.map((wave) => (
          <mesh
            key={wave.id}
            position={wave.position}
            quaternion={wave.quaternion}
            scale={[wave.scale, wave.scale, wave.scale]}
          >
            <ringGeometry args={[0, 0.45, 64]} />
            <meshBasicMaterial
              color={0x00e5ff}
              transparent
              opacity={wave.opacity}
              side={THREE.DoubleSide}
              depthWrite={false}
              blending={THREE.AdditiveBlending}
            />
          </mesh>
        ))}
      </mesh>

      {/* High-Tech Glowing Wireframe Overlay */}
      <mesh>
        <sphereGeometry args={[1.503, 32, 32]} />
        <meshBasicMaterial 
          color={0x00c8ff} 
          wireframe={true} 
          transparent={true} 
          opacity={0.08} 
          blending={THREE.AdditiveBlending} 
        />
      </mesh>

      {/* Cloud Layer Mesh */}
      <mesh ref={cloudsRef}>
        <sphereGeometry args={[1.515, 64, 64]} />
        <meshPhongMaterial
          map={cloudMap}
          transparent={true}
          opacity={0.65} // Increased cloud opacity for dramatic effect (Image 2)
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Atmosphere glow shell */}
      <mesh>
        <sphereGeometry args={[1.54, 64, 64]} />
        <shaderMaterial
          ref={atmosphereRef as any}
          vertexShader={AtmosphereShader.vertexShader}
          fragmentShader={AtmosphereShader.fragmentShader}
          uniforms={{
            uColor: { value: new THREE.Color(0.0, 0.6, 1.0) }, // Deeper blue
            uIntensityMultiplier: { value: 1.8 }
          }}
          blending={THREE.AdditiveBlending}
          side={THREE.BackSide}
          transparent={true}
          depthWrite={false}
        />
      </mesh>

      {/* Animated Equator Energy Rings (Elliptical look via scaling/rotation) */}
      <mesh ref={energyRingRef1} rotation={[Math.PI / 2.2, 0.3, 0]} scale={[1, 1.05, 1]}>
        <ringGeometry args={[1.95, 1.955, 64]} />
        <meshBasicMaterial
          color={0x00ffff}
          side={THREE.DoubleSide}
          transparent={true}
          opacity={0.4}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      <mesh ref={energyRingRef2} rotation={[Math.PI / 1.8, -0.4, 0]} scale={[1, 1.1, 1]}>
        <ringGeometry args={[2.2, 2.205, 64]} />
        <meshBasicMaterial
          color={0x00d8ff}
          side={THREE.DoubleSide}
          transparent={true}
          opacity={0.3}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      <mesh ref={energyRingRef3} rotation={[Math.PI / 6, 0.6, 0.2]} scale={[1, 1.15, 1]}>
        <ringGeometry args={[2.4, 2.403, 64]} />
        <meshBasicMaterial
          color={0x00aaff}
          side={THREE.DoubleSide}
          transparent={true}
          opacity={0.25}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Orbit paths for Satellites */}
      <mesh rotation={[0.2, 0.4, 0.1]} scale={[1, 1.02, 1]}>
        <ringGeometry args={[2.698, 2.702, 64]} />
        <meshBasicMaterial color={0x00aaff} side={THREE.DoubleSide} transparent opacity={0.15} blending={THREE.AdditiveBlending} />
      </mesh>
      <mesh rotation={[1.1, -0.2, 0.5]} scale={[1, 1.05, 1]}>
        <ringGeometry args={[3.098, 3.102, 64]} />
        <meshBasicMaterial color={0x00aaff} side={THREE.DoubleSide} transparent opacity={0.1} blending={THREE.AdditiveBlending} />
      </mesh>

      {/* Custom High-Fidelity Orbiting Satellites */}
      <Satellite ref={sat1Ref} scale={0.045} bodyColor={0xdfe6e9} panelColor={0x00c8ff} />
      <Satellite ref={sat2Ref} scale={0.04} bodyColor={0xb2bec3} panelColor={0x00e5ff} />

      {/* Floating 3D HUD label - Projects HTML card into 3D WebGL scene */}
      <Html
        position={[2.1, 1.3, 0]}
        center
        distanceFactor={6}
        className="pointer-events-none select-none"
      >
        <div className="glass-panel p-3.5 min-w-[200px] border border-cyan-400/35 shadow-[0_0_20px_rgba(0,229,255,0.25)] text-left backdrop-blur-md select-none">
          <div className="flex items-center gap-1.5 text-[8px] font-mono text-cyan-400 font-bold uppercase tracking-[2px]">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
            LIVE TELEMETRY
          </div>
          <div className="text-[11px] font-mono text-white mt-1.5 uppercase font-black tracking-widest">
            SEC_A // ORBIT_ACTIVE
          </div>
          <div className="text-[9px] font-mono text-slate-400 mt-1">
            RES: 10M / SENTINEL-2B
          </div>
          <div className="border-t border-cyan-500/10 mt-2.5 pt-2 flex justify-between text-[7.5px] font-mono text-cyan-300/80">
            <span>AZ: 142.84°</span>
            <span>ELEV: 52.17%</span>
          </div>
        </div>
      </Html>
    </group>
  )
}

class ErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: any) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch(error: any, errorInfo: any) {
    console.error("EarthScene WebGL Error caught by boundary:", error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback
    }
    return this.props.children
  }
}

function EarthFallback({ onClick }: Props) {
  return (
    <div 
      onClick={onClick}
      className="w-full h-full flex flex-col items-center justify-center cursor-pointer group relative"
    >
      <div className="absolute w-[350px] h-[350px] rounded-full bg-cyan-500/5 blur-3xl group-hover:bg-cyan-500/10 transition-all duration-1000" />
      <div className="relative w-[300px] h-[300px] rounded-full border border-cyan-500/20 flex items-center justify-center animate-pulse" style={{ animationDuration: '4s' }}>
        <div className="absolute w-[340px] h-[340px] rounded-full border border-dashed border-cyan-500/10 animate-spin" style={{ animationDuration: '40s' }} />
        <div className="w-64 h-64 rounded-full border border-cyan-400/30 flex items-center justify-center overflow-hidden relative shadow-[0_0_40px_rgba(0,240,255,0.2)] group-hover:shadow-[0_0_60px_rgba(0,240,255,0.35)] transition-all duration-500 bg-black">
          <img 
            src="/realistic_earth.png" 
            alt="Earth" 
            className="w-[110%] h-[110%] object-cover animate-[spin_30s_linear_infinite] group-hover:scale-110 transition-transform duration-700"
          />
          <div className="absolute inset-0 scanline opacity-30 pointer-events-none" />
          <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(0,240,255,0.05)_1px,transparent_1px),linear-gradient(to_bottom,rgba(0,240,255,0.05)_1px,transparent_1px)] bg-[size:14px_14px]" />
        </div>
      </div>
    </div>
  )
}


export default function EarthScene({ onClick }: Props) {
  const earthHovered = useStore((s) => s.earthHovered)

  return (
    <ErrorBoundary fallback={<EarthFallback onClick={onClick} />}>
      <Canvas
        camera={{ position: [0, 0, 4.0], fov: 45 }}
        style={{ width: '100%', height: '100%', cursor: earthHovered ? 'crosshair' : 'default' }}
      >
        <color attach="background" args={['#031321']} />
        <ambientLight intensity={0.25} />

        {/* Key directional light simulating solar illumination */}
        <directionalLight position={[6, 3, 5]} intensity={2.2} color="#e6f4ff" />

        {/* Soft fill light */}
        <pointLight
          position={[0, 0, 2.5]}
          intensity={earthHovered ? 3.0 : 1.5}
          color={0x00f0ff}
          distance={8}
        />

        <Suspense fallback={null}>
          <EarthMesh onClick={onClick} />
          <Starfield />
          <DustParticles />
          <Particles />
          <Water />
          <CinematicCamera />
        </Suspense>

        {/* Premium cinematic Postprocessing Bloom & Vignette */}
        <EffectComposer>
          <Bloom
            intensity={earthHovered ? 1.6 : 0.8}
            luminanceThreshold={0.15}
            luminanceSmoothing={0.9}
            mipmapBlur
          />
          <Vignette eskil={false} offset={0.35} darkness={0.65} />
        </EffectComposer>
      </Canvas>
    </ErrorBoundary>
  )
}
