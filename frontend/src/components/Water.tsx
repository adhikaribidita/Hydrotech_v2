import React, { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export default function Water() {
  const materialRef = useRef<THREE.ShaderMaterial | null>(null)

  useFrame(({ clock }) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = clock.getElapsedTime()
    }
  })

  // Vertex shader deforms the plane to create realistic moving waves
  const vs = `
    varying vec2 vUv;
    varying vec3 vPosition;
    varying float vElevation;
    uniform float uTime;
    
    void main() {
      vUv = uv;
      vec3 pos = position;
      
      // Calculate wave elevation using multiple wave octaves for natural motion (Increased amplitudes)
      float elevation = sin(pos.x * 0.8 + uTime * 1.0) * 0.18;
      elevation += sin(pos.y * 1.2 + uTime * 0.8) * 0.12;
      elevation += cos(pos.x * 2.0 + pos.y * 1.5 + uTime * 1.3) * 0.08;
      elevation += sin((pos.x - pos.y) * 3.5 + uTime * 1.8) * 0.04;
      
      pos.z += elevation;
      vElevation = elevation;
      vPosition = pos;
      
      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
  `

  // Fragment shader renders deep blue water with glowing cyan swells, specular highlights, and edge-feathering
  const fs = `
    varying vec2 vUv;
    varying vec3 vPosition;
    varying float vElevation;
    uniform float uTime;
    
    void main() {
      // Calculate distance from center to fade out the plane edges smoothly
      float distFromCenter = distance(vUv, vec2(0.5));
      float edgeAlpha = smoothstep(0.5, 0.3, distFromCenter);
      
      // Fake normal to avoid WebGL cross-platform dFdx/dFdy issues
      vec3 normal = normalize(vec3(0.0, 0.0, 1.0));
      
      // Virtual light source for specular highlights (deep blue lighting / sun reflection)
      vec3 lightDir = normalize(vec3(2.0, 5.0, 3.0));
      vec3 viewDir = normalize(vec3(0.0, 0.0, 5.0) - vPosition);
      vec3 halfDir = normalize(lightDir + viewDir);
      
      // Specular highlights
      float spec = pow(max(dot(normal, halfDir), 0.0), 64.0);
      vec3 specColor = vec3(0.4, 0.95, 1.0) * spec * 1.5;
      
      // Base color palette: Deep navy bottom to cyber-cyan crests
      vec3 deepWater = vec3(0.01, 0.06, 0.12);
      vec3 shallowWater = vec3(0.0, 0.22, 0.45);
      vec3 crestGlow = vec3(0.0, 0.78, 1.0);
      vec3 foamColor = vec3(0.85, 0.98, 1.0);
      
      // Interpolate colors based on elevation
      float normElevation = clamp((vElevation + 0.2) * 2.5, 0.0, 1.0);
      vec3 waterColor = mix(deepWater, shallowWater, normElevation);
      
      // Highlight the crests of the waves with a neon cyan glow
      float crestIntensity = smoothstep(0.04, 0.15, vElevation);
      waterColor = mix(waterColor, crestGlow, crestIntensity * 0.6);
      
      // Foam lines on wave crests
      float foamIntensity = smoothstep(0.08, 0.18, vElevation + sin(vPosition.x * 20.0 + uTime * 4.0) * 0.01);
      waterColor = mix(waterColor, foamColor, foamIntensity * 0.5);
      
      // Fresnel effect for atmospheric blue glow edge
      float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 4.0);
      waterColor += crestGlow * fresnel * 0.35;
      
      // Add specular highlights to final color
      waterColor += specColor;
      
      // Add subtle shimmering light streaks
      float shimmer = sin(vUv.x * 45.0 + uTime * 2.5) * cos(vUv.y * 45.0 - uTime * 2.0);
      waterColor += crestGlow * clamp(shimmer * 0.05, 0.0, 0.08);
      
      // Set transparency
      float baseOpacity = 0.9;
      gl_FragColor = vec4(waterColor, baseOpacity * edgeAlpha);
    }
  `

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.8, 0]}>
      <planeGeometry args={[20, 20, 256, 256]} />
      <shaderMaterial
        ref={materialRef as any}
        vertexShader={vs}
        fragmentShader={fs}
        uniforms={{
          uTime: { value: 0 },
        }}
        transparent={true}
        depthWrite={false}
      />
    </mesh>
  )
}
