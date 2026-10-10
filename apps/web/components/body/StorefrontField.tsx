"use client";

/**
 * Shared 3D backdrop for every storefront, tinted by the body's accent.
 *
 * The hub has an interactive module graph; a body does not need a second hero scene, it needs
 * depth and motion that make it feel like a place rather than a page. So this is deliberately
 * the cheapest thing that is genuinely 3D: one `THREE.Points` cloud, one geometry, one material,
 * no lights, no shadows, no post-processing.
 *
 * Three constraints that are not negotiable here:
 *
 *  1. **Resource disposal.** WebGL buffers are not garbage collected. The geometry and material
 *     are built in `useMemo` and disposed on unmount, or every navigation between bodies leaks
 *     a GPU context — and there are eight of them.
 *  2. **`prefers-reduced-motion`.** A constantly drifting field is a genuine accessibility
 *     problem, not a preference. When it is set, the cloud renders static.
 *  3. **No client-only crash on the server.** `Canvas` touches WebGL, so the whole thing is
 *     behind a mounted flag; the server renders the fallback gradient instead.
 */

import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useEffect, useMemo, useRef, useState } from "react";

interface FieldProps {
  accent: string;
  count?: number;
}

function PointField({ accent, count = 900 }: FieldProps) {
  const ref = useRef<THREE.Points>(null);
  const reduce = usePrefersReducedMotion();

  // Deterministic scatter: a seeded LCG rather than Math.random, so the field looks the same on
  // every render and does not visibly reshuffle on a hot reload.
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    let seed = 20260101;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };
    for (let i = 0; i < count; i++) {
      // A flattened ellipsoid: wide and shallow reads as a horizon, which suits a page backdrop
      // far better than an evenly-filled cube.
      const r = 9 + rand() * 13;
      const theta = rand() * Math.PI * 2;
      const phi = Math.acos(2 * rand() - 1);
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.cos(phi) * 0.38;
      positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
    }
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return g;
  }, [count]);

  const material = useMemo(
    () =>
      new THREE.PointsMaterial({
        color: new THREE.Color(accent),
        size: 0.055,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.7,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [accent],
  );

  useEffect(
    () => () => {
      geometry.dispose();
      material.dispose();
    },
    [geometry, material],
  );

  useFrame(({ clock }) => {
    if (reduce || !ref.current) return;
    const t = clock.getElapsedTime();
    ref.current.rotation.y = t * 0.02;
    ref.current.rotation.x = Math.sin(t * 0.05) * 0.08;
  });

  return <points ref={ref} geometry={geometry} material={material} />;
}

export function StorefrontField({ accent, count }: FieldProps) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      {/* Always present: it is the visual fallback when WebGL is unavailable, and it is what the
          server renders, so there is no flash of an empty page before hydration. */}
      <div
        className="absolute inset-0"
        style={{
          background: `radial-gradient(58rem 30rem at 100% -12%, ${accent}26, transparent 62%), radial-gradient(40rem 26rem at -8% 30%, ${accent}14, transparent 60%)`,
        }}
      />

      {mounted && (
        <Canvas
          camera={{ position: [0, 0, 26], fov: 42 }}
          dpr={[1, 1.5]}
          gl={{ antialias: false, alpha: true, powerPreference: "low-power" }}
          style={{ position: "absolute", inset: 0 }}
        >
          <PointField accent={accent} count={count} />
        </Canvas>
      )}
    </div>
  );
}

function usePrefersReducedMotion(): boolean {
  const [reduce, setReduce] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduce(mq.matches);
    const onChange = (e: MediaQueryListEvent) => setReduce(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduce;
}
