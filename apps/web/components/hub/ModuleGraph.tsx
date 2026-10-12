"use client";

/**
 * The hub's 3D module graph — and the point of it is that the 3D *is* the interface.
 *
 * Six bodies orbit a core; hovering a node names it, clicking it navigates to that body's own
 * storefront. Earning its frame budget by being navigation rather than decoration is what
 * separates this from a spinning logo.
 *
 * Budget rules applied here: capped `dpr`, `frameloop="demand"` is deliberately NOT used
 * (the orbit needs continuous frames) but the canvas is lazy-mounted on intersection and the
 * whole scene is skipped for `prefers-reduced-motion`.
 */

import { Canvas, useFrame } from "@react-three/fiber";
import { Float, Html } from "@react-three/drei";
import { useRouter } from "next/navigation";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import type { ModuleManifest } from "@/lib/types";
import { KNOWN_BODIES } from "@/lib/types";

interface Node {
  id: string;
  title: string;
  blurb: string;
  accent: string;
  live: boolean;
  /** The body's own storefront. Null only for a body that genuinely does not exist yet. */
  href: string | null;
  radius: number;
  speed: number;
  phase: number;
  tilt: number;
}

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return reduced;
}

function Core() {
  const mesh = useRef<THREE.Mesh>(null);
  const glow = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (mesh.current) {
      mesh.current.rotation.y = t * 0.18;
      mesh.current.rotation.x = Math.sin(t * 0.22) * 0.25;
    }
    if (glow.current) {
      const s = 1 + Math.sin(t * 1.6) * 0.045;
      glow.current.scale.setScalar(s);
    }
  });

  return (
    <group>
      <mesh ref={mesh}>
        <icosahedronGeometry args={[0.9, 1]} />
        <meshStandardMaterial
          color="#7c5cff"
          emissive="#4c2fd6"
          emissiveIntensity={0.9}
          roughness={0.25}
          metalness={0.6}
          flatShading
        />
      </mesh>
      <mesh ref={glow}>
        <sphereGeometry args={[1.25, 32, 32]} />
        <meshBasicMaterial color="#7c5cff" transparent opacity={0.07} />
      </mesh>
      <pointLight color="#9d86ff" intensity={28} distance={14} />
    </group>
  );
}

function OrbitRing({ radius, tilt }: { radius: number; tilt: number }) {
  // `<line>` in JSX resolves to the *SVG* line element, not THREE.Line, so the ring has to be
  // built as an object and mounted with `<primitive>`.
  const ring = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i <= 128; i++) {
      const a = (i / 128) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius));
    }
    const geometry = new THREE.BufferGeometry().setFromPoints(pts);
    const material = new THREE.LineBasicMaterial({ color: "#2a3149", transparent: true, opacity: 0.5 });
    return new THREE.Line(geometry, material);
  }, [radius]);

  // WebGL resources are not garbage collected — dispose on unmount or the context leaks.
  useEffect(
    () => () => {
      ring.geometry.dispose();
      (ring.material as THREE.Material).dispose();
    },
    [ring],
  );

  return (
    <group rotation={[tilt, 0, 0]}>
      <primitive object={ring} />
    </group>
  );
}

function Body({ node, onHover }: { node: Node; onHover: (id: string | null) => void }) {
  const group = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState(false);
  const router = useRouter();

  useFrame(({ clock }) => {
    if (!group.current) return;
    const t = clock.getElapsedTime() * node.speed + node.phase;
    group.current.position.set(
      Math.cos(t) * node.radius,
      Math.sin(t * 0.9) * node.tilt,
      Math.sin(t) * node.radius,
    );
    group.current.rotation.y += 0.01;
  });

  return (
    <group ref={group}>
      <Float speed={1.4} rotationIntensity={0.35} floatIntensity={0.5}>
        <mesh
          onPointerOver={(e) => {
            e.stopPropagation();
            setHovered(true);
            onHover(node.id);
            document.body.style.cursor = "pointer";
          }}
          onPointerOut={() => {
            setHovered(false);
            onHover(null);
            document.body.style.cursor = "auto";
          }}
          onClick={(e) => {
            e.stopPropagation();
            if (node.href) router.push(node.href);
          }}
          scale={hovered ? 1.35 : 1}
        >
          <octahedronGeometry args={[node.live ? 0.34 : 0.24, 0]} />
          <meshStandardMaterial
            color={node.accent}
            emissive={node.accent}
            emissiveIntensity={hovered ? 1.5 : node.live ? 0.75 : 0.25}
            roughness={0.2}
            metalness={0.75}
            transparent
            opacity={node.live ? 1 : 0.45}
            flatShading
          />
        </mesh>
      </Float>
      {hovered && (
        <Html center distanceFactor={9} style={{ pointerEvents: "none" }}>
          <div className="w-max -translate-y-10 rounded-lg border border-white/15 bg-void-2/95 px-3 py-2 text-center shadow-2xl backdrop-blur">
            <div className="text-[11px] font-semibold tracking-wide text-white">{node.title}</div>
            <div className="mt-0.5 text-[10px] uppercase tracking-wider" style={{ color: node.accent }}>
              {node.live ? "live" : "planned"}
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}

export function ModuleGraph({ live }: { live: ModuleManifest[] }) {
  const reduced = usePrefersReducedMotion();
  const [mounted, setMounted] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState<string | null>(null);

  // Lazy-mount: do not create a WebGL context until the section is actually on screen.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setMounted(true);
          io.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const nodes: Node[] = useMemo(() => {
    // Live manifests win; when the backend is asleep the static mirror keeps the graph whole.
    const base = live.length
      ? live.map((m) => ({
          id: m.id,
          title: m.title,
          blurb: m.summary,
          accent: m.accent,
          live: m.status !== "planned",
          href: m.routes?.ui ?? null,
        }))
      : KNOWN_BODIES.map((b) => ({
          id: b.id,
          title: b.title,
          blurb: b.blurb,
          accent: b.accent,
          live: true,
          href: b.href as string | null,
        }));
    return base.map((n, i) => ({
      ...n,
      radius: 3.0 + (i % 3) * 0.85,
      speed: 0.22 + (i % 4) * 0.045,
      phase: (i / base.length) * Math.PI * 2,
      tilt: 0.35 + (i % 3) * 0.22,
    }));
  }, [live]);

  const activeNode = nodes.find((n) => n.id === active);

  return (
    <div ref={containerRef} className="relative h-[520px] w-full sm:h-[600px]">
      {mounted && !reduced ? (
        <Canvas
          camera={{ position: [0, 2.4, 8.2], fov: 46 }}
          dpr={[1, 1.75]}
          gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
        >
          <Suspense fallback={null}>
            <ambientLight intensity={0.35} />
            <directionalLight position={[6, 8, 4]} intensity={0.7} color="#ffffff" />
            <fog attach="fog" args={["#05060a", 9, 20]} />

            <Core />
            {[3.0, 3.85, 4.7].map((r, i) => (
              <OrbitRing key={r} radius={r} tilt={i * 0.18 - 0.12} />
            ))}
            {nodes.map((n) => (
              <Body key={n.id} node={n} onHover={setActive} />
            ))}
          </Suspense>
        </Canvas>
      ) : (
        /* Reduced-motion / pre-mount fallback: the same information, statically. */
        <div className="flex h-full w-full items-center justify-center">
          <div className="relative flex h-64 w-64 items-center justify-center rounded-full border border-white/10">
            <div className="h-20 w-20 rounded-2xl bg-hub shadow-[0_0_80px_20px_rgba(124,92,255,0.45)]" />
            {nodes.map((n, i) => {
              const a = (i / nodes.length) * Math.PI * 2;
              return (
                <span
                  key={n.id}
                  className="absolute h-3 w-3 rotate-45 rounded-[2px]"
                  style={{
                    background: n.accent,
                    opacity: n.live ? 1 : 0.4,
                    left: `${50 + Math.cos(a) * 42}%`,
                    top: `${50 + Math.sin(a) * 42}%`,
                  }}
                />
              );
            })}
          </div>
        </div>
      )}

      {/* Readout under the canvas — keeps the scene legible without hovering. */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-center">
        <div className="glass h-12 min-w-[280px] rounded-xl px-5 text-center text-xs">
          {activeNode ? (
            <div className="flex h-full flex-col justify-center">
              <span className="font-semibold" style={{ color: activeNode.accent }}>
                {activeNode.title}
              </span>
              <span className="text-[11px] text-ink-dim">
                {activeNode.live ? "click to open the storefront" : "on the roadmap"}
              </span>
            </div>
          ) : (
            <div className="flex h-full items-center justify-center text-ink-faint">
              hover a body to inspect it · click a live one to enter
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
