import type { NextConfig } from "next";

/**
 * The browser must never call the backend directly — it only ever speaks to this origin, and
 * Next proxies to FastAPI. That keeps the preview host working and avoids CORS in dev.
 *
 * `API_ORIGIN` points at the local FastAPI in dev and at the deployed backend in production.
 */
const DEFAULT_ORIGIN = "http://127.0.0.1:8000";
const API_ORIGIN = process.env.API_ORIGIN ?? DEFAULT_ORIGIN;

/**
 * Fail loudly, at build time, on the one misconfiguration that produces a site which *looks*
 * deployed and answers nothing.
 *
 * With `API_ORIGIN` unset, every rewrite silently targets 127.0.0.1:8000. On a laptop that is
 * the FastAPI dev server, so it works. On Vercel it is the serverless function's own loopback,
 * where nothing listens — so the hub renders, the bodies render, and every API call fails with
 * no clue why. Detecting a production build (Vercel sets VERCEL=1) and an unset or default
 * origin turns that into a build error instead of a mystery.
 */
const isProductionBuild = process.env.NODE_ENV === "production" && !process.env.API_ORIGIN;
if (isProductionBuild && process.env.VERCEL) {
  throw new Error(
    `API_ORIGIN is not set, so every /api request would proxy to ${DEFAULT_ORIGIN} — which is ` +
      "this serverless function's own loopback, where nothing is listening. Set API_ORIGIN to " +
      "the deployed FastAPI URL (e.g. https://saros-api.onrender.com) in the Vercel project " +
      "settings. The pages would still render, which is exactly why this must fail the build.",
  );
}
if (isProductionBuild) {
  // eslint-disable-next-line no-console
  console.warn(
    `[saros] API_ORIGIN is unset; defaulting to ${DEFAULT_ORIGIN}. Fine for a local production ` +
      "build, broken on any host where the API is not on localhost.",
  );
}


/**
 * Which origins may load Next's *dev* resources (HMR socket, dev chunks).
 *
 * This is not cosmetic. Next 16 blocks dev resources from any origin not listed here, and when
 * the HMR socket is refused the dev client never hydrates: the page renders as static HTML, so
 * every button, file picker and select silently does nothing, with no visible error. That is
 * exactly what happened behind a preview host — `next dev` was reached at a host it did not
 * recognise, `/_next/hmr` was answered with "Blocked cross-origin request", and the whole UI
 * became decorative.
 *
 * So the list deliberately covers the hosts this project is actually run behind: localhost, the
 * container's own loopback, the sandbox preview host (`*.e2b.app`), and Vercel previews. Add
 * more with `SAROS_DEV_ORIGINS=host-a,host-b` without editing this file. Wildcards are supported
 * by Next. Production is unaffected: `allowedDevOrigins` is a dev-only setting.
 */
const DEV_ORIGINS = [
  "localhost",
  "127.0.0.1",
  "0.0.0.0",
  "*.local",
  "*.e2b.app",
  "*.e2b.dev",
  "*.vercel.app",
  ...(process.env.SAROS_DEV_ORIGINS ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean),
];

const nextConfig: NextConfig = {
  reactStrictMode: true,
  allowedDevOrigins: DEV_ORIGINS,
  async rewrites() {
    return [
      // Host-based routing: one deploy, many storefronts. Each body can later be given its
      // own subdomain (churn.saros.dev) that renders a completely different root layout,
      // exactly like a retailer pointing /electronics at a separate storefront.
      ...(process.env.CHURN_HOST
        ? [
            {
              source: "/:path*",
              has: [{ type: "host" as const, value: process.env.CHURN_HOST }],
              destination: "/churn/:path*",
            },
          ]
        : []),
      { source: "/api/:path*", destination: `${API_ORIGIN}/api/:path*` },
      { source: "/health", destination: `${API_ORIGIN}/health` },
    ];
  },
};

export default nextConfig;
