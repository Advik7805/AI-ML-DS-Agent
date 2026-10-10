/**
 * Optional background video, layered beneath the 3D field.
 *
 * Off unless `BACKDROP_VIDEO_URL` is set. That is deliberate, and it is the reason this is a
 * server component: the URL is read here and the markup is emitted, so nothing needs
 * `NEXT_PUBLIC_` and no URL leaks into the client bundle.
 *
 * Why it is not bundled: a background video is 5–30 MB. Committing one to git means every clone
 * and every deploy carries it, which is a poor trade on a free tier with an egress cap. Point
 * this at a CDN instead — Cloudflare R2's free tier is 10 GB storage and 10 GB egress, which
 * covers it comfortably.
 *
 * Without the variable the procedural field renders alone and nothing changes. There is no
 * asset this component requires.
 */

/** Accept only http(s) URLs. A stray filesystem path or javascript: URI would be worse than none. */
function safeVideoUrl(raw: string | undefined): string | null {
  const v = (raw ?? "").trim();
  if (!v) return null;
  try {
    const u = new URL(v);
    return u.protocol === "http:" || u.protocol === "https:" ? u.toString() : null;
  } catch {
    return null;
  }
}

export function BackdropVideo({ accent }: { accent: string }) {
  const src = safeVideoUrl(process.env.BACKDROP_VIDEO_URL);
  if (!src) return null;

  return (
    <>
      <video
        className="pointer-events-none absolute inset-0 h-full w-full object-cover"
        src={src}
        autoPlay
        muted
        loop
        playsInline
        preload="metadata"
        aria-hidden="true"
        tabIndex={-1}
      />
      {/* Keeps foreground text legible over arbitrary footage, and tints the video toward the
          body's accent so it still reads as that storefront. */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background: `linear-gradient(160deg, ${accent}33 0%, rgba(3,4,9,0.82) 55%, rgba(3,4,9,0.94) 100%)`,
        }}
      />
    </>
  );
}
