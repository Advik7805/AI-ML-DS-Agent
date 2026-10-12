"use client";

export function WakeRetry({
  error,
  onRetry,
  busy = false,
  label = "Retry",
}: {
  error: string;
  onRetry: () => void;
  busy?: boolean;
  label?: string;
}) {
  return (
    <div role="status" aria-live="polite" className="rounded-2xl border border-line bg-void-3 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="text-[12.5px] font-medium text-ink">Request failed. The free-tier API may be waking after idle.</p>
          <p className="mt-1 break-words text-[11.5px] leading-relaxed text-ink-dim">{error}</p>
        </div>
        <button
          type="button"
          onClick={onRetry}
          disabled={busy}
          className="shrink-0 rounded-lg border border-line px-3 py-2 text-[11.5px] font-medium text-ink-dim transition hover:bg-void-2 hover:text-ink disabled:cursor-wait disabled:opacity-50"
        >
          {busy ? "Trying…" : label}
        </button>
      </div>
    </div>
  );
}
