import { BodyExplorer } from "@/components/body/BodyExplorer";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <>
      <div className="mx-auto max-w-6xl px-5 pb-8">
        <p className="max-w-2xl text-[15px] leading-relaxed text-[#57534e]">Host-allowlisted and size-capped so the agent cannot be directed at an arbitrary site. Mocked by default; the production path is Firecrawl behind the same interface.</p>
      </div>
      <BodyExplorer moduleId="scraper" accent="#f97316" />
    </>
  );
}
