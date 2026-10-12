import { BodyExplorer } from "@/components/body/BodyExplorer";
import { ProvenanceCard } from "@/components/body/ProvenanceCard";
import { AcquisitionConsole } from "@/components/scraper/AcquisitionConsole";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <>
      <div className="mx-auto max-w-6xl px-5 pb-8">
        <p className="max-w-2xl text-[15px] leading-relaxed text-ink-dim">
          <strong className="font-medium text-ink">Northwind Labs</strong> acquires public pages — for
          competitive price intel the retail body can use, and for documents the knowledge body can
          index. Three backends with genuinely different trade-offs: plain HTTP is free but cannot
          render JavaScript, Tavily renders JS and costs credits. Every result says which one produced it.
        </p>
      </div>

      <section id="data" className="mx-auto max-w-6xl px-5 pb-10">
        <ProvenanceCard moduleId="scraper" accent="#f97316" />
      </section>

      <section id="board" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-10">
        <AcquisitionConsole />
      </section>

      <section id="console" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-16">
        <h2 className="pb-4 text-[18px] font-semibold tracking-tight">Tool console</h2>
        <BodyExplorer moduleId="scraper" accent="#f97316" />
      </section>
    </>
  );
}
