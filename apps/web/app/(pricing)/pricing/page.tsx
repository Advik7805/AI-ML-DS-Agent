import { BodyExplorer } from "@/components/body/BodyExplorer";
import { OpportunityBoard } from "@/components/pricing/OpportunityBoard";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <>
      <div className="mx-auto max-w-6xl px-5 pb-8">
        <p className="max-w-2xl text-[15px] leading-relaxed text-ink-dim">Own-price elasticity per SKU from point-of-sale history, then a revenue-optimal price under a demand floor. Recommendations are clipped to the observed price range and carry a confidence interval, because an optimum outside your price history is an extrapolation, not a finding.</p>
      </div>

      <section id="board" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-10">
        <OpportunityBoard />
      </section>

      <section id="console" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-16">
        <h2 className="pb-4 text-[18px] font-semibold tracking-tight">Tool console</h2>
        <BodyExplorer moduleId="pricing" accent="#34d399" />
      </section>
    </>
  );
}
