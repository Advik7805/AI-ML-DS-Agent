import { BodyExplorer } from "@/components/body/BodyExplorer";
import { ProvenanceCard } from "@/components/body/ProvenanceCard";
import { FairnessAudit } from "@/components/diagnostics/FairnessAudit";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <>
      <div className="mx-auto max-w-6xl px-5 pb-8">
        <p className="max-w-2xl text-[15px] leading-relaxed text-ink-dim">
          <strong className="font-medium text-ink">Northwind Care</strong> runs the employee health
          screening pilot — 768 records from the public Pima Indians Diabetes study, standing in for
          the occupational-health feed. Diabetes risk scoring with per-patient SHAP explanations and a
          Fairlearn group audit. Optimises against false-negative rate, because in screening a missed
          diagnosis is the harm that matters.
        </p>
      </div>

      <section id="data" className="mx-auto max-w-6xl px-5 pb-10">
        <ProvenanceCard moduleId="xai_health" accent="#a78bfa" />
      </section>

      <section id="board" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-10">
        <FairnessAudit />
      </section>

      <section id="console" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-16">
        <h2 className="pb-4 text-[18px] font-semibold tracking-tight">Tool console</h2>
        <BodyExplorer moduleId="xai_health" accent="#a78bfa" />
      </section>
    </>
  );
}
