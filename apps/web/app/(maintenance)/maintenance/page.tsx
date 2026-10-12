import { BodyExplorer } from "@/components/body/BodyExplorer";
import { ProvenanceCard } from "@/components/body/ProvenanceCard";
import { DriftPanel } from "@/components/body/DriftPanel";
import { FleetBoard } from "@/components/maintenance/FleetBoard";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <>
      <div className="mx-auto max-w-6xl px-5 pb-8">
        <p className="max-w-2xl text-[15px] leading-relaxed text-ink-dim">
          <strong className="font-medium text-ink">Northwind Industrial</strong> maintains the engines
          it leases to airlines. This is remaining-useful-life regression on that fleet's telemetry —
          NASA's CMAPSS FD001 record (100 engines, 20,631 cycles) standing in for the MRO feed —
          cross-validated <em>by engine</em> so the error is measured on hardware the model never saw,
          and reported per remaining-life band, because an overall RMSE hides exactly where it fails.
        </p>
      </div>

      <section id="data" className="mx-auto max-w-6xl px-5 pb-10">
        <ProvenanceCard moduleId="maintenance" accent="#22d3ee" />
      </section>

      <section id="board" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-10">
        <FleetBoard />
      </section>

      <section id="drift" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-10">
        <h2 className="pb-1.5 text-[18px] font-semibold tracking-tight">Model health</h2>
        <p className="max-w-2xl pb-4 text-[13.5px] leading-relaxed text-ink-dim">Fleet sensors shift for ordinary reasons — a new engine model, a recalibrated sensor, a different operating profile — and the RUL model keeps producing confident cycle counts regardless. The panel below makes that visible.</p>
        <div className="max-w-xl">
          <DriftPanel tool="check_fleet_drift" accent="#22d3ee" label="Fleet telemetry drift" />
        </div>
      </section>

      <section id="console" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-16">
        <h2 className="pb-4 text-[18px] font-semibold tracking-tight">Tool console</h2>
        <BodyExplorer moduleId="maintenance" accent="#22d3ee" />
      </section>
    </>
  );
}
