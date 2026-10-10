import { BodyExplorer } from "@/components/body/BodyExplorer";
import { FleetBoard } from "@/components/maintenance/FleetBoard";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <>
      <div className="mx-auto max-w-6xl px-5 pb-8">
        <p className="max-w-2xl text-[15px] leading-relaxed text-[#57534e]">Remaining-useful-life regression on turbofan telemetry, cross-validated by engine so the error is measured on hardware the model never saw — and reported per remaining-life band, because an overall RMSE hides exactly where it fails.</p>
      </div>

      <section id="board" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-10">
        <FleetBoard />
      </section>

      <section id="console" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-16">
        <h2 className="pb-4 text-[18px] font-semibold tracking-tight">Tool console</h2>
        <BodyExplorer moduleId="maintenance" accent="#22d3ee" />
      </section>
    </>
  );
}
