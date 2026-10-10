import { BodyExplorer } from "@/components/body/BodyExplorer";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <>
      <div className="mx-auto max-w-6xl px-5 pb-8">
        <p className="max-w-2xl text-[15px] leading-relaxed text-[#57534e]">Remaining-useful-life regression on turbofan telemetry, cross-validated by engine so the error is measured on hardware the model never saw. Reported per remaining-life band, because overall RMSE hides where it fails.</p>
      </div>
      <BodyExplorer moduleId="maintenance" accent="#22d3ee" />
    </>
  );
}
