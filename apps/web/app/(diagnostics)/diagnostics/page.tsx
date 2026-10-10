import { BodyExplorer } from "@/components/body/BodyExplorer";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <>
      <div className="mx-auto max-w-6xl px-5 pb-8">
        <p className="max-w-2xl text-[15px] leading-relaxed text-[#57534e]">Diabetes risk scoring with per-patient SHAP explanations and a Fairlearn group audit. Optimises against false-negative rate, because in screening a missed diagnosis is the harm that matters.</p>
      </div>
      <BodyExplorer moduleId="xai_health" accent="#a78bfa" />
    </>
  );
}
