import { BodyExplorer } from "@/components/body/BodyExplorer";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <>
      <div className="mx-auto max-w-6xl px-5 pb-8">
        <p className="max-w-2xl text-[15px] leading-relaxed text-[#57534e]">Own-price elasticity per SKU from point-of-sale history, then a revenue-optimal price under a demand floor. Recommendations are clipped to the observed price range and carry a confidence interval.</p>
      </div>
      <BodyExplorer moduleId="pricing" accent="#34d399" />
    </>
  );
}
