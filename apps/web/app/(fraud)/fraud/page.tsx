import { BodyExplorer } from "@/components/body/BodyExplorer";
import { FraudStream } from "@/components/fraud/FraudStream";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <>
      <div className="mx-auto max-w-6xl px-5 pb-8">
        <p className="max-w-2xl text-[15px] leading-relaxed text-[#57534e]">A supervised model for the fraud you have seen, an isolation forest for the fraud you have not, combined by max. The demo stream is labelled, so the panel can show the misses and not just the catches.</p>
      </div>

      <section id="board" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-10">
        <FraudStream />
      </section>

      <section id="console" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-16">
        <h2 className="pb-4 text-[18px] font-semibold tracking-tight">Tool console</h2>
        <BodyExplorer moduleId="fraud" accent="#f43f5e" />
      </section>
    </>
  );
}
