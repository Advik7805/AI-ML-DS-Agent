import { BodyExplorer } from "@/components/body/BodyExplorer";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <>
      <div className="mx-auto max-w-6xl px-5 pb-8">
        <p className="max-w-2xl text-[15px] leading-relaxed text-[#57534e]">Heading-aware chunking and TF-IDF ranking over an enterprise policy corpus. Retrieval lives here; synthesis lives in the hub agent, so any provider can cite the same passages.</p>
      </div>
      <BodyExplorer moduleId="rag" accent="#38bdf8" />
    </>
  );
}
