import { BodyExplorer } from "@/components/body/BodyExplorer";
import { KnowledgeSearch } from "@/components/rag/KnowledgeSearch";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <>
      <div className="mx-auto max-w-6xl px-5 pb-8">
        <p className="max-w-2xl text-[15px] leading-relaxed text-[#57534e]">Heading-aware chunking and TF-IDF ranking over an enterprise policy corpus, with a grounding verdict attached to every answer. Retrieval lives here; synthesis lives in the hub agent, so any provider can cite the same passages.</p>
      </div>

      <section id="board" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-10">
        <KnowledgeSearch />
      </section>

      <section id="console" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-16">
        <h2 className="pb-4 text-[18px] font-semibold tracking-tight">Tool console</h2>
        <BodyExplorer moduleId="rag" accent="#38bdf8" />
      </section>
    </>
  );
}
