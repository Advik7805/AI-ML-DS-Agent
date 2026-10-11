-- Saros: extra knowledge-base documents, stored in Supabase.
--
-- Run this once in the Supabase SQL editor (Dashboard -> SQL Editor -> New query -> Run).
-- It is idempotent, so re-running it is harmless.
--
-- This table is OPTIONAL. The RAG body ships six markdown documents in modules/rag/corpus/ and
-- always loads them; rows here are merged in on top. So if this migration never runs, the
-- knowledge base still works — it just cannot grow without a redeploy.
--
-- What this deliberately is not: a vector store. At 25 passages, pgvector would be a network
-- round trip that buys nothing over the TF-IDF ranking that already retrieves correctly, and it
-- would add an embedding dependency to a body that has none. The table stores plain text.
-- Revisit past a few hundred passages.
--
-- Schema mirrored in core/integrations/supabase.py — keep the two in sync.

create table if not exists public.knowledge_documents (
  id         bigint generated always as identity primary key,
  -- Slug that becomes the citation label shown in the UI, e.g. 'escalation-policy'.
  -- Must be non-empty: a row with no source is dropped at load time.
  source     text not null,
  -- Optional section heading, chunked along with the body.
  heading    text not null default '',
  -- The document text. Must be non-empty: a row with no body can never be matched but would
  -- still inflate the reported corpus size, which is worse than no row at all.
  body       text not null,
  -- Lets you retire a document without deleting it. Inactive rows are never fetched.
  active     boolean not null default true,
  updated_at timestamptz not null default now()
);

-- Index the only two columns the query filters or orders on.
create index if not exists knowledge_documents_active_idx
  on public.knowledge_documents (active, id);

-- Row Level Security ON with no policies. The API uses the service_role key, which bypasses RLS,
-- so the server keeps working; the anon key gets no access at all. That is the intended
-- configuration, and it means a leaked anon key cannot read or write the corpus.
alter table public.knowledge_documents enable row level security;

-- updated_at should mean something. Without this trigger it silently stays at insert time.
create or replace function public.touch_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists knowledge_documents_touch on public.knowledge_documents;
create trigger knowledge_documents_touch
  before update on public.knowledge_documents
  for each row execute function public.touch_updated_at();

-- Seed one row so you can confirm the wiring end to end before writing real content.
-- Delete it once the integration is verified.
insert into public.knowledge_documents (source, heading, body)
select 'demo-note', 'Deployment check',
       'This row exists to verify the Supabase corpus integration. Delete it once '
       || 'GET /api/v1/agent/tools/list_knowledge_sources reports external_live as true.'
where not exists (select 1 from public.knowledge_documents where source = 'demo-note');

-- Verify: should return 1 row after the seed.
-- select source, heading, active, updated_at from public.knowledge_documents;
