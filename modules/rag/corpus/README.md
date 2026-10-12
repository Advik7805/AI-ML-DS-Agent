# Northwind Global — Knowledge Base

Owner: Northwind Knowledge (group function) · Index: `rag-tfidf-v1`

This directory is the shipped corpus the Knowledge RAG body retrieves from. It is deliberately
small and readable: every answer the retrieval layer returns can be checked by hand against the
source document, which is the only honest way to demo citations.

## Contents

| Document | Owner | Answers questions about |
|---|---|---|
| `refund-policy.md` | Group Customer Operations | refund windows, payment methods, exceptions |
| `data-privacy.md` | Group Privacy Office | what customer data may be used for, retention, subject access |
| `fraud-protocol.md` | Payment Risk, Northwind Pay | hold vs decline, review SLAs, escalation |
| `retention-playbook.md` | Customer Retention, Northwind Connect | which risk band gets which save action |
| `churn-model-card.md` | Data Science, Northwind Connect | the churn model's task, data, metrics and limits |
| `sla.md` | Group Service Management | availability commitments, credits, maintenance windows |
| `README.md` | Northwind Knowledge | this index — what the knowledge base contains |

## Adding documents

Upload `.md`, `.txt` or `.pdf` from the hub's **Analyze your data** studio and choose the
Knowledge body. The document is appended to a **session-scoped overlay**: it is retrievable in
your session, and it is never written to disk or into this directory. The shipped corpus above
stays exactly as it is, so a demo cannot be corrupted by an upload.

## How retrieval works, stated plainly

Heading-aware chunking (each chunk carries its file and heading path, which is what the
citation shows), then TF-IDF cosine ranking over the chunks. No embeddings, no model download:
the trade-off is recall on paraphrase, and it is stated in the UI rather than hidden. A question
with no passage above the floor returns **no hits** instead of padding an answer with an
irrelevant document.
