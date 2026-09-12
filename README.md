# Kivi - semantic memory for a voice dictation app

Kivi listens to what you say, decides what is worth remembering, and answers
questions from what it kept - with a citation for every claim, or an honest
"I don't know".

Built on the **Sarvam** stack (Sarvam chat, Saaras ASR, Bulbul TTS) so it works
in Indian languages and code-mixed speech, not just English, with Groq as a
fallback. Which one is actually serving is reported, never assumed - see
[Before any demo](#before-any-demo).

---

## The idea worth reading

Most "memory" systems detect a contradiction by similarity: if a new fact is
very close to an old one, treat it as a replacement. **That is backwards.**

Two facts contradict each other precisely when they fill the *same slot* with
*different values* - and different values push cosine similarity **down**.
Measured on real data:

| Pair | Cosine | What it actually is |
|---|---|---|
| "works at Acme" → "works at Google" | **0.324** | a genuine job change |
| "works at Google" vs "has a cat named Pixel" | **0.357** | completely unrelated |

The real contradiction scores *lower* than the unrelated pair. The
distributions overlap, so **no similarity threshold can separate them** - a
0.95 cutoff catches only verbatim restatements, which are duplicates, not
contradictions.

Kivi therefore shortlists candidates by **rank** (nearest by embedding ∪ top
BM25 hits, same tag) and asks the model for a **semantic verdict**:
`supersedes` / `refines` / `duplicate` / `coexists`. Superseded memories are
archived with a pointer to what replaced them, never deleted.

## The second idea: citation gating

The anti-hallucination guarantee is *not* "Kivi only knows your memories". It
is **"Kivi never presents anything as your memory unless a memory backs it"**.

If the model writes an answer but cites no memory, the prose is **discarded**
and the honest decline is shown instead. Answers from general knowledge or the
web are labelled as such in the UI. Hallucination cannot surface as memory,
structurally.

---

## Architecture

```
Voice / text
    │
    ├─ fast_paths ──────── "ok", "hi", "thanks" → 0 LLM calls, 0 ms
    │
    ├─ router ──────────── intent (statement / question / request)
    │                      scope  (personal / general / mixed)
    │                      resolved_query ← pronouns bound to referents
    │
    ├─ INGEST  filter → conflict verdict → link → save
    │          (episode + preference from one sentence)
    │
    └─ RECALL  hybrid retrieve → decide → answer → cite
               memory · general · web   + follow-up questions on a gap
```

| Layer | Choice | Why |
|---|---|---|
| Chat | Sarvam (`sarvam-105b`), Groq fallback | Indic-native. Sarvam-M was retired and `sarvam-30b` is not on the GA endpoint; the API itself names the live set |
| ASR | Saaras `v4`, Whisper fallback | handles code-mixed Tamil/Hindi–English |
| TTS | Bulbul `v3` | answers spoken back in 11 Indian languages |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | 384-dim, cross-lingual retrieval |
| Lexical | SQLite FTS5 (BM25) | names — "Pixel", "Meera", "Zoho" — where embeddings are weakest |
| Fusion | Reciprocal Rank Fusion (k=60) | no score calibration needed between the two |
| Store | SQLite, `float16` embeddings | 1.22 KB per memory end-to-end |

---

## Measured results

Optimisation pass, same machine, same data:

| | Before | After |
|---|---|---|
| Trivial input ("ok", "hi") | ~1,400 ms, 2 LLM calls | **0 ms, 0 calls** |
| Question, cold | ~2,400 ms avg | **~840 ms** |
| Question, repeated | ~2,400 ms | **1 ms** (cached) |
| Retrieval score on answered queries | 0.363 | **0.54–0.69** |
| Embedding storage | 1,536 B | **768 B** |

Scaling, `python bench/scale.py 50000`:

```
50,000 memories · 60.8 MB db · 1.22 KB per memory
cold  (loads all vectors) : 1,046 ms
warm  (cached matrix)     : 113–341 ms
```

Retrieval is one matmul against an in-process matrix, invalidated on write.
Past ~100k memories the honest next step is an ANN index (hnswlib/FAISS);
below that it is not worth the dependency.

---

## Tests

```bash
python -m pytest tests/ -q      # 47 passing
```

The LLM is stubbed, so the tests pin **pipeline behaviour** rather than model
output. Every case is a bug that actually shipped during development and was
caught by a screenshot - contradiction archiving, refinement, duplicate
reinforcement, the empty-extraction ledger entry, U+202F normalisation,
solicited answers, bidirectional links, BM25 injection safety.

This matters because both providers deprecated a model mid-build
(Groq dropped `llama-3.3-70b-versatile`, Sarvam retired `sarvam-m`). Model ids
live in exactly one file, and the tests fail loudly if a swap changes behaviour.

---

## Running it

```bash
pip install -r requirements.txt
copy .env.example .env          # then add SARVAM_API_KEY
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

Kivi runs on Groq alone if that is all you have, but without a Sarvam key
there is no Indic ASR and no speech output.

---

## Before any demo

Presence of an API key proves nothing - a key with no credits authenticates
fine and fails every request. Check what is actually serving:

```bash
curl -s http://127.0.0.1:8000/api/integrations/preflight
```

The server also prints a loud banner at startup when Sarvam is configured but
not serving, and `KIVI_REQUIRE_SARVAM=true` makes it refuse to start at all
rather than quietly answer on Groq.

## Honest limitations

- **Single user.** Signing in with Google identifies the owner and gates
  connections; memories are not scoped per user. Multi-tenancy means a
  `user_id` on every table - a real change, not a flag.
- **OAuth is untested against live accounts.** The flows follow each
  provider's documented spec; first connection may surface a redirect-URI
  mismatch.
- **Live data is search snippets, not instruments.** Weather can be hours
  stale. Kivi discloses the staleness rather than hiding it; precise data
  wants a purpose-built API.
- **Retrieval is exact-ish, not semantic-perfect.** Citation gating is what
  makes a weak retrieval safe rather than confidently wrong.
- **No offline eval set yet.** Behaviour is pinned by 47 tests, but there is
  no labelled corpus reporting filter precision/recall or retrieval hit@k.
  That is the next thing worth building.
