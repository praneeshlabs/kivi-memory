# Kivi

Kivi is a memory layer for a voice dictation app. You talk to it. It decides
what is worth remembering, keeps that, and answers questions from what it
kept. Every answer either points to the memory it came from or says plainly
that it does not know.

Built for Indian users, on Sarvam's models, so it can hear Tamil, Hindi and
the code-mixed English people actually speak. Groq is a fallback so the app
still runs on days that stack is not available.

The product thinking behind this is in [product-position](product-position)
and [product-vision](product-vision). This file covers the build: what it
does, how, what was tested, and what is not done yet. How to run it
yourself is in [RUN.md](RUN.md).

---

## What it is for

Three real situations, not three feature bullets.

**You say something once and want it remembered.**
"My manager is Rahul." "I'm allergic to peanuts." "I always order filter
coffee when I'm running late." Kivi decides on its own whether this is a
fact, a one-off event, or a pattern, and stores it that way. You never fill
in a form.

**You ask something later and want a straight answer.**
"Who's my manager?" "What did I say about coffee?" Kivi answers from what
you told it and shows the memory it used. If it does not have the answer, it
says so instead of guessing.

**Things you told it stop being true.**
You change jobs, move city, or the vague thing becomes a specific thing.
Kivi is supposed to notice and update instead of quietly piling up five
versions of the same fact.

---

## What actually happens when you talk to it

1. You say or type something.
2. Kivi decides: is this something to remember, a question, or a request to
   go do something in the world. It cannot do the last one yet, but it says
   so honestly instead of pretending.
3. If it's worth remembering, Kivi checks what it already knows. If the new
   thing replaces an old one, the old one is archived, not deleted. If it's
   the same thing said differently, nothing new is stored.
4. If you ask something, Kivi searches what it has, drafts an answer, and
   only shows that answer if it can point to the memory behind it. If it
   can't, you get an honest "I don't know" instead of a confident guess.

That last rule is the one thing in this project I would defend hardest: an
answer is never shown unless a memory backs it. Not "Kivi mostly sticks to
memory." Never, by construction.

---

## The two decisions worth explaining

### Why a fact changing does not look like a fact changing

The obvious way to catch a contradiction is: if the new thing you said is
very similar to something you said before, treat it as a correction.

That is wrong, and it took a wrong version shipping before I understood why.
Two facts contradict each other exactly when they fill the same slot with
different values. Different values push a similarity score down, not up. I
measured it on this app's own data.

| Pair | Similarity | What it is |
|---|---|---|
| "works at Acme" to "works at Google" | 0.32 | a real job change |
| "works at Google" vs "has a cat named Pixel" | 0.36 | two unrelated facts |

The real contradiction scores lower than two things that have nothing to do
with each other. There is no cutoff number that separates those two rows,
because they overlap. So Kivi does not use a number here. It shortlists a
few memories that are plausibly about the same thing, nearest by meaning
plus anything sharing a keyword, and asks the model directly: does this
replace that, refine it, repeat it, or is it just a different fact?
Superseded memories move to an archive table with a pointer to what replaced
them. Nothing is deleted.

### Why an answer with no citation gets thrown away, not shown

A model asked to answer from memory will sometimes write something
reasonable sounding that is not actually grounded in anything it was given.
The fix is not a stricter prompt. Prompts get ignored under pressure. The
fix is structural: if the model's answer does not name which memory it
used, the text is discarded and a plain decline is shown instead. The user
never sees prose that is not backed by something real, because that prose
never leaves the server.

---

## Architecture

```
you speak or type
      |
      v
routing: is this a memory, a question, or a request?
      |   (also strips "ok", "hi", "thanks" before any model runs)
      v
  remembering                          answering
  filter, conflict check, save         hybrid search, answer, cite
  (one sentence can produce both       or an honest "I don't know"
   an event and a habit)
```

**Storage.** SQLite, one file. `schema.sql` builds it from nothing.
`migrations/` carries every change since, in order, and both a brand new
database and an old one end up in the same shape.

**Retrieval.** Two search methods, combined. Meaning based search, an
embedding model, finds things that are phrased differently but mean the
same thing. Keyword search, SQLite's built in full text index, finds names
and proper nouns, which the meaning based search is weak on, because a name
like "Pixel" does not carry much meaning as a word. The two rankings are
merged by position, not by score, so neither method needs its numbers
recalibrated when the other one changes.

**Speech.** Sarvam's Saaras model for speech to text, Bulbul for text to
speech, in eleven Indian languages plus English, including code mixed
speech. Whisper, through Groq, is the fallback.

**Language model.** Sarvam's chat models by default, Groq as fallback. One
file, `app/services/llm_client.py`, holds every model name. Both providers
have already quietly retired a model name on me mid build. Centralising
this meant the fix was one line, not a search through every prompt in the
app.

**Everything else that changes how it feels to use:**

- A small set of Indian English phrases, tone that turns off automatically
  when you mention something hard, in `app/services/persona.py`.
- Follow up questions when Kivi does not know something, so it can ask
  rather than just refuse.
- A consolidation pass, `POST /api/memories/consolidate`, that finds
  memories saying the same thing in different words and collapses them,
  since the filter alone lets a few through.

---

## What I checked, and what I did not just assume

52 automated tests run against a stubbed model, so they check the app's own
logic rather than the model's mood that day. Does a superseded fact
actually get archived. Does an answer with no citation actually get thrown
away. Does an old database actually pick up new columns without losing
data. Run them with:

```bash
python -m pytest tests/ -q
```

Separately, an evaluation runs against the real model and produces numbers
that move a little each time, because a real model is not perfectly
consistent. This is the difference between "the code does what I meant" and
"the model is actually any good at the job." Both matter and they are not
the same test. Run it with:

```bash
python seed.py --reset
python eval/run_eval.py
```

### Results

The full breakdown, including every case, is in `eval/results/RESULTS.md`
and `eval/results/results.json`. This is what the last run against the
seeded database produced, real model, nothing removed to make the numbers
look better.

**Memory filter, 18 cases.** Decides correctly whether to keep, watch, or
drop 89 percent of the time. Never wrongly threw away something worth
keeping. Three misses, kept in the report rather than removed: a Hindi
sentence got tagged as a permanent fact instead of a dated event, a hedged
sentence, "I think I might be coming down with something," got saved
outright instead of watched first, and a task request got classified as a
habit instead of an event.

**Retrieval, 17 queries.** Finds the right memory in the top 3 results 79
percent of the time. Two real misses worth naming: "Where do I work?" did
not surface the Zoho memory, and the same question in Hindi also missed.
Both are in the report, not hidden. Declines correctly on all three
questions that have no answer in memory.

**Speech to text, 8 scored English cases.** 4.6 percent word error rate.
Three Tamil, Hindi and code mixed cases are in the corpus but marked
untested, because a machine voice reading Tamil text does not tell you
anything about how a real person speaking Tamil sounds. Those need an
actual human recording, which I did not have for this submission.

The numbers in the eval run above were produced while Sarvam was not
serving requests, see the note in `eval/results/RESULTS.md`. The pipeline
and the corpus are what is being evaluated here. Rerun after Sarvam credit
is available to get Sarvam's own numbers on the same corpus.

### Performance

Before the optimisation pass, the app averaged 2.4 seconds per message and
made 2 to 4 model calls for things like "ok" or "thanks."

| | Before | After |
|---|---|---|
| "ok", "hi", "thanks" | about 1.4s, 2 model calls | 0ms, 0 calls |
| A real question, first time | about 2.4s | about 0.8s |
| The same question again | about 2.4s | about 1ms, cached |
| Retrieval score on answered questions | 0.36 | 0.54 to 0.69 |
| Storage per memory | 1536 bytes | 768 bytes |

At 50,000 memories (`python bench/scale.py 50000`): 61MB database, a cold
search takes about a second, a warm one 100 to 300ms. Past roughly 100,000
memories the honest next step is a proper approximate search index. Below
that it is not worth the extra dependency.

---

## Running it

```bash
pip install -r requirements.txt
copy .env.example .env
```

Put your Sarvam key in `.env`. Groq works as a fallback if that is all you
have, but you lose Indian language speech.

```bash
python seed.py --reset
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Full steps, including how to run the tests
and the evaluation, are in `RUN.md`.

**Before you rely on a demo.** Having a Sarvam key in `.env` does not mean
Sarvam is actually answering. A key with no credit on the account
authenticates fine and then fails every real request. Check what is
actually serving:

```bash
curl -s http://127.0.0.1:8000/api/integrations/preflight
```

The server also prints this loudly at startup if Sarvam is configured but
not working, rather than silently falling back and looking fine.

---

## What I used Claude Code for

I used Claude Code to write, debug and test the code in this repository,
through an iterative back and forth where I reviewed what it built, pushed
back on specific behaviour, and redirected it when something was wrong. A
number of the technical design decisions described above, including the
contradiction detection approach, the citation gate, and the hybrid search
method, came out of that process rather than being specified by me in
advance. I made the calls on what to build, what to cut, and what the
product needed to do. Claude wrote the implementation and, in several
places, proposed the approach after I described the problem.

The product positioning and vision documents were written independently
and are not generated by an AI tool.

---

## What I know is missing or weak

- **One user.** Signing in with Google identifies whoever is running the
  app, but nothing in the data is scoped per person. Making it multi-user
  is a real change, a user column on every table, not a flag to flip.
- **The connected app sign-ins, Gmail, Notion, Slack, are untested against
  real accounts.** They follow each provider's documented flow. The first
  real connection might hit a redirect URL mismatch or something else the
  docs did not warn about.
- **Live web answers, weather, prices, come from search result snippets,
  not a live feed.** They can be hours old. Kivi says so in the answer
  rather than stating a number like it is certain.
- **No Tamil, Hindi or code mixed audio in the evaluation.** The corpus has
  the sentences written down. Getting real speech samples for those and
  scoring them properly is the most useful thing left to do here.
- **Retrieval is good, not perfect,** which is exactly why the citation
  gate matters as much as it does. It is the backstop for when retrieval
  gets it wrong.
- **The evaluation ran on Groq, not Sarvam,** because the Sarvam account
  had no credit at the time. The corpus and the method are real. The
  numbers should be rerun on Sarvam before anyone treats them as Sarvam's
  own performance.
