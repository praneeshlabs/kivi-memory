# How to review this

**Primary review method: run the app and talk to it.**

Reading the code will tell you the design is sound. It will not tell you
whether Kivi is actually pleasant to use, whether it remembers the right
things, or whether it admits it does not know something instead of making
something up. That only shows up when you use it. Fifteen minutes with the
seeded data and the steps below will tell you more than the code will.

If you only have five minutes, do steps 1 to 4 and skip the rest.

---

## 1. Install

```bash
pip install -r requirements.txt
```

Needs Python 3.11 or newer.

## 2. Set up the environment file

```bash
copy .env.example .env
```

(`cp .env.example .env` on Mac or Linux.)

Open `.env` and add at least one key:

- `SARVAM_API_KEY` for the real stack this was built for. Get one at
  dashboard.sarvam.ai.
- `GROQ_API_KEY` works as a fallback and is enough to review everything
  except Indian language speech.

You need at least one of the two. Without either, the server refuses to
start and tells you why.

## 3. Load it with sample data

```bash
python seed.py --reset
```

This deletes any existing local database and writes in 16 memories: a mix
of facts, events and habits, some in Tamil and Hindi, the kind of thing a
real person would tell a voice assistant. It does not call the model, so it
costs nothing and gives you the same starting point every time.

## 4. Start the server

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser. Leave this running in its own
terminal window.

**Before you judge anything Sarvam related**, check what is actually
serving. A Sarvam key with no credit on the account authenticates fine and
then fails every real request, and the app quietly falls back to Groq
unless you check:

```bash
curl -s http://127.0.0.1:8000/api/integrations/preflight
```

`"ok": true` means Sarvam is answering. `"ok": false` means you are looking
at Groq right now, and the reason is in the response.

---

## 5. Things worth trying by hand

Type or say these into the app. Each one is testing something specific.

- **"Who is my manager?"** Answers from memory and shows which memory it
  used.
- **"What is the capital of France?"** Answers from general knowledge, and
  the interface marks it as not from your memory. Different from the first
  one on purpose.
- **"What is my phone number?"** Kivi does not know this. It should say so
  plainly, not guess.
- **"I moved to Bangalore last week."** Then ask "where do I live?" The
  city you gave in the seed data should be replaced, not sitting next to
  the new one.
- **"I always order filter coffee when I'm running late."** Say something
  close to this again in a separate message. The second time should turn
  it from a watched pattern into a stored preference, not a duplicate.
- Type something in Tamil or Hindi, in the browser's own script or in
  plain letters like "nalaiku doctor kitta poganum." It should be
  understood.

Open the Memory tab. Every memory shown should trace back to something you
actually said, with a timestamp. Archived memories, from the city change
above, should still be visible there, not gone.

## 6. Run the automated tests

```bash
python -m pytest tests/ -q
```

52 tests, no API key needed for these because the model is stubbed. This
checks the app's own logic: does a superseded fact get archived, does an
answer with no citation get thrown away instead of shown, does an old
database pick up new columns without losing data. Should finish in under a
minute and pass cleanly.

## 7. Run the evaluation against the real model

```bash
python eval/run_eval.py
```

This calls the real model, so it uses API credit and takes a couple of
minutes. Unlike the tests, this checks whether the model is actually good
at the job, not just whether the code is wired correctly. Results are
written to `eval/results/RESULTS.md` and `eval/results/results.json`,
overwriting what is already there. If you want to keep the existing
results and compare, copy the file somewhere first.

You can run one suite at a time:

```bash
python eval/run_eval.py --suite filter
python eval/run_eval.py --suite retrieval
python eval/run_eval.py --suite asr
```

---

## What "done" looks like at each step

| Step | You should see |
|---|---|
| Install | No errors. First run downloads a small language model, one time only. |
| Seed | "seeded 16 memories" with a breakdown by type. |
| Server start | "Application startup complete" with no port or key errors. |
| Preflight | A clear `ok: true` or `ok: false` with a reason. |
| Manual questions | Citations on memory answers, an honest decline on the phone number question, the city actually replaced not duplicated. |
| Tests | `52 passed` in the last line. |
| Evaluation | A written report. Read it, not just the console summary. The report explains the misses, it does not hide them. |

---

## If something does not start

**"No model provider configured"** — you did not put a key in `.env`, or
the file is not where the app expects it. It has to be named `.env`, not
`.env.example`, sitting next to `requirements.txt`.

**Port 8000 already in use** — something else is already running there.

```bash
netstat -ano | findstr :8000
taskkill /F /PID <the number in the last column>
```

**The evaluation looks slow or expensive** — the ASR suite generates audio
locally with the operating system's own speech voice, so it needs no extra
setup on Windows, and should degrade gracefully on other systems by
marking those cases untested rather than failing.

**Sarvam preflight always fails** — check the account has credit, not just
a valid key. Both look identical from a missing key, but a key with no
credit produces a different, specific error message in the preflight
response.
