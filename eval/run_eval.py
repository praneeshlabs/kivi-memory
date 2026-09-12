"""Run the evaluation and write the results.

    python eval/run_eval.py              # all three suites
    python eval/run_eval.py --suite filter
    python eval/run_eval.py --suite retrieval
    python eval/run_eval.py --suite asr

This calls the real model, so it costs API credits and the numbers move a
little between runs. That is the point. The unit tests pin behaviour with a
stubbed model; this measures how well the real thing actually does.

Seed the database first, or retrieval has nothing to search:

    python seed.py --reset
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

CORPUS = Path(__file__).resolve().parent / "corpus"
RESULTS = Path(__file__).resolve().parent / "results"


def load(name: str) -> list[dict]:
    path = CORPUS / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------------- filter

def run_filter() -> dict:
    """Does Kivi keep what matters and drop what does not?"""
    from app.services.memory_pipeline import classify_take

    cases = load("filter")
    rows = []

    for case in cases:
        items = classify_take(case["take"])
        decisions = [str(i.get("decision", "")).lower() for i in items]
        tags = [i.get("tag") for i in items]

        if not items:
            got = "drop"
        elif "save" in decisions:
            got = "save"
        elif "watch" in decisions:
            got = "watch"
        else:
            got = "drop"

        expected = case["expect"]
        tag_ok = case.get("tag") is None or case["tag"] in tags
        also = case.get("also_expect")
        also_ok = also is None or also in tags

        rows.append({
            "id": case["id"],
            "take": case["take"],
            "expected": expected,
            "got": got,
            "pass": got == expected and tag_ok and also_ok,
            "expected_tag": case.get("tag"),
            "got_tags": tags,
            "also_expected": also,
            "also_found": also_ok,
            "note": case["note"],
        })

    return {"suite": "filter", "cases": rows, **_filter_metrics(rows)}


def _filter_metrics(rows: list[dict]) -> dict:
    """Per-class precision and recall, plus the number that matters most.

    A wrongly dropped memory is the expensive error: the user told Kivi
    something and it was thrown away. A wrongly kept one is clutter.
    """
    metrics = {}
    for label in ("save", "watch", "drop"):
        predicted = [r for r in rows if r["got"] == label]
        actual = [r for r in rows if r["expected"] == label]
        hits = [r for r in predicted if r["expected"] == label]
        metrics[label] = {
            "precision": round(len(hits) / len(predicted), 3) if predicted else None,
            "recall": round(len(hits) / len(actual), 3) if actual else None,
            "support": len(actual),
        }

    kept = [r for r in rows if r["expected"] in ("save", "watch")]
    wrongly_dropped = [r for r in kept if r["got"] == "drop"]

    return {
        "accuracy": round(sum(r["pass"] for r in rows) / len(rows), 3),
        "decision_accuracy": round(
            sum(r["got"] == r["expected"] for r in rows) / len(rows), 3
        ),
        "per_class": metrics,
        "wrongly_dropped": [r["id"] for r in wrongly_dropped],
        "wrongly_dropped_count": len(wrongly_dropped),
    }


# ---------------------------------------------------------------- retrieval

def run_retrieval() -> dict:
    """Does the right memory come back, and does Kivi decline when it should?"""
    from app.db import get_connection
    from app.services.qa_pipeline import cached_embed
    from app.services.retrieval import hybrid_search

    cases = load("retrieval")
    conn = get_connection()
    rows = []
    try:
        for case in cases:
            ranked = hybrid_search(conn, case["query"], cached_embed(case["query"]), 5)
            contents = [row["content"] for _, row in ranked]
            top_score = ranked[0][0] if ranked else 0.0
            wanted = case["expect_contains"]

            if wanted is None:
                # Nothing should match. Success is the right memory NOT being
                # near the top, which the citation gate then turns into a
                # decline. Retrieval alone cannot refuse.
                rank = None
                passed = True
                note_result = "no target, handled by the citation gate"
            else:
                rank = next(
                    (i + 1 for i, c in enumerate(contents)
                     if wanted.lower() in c.lower()), None
                )
                passed = rank is not None and rank <= 3
                note_result = f"rank {rank}" if rank else "not retrieved"

            rows.append({
                "id": case["id"],
                "query": case["query"],
                "expect_contains": wanted,
                "rank": rank,
                "top_score": round(float(top_score), 3),
                "pass": passed,
                "result": note_result,
                "note": case["note"],
            })
    finally:
        conn.close()

    targeted = [r for r in rows if r["expect_contains"] is not None]
    return {
        "suite": "retrieval",
        "cases": rows,
        "hit_at_1": round(sum(r["rank"] == 1 for r in targeted) / len(targeted), 3),
        "hit_at_3": round(
            sum(r["rank"] is not None and r["rank"] <= 3 for r in targeted) / len(targeted), 3
        ),
        "hit_at_5": round(
            sum(r["rank"] is not None for r in targeted) / len(targeted), 3
        ),
        "missed": [r["id"] for r in targeted if r["rank"] is None],
        "mean_top_score": round(
            sum(r["top_score"] for r in targeted) / len(targeted), 3
        ),
    }


# ---------------------------------------------------------------------- asr

def _normalise(text: str) -> list[str]:
    keep = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
    return keep.split()


def _wer(reference: str, hypothesis: str) -> float:
    """Word error rate by edit distance."""
    ref, hyp = _normalise(reference), _normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1] / len(ref)


def _synthesize(text: str, out_path: Path) -> bool:
    """Make a WAV with the Windows speech engine. Returns False if unavailable."""
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{out_path}'); "
        f"$s.Speak('{text}'); $s.Dispose()"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=True, capture_output=True, timeout=60,
        )
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def run_asr() -> dict:
    """Word error rate on spoken dictation.

    English cases are synthesised locally so the suite runs anywhere. The
    Tamil, Hindi and code-mixed cases are marked synth=false: a text to speech
    voice reading them is not a fair test of how a person actually speaks, so
    they are reported as untested rather than given a fake score.
    """
    from app.services.transcription import transcribe_with_language

    cases = load("asr")
    rows = []
    tmp = Path(tempfile.mkdtemp())

    for case in cases:
        if not case.get("synth"):
            rows.append({
                "id": case["id"], "lang": case["lang"],
                "reference": case["reference"], "hypothesis": None,
                "wer": None, "status": "untested, needs a human recording",
                "note": case["note"],
            })
            continue

        wav = tmp / f"{case['id']}.wav"
        if not _synthesize(case["reference"], wav):
            rows.append({
                "id": case["id"], "lang": case["lang"],
                "reference": case["reference"], "hypothesis": None,
                "wer": None, "status": "could not synthesise audio on this machine",
                "note": case["note"],
            })
            continue

        result = transcribe_with_language(wav.name, wav.read_bytes())
        if not result:
            rows.append({
                "id": case["id"], "lang": case["lang"],
                "reference": case["reference"], "hypothesis": None,
                "wer": None, "status": "transcription failed", "note": case["note"],
            })
            continue

        hypothesis, detected = result
        rows.append({
            "id": case["id"], "lang": case["lang"],
            "reference": case["reference"], "hypothesis": hypothesis,
            "detected_language": detected,
            "wer": round(_wer(case["reference"], hypothesis), 3),
            "status": "scored", "note": case["note"],
        })

    scored = [r for r in rows if r["wer"] is not None]
    return {
        "suite": "asr",
        "cases": rows,
        "scored_count": len(scored),
        "untested_count": len(rows) - len(scored),
        "mean_wer": round(sum(r["wer"] for r in scored) / len(scored), 3) if scored else None,
        "perfect": [r["id"] for r in scored if r["wer"] == 0.0],
        "worst": sorted(scored, key=lambda r: -r["wer"])[0]["id"] if scored else None,
    }


# -------------------------------------------------------------------- report

def write_report(results: list[dict], provider: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "suites": results,
    }
    (RESULTS / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Evaluation results",
        "",
        f"Run on {stamp}.",
        f"Chat model: `{provider.get('chat_model')}` (provider: {provider.get('active')}).",
        f"Speech model: `{provider.get('stt_model')}`.",
    ]
    if provider.get("sarvam_note"):
        lines.append(
            f"Sarvam was configured but not serving this run: {provider['sarvam_note']}. "
            f"Rerun after fixing that to get Sarvam's own numbers."
        )
    lines += [
        "",
        "Regenerate with `python eval/run_eval.py`. Numbers move slightly between",
        "runs because the model is not deterministic.",
        "",
    ]

    for suite in results:
        if suite["suite"] == "filter":
            lines += _filter_report(suite)
        elif suite["suite"] == "retrieval":
            lines += _retrieval_report(suite)
        elif suite["suite"] == "asr":
            lines += _asr_report(suite)

    (RESULTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return RESULTS / "RESULTS.md"


def _filter_report(suite: dict) -> list[str]:
    lines = [
        "## Memory filter",
        "",
        "Does Kivi keep what matters and drop what does not? 18 labelled takes.",
        "",
        f"Full accuracy (decision and tag correct): **{suite['accuracy']:.0%}**  ",
        f"Decision only: **{suite['decision_accuracy']:.0%}**  ",
        f"Wrongly dropped: **{suite['wrongly_dropped_count']}**"
        f" {suite['wrongly_dropped'] or ''}",
        "",
        "| Decision | Precision | Recall | Cases |",
        "|---|---|---|---|",
    ]
    for label, m in suite["per_class"].items():
        p = f"{m['precision']:.2f}" if m["precision"] is not None else "n/a"
        r = f"{m['recall']:.2f}" if m["recall"] is not None else "n/a"
        lines.append(f"| {label} | {p} | {r} | {m['support']} |")

    failures = [c for c in suite["cases"] if not c["pass"]]
    lines += ["", f"### Failures ({len(failures)} of {len(suite['cases'])})", ""]
    if not failures:
        lines.append("None.")
    else:
        lines += ["| Take | Wanted | Got | Why it matters |", "|---|---|---|---|"]
        for c in failures:
            take = c["take"][:52].replace("|", " ")
            lines.append(
                f"| {take} | {c['expected']} ({c['expected_tag']}) | "
                f"{c['got']} ({', '.join(str(t) for t in c['got_tags']) or 'none'}) | {c['note']} |"
            )
    return lines + [""]


def _retrieval_report(suite: dict) -> list[str]:
    lines = [
        "## Retrieval",
        "",
        "Does the right memory come back for a question? 14 targeted queries",
        "plus 3 that should find nothing.",
        "",
        f"hit@1: **{suite['hit_at_1']:.0%}**  ",
        f"hit@3: **{suite['hit_at_3']:.0%}**  ",
        f"hit@5: **{suite['hit_at_5']:.0%}**  ",
        f"Mean top score: {suite['mean_top_score']}",
        "",
        "| Query | Wanted | Result |",
        "|---|---|---|",
    ]
    for c in suite["cases"]:
        query = c["query"][:42].replace("|", " ")
        wanted = c["expect_contains"] or "nothing"
        mark = "" if c["pass"] else " **FAIL**"
        lines.append(f"| {query} | {wanted} | {c['result']}{mark} |")
    return lines + [""]


def _asr_report(suite: dict) -> list[str]:
    mean = f"{suite['mean_wer']:.1%}" if suite["mean_wer"] is not None else "n/a"
    lines = [
        "## Speech to text",
        "",
        "Word error rate on dictation. English cases are synthesised locally so",
        "the suite runs on any machine. Tamil, Hindi and code-mixed cases are",
        "listed but not scored: a synthetic voice reading them would not tell",
        "you anything about how a person actually speaks.",
        "",
        f"Mean WER on scored cases: **{mean}** ({suite['scored_count']} scored, "
        f"{suite['untested_count']} untested)",
        "",
        "| Case | WER | Heard |",
        "|---|---|---|",
    ]
    for c in suite["cases"]:
        if c["wer"] is None:
            lines.append(f"| {c['id']} ({c['lang']}) | not scored | {c['status']} |")
        else:
            heard = (c["hypothesis"] or "")[:56].replace("|", " ")
            lines.append(f"| {c['id']} ({c['lang']}) | {c['wer']:.1%} | {heard} |")
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Kivi evaluation.")
    parser.add_argument("--suite", choices=["filter", "retrieval", "asr"],
                        help="run one suite instead of all three")
    args = parser.parse_args()

    from app.db import init_db
    from app.services.llm_client import GROQ_MODEL, GROQ_TRANSCRIBE_MODEL, preflight, provider_status

    init_db()
    # A live check, not just "is a key present" — the eval report must say
    # which model actually answered, not which one was merely configured.
    check = preflight()
    provider = provider_status()
    if not check["ok"]:
        # provider_status() reports the configured model optimistically; once
        # a live call has actually failed, correct it to what really served.
        provider = {
            **provider,
            "active": "groq (sarvam fallback)",
            "chat_model": GROQ_MODEL,
            "stt_model": GROQ_TRANSCRIBE_MODEL,
            "sarvam_note": check["reason"],
        }
        print(f"NOTE: Sarvam is not serving ({check['reason']}). "
              f"This run used the Groq fallback.")
    print(f"provider: {provider['active']}  chat: {provider['chat_model']}")

    suites = [args.suite] if args.suite else ["filter", "retrieval", "asr"]
    results = []

    for name in suites:
        print(f"\nrunning {name} ...")
        started = time.perf_counter()
        results.append({"filter": run_filter, "retrieval": run_retrieval, "asr": run_asr}[name]())
        print(f"  done in {time.perf_counter() - started:.0f}s")

    path = write_report(results, provider)
    print(f"\nwrote {path}")
    for suite in results:
        headline = {
            "filter": lambda s: f"accuracy {s['accuracy']:.0%}, wrongly dropped {s['wrongly_dropped_count']}",
            "retrieval": lambda s: f"hit@3 {s['hit_at_3']:.0%}, hit@1 {s['hit_at_1']:.0%}",
            "asr": lambda s: f"mean WER {s['mean_wer']:.1%}" if s["mean_wer"] is not None else "not scored",
        }[suite["suite"]]
        print(f"  {suite['suite']}: {headline(suite)}")


if __name__ == "__main__":
    main()
