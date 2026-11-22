"""Reporting only: never changes generation, selection, or frozen predictions."""
import csv
import html
import itertools
import json
import random
import statistics
from pathlib import Path

import numpy as np
import final_benchmark as b


def mean(values):
    return float(np.mean(values)) if values else None


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def collect(manifest, sources):
    records = {}
    aggregates = {}
    flat = []
    for track in b.TRACKS:
        aggregates[track] = {}
        for model in b.MODELS:
            cases = []
            for source in sources:
                path = b.WORK / "predictions" / track / model / (source["id"] + ".json")
                if not path.exists():
                    continue
                row = b.load(path)
                assert row["fingerprint"] == manifest["fingerprint"]
                supplied = source["common_input"] if track == "common_input" else source["article"]
                assert row["input_sha256"] == b.sha(supplied)
                cases.append(row)
                flat.append({"id": row["id"], "row_idx": row["row_idx"], "track": track, "model": model,
                             "status": row["status"], "summary": row["summary"], "error": row.get("error", ""),
                             **row["scores"], **{k: row.get(k) for k in (
                                 "output_words", "compression_original", "compression_supplied", "repeated_trigram_rate",
                                 "generation_seconds", "request_wall_seconds", "empty", "had_max_length_attempt",
                                 "unfinished_error", "unfinished_punctuation_heuristic", "source_chunks", "consolidation_passes",
                                 "covered_characters", "supplied_characters", "supplied_source_word_coverage")}})
            records[track, model] = {r["id"]: r for r in cases}
            if not cases:
                continue
            completed = [r for r in cases if r["status"] == "completed"]
            failed = [r for r in cases if r["status"] != "completed"]
            loads = [b.load(p) for p in (b.WORK / "loads").glob(model + "_*.json")]
            aggregate = {"attempted": len(cases), "completed": len(completed), "failed": len(failed),
                         "pending": 1000 - len(cases), "failure_rate": len(failed) / len(cases),
                         "scores": {m: mean([r["scores"][m] for r in cases]) for m in b.METRICS},
                         "mean_output_words": mean([r["output_words"] for r in cases]),
                         "mean_compression_original": mean([r["compression_original"] for r in cases]),
                         "mean_compression_supplied": mean([r["compression_supplied"] for r in cases]),
                         "mean_repeated_trigram_rate": mean([r["repeated_trigram_rate"] for r in cases]),
                         "empty": sum(r["empty"] for r in cases),
                         "unfinished_cap_failures": sum(r["unfinished_error"] for r in cases),
                         "unfinished_punctuation_heuristic": sum(r["unfinished_punctuation_heuristic"] for r in cases),
                         "max_length_attempts_including_recovered": sum(r["had_max_length_attempt"] for r in cases),
                         "mean_supplied_word_coverage": mean([r["supplied_source_word_coverage"] for r in cases]),
                         "mean_supplied_character_coverage": mean([r["supplied_source_character_coverage"] for r in cases]),
                         "completed_input_coverage": mean([r["covered_characters"] / r["supplied_characters"] for r in completed]),
                         "failure_input_coverage": "Unknown when request aborted; inspect saved traces. Not counted as complete.",
                         "mean_source_chunks_completed": mean([r["source_chunks"] for r in completed]),
                         "max_source_chunks_completed": max((r["source_chunks"] for r in completed), default=None),
                         "multichunk_completed": sum(r["source_chunks"] > 1 for r in completed),
                         "mean_consolidation_passes_completed": mean([r["consolidation_passes"] for r in completed]),
                         "max_consolidation_passes_completed": max((r["consolidation_passes"] for r in completed), default=None),
                         "generation_median_seconds_completed": percentile([r["generation_seconds"] for r in completed], 50),
                         "generation_p95_seconds_completed": percentile([r["generation_seconds"] for r in completed], 95),
                         "generation_median_seconds_all_attempted": percentile([r["generation_seconds"] for r in cases], 50),
                         "generation_p95_seconds_all_attempted": percentile([r["generation_seconds"] for r in cases], 95),
                         "request_median_seconds_all_attempted": percentile([r["request_wall_seconds"] for r in cases], 50),
                         "request_p95_seconds_all_attempted": percentile([r["request_wall_seconds"] for r in cases], 95),
                         "loads": loads, "parameters": manifest["model_verification"].get(model, {}).get("parameters", 0),
                         "failed_cases": [{"id": r["id"], "error": r["error"]} for r in failed]}
            aggregates[track][model] = aggregate
    b.save(b.REPORT / "aggregate_metrics.json", {"fingerprint": manifest["fingerprint"], "tracks": aggregates})
    if flat:
        with (b.REPORT / "predictions_and_scores.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    return aggregates, records


def bootstrap(manifest, sources, records):
    intervals = []
    rng = np.random.default_rng(manifest["bootstrap"]["seed"])
    for track in b.TRACKS:
        for first, second in itertools.combinations(b.MODELS, 2):
            common = [s["id"] for s in sources if s["id"] in records[track, first] and s["id"] in records[track, second]]
            if not common:
                continue
            indices = rng.integers(0, len(common), size=(manifest["bootstrap"]["replicates"], len(common)))
            for metric in b.METRICS:
                differences = np.array([records[track, first][i]["scores"][metric] - records[track, second][i]["scores"][metric] for i in common])
                samples = differences[indices].mean(axis=1)
                lo, hi = np.percentile(samples, [2.5, 97.5])
                intervals.append({"track": track, "first": first, "second": second, "metric": metric,
                                  "paired_articles": len(common), "mean_difference": float(differences.mean()),
                                  "ci_low": float(lo), "ci_high": float(hi)})
    b.save(b.REPORT / "paired_bootstrap.json", {"fingerprint": manifest["fingerprint"], "settings": manifest["bootstrap"],
                                               "note": "Pointwise percentile intervals; not multiplicity-adjusted. Failures score zero; pending rows excluded, n explicit.", "intervals": intervals})
    return intervals


def review(manifest, sources, records):
    rng = random.Random(manifest["blinding_seed"])
    source_by_id = {s["id"]: s for s in sources}
    cards = []
    keys = []
    ratings = []
    for index, identifier in enumerate(manifest["review_ids"], 1):
        source = source_by_id[identifier]
        for track in b.TRACKS:
            entries = [(model, records[track, model].get(identifier)) for model in b.MODELS]
            rng.shuffle(entries)
            supplied = source["common_input"] if track == "common_input" else source["article"]
            body = []
            for letter, (model, row) in zip("ABCDE", entries):
                label = f"{index:02d}-{track}-{letter}"
                keys.append({"review_id": label, "article_id": identifier, "model": model})
                ratings.append({"review_id": label, "main_event_coverage": "", "entity_relationship_errors": "",
                                "unsupported_claims": "", "attribution_uncertainty": "", "severity": "", "reviewer": "", "notes": ""})
                text = row["summary"] if row and row["status"] == "completed" else "[No completed summary]" if row else "[Pending]"
                body.append(f"<section><h3>Candidate {letter} — {label}</h3><p>{html.escape(text)}</p><p class='blank'>Human judgments: ____________________________</p></section>")
            cards.append(f"<article><h2>Article {index}: {track}</h2><p>ID: {identifier}</p>"
                         f"<details><summary>Complete original source</summary><pre>{html.escape(source['article'])}</pre></details>"
                         f"<details><summary>Exact supplied input</summary><pre>{html.escape(supplied)}</pre></details>"
                         f"<h3>Original reference</h3><p>{html.escape(source['reference'])}</p>{''.join(body)}</article>")
    document = """<!doctype html><meta charset="utf-8"><title>Blinded final benchmark review</title>
<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#192738}article{border-top:3px solid #234;padding:20px 0}section{background:#f3f6f8;padding:12px;margin:12px 0}pre{white-space:pre-wrap;font:inherit}.blank{color:#667}</style>
<h1>Blinded final benchmark review</h1><p>20 randomly selected frozen test articles, seed 142. Each track has five independently shuffled candidates (seed 242). Human ratings are blank; no factuality rate has been measured. Style can reveal model identity despite label blinding. Review main event, entities/relationships, unsupported claims and attribution against supplied and complete source. Common prefixes may omit reference facts. Keep review separate from automated ROUGE and error heuristics.</p>""" + "".join(cards)
    (b.REPORT / "blinded_review.html").write_text(document, encoding="utf-8")
    b.save(b.REPORT / "review_key.json", keys)
    # Never overwrite completed human work on a later report refresh.
    rating_path = b.REPORT / "human_review_blank.csv"
    if not rating_path.exists():
        with rating_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ratings[0]))
            writer.writeheader()
            writer.writerows(ratings)


def chart(aggregates, metric, title, filename):
    # Dependency-free standalone SVG: installed environment has no plotting library.
    values = [(t, m, aggregates[t][m]["scores"]["rougeLsum"] if metric == "rouge" else aggregates[t][m]["generation_median_seconds_completed"])
              for t in b.TRACKS for m in b.MODELS if m in aggregates[t] and (metric == "rouge" or m != "Lead-3")]
    if not values:
        return
    maximum = max(v or 0 for _, _, v in values) or 1
    height = 95 + 38 * len(values)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" viewBox="0 0 960 {height}" role="img"><title>{title}</title><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" fill="#243448"><text x="20" y="30" font-size="21">{title}</text>']
    for i, (track, model, value) in enumerate(values):
        y = 65 + i * 38
        color = "#3267b1" if track == "common_input" else "#218477"
        parts.append(f'<text x="20" y="{y+17}" font-size="14">{track} / {model}</text><rect x="270" y="{y}" width="{(value or 0)/maximum*560:.1f}" height="25" fill="{color}"/><text x="{280+(value or 0)/maximum*560:.1f}" y="{y+17}" font-size="14">{value or 0:.2f}</text>')
    parts.append('</g></svg>')
    (b.REPORT / filename).write_text("".join(parts), encoding="utf-8")


def examples(sources, records):
    """Transparent illustrative selection, not a factuality sample or rating."""
    source_map = {s["id"]: s for s in sources}
    selected = []
    cards = []
    for (track, model), mapping in records.items():
        ordered = sorted(mapping.values(), key=lambda r: (r["scores"]["rougeLsum"], r["id"]))
        if not ordered:
            continue
        choices = [("lowest ROUGE-Lsum", ordered[0]), ("median ROUGE-Lsum", ordered[len(ordered)//2]),
                   ("highest ROUGE-Lsum", ordered[-1])]
        choices += [("failed request", r) for r in ordered if r["status"] != "completed"]
        for reason, row in choices:
            source = source_map[row["id"]]
            entry = {"selection": reason, "track": track, "model": model, "id": row["id"],
                     "source": source["article"], "reference": source["reference"],
                     "supplied_input": source["common_input"] if track == "common_input" else source["article"],
                     "summary": row["summary"], "status": row["status"], "error": row.get("error"), "scores": row["scores"]}
            selected.append(entry)
            cards.append(f"<article><h2>{track} / {model}: {reason}</h2><p>ID {row['id']}; R-Lsum {row['scores']['rougeLsum']:.2f}</p>"
                         f"<details><summary>Full source</summary><pre>{html.escape(entry['source'])}</pre></details>"
                         f"<details><summary>Exact supplied input</summary><pre>{html.escape(entry['supplied_input'])}</pre></details>"
                         f"<h3>Reference</h3><p>{html.escape(entry['reference'])}</p><h3>Output</h3><p>{html.escape(entry['summary'])}</p>"
                         f"<p>{html.escape(entry['error'] or '')}</p></article>")
    b.save(b.REPORT / "illustrative_examples.json", selected)
    (b.REPORT / "illustrative_examples.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Illustrative benchmark outputs</title><style>body{font:16px/1.6 system-ui;max-width:1000px;margin:30px auto}pre{white-space:pre-wrap;font:inherit}article{border-top:2px solid #456;padding:20px}</style><h1>Illustrative outputs, including failures</h1><p>Lowest, median and highest ROUGE-Lsum per system/track, plus every failed request. Selection is transparent but not representative of factual accuracy. No human ratings or AI factuality scores are assigned. Use the random blinded packet for review.</p>' + ''.join(cards), encoding='utf-8')


def table(aggregates, track):
    lines = ["| System | Attempted / 1000 | Failed | R-1 | R-2 | R-L | R-Lsum | Words | Median / p95 seconds |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in b.MODELS:
        if name not in aggregates[track]:
            lines.append(f"| {name} | 0 | — | — | — | — | — | — | — |")
            continue
        a = aggregates[track][name]
        scores = " | ".join(f"{a['scores'][m]:.2f}" for m in b.METRICS)
        digits = 6 if name == "Lead-3" else 3
        timing = f"{a['generation_median_seconds_completed']:.{digits}f} / {a['generation_p95_seconds_completed']:.{digits}f}" if a['completed'] else "—"
        lines.append(f"| {name} | {a['attempted']} | {a['failed']} | {scores} | {a['mean_output_words']:.1f} | {timing} |")
    return "\n".join(lines)


def markdown(manifest, aggregates, intervals, records, sources):
    attempted = sum(a["attempted"] for t in aggregates.values() for a in t.values())
    completed_run = attempted == 10000
    text = f"""# Final CNN/DailyMail benchmark

**{'Completed' if completed_run else 'IN PROGRESS / PARTIAL'}: {attempted:,} of 10,000 system/article/track evaluations attempted.**
Each completed track targets the same 1,000 randomly sampled official CNN/DailyMail
3.0.0 test articles, seed 42. This is a complete-system comparison, not an isolated
architecture experiment. Production policy: **natural-checkpoint-v1**.

No training, checkpoint replacement, dependency reinstall, UI change or test-based
generation tuning was performed. Riverton remained separate synthetic diagnostic
data. Its PEGASUS omission appears in raw direct inference with the complete input;
[the diagnosis](reports/RIVERTON_DIAGNOSIS.md) explains why no generation fix was justified.

## A. Common-input comparison

{table(aggregates, 'common_input')}

Each system receives the **same exact original text prefix**, bounded by 400
production OpenNMT tokens and each HF tokenizer's 1024-token capacity including
specials. OpenNMT retains its existing approximate PTB preprocessing. References
are the complete original highlights, so they may describe omitted source text.
This is explicitly a prefix diagnostic/comparison, not full-article coverage.

## B. Full-article application comparison

{table(aggregates, 'full_article')}

All original text is submitted to the production coverage-checked pipeline.
Completed neural requests cover every source character; aborted requests are
reported as failures, never partial completed summaries. Lead-3 selects the first
three sentences using the same sentence segmenter. It reads full input but does
not claim to represent later facts. Source delivery is not factual coverage.

## Measurement and denominators

ROUGE-1/2/L/Lsum are F1 ×100, rouge-score 0.1.2 with Porter stemming. Lsum receives
newline-separated sentences from the same deterministic Punkt segmenter for both
reference and output. Failed/empty final summaries score **zero** and remain in
all attempted-case quality means. Pending cases are excluded and explicitly shown.
Words use whitespace splitting. Compression is summary words / original article
words, averaged per case; supplied-input compression is also saved. Repetition is
excess duplicate lowercase regex-word trigrams divided by all output trigrams.

Timing above is median/p95 **completed-request generation time**, summed across
all decode calls and retries, synchronized on the RTX 4090; excludes model loading,
queueing and input tokenization. All neural systems use CUDA FP32, TF32 disabled,
batch size one and one excluded fixed warmup per process. Full request wall times
and all-attempt latency statistics are saved separately. Lead-3 timing is **CPU
sentence extraction**, not a GPU/model speed comparison. This is a local Windows
desktop measurement, not an isolated hardware throughput study.

**GPU contention limitation:** a spot check found Fortnite and other desktop GPU
contexts present alongside Python. Per-process utilization/memory was unavailable
under Windows WDDM, so contention cannot be quantified or corrected. These times
are observed local latencies, **not evidence for a controlled speed ranking**.
No unrelated application was stopped. See
[hardware observation](reports/final_test_1000/hardware_observation.json).

BERTScore was omitted: its package and scorer weights were not installed/cached;
the working environment was preserved. ROUGE is reference similarity, **not proof
of factual correctness**. Neither longer output nor successful execution establishes
faithfulness. Native EOS does not establish grammatical or factual completeness.

## Coverage, repetition and termination

| Track | System | Compression % | Repeated trigrams % | Supplied word coverage % | Mean chunks* | Max chunks* | Extra consolidation passes* | Empty/failed | Cap failures | Punctuation flags |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for track in b.TRACKS:
        for name, a in aggregates[track].items():
            text += f"| {track} | {name} | {a['mean_compression_original']*100:.1f} | {a['mean_repeated_trigram_rate']*100:.2f} | {a['mean_supplied_word_coverage']*100:.1f} | {a['mean_source_chunks_completed'] or 0:.2f} | {a['max_source_chunks_completed']} | {a['mean_consolidation_passes_completed'] or 0:.3f} | {a['empty']} | {a['unfinished_cap_failures']} | {a['unfinished_punctuation_heuristic']} |\n"
    text += "\n*Completed requests only. Extra passes exclude source-order concatenation/exact-sentence deduplication. Punctuation flags are an automated heuristic, not human grammar ratings. Recovered maximum-length attempts are recorded separately from failed final outputs. Failure-stage coverage can be unknown; traces are preserved.\n\n## Parameters and loading\n\n| Model | Parameters | Cold load/transfer observations (seconds) |\n|---|---:|---|\n"
    for name in b.MODELS:
        a = aggregates['common_input'].get(name) or aggregates['full_article'].get(name)
        if a:
            loads = ", ".join(f"{r['cold_load_transfer_seconds']:.3f}" for r in a['loads']) or "N/A"
            text += f"| {name} | {a['parameters']:,} | {loads} |\n"
    text += "\nPEGASUS includes 2,097,152 fixed sinusoidal positional values; learned count 568,699,904. Load observations include process-start initialization/device transfer and filesystem cache variation; operational/resume processes are listed rather than folded into decode latency.\n\n## Paired uncertainty\n\n[All paired ROUGE differences](reports/final_test_1000/paired_bootstrap.json) use 2,000 article-level paired bootstrap replicates, seed 42, 95% percentile intervals. Failures retain zero scores. These are pointwise intervals without multiple-comparison correction.\n\n| Track | Difference (first − second) | R-Lsum difference | 95% CI | Paired n |\n|---|---|---:|---|---:|\n"
    for ci in intervals:
        if ci['metric'] == 'rougeLsum':
            text += f"| {ci['track']} | {ci['first']} − {ci['second']} | {ci['mean_difference']:.2f} | [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}] | {ci['paired_articles']} |\n"
    text += r"""
## Review and interpretation

The [blinded review packet](reports/final_test_1000/blinded_review.html) uses 20
random frozen sample IDs (seed 142), with shuffled anonymous candidates for both
tracks. [Human rating fields](reports/final_test_1000/human_review_blank.csv) are
blank. **No human factuality ratings or population factual-error rates have been
measured; no factual-quality winner is established.** Candidate style can partially
reveal identity. The separate key is for unblinding after review.

Main-event coverage, entity/relationship errors, unsupported claims, attribution
and uncertainty require source-grounded review. Automatic repetition, punctuation,
EOS and overlap metrics cannot substitute for it. Riverton and the earlier five
known development failures are diagnostic evidence, not estimates of test accuracy.

The released systems differ in size, training/pretraining, input convention and
decoding. OpenNMT preprocessing approximates historical Stanford CoreNLP and its
full-article window exceeds the 400-token training convention. Concatenated chunk
summaries may miss global relationships or repeat paraphrases even with complete
source delivery. Full-article and common-prefix outputs must not be pooled.

[Illustrative outputs and all request failures](reports/final_test_1000/illustrative_examples.html)
include lowest, median and highest ROUGE-Lsum examples under a transparent selection
rule. These are diagnostic examples, not representative factuality judgments.

## Reproduce and resume

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe tests\fetch_final_sources.py
.\.venv\Scripts\python.exe tests\final_benchmark.py prepare
.\.venv\Scripts\python.exe tests\final_benchmark.py run
.\.venv\Scripts\python.exe tests\report_final_benchmark.py
```

The run command validates code, dependency versions, model files, source hashes and
the complete frozen fingerprint before reusing any prediction. Existing failed
attempts are preserved rather than silently retried/removed. The first five IDs
were the operational run and remain included. No generation tuning followed.
An incompatible environment fails validation; recover the recorded environment
and `test-results/final_test_1000/frozen_code.zip` rather than overwrite results.

Local exact sources and shared prefixes: `data/final_test_1000/prepared_sources.json`.
Exact per-call inputs, raw outputs/IDs, decoded/displayed text, budgets, coverage,
timing and scores: `test-results/final_test_1000/predictions/<track>/<model>/<id>.json`.
These bulk files remain ignored by Git. Portable summaries and predictions are in
[predictions_and_scores.csv](reports/final_test_1000/predictions_and_scores.csv),
[aggregate metrics](reports/final_test_1000/aggregate_metrics.json), and
[frozen configuration/IDs](reports/final_test_1000/frozen_config.json).

![ROUGE-Lsum](reports/final_test_1000/rouge_comparison.svg)

![Generation latency](reports/final_test_1000/latency_comparison.svg)
"""
    (b.ROOT / "BENCHMARK_RESULTS.md").write_text(text, encoding="utf-8")


def main():
    manifest, sources = b.validate()
    aggregates, records = collect(manifest, sources)
    intervals = bootstrap(manifest, sources, records)
    review(manifest, sources, records)
    examples(sources, records)
    chart(aggregates, "rouge", "ROUGE-Lsum F1 x100 (failures included as zero)", "rouge_comparison.svg")
    chart(aggregates, "latency", "Median completed generation seconds (CUDA FP32, batch 1)", "latency_comparison.svg")
    markdown(manifest, aggregates, intervals, records, sources)
    for track in b.TRACKS:
        print(track)
        print(table(aggregates, track))


if __name__ == "__main__":
    main()
