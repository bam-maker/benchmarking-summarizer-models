"""Small requested ROUGE comparison; reuse the first 50 fixed IDs, no inference."""
import csv
import json
import statistics
import final_benchmark as b


def main():
    manifest, sources = b.validate()
    sources = sources[:50]
    destination = b.ROOT / "reports/pilot50"
    destination.mkdir(parents=True, exist_ok=True)
    results = {}
    rows = []
    for track in b.TRACKS:
        results[track] = {}
        for model in b.MODELS:
            cases = [b.load(b.WORK / "predictions" / track / model / (s["id"] + ".json")) for s in sources]
            assert len(cases) == 50 and all(c["fingerprint"] == manifest["fingerprint"] for c in cases)
            for source, case in zip(sources, cases):
                assert case["id"] == source["id"]
                expected = source["common_input"] if track == "common_input" else source["article"]
                assert case["input_sha256"] == b.sha(expected)
                rows.append({"track": track, "model": model, "id": case["id"], "reference": source["reference"],
                             "summary": case["summary"], "status": case["status"], **case["scores"]})
            results[track][model] = {
                "n": 50, "failed": sum(c["status"] != "completed" for c in cases),
                **{m: statistics.mean(c["scores"][m] for c in cases) for m in b.METRICS},
                "mean_words": statistics.mean(c["output_words"] for c in cases),
                "median_generation_seconds": statistics.median(c["generation_seconds"] for c in cases),
                "repeated_trigram_rate": statistics.mean(c["repeated_trigram_rate"] for c in cases),
            }
    b.save(destination / "results.json", {"kind": "Small 50-article comparison; not the completed 1000-article benchmark",
            "selection": "First 50 IDs in the previously frozen random seed-42 sample; identical for every system; no generation tuning",
            "fingerprint": manifest["fingerprint"], "ids": [s["id"] for s in sources], "results": results})
    with (destination / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = """# Small summarization benchmark: 50 articles

The scope was reduced at the user's request. This is a quick reference-overlap
comparison, **not a completed 1,000-article final benchmark**. The larger run was
stopped; its saved predictions and frozen settings are preserved separately.

All five systems use the same first 50 articles from the frozen random CNN/DailyMail
3.0.0 test sample (seed 42), with original references. No generation settings,
models, dependencies or UI were changed. Lead-3 extracts the first three sentences.

ROUGE values below are F1 ×100 with Porter stemming; failures would score zero.
Higher ROUGE means greater reference overlap, **not verified factual accuracy**.
This small sample supports a rough comparison, not a broad factual-quality ranking.
"""
    for track, title in (("full_article", "Full-article application"), ("common_input", "Shared-prefix comparison")):
        text += f"\n## {title}\n\n| System | Articles | R-1 | R-2 | R-L | R-Lsum | Mean words | Failed |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
        for model, a in results[track].items():
            text += f"| {model} | 50 | {a['rouge1']:.2f} | {a['rouge2']:.2f} | {a['rougeL']:.2f} | {a['rougeLsum']:.2f} | {a['mean_words']:.1f} | {a['failed']} |\n"
    text += r"""

Full-article runs use the existing natural-checkpoint-v1 production pipeline.
The shared-prefix track uses identical text bounded by 400 OpenNMT tokens and
each HF tokenizer's capacity. It still uses full original references, which may
mention facts outside the prefix. Keep the two tracks separate.

Generation times remain in the JSON, but Fortnite/desktop GPU contexts were present;
these are not controlled speed comparisons. No human factuality ratings were made.
Riverton remains a separate synthetic diagnostic: PEGASUS's omission also occurs in
direct inference with complete input. That did not justify a pipeline change.

Files: [metrics and IDs](reports/pilot50/results.json),
[outputs and references](reports/pilot50/predictions.csv),
[Riverton diagnosis](reports/RIVERTON_DIAGNOSIS.md).
The interrupted larger run remains in `test-results/final_test_1000/` and
`reports/final_test_1000/`; its partial results must not be called a completed final benchmark.

To reproduce this small comparison from the repository in PowerShell:

```powershell
.\.venv\Scripts\python.exe tests\final_benchmark.py run --limit 50
.\.venv\Scripts\python.exe tests\report_pilot50.py
```

No more large runs, tuning or analysis are needed for this rough comparison.
"""
    (b.ROOT / "BENCHMARK_RESULTS.md").write_text(text, encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
