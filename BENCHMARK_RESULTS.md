# Small summarization benchmark: 50 articles

The scope was reduced at the user's request. This is a quick reference-overlap
comparison, **not a completed 1,000-article final benchmark**. The larger run was
stopped; its saved predictions and frozen settings are preserved separately.

All five systems use the same first 50 articles from the frozen random CNN/DailyMail
3.0.0 test sample (seed 42), with original references. No generation settings,
models, dependencies or UI were changed. Lead-3 extracts the first three sentences.

ROUGE values below are F1 ×100 with Porter stemming; failures would score zero.
Higher ROUGE means greater reference overlap, **not verified factual accuracy**.
This small sample supports a rough comparison, not a broad factual-quality ranking.

## Full-article application

| System | Articles | R-1 | R-2 | R-L | R-Lsum | Mean words | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 50 | 40.95 | 20.34 | 28.71 | 38.22 | 55.0 | 0 |
| Transformer | 50 | 38.75 | 17.35 | 26.99 | 36.09 | 47.4 | 0 |
| BART | 50 | 45.21 | 22.50 | 30.88 | 42.73 | 77.4 | 0 |
| PEGASUS | 50 | 43.81 | 22.43 | 30.92 | 40.91 | 68.9 | 0 |
| Lead-3 | 50 | 40.36 | 18.28 | 24.91 | 36.30 | 74.4 | 0 |

## Shared-prefix comparison

| System | Articles | R-1 | R-2 | R-L | R-Lsum | Mean words | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 50 | 38.14 | 18.33 | 26.82 | 35.53 | 50.2 | 0 |
| Transformer | 50 | 39.09 | 18.61 | 28.18 | 35.95 | 46.7 | 0 |
| BART | 50 | 43.83 | 20.71 | 30.37 | 40.73 | 62.5 | 0 |
| PEGASUS | 50 | 42.59 | 21.34 | 30.24 | 39.88 | 53.9 | 0 |
| Lead-3 | 50 | 40.36 | 18.28 | 24.91 | 36.30 | 74.4 | 0 |


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
