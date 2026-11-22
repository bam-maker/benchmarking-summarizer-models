"""Evidence synthesis only: no model imports, inference, downloads or metric tuning."""

import csv
import hashlib
import html
import json
import random
import statistics
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/model_capabilities"
GROUPS = (
    ("short", 0, 400),
    ("medium", 400, 800),
    ("long_single_pass", 800, 1024),
    ("beyond_window", 1024, float("inf")),
)


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save(name, value):
    (OUT / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = read("reports/validation_40_results.json")
    manifest = read("reports/validation_40_manifest.json")
    audit = read("reports/factual_audit_results.json")
    sources = read("data/validation_40/sources.json")
    annotations = read("reports/validation_40_ai_review.json")
    protected = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (ROOT / "reports").iterdir()
        if p.is_file()
    }
    protected.update(manifest["generation_code_sha256"])
    for path, expected in protected.items():
        assert digest(ROOT / path) == expected
    archive = OUT / "application_snapshot.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as package:
            for path in manifest["generation_code_sha256"]:
                package.write(ROOT / path, path)
    with zipfile.ZipFile(archive) as package:
        assert set(package.namelist()) == set(manifest["generation_code_sha256"])
        for path, expected in manifest["generation_code_sha256"].items():
            assert hashlib.sha256(package.read(path)).hexdigest() == expected
    save(
        "snapshot_manifest.json",
        {
            "archive": archive.name,
            "archive_sha256": digest(archive),
            "files": manifest["generation_code_sha256"],
            "restore": "Extract into a separate review directory first; compare checksums, then deliberately copy selected files into the project after saving intervening work. No weights/environment included. Existing pinned recovery scripts restore those separately.",
            "preserved_evidence_sha256": protected,
        },
    )
    groups = []
    for model in results["models"]:
        for label, low, high in GROUPS:
            rows = [
                r
                for r in results["cases"]
                if r["model"] == model
                and low < r["input_tokens_including_source_specials"] <= high
            ]
            groups.append(
                {
                    "model": model,
                    "group": label,
                    "n": len(rows),
                    "row_ids": [
                        {"row_idx": r["row_idx"], "id": r["article_id"]} for r in rows
                    ],
                    "token_range": [
                        min(r["input_tokens_including_source_specials"] for r in rows),
                        max(r["input_tokens_including_source_specials"] for r in rows),
                    ],
                    "word_range": [
                        min(r["source_words"] for r in rows),
                        max(r["source_words"] for r in rows),
                    ],
                    "mean_rouge_f1_percent": {
                        k: statistics.mean(r["rouge_f1"][k] * 100 for r in rows)
                        for k in ("rouge1", "rouge2", "rougeL")
                    },
                    "mean_output_words": statistics.mean(
                        r["output_words"] for r in rows
                    ),
                    "mean_generation_seconds": statistics.mean(
                        r["generation_seconds"] for r in rows
                    ),
                    "mean_shorter_by_percent": statistics.mean(
                        r["shorter_by_percent"] for r in rows
                    ),
                    "mean_excess_trigram_percent": statistics.mean(
                        r["repetition"]["trigram_excess_fraction"] * 100 for r in rows
                    ),
                    "failed": sum(r["status"] != "completed" for r in rows),
                    "no_final_punctuation": sum(
                        r["missing_terminal_punctuation"] for r in rows
                    ),
                    "complete_input_coverage": sum(
                        r["completed_full_article_summary"] for r in rows
                    ),
                    "experiment": "C: full-article chunk method"
                    if low == 1024
                    else "Application Medium single-pass; not identical to checkpoint A",
                    "ai_notes": [
                        {
                            "row_idx": r["row_idx"],
                            "note": annotations[str(r["row_idx"])][model],
                        }
                        for r in rows
                    ],
                }
            )
    save(
        "input_ranges.json",
        {
            "population": "Inspected development validation rows2-41; not untouched final test",
            "groups": "Model-specific source tokens including specials where applicable: short<=400, medium401-800, long801-1024, beyond>1024. Same underlying40 IDs; groups may differ between tokenizers. These are descriptive post-hoc strata, not causal comparisons or safety thresholds.",
            "memory": "No per-request allocated/reserved/peak memory series exists in the reused reports; do not estimate measured VRAM from parameter counts.",
            "rows": groups,
        },
    )
    with (OUT / "input_ranges.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model",
                "group",
                "n",
                "tokens_min",
                "tokens_max",
                "words_min",
                "words_max",
                "R1_F1_percent",
                "R2_F1_percent",
                "RL_F1_percent",
                "output_words_mean",
                "generation_seconds_mean",
                "repeated_trigrams_percent_mean",
                "no_final_punctuation",
            ]
        )
        for r in groups:
            writer.writerow(
                [
                    r["model"],
                    r["group"],
                    r["n"],
                    *r["token_range"],
                    *r["word_range"],
                    *r["mean_rouge_f1_percent"].values(),
                    r["mean_output_words"],
                    r["mean_generation_seconds"],
                    r["mean_excess_trigram_percent"],
                    r["no_final_punctuation"],
                ]
            )
    # Preserve defaults as data, not new product presets. Explicitly separate A/B/C.
    save(
        "baseline_protocol.json",
        {
            "A": {
                "method": "Direct single pass; original tokenizer/checkpoint vocabulary; deterministic search. Full source only when it fits. Any truncated reference run labeled prefix diagnostic.",
                "OpenNMT": read("reports/factual_audit_manifest.json")[
                    "opennmt_recipe"
                ],
                "HF_literal_checkpoint_generation_configs": results[
                    "checkpoint_generation_configs"
                ],
                "EOS": "HF literal defaults can force EOS at cap; preserve that reproducibility control. Natural-EOS baseline is the identical configuration with only forced_eos_token_id=None, already tested in factual audit. Report a cap without EOS as unfinished; do not silently append EOS or extend baseline.",
                "source_window": {
                    "LSTM": 400,
                    "Transformer": 400,
                    "BART": 1024,
                    "PEGASUS": 1024,
                },
            },
            "B": {
                "status": "No calibrated production presets promoted",
                "Medium": "Candidate starting point is natural-EOS checkpoint A, not an article percentage.",
                "Short": "Future candidate changes only A minimum downward; hold maximum/search fixed, judge event preservation. Lower ceilings need separate termination checks.",
                "Detailed": "Future candidate changes only A maximum upward within app/decoder limits; hold minimum/search fixed, accept identical EOS output.",
                "existing_evidence": "172-call audit varies length/EOS controls independently. Most outputs unchanged; bombing example adds secondary detail at native minimum without restoring event. No separate Short/Medium/Detailed factuality validation supports promotion.",
                "promotion_gate": "Freeze one numeric candidate per model before inference; use separately saved validation IDs outside inspected0-41 (42-47 reserved but not fetched). Human source review must support fidelity and desired preference without worsened errors; otherwise reject. No repeated tuning on those rows. No candidate run is needed merely to retain existing behavior.",
            },
            "C": {
                "method": "Current full-article chunk/merge/consolidation with safeguards, evaluated independently from A/B",
                "measured": "Rows33/39 need two source chunks per model; no additional generative consolidation in40-case evaluation. Row39 omission occurs before merge. Earlier synthetic deep-loop checks establish execution, not news factuality.",
            },
            "audit_checks_reused": audit["automated_checks"],
        },
    )
    blind(sources)
    for path, expected in protected.items():
        assert digest(ROOT / path) == expected
    print(
        json.dumps(
            {
                "groups": len(groups),
                "blind_cases": 6,
                "blind_summaries": 24,
                "protected_files_unchanged": len(protected),
                "new_inference_calls": 0,
            }
        )
    )


def blind(sources):
    selected = [2, 19, 21, 30, 35, 39]
    rng = random.Random(73219)
    rng.shuffle(selected)
    key, rating_rows, blocks = [], [], []
    rubric = "Read the article first; identify actor/action/outcome. Rate each anonymous summary independently before opening the reference. Main-event correctness:0 absent/wrong,1 partial or ambiguous,2 correct core event. Entity/relationship errors:count distinct erroneous claims (0 if none). Unsupported claims:count claims not supported by article. Important omissions:0 none,1 secondary,2 main event/key outcome. Readability:1 severely broken,2 difficult,3 understandable,4 clear,5 fluent. Attribution/uncertainty:preserved, changed, or not_applicable. For each error cite a summary phrase and supporting source sentence; use uncertain when evidence is ambiguous. Do not infer author names from the reference or outside knowledge. Reference highlights are incomplete and may contain unsupported material. Repetition and incomplete sentences should be noted even with final punctuation. Leave blanks until a person reviews; do not treat ROUGE as factuality. No AI notes or model identities are supplied in this packet."
    for index, number in enumerate(selected, 1):
        source = next(s for s in sources if s["row_idx"] == number)
        names = ["LSTM", "Transformer", "BART", "PEGASUS"]
        rng.shuffle(names)
        case = f"Case{index:02d}"
        block = f"<article><h2>{case}</h2><h3>Article</h3><p>{html.escape(source['article'])}</p>"
        for code, model in zip("ABCD", names, strict=True):
            saved = read(f"test-results/validation_40/cases/{number:03d}_{model}.json")
            candidate = f"{case}-{code}"
            key.append(
                {
                    "candidate": candidate,
                    "row_idx": number,
                    "article_id": source["id"],
                    "model": model,
                    "summary_sha256": hashlib.sha256(
                        saved["summary"].encode()
                    ).hexdigest(),
                }
            )
            rating_rows.append(
                {
                    "candidate": candidate,
                    **{
                        k: ""
                        for k in (
                            "reviewer",
                            "main_event_0_2",
                            "entity_relationship_error_count",
                            "unsupported_claim_count",
                            "important_omissions_0_2",
                            "readability_1_5",
                            "attribution_uncertainty",
                            "repetition_or_incomplete",
                            "evidence_and_uncertainty",
                        )
                    },
                }
            )
            block += f"<section><h3>{candidate}</h3><p>{html.escape(saved['summary'])}</p></section>"
        block += f"<details><summary>Reference — open after initial source-based ratings</summary><p>{html.escape(source['reference'])}</p></details></article>"
        blocks.append(block)
    save(
        "blinding_key_DO_NOT_SHARE_WITH_REVIEWER.json",
        {
            "selection": "Purposive six-case development review: one plausible success and five diagnosed cases, not an unbiased error-rate estimate. Independently shuffled model order per case; partial blinding only because style/casing or prior exposure may reveal identity.",
            "mapping": key,
        },
    )
    with (OUT / "human_ratings_BLANK.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rating_rows[0]))
        writer.writeheader()
        writer.writerows(rating_rows)
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Anonymous source-summary review</title><style>body{font:16px/1.6 system-ui;max-width:1050px;margin:25px auto;padding:20px}article{border-top:2px solid #888;margin-top:35px}section{padding:15px;background:#f3f5f4;margin:15px 0}details{padding:15px}p{white-space:pre-wrap}</style><h1>Anonymous source-summary comparison</h1><p>'
        + html.escape(rubric)
        + '</p><p>Copy <a href="human_ratings_BLANK.csv">the blank rating CSV</a> to a reviewer-specific filename before entering ratings. This packet reuses inspected development examples; it cannot estimate general error rates. Casing/style are preserved, so blinding is imperfect. Please report if you recognize a summary.</p>'
        + "".join(blocks)
        + "</html>"
    )
    (OUT / "blind_review.html").write_text(page, encoding="utf-8")
    assert len(key) == 24 and len({k["candidate"] for k in key}) == 24
    assert all(
        not value for r in rating_rows for k, value in r.items() if k != "candidate"
    )


if __name__ == "__main__":
    main()
