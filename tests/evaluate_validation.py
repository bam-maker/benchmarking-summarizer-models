"""Frozen 40-item validation evaluation, not training or the final test benchmark.

Uses the app unchanged. Public validation text/raw traces stay ignored locally.
ROUGE is isolated under .cache/eval-metrics; no app dependency is upgraded.
"""

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".cache" / "eval-metrics"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from diagnose_quality import Trace
from rouge_score import rouge_scorer

import article_summarizer as full
import summarizer as engine
from runtime import synchronize

WORK = ROOT / "test-results" / "validation_40"
DATA = ROOT / "data" / "validation_40" / "sources.json"
MANIFEST = ROOT / "reports" / "validation_40_manifest.json"
FROZEN_FILES = (
    "app.py",
    "article_summarizer.py",
    "summarizer.py",
    "hf_summarizer.py",
    "runtime.py",
    "processing_config.json",
    "model_manifest.json",
    "requirements.txt",
    "OpenNMT_Verified_Setup/checkpoint_manifest.json",
)
SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def hashes():
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in FROZEN_FILES
    }


def prepare():
    if DATA.exists():
        load_frozen()
        print("Existing frozen sources verified; no download or replacement.")
        return
    existing = (
        json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else None
    )
    if existing:
        assert hashes() == existing["generation_code_sha256"], (
            "Restore the frozen generation code before recovering this sample."
        )
    url = "https://datasets-server.huggingface.co/rows?dataset=abisee%2Fcnn_dailymail&config=3.0.0&split=validation&offset=2&length=40"
    with urlopen(url, timeout=60) as response:
        payload = json.load(response)
    sources = []
    for entry in payload["rows"]:
        row = entry["row"]
        assert row["article"].strip() and row["highlights"].strip()
        sources.append(
            {
                "row_idx": entry["row_idx"],
                "id": row["id"],
                "article": row["article"],
                "reference": row["highlights"],
                "article_sha256": sha(row["article"]),
                "reference_sha256": sha(row["highlights"]),
            }
        )
    assert [s["row_idx"] for s in sources] == list(range(2, 42))
    assert len({s["id"] for s in sources}) == 40
    if existing:
        metadata = [
            {k: v for k, v in s.items() if k not in {"article", "reference"}}
            for s in sources
        ]
        assert metadata == existing["articles"], (
            "Dataset rows or content changed; refusing replacement."
        )
        serialized = json.dumps(sources, indent=2, ensure_ascii=False)
        candidates = [
            serialized.encode("utf-8"),
            serialized.replace("\n", "\r\n").encode("utf-8"),
        ]
        snapshot = next(
            (
                b
                for b in candidates
                if hashlib.sha256(b).hexdigest() == existing["source_snapshot_sha256"]
            ),
            None,
        )
        assert snapshot is not None, (
            "Snapshot bytes do not match the frozen source manifest."
        )
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_bytes(snapshot)
        print("Recovered exact frozen sources; original manifest unchanged.")
        return
    save(DATA, sources)
    manifest = {
        "kind": "Fixed development validation sample; NOT final-test benchmark",
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset": "abisee/cnn_dailymail",
        "config": "3.0.0",
        "split": "validation",
        "sample_policy": "Rows 2-41 inclusive, selected before generation. Rows 0/1 and Jensen input excluded. Consecutive fixed sample, not random or population-representative.",
        "url": url,
        "size": 40,
        "articles": [
            {k: v for k, v in s.items() if k not in {"article", "reference"}}
            for s in sources
        ],
        "source_snapshot_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "generation_code_sha256": hashes(),
        "processing_config": json.loads(
            (ROOT / "processing_config.json").read_text(encoding="utf-8")
        ),
        "models": json.loads(
            (ROOT / "model_manifest.json").read_text(encoding="utf-8")
        ),
        "opennmt_checkpoints": json.loads(
            (ROOT / "OpenNMT_Verified_Setup/checkpoint_manifest.json").read_text(
                encoding="utf-8"
            )
        ),
        "length_preference": "Medium",
        "device_policy": "cuda for every completed comparison; fallback/device mismatch is recorded as a failed comparison, not mixed into GPU timings",
        "rouge": {
            "package": "rouge-score",
            "version": importlib.metadata.version("rouge-score"),
            "use_stemmer": True,
            "variants": ["rouge1", "rouge2", "rougeL"],
            "aggregation": "mean per-article F1, scaled 0-100; failed jobs score zero; successful-only means also reported",
            "rougeL": "flat sequence LCS, newlines ignored; NOT rougeLsum",
            "stopword_removal": False,
        },
        "repetition": "Excess repeated trigrams / total trigrams over casefolded regex word tokens; duplicate whole sentences are a separate heuristic. Neither proves factual accuracy.",
        "human_review": "No human ratings collected. Review-sheet human fields start blank. Any later AI reading annotations must be labeled separately.",
        "jensen_input": {
            "included_in_evaluation": False,
            "words": 518,
            "characters": 3066,
            "sha256": "fd48270a2513baf53a5598394b934d8a9d8c4c016f5758373f936bd0bbf926fd",
            "word_rule": "len(text.split()); whitespace-separated items, including standalone dashes",
        },
    }
    save(MANIFEST, manifest)
    print(
        f"FROZEN: {len(sources)} validation IDs and exact generation-code hashes.",
        flush=True,
    )


def load_frozen():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = json.loads(DATA.read_text(encoding="utf-8"))
    assert hashes() == manifest["generation_code_sha256"], (
        "Generation source/config changed; refusing mixed-settings evaluation"
    )
    assert (
        hashlib.sha256(DATA.read_bytes()).hexdigest()
        == manifest["source_snapshot_sha256"]
    )
    assert [s["id"] for s in sources] == [s["id"] for s in manifest["articles"]]
    return manifest, sources


def repetition(text):
    tokens = re.findall(r"\b\w+\b", text.casefold())
    triples = [tuple(tokens[i : i + 3]) for i in range(max(0, len(tokens) - 2))]
    sentences = [
        " ".join(text[a:b].casefold().split()) for a, b in full.sentence_spans(text)
    ]
    duplicates = sum(c - 1 for c in Counter(sentences).values())
    return {
        "trigram_total": len(triples),
        "trigram_excess": len(triples) - len(set(triples)),
        "trigram_excess_fraction": (len(triples) - len(set(triples))) / len(triples)
        if triples
        else 0.0,
        "duplicate_sentences": duplicates,
    }


def run(models):
    manifest, sources = load_frozen()
    if all(
        (WORK / "cases" / f"{s['row_idx']:03d}_{name}.json").exists()
        for name in models
        for s in sources
    ):
        print(
            "All requested case traces already exist. Use report to rescore; no inference repeated."
        )
        return
    config = full.ProcessingConfig(**manifest["processing_config"])
    assert config.processing_mode == "full"
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This frozen comparison requires the shared CUDA device; no CPU substitution."
        )
    os.environ["SUMMARIZER_DEVICE"] = "cuda"
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "dtype": "float32",
        "timing": "Synchronized existing backend generation regions; no repeated latency benchmark or explicit warmup. First inference includes first-use kernels; model load/transfer reported separately.",
    }
    save(WORK / "environment.json", environment)
    for name in models:
        load_frozen()
        synchronize("cuda")
        started = time.perf_counter()
        backend = engine.get_model(name, "cuda")
        synchronize("cuda")
        cold_load = time.perf_counter() - started
        load_path = WORK / f"loading_{name}.json"
        if not load_path.exists():
            save(load_path, {"model": name, "cold_load_transfer_seconds": cold_load})
        identity = id(backend)
        for source in sources:
            path = WORK / "cases" / f"{source['row_idx']:03d}_{name}.json"
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                assert previous["article_id"] == source["id"]
                continue
            words = full.word_count(source["article"])
            row = {
                "model": name,
                "row_idx": source["row_idx"],
                "article_id": source["id"],
                "source_words": words,
                "reference_words": full.word_count(source["reference"]),
                "article_sha256": source["article_sha256"],
                "summary": "",
                "status": "failed",
                "error": None,
                "human_review_status": "not_reviewed",
                "human_main_event": "",
                "human_entity_relationship_errors": "",
                "human_unsupported_claims": "",
                "human_attribution": "",
                "human_reviewer": "",
            }
            with engine._execution_lock:
                capacity = full.source_capacity(backend, config)
                with patch.object(backend, "source_limit", capacity):
                    total = backend.prepare(source["article"], True).total_tokens
                    chunks = full.ChunkPlanner(backend).plan(
                        source["article"], config.max_source_chunks
                    )
                row.update(
                    input_capacity=capacity,
                    input_tokens_including_source_specials=total,
                    planned_chunks=len(chunks),
                    fits_single_pass=total <= capacity,
                )
                trace = Trace(backend)
                started = time.perf_counter()
                try:
                    with trace:
                        result = full.summarize_article(
                            source["article"], name, "Medium", config=config
                        )
                    if result.device != "cuda":
                        raise RuntimeError(
                            "Device mismatch: CPU fallback is excluded from this common-GPU comparison"
                        )
                    row.update(
                        status="completed",
                        summary=result.summary,
                        diagnostics=asdict(result),
                    )
                except (ValueError, RuntimeError) as error:
                    row["error"] = repr(error)
                row["request_wall_seconds"] = time.perf_counter() - started
                row["decode_trace"] = trace.calls
                presented = {
                    c["input"].get("text") for c in trace.calls if "raw_output_ids" in c
                }
                covered = sum(c.end - c.start for c in chunks if c.text in presented)
                row["source_characters_presented"] = covered
                row["source_characters"] = len(source["article"])
                row["input_presented_fraction"] = covered / max(
                    1, len(source["article"])
                )
                row["completed_full_article_summary"] = (
                    row["status"] == "completed"
                    and row["diagnostics"]["coverage_complete"]
                    if "coverage_complete" in row.get("diagnostics", {})
                    else row["status"] == "completed"
                    and row["diagnostics"]["covered_characters"]
                    == len(source["article"])
                )
                row["unfinished_decode_attempts"] = sum(
                    c.get("eos") is False for c in trace.calls
                )
                row["final_attempt_eos"] = (
                    trace.calls[-1].get("eos") if trace.calls else None
                )
                row["missing_terminal_punctuation"] = bool(row["summary"]) and not bool(
                    re.search(r"[.!?][\"'\u201d\u2019)\]]*$", row["summary"])
                )
                row["output_words"] = full.word_count(row["summary"])
                row["shorter_by_percent"] = (
                    100 * (1 - row["output_words"] / words)
                    if row["status"] == "completed"
                    else None
                )
                row["repetition"] = repetition(row["summary"])
                row["rouge_f1"] = {
                    key: score.fmeasure
                    for key, score in SCORER.score(
                        source["reference"], row["summary"]
                    ).items()
                }
                row["generation_seconds"] = row.get("diagnostics", {}).get(
                    "generation_seconds"
                )
                assert id(engine._models[name]) == identity
                assert backend.source_limit == (
                    1024 if hasattr(backend, "tokenizer") else 400
                )
                save(path, row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "model",
                            "row_idx",
                            "status",
                            "output_words",
                            "planned_chunks",
                            "input_presented_fraction",
                            "generation_seconds",
                            "error",
                        )
                    }
                ),
                flush=True,
            )
        load_frozen()


def aggregate():
    manifest, sources = load_frozen()
    rows = [
        json.loads(
            (WORK / "cases" / f"{s['row_idx']:03d}_{name}.json").read_text(
                encoding="utf-8"
            )
        )
        for name in engine.MODEL_CHOICES
        for s in sources
    ]
    result = {
        "kind": "40 fixed validation articles; frozen Medium settings; NOT final test or factuality certification",
        "manifest": "validation_40_manifest.json",
        "rouge_definition": manifest["rouge"],
        "sample_policy": manifest["sample_policy"],
        "models": {},
        "human_review": "No human ratings collected; numerical metrics and software checks are not factuality evidence.",
    }
    for name in engine.MODEL_CHOICES:
        group = [r for r in rows if r["model"] == name]
        good = [r for r in group if r["status"] == "completed"]
        rng = np.random.default_rng(20261121)
        scores = {
            key: np.array([r["rouge_f1"][key] * 100 for r in group])
            for key in ("rouge1", "rouge2", "rougeL")
        }
        indices = rng.integers(0, len(group), size=(2000, len(group)))
        result["models"][name] = {
            "articles": len(group),
            "completed": len(good),
            "failed": len(group) - len(good),
            "rouge_f1_mean_all_articles_percent": {
                k: float(v.mean()) for k, v in scores.items()
            },
            "rouge_f1_bootstrap_95_ci": {
                k: np.quantile(v[indices].mean(axis=1), [0.025, 0.975]).tolist()
                for k, v in scores.items()
            },
            "rouge_f1_successful_only_percent": {
                k: statistics.mean(r["rouge_f1"][k] * 100 for r in good)
                if good
                else None
                for k in scores
            },
            "output_words_success_mean": statistics.mean(
                r["output_words"] for r in good
            )
            if good
            else None,
            "output_words_success_median": statistics.median(
                r["output_words"] for r in good
            )
            if good
            else None,
            "shorter_by_success_mean_percent": statistics.mean(
                r["shorter_by_percent"] for r in good
            )
            if good
            else None,
            "generation_success_mean_seconds": statistics.mean(
                r["generation_seconds"] for r in good
            )
            if good
            else None,
            "generation_success_median_seconds": statistics.median(
                r["generation_seconds"] for r in good
            )
            if good
            else None,
            "cold_load_transfer_seconds": json.loads(
                (WORK / f"loading_{name}.json").read_text(encoding="utf-8")
            )["cold_load_transfer_seconds"],
            "input_presented_mean_fraction": statistics.mean(
                r["input_presented_fraction"] for r in group
            ),
            "completed_summaries_full_source_coverage": sum(
                r["completed_full_article_summary"] for r in group
            ),
            "source_chunk_histogram": dict(
                Counter(str(r["planned_chunks"]) for r in group)
            ),
            "excess_trigram_success_mean_percent": 100
            * statistics.mean(r["repetition"]["trigram_excess_fraction"] for r in good)
            if good
            else None,
            "summaries_with_duplicate_sentences": sum(
                r["repetition"]["duplicate_sentences"] > 0 for r in good
            ),
            "summaries_with_ge_10percent_excess_trigrams": sum(
                r["repetition"]["trigram_excess_fraction"] >= 0.1 for r in good
            ),
            "unfinished_attempts": sum(r["unfinished_decode_attempts"] for r in group),
            "returned_unfinished_summaries": sum(
                r["status"] == "completed" and r["final_attempt_eos"] is False
                for r in group
            ),
            "missing_terminal_punctuation_successes": sum(
                r["missing_terminal_punctuation"] for r in good
            ),
            "failed_article_ids": [
                r["article_id"] for r in group if r["status"] != "completed"
            ],
        }
    # Same article bootstrap resamples for paired metric differences; this is
    # lexical-overlap uncertainty only, not factuality uncertainty.
    result["paired_rougeL_differences"] = {}
    for left, right in (("BART", "PEGASUS"), ("BART", "LSTM"), ("BART", "Transformer")):
        a = np.array(
            [r["rouge_f1"]["rougeL"] * 100 for r in rows if r["model"] == left]
        )
        b = np.array(
            [r["rouge_f1"]["rougeL"] * 100 for r in rows if r["model"] == right]
        )
        result["paired_rougeL_differences"][left + " minus " + right] = {
            "mean": float((a - b).mean()),
            "bootstrap_95_ci": np.quantile(
                (a - b)[indices].mean(axis=1), [0.025, 0.975]
            ).tolist(),
        }
    # Post-hoc sensitivity for reference templates; preserve the primary sample.
    template_rows = [
        s
        for s in sources
        if "CNN Student News Roll Call" in s["article"] and "TRANSCRIPT" in s["article"]
    ]
    template_ids = {s["id"] for s in template_rows}
    reference_groups = {}
    for source in sources:
        reference_groups.setdefault(source["reference_sha256"], []).append(source["id"])
    result["reference_audit"] = {
        "identical_reference_groups": [
            ids for ids in reference_groups.values() if len(ids) > 1
        ],
        "template_rows": [
            {"row_idx": s["row_idx"], "id": s["id"]} for s in template_rows
        ],
        "sensitivity_label": "Post-hoc descriptive sensitivity, NOT a replacement primary sample. Template-page references may reward boilerplate unrelated to episode events; memorization or dataset leakage is not established.",
        "non_template_rouge_f1_mean_percent": {
            name: {
                key: statistics.mean(
                    r["rouge_f1"][key] * 100
                    for r in rows
                    if r["model"] == name and r["article_id"] not in template_ids
                )
                for key in ("rouge1", "rouge2", "rougeL")
            }
            for name in engine.MODEL_CHOICES
        },
    }
    result["uncertainty_note"] = (
        "Bootstrap intervals describe resampling variation within these 40 consecutive articles. They are not population-representative confidence or factuality estimates. Paired differences reuse the same article indices. All runs used Medium; no length-setting comparison was performed."
    )
    result["environment"] = json.loads(
        (WORK / "environment.json").read_text(encoding="utf-8")
    )
    result["checkpoint_generation_configs"] = {}
    for name, model in manifest["models"].items():
        config_path = (
            ROOT
            / ".cache"
            / "huggingface"
            / "hub"
            / ("models--" + model["id"].replace("/", "--"))
            / "snapshots"
            / model["revision"]
            / "generation_config.json"
        )
        result["checkpoint_generation_configs"][name] = json.loads(
            config_path.read_text(encoding="utf-8")
        )
    result["settings_note"] = (
        "Per-call model_generation_kwargs are actual explicit overrides. HF calls inherit checkpoint configuration above; max_new_tokens overrides max_length and min_new_tokens overrides min_length. forced_eos_token_id is explicitly disabled. OpenNMT settings are recorded per call. Token counts use each backend's tokenizer, including source specials where applicable."
    )
    result["cases"] = []
    for row in rows:
        record = {
            key: row[key]
            for key in (
                "row_idx",
                "article_id",
                "model",
                "status",
                "source_words",
                "reference_words",
                "input_capacity",
                "input_tokens_including_source_specials",
                "planned_chunks",
                "fits_single_pass",
                "input_presented_fraction",
                "completed_full_article_summary",
                "output_words",
                "shorter_by_percent",
                "generation_seconds",
                "request_wall_seconds",
                "rouge_f1",
                "repetition",
                "unfinished_decode_attempts",
                "final_attempt_eos",
                "missing_terminal_punctuation",
                "error",
            )
        }
        record["pipeline"] = {
            key: row.get("diagnostics", {}).get(key)
            for key in (
                "requested_words",
                "source_chunks",
                "consolidation_passes",
                "model_calls",
                "total_input_tokens",
                "total_generated_tokens",
                "device",
                "warnings",
                "retained_fragments",
                "returned_input",
            )
        }
        record["calls"] = [
            {
                "input_tokens": c["input"]["total_tokens"],
                "truncated": c["input"]["truncated"],
                **{
                    key: c.get(key)
                    for key in (
                        "budget",
                        "model_generation_kwargs",
                        "output_tokens",
                        "output_words",
                        "eos",
                        "termination",
                        "generation_seconds",
                    )
                },
            }
            for c in row["decode_trace"]
        ]
        result["cases"].append(record)
    annotations_path = ROOT / "reports" / "validation_40_ai_review.json"
    annotations = (
        json.loads(annotations_path.read_text(encoding="utf-8"))
        if annotations_path.exists()
        else {}
    )
    result["ai_review"] = {
        "notes": sum(
            name in annotations.get(str(s["row_idx"]), {})
            for s in sources
            for name in engine.MODEL_CHOICES
        ),
        "expected": len(rows),
        "kind": "AI-assisted source/reference reading, not human ratings; see validation_40_ai_review.json",
        "human_ratings": 0,
    }
    save(ROOT / "reports" / "validation_40_results.json", result)
    export_review(rows, sources, annotations, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "cases"}, indent=2
        )
    )


def export_review(rows, sources, annotations, result):
    fields = [
        "row_idx",
        "article_id",
        "model",
        "status",
        "output_words",
        "shorter_by_percent",
        "generation_seconds",
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
        "input_presented_fraction",
        "summary",
        "reference",
        "ai_reading_note",
        "human_review_status",
        "human_main_event",
        "human_entity_relationship_errors",
        "human_unsupported_claims",
        "human_attribution",
        "human_reviewer",
    ]
    with (ROOT / "reports" / "validation_40_review.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            record = {key: r.get(key, "") for key in fields}
            reference = next(
                s["reference"] for s in sources if s["id"] == r["article_id"]
            )
            record.update(
                reference=reference,
                ai_reading_note=annotations.get(str(r["row_idx"]), {}).get(
                    r["model"], "Not reviewed by AI; human review needed."
                ),
            )
            record.update(
                {
                    key + "_f1": r["rouge_f1"][key]
                    for key in ("rouge1", "rouge2", "rougeL")
                }
            )
            writer.writerow(record)
    blocks = []
    for source in sources:
        cards = []
        for r in [r for r in rows if r["article_id"] == source["id"]]:
            note = annotations.get(str(r["row_idx"]), {}).get(
                r["model"], "No AI assessment yet. Human review needed."
            )
            cards.append(
                f'<section class="card"><h3>{r["model"]}</h3><p class="meta">{r["status"]} · {r["output_words"]} words · ROUGE-L F1 {r["rouge_f1"]["rougeL"] * 100:.2f} · {r["input_presented_fraction"]:.0%} input presented</p><p>{html.escape(r["summary"] or "No completed summary returned.")}</p><p class="note"><strong>AI reading note—not a human rating:</strong> {html.escape(note)}</p><p class="meta">Human reviewer: not reviewed. Enter ratings/evidence in the CSV.</p></section>'
            )
        blocks.append(
            f'<article id="row-{source["row_idx"]}"><h2>Validation row {source["row_idx"]} · {source["id"]}</h2><details><summary>Original article ({full.word_count(source["article"])} words)</summary><p class="source">{html.escape(source["article"])}</p></details><h3>Original reference highlights</h3><p class="source">{html.escape(source["reference"])}</p><div class="grid">{"".join(cards)}</div></article>'
        )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>40-article validation review</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1300px;margin:30px auto;padding:0 20px;color:#1b3030;background:#f7f9f8}article{margin:36px 0;padding:22px;border:1px solid #cdd9d3;background:white;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{padding:18px;border:1px solid #d6dfdb;border-radius:10px}.source{white-space:pre-wrap}.meta{font-size:13px;color:#486258}.note{background:#f0f5f2;padding:12px}summary{cursor:pointer}h2{overflow-wrap:anywhere;font-size:20px}@media(max-width:750px){.grid{grid-template-columns:1fr}}@media print{details>p{display:block}.grid{display:block}.card{break-inside:avoid}}</style><h1>Frozen 40-article validation review</h1><p>All four models use the same validation IDs and frozen Medium settings. References are original dataset highlights. Numerical checks are automated, not factuality judgments. Input coverage means text was processed, not that facts survived.</p><p><strong>No human ratings have been collected.</strong> AI notes, when present, are provisional source-comparison observations and require human confirmation. Blank human-review columns are in <a href="validation_40_review.csv">the editable CSV</a>.</p><p>Review rubric: identify the main event and actor/action/outcome; check names, dates, quantities and relationships; look for unsupported additions or changed negation/certainty; verify who said or alleged each claim. Cite evidence from both source and summary. References omit details and are not a complete factuality oracle.</p>'
        + "".join(blocks)
        + "</html>"
    )
    table_rows = []
    for name, stats in result["models"].items():
        scores = stats["rouge_f1_mean_all_articles_percent"]
        table_rows.append(
            f"<tr><th>{name}</th><td>{scores['rouge1']:.2f}</td><td>{scores['rouge2']:.2f}</td><td>{scores['rougeL']:.2f}</td><td>{stats['output_words_success_mean']:.1f}</td><td>{stats['shorter_by_success_mean_percent']:.1f}%</td><td>{stats['generation_success_mean_seconds']:.3f}</td><td>{stats['cold_load_transfer_seconds']:.3f}</td></tr>"
        )
    overview = (
        '<h2>Automated results</h2><table cellpadding="8"><thead><tr><th>Model</th><th>R1</th><th>R2</th><th>RL</th><th>Mean words</th><th>Shorter by</th><th>Mean generation s</th><th>Load s</th></tr></thead><tbody>'
        + "".join(table_rows)
        + '</tbody></table><p>ROUGE F1 means on 0-100 scale, Porter stemming, flat ROUGE-L (not Lsum). CSV per-case scores are 0-1. Forty consecutive validation rows 2-41; not a random population sample. All runs: Medium, RTX 4090 CUDA, float32. One timed run per case without explicit warmup; loading separate. All 160 jobs completed with complete input coverage and EOS. EOS does not certify a complete or faithful sentence.</p><p>Two CNN Student News pages have identical template references. Primary scores retain them; JSON includes an explicitly post-hoc 38-page sensitivity. References themselves can omit the main event. Bootstrap intervals measure only within-sample resampling variability. No default model is recommended from these results.</p><p><a href="validation_40_results.json">Detailed results, actual per-call decoding settings and coverage</a> | <a href="validation_40_manifest.json">Frozen IDs, input hashes and protocol</a></p>'
    )
    page = page.replace("<article id=", overview + "<article id=", 1)
    (ROOT / "reports" / "validation_40_review.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run", "report"])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=engine.MODEL_CHOICES,
        default=list(engine.MODEL_CHOICES),
    )
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "run":
        run(args.models)
    else:
        aggregate()
