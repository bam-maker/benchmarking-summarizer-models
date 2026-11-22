"""Frozen, resumable 1000-test-article evaluation. No tuning or training.

Commands: prepare; run --limit 5; run; report (separate report_final_benchmark.py).
Artifacts under data/ and test-results/ are intentionally ignored by Git.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".cache/eval-metrics"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SUMMARIZER_DEVICE"] = "cuda"
os.environ["SUMMARIZER_PROCESSING_MODE"] = "full"
import numpy as np
import torch
from rouge_score import rouge_scorer
import article_summarizer as full
import summarizer as engine
from diagnose_quality import Trace
from runtime import synchronize

WORK = ROOT / "test-results/final_test_1000"
REPORT = ROOT / "reports/final_test_1000"
DATA = ROOT / "data/final_test_1000/sources.json"
INPUTS = ROOT / "data/final_test_1000/prepared_sources.json"
MANIFEST = REPORT / "frozen_config.json"
TRACKS = ("common_input", "full_article")
MODELS = (*engine.MODEL_CHOICES, "Lead-3")
METRICS = ("rouge1", "rouge2", "rougeL", "rougeLsum")
SCORER = rouge_scorer.RougeScorer(METRICS, use_stemmer=True)
FILES = ("app.py", "article_summarizer.py", "summarizer.py", "hf_summarizer.py", "runtime.py",
         "processing_config.json", "model_manifest.json", "requirements.txt",
         "OpenNMT_Verified_Setup/checkpoint_manifest.json", "tests/final_benchmark.py",
         "tests/diagnose_quality.py", "tests/fetch_final_sources.py")


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def code_hashes():
    paths = [ROOT / name for name in FILES]
    paths += sorted((ROOT / "OpenNMT_Verified_Setup/legacy/onmt").rglob("*.py"))
    return {p.relative_to(ROOT).as_posix(): file_sha(p) for p in paths}


def segmented(text):
    return "\n".join(text[a:b].strip() for a, b in full.sentence_spans(text))


def score(summary, reference):
    values = SCORER.score(reference, summary)
    values["rougeLsum"] = SCORER.score(segmented(reference), segmented(summary))["rougeLsum"]
    return {name: value.fmeasure * 100 for name, value in values.items()}


def repetition(text):
    words = re.findall(r"\w+", text.lower())
    grams = list(zip(words, words[1:], words[2:]))
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.0


def common_prefix(article, tokenizers):
    def counts(text):
        old = engine.prepare_article(text, True, source_limit=100000).total_tokens
        return {"LSTM": old, "Transformer": old, **{
            name: len(tok.encode(text, add_special_tokens=True, truncation=False))
            for name, tok in tokenizers.items()}}
    def fits(text):
        c = counts(text)
        return c["LSTM"] <= 400 and all(c[n] <= 1024 for n in tokenizers)
    # Same original text/case for all systems. Cut only after complete whitespace
    # words; verification below is authoritative even if token counts fluctuate.
    ends = [m.end() for m in re.finditer(r"\S+\s*", article)]
    lo, hi = 0, len(ends)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(article[:ends[mid - 1]]):
            lo = mid
        else:
            hi = mid - 1
    if lo == 0:
        raise ValueError("No whole first word fits the common input window")
    end = ends[lo - 1]
    prefix = article[:end]
    assert fits(prefix) and article.startswith(prefix)
    return prefix, counts(prefix)


def prepare():
    if MANIFEST.exists():
        validate()
        print("Existing freeze verified; no replacement.")
        return
    verified = load(ROOT / "reports/final_model_verification.json")
    assert set(verified["models"]) == set(engine.MODEL_CHOICES)
    rows = load(DATA)
    assert [r["row_idx"] for r in rows] == random.Random(42).sample(range(11490), 1000)
    from transformers import AutoTokenizer
    specs = load(ROOT / "model_manifest.json")
    tokenizers = {name: AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"],
                  cache_dir=str(ROOT / ".cache/huggingface/hub"), use_fast=False, local_files_only=True)
                  for name, spec in specs.items()}
    for i, row in enumerate(rows):
        assert sha(row["article"]) == row["article_sha256"]
        assert sha(row["reference"]) == row["reference_sha256"]
        prefix, counts = common_prefix(row["article"], tokenizers)
        row.update(common_input=prefix, common_input_sha256=sha(prefix), common_tokens=counts,
                   common_characters=len(prefix), common_word_coverage=len(prefix.split()) / len(row["article"].split()),
                   common_character_coverage=len(prefix) / len(row["article"]))
        if i % 100 == 0:
            print(f"Prepared identical prefixes {i + 1}/1000", flush=True)
    save(INPUTS, rows)
    review_ids = random.Random(142).sample([r["id"] for r in rows], 20)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": "abisee/cnn_dailymail", "dataset_version": "3.0.0", "split": "test",
                "dataset_revision_at_retrieval": "96df5e686bee6baa90b8bee7c28b81fa3fa6223d",
                "retrieval_note": "Dataset-server rows at recorded revision time; exact text hashes authoritative. API does not pin a revision.",
                "population": 11490, "sample_size": 1000, "seed": 42,
                "sampling": "random.Random(42).sample(range(11490), 1000), preserved random order",
                "sample": [{k: r[k] for k in ("row_idx", "id", "article_sha256", "reference_sha256", "common_input_sha256")} for r in rows],
                "review_ids": review_ids, "review_seed": 142, "blinding_seed": 242,
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "working_tree_sha256": code_hashes(), "prepared_sources_sha256": file_sha(INPUTS),
                "verification_sha256": file_sha(ROOT / "reports/final_model_verification.json"),
                "model_verification": {name: {k: v for k, v in info.items() if k in
                    ("checkpoint", "snapshot_sha256", "inference_file_sha256", "parameters", "loading_info", "checkpoint_generation_config")}
                    for name, info in verified["models"].items()},
                "generation_policy": full.NATURAL_POLICY, "processing_config": asdict(full.load_config()),
                "natural_settings": "Native beam/search; min <= source_tokens//8; maxima 100/100/141/127, no forced EOS, retry x1.75 to hard caps 256/256/512/512; no word-percentage target.",
                "tracks": {"common_input": "Identical original-text whitespace-boundary prefix, <=400 production OpenNMT tokens, <=1024 each HF including specials; natural single pass. Historical Stanford preprocessing remains approximated by existing NLTK adapter.",
                           "full_article": "Original article through unchanged full coverage natural pipeline, sentence-aware chunks, exact-sentence dedup, bounded consolidation."},
                "references": "Same full original highlights in both tracks; common input may omit reference-supported facts.",
                "device": "cuda", "hardware": torch.cuda.get_device_name(0), "precision": "float32", "TF32": False,
                "batch_size": 1, "warmups_per_model_per_process": 1,
                "warmup_text": "The council approved new electric buses on Monday. Officials said the first vehicles will arrive in September. Existing fares will remain unchanged.",
                "timing": "CUDA synchronized inside backend calls; sum of decode durations including retries, excluding tokenization/load/queue. Request wall time also recorded. Warmup excluded; cold loading recorded separately. Lead-3 CPU extraction separately.",
                "rouge": {"package": "rouge-score", "version": importlib.metadata.version("rouge-score"), "stemming": True,
                          "scale": "F1 x100", "Lsum_segmentation": "same article_summarizer.sentence_spans for reference and output"},
                "failure_scoring": "Every attempted failure has empty final output and zero ROUGE; unfinished-cap intermediates never used as completed summaries.",
                "repetition": "excess repeated lowercase regex-word trigrams / all output trigrams",
                "bootstrap": {"paired": True, "replicates": 2000, "seed": 42, "confidence": .95},
                "bertscore": "Omitted: package and scorer weights not installed/cached; preserve working environment.",
                "python": platform.python_version(), "platform": platform.platform(), "cuda_runtime": torch.version.cuda,
                "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}}
    metadata["fingerprint"] = sha(json.dumps(metadata, sort_keys=True))
    save(MANIFEST, metadata)
    WORK.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(WORK / "frozen_code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in metadata["working_tree_sha256"]:
            archive.write(ROOT / name, name)
    print("FROZEN", metadata["fingerprint"], flush=True)


def validate():
    manifest = load(MANIFEST)
    assert code_hashes() == manifest["working_tree_sha256"], "Frozen evaluation/production code changed"
    assert file_sha(INPUTS) == manifest["prepared_sources_sha256"], "Frozen inputs changed"
    assert asdict(full.load_config()) == manifest["processing_config"], "Config override differs"
    assert file_sha(ROOT / "reports/final_model_verification.json") == manifest["verification_sha256"]
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    assert versions == manifest["dependencies"], "Dependency environment differs from freeze"
    check = dict(manifest)
    fingerprint = check.pop("fingerprint")
    assert sha(json.dumps(check, sort_keys=True)) == fingerprint
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == manifest["hardware"]
    return manifest, load(INPUTS)


def run(limit, models):
    manifest, rows = validate()
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    rows = rows[:limit]
    for name in models:
        pending = [(r, track) for r in rows for track in TRACKS
                   if not (WORK / "predictions" / track / name / (r["id"] + ".json")).exists()]
        # Validate all reused cases, including exact input and metric configuration.
        for r in rows:
            for track in TRACKS:
                path = WORK / "predictions" / track / name / (r["id"] + ".json")
                if path.exists():
                    old = load(path)
                    supplied = r["common_input"] if track == "common_input" else r["article"]
                    assert old["fingerprint"] == manifest["fingerprint"] and old["input_sha256"] == sha(supplied)
                    assert old["id"] == r["id"] and old["reference_sha256"] == r["reference_sha256"]
        if not pending:
            continue
        if name != "Lead-3":
            evidence = manifest["model_verification"][name]
            if name in {"LSTM", "Transformer"}:
                assert file_sha(engine.SETUP / f"{name.lower()}_inference.pt") == evidence["inference_file_sha256"]
            else:
                spec = evidence["checkpoint"]
                folder = ROOT / ".cache/huggingface/hub" / ("models--" + spec["id"].replace("/", "--")) / "snapshots" / spec["revision"]
                assert {p.name: file_sha(p) for p in folder.iterdir() if p.is_file()} == evidence["snapshot_sha256"]
            synchronize("cuda")
            began = time.perf_counter()
            backend = engine.get_model(name, "cuda")
            synchronize("cuda")
            loading = time.perf_counter() - began
            assert {p.dtype for p in backend.model.parameters()} == {torch.float32}
            assert not backend.model.training
            warm = full.summarize_article(manifest["warmup_text"], name, generation_policy=full.NATURAL_POLICY)
            assert warm.device == "cuda" and warm.coverage_complete
            save(WORK / "loads" / f"{name}_{time.time_ns()}.json",
                 {"model": name, "cold_load_transfer_seconds": loading, "warmup_generation_seconds": warm.generation_seconds,
                  "fingerprint": manifest["fingerprint"], "parameters": sum(p.numel() for p in backend.model.parameters())})
        started_model = time.perf_counter()
        for index, (row, track) in enumerate(pending):
            supplied = row["common_input"] if track == "common_input" else row["article"]
            record = {"id": row["id"], "row_idx": row["row_idx"], "track": track, "model": name,
                      "fingerprint": manifest["fingerprint"], "input_sha256": sha(supplied),
                      "reference_sha256": row["reference_sha256"], "original_source_words": len(row["article"].split()),
                      "supplied_words": len(supplied.split()), "supplied_characters": len(supplied),
                      "supplied_source_character_coverage": len(supplied) / len(row["article"]),
                      "supplied_source_word_coverage": len(supplied.split()) / len(row["article"].split())}
            start = time.perf_counter()
            trace = None
            try:
                if name == "Lead-3":
                    summary = " ".join(supplied[a:b].strip() for a, b in full.sentence_spans(supplied)[:3])
                    elapsed = time.perf_counter() - start
                    record.update(device="cpu", generation_seconds=elapsed, source_chunks=1,
                                  consolidation_passes=0, covered_characters=len(supplied), model_calls=0,
                                  summary=summary, trace=[], warnings=[])
                else:
                    with Trace(backend) as trace:
                        result = full.summarize_article(supplied, name, generation_policy=full.NATURAL_POLICY)
                    assert result.device == "cuda", "CPU fallback disallowed for comparable neural timing"
                    assert result.coverage_complete
                    if track == "common_input":
                        assert result.source_chunks == 1 and result.consolidation_passes == 0
                    record.update(asdict(result))
                    record["trace"] = trace.calls
                    assert all(not c["input"].get("truncated", False) for c in trace.calls)
                record["status"] = "completed"
            except (ValueError, RuntimeError) as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}", summary="",
                              device=getattr(locals().get("backend"), "device", None),
                              trace=trace.calls if trace else [], covered_characters=None,
                              source_chunks=None, consolidation_passes=None)
                record["generation_seconds"] = sum(c.get("generation_seconds", 0) for c in record["trace"])
            record["request_wall_seconds"] = time.perf_counter() - start
            record["scores"] = score(record["summary"], row["reference"])
            record["output_words"] = len(record["summary"].split())
            record["compression_original"] = record["output_words"] / record["original_source_words"]
            record["compression_supplied"] = record["output_words"] / record["supplied_words"]
            record["repeated_trigram_rate"] = repetition(record["summary"])
            record["empty"] = not bool(record["summary"].strip())
            record["had_max_length_attempt"] = any(c.get("eos") is False for c in record["trace"])
            record["unfinished_error"] = record["status"] == "failed" and "output limit" in record.get("error", "")
            record["unfinished_punctuation_heuristic"] = bool(record["summary"]) and not bool(re.search(r"[.!?][\"'\u201d\u2019)\]]*$", record["summary"]))
            save(WORK / "predictions" / track / name / (row["id"] + ".json"), record)
            if (index + 1) % 10 == 0 or index == len(pending) - 1:
                elapsed = time.perf_counter() - started_model
                print(f"{name}: {index + 1}/{len(pending)} newly saved; {elapsed:.1f}s; last={record['status']}", flush=True)
        elapsed = time.perf_counter() - started_model
        print(f"{name} operational elapsed/prediction={elapsed / len(pending):.3f}s; projected 2000 calls={elapsed / len(pending) * 2000 / 60:.1f}min", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        assert 1 <= args.limit <= 1000
        run(args.limit, args.models)
