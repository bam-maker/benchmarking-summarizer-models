"""Extend the frozen protocol to its first 250 IDs without editing historical runs."""
import argparse
import csv
import importlib.metadata
import json
import os
import shutil
import statistics
import subprocess
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import final_benchmark as b

HISTORICAL = b.WORK
DEST = b.ROOT / "reports/benchmark250"
WORK = b.ROOT / "test-results/benchmark250"
CONFIG = DEST / "frozen_config.json"
REUSE = DEST / "reuse_manifest.json"
PARENT = b.MANIFEST
MODELS = b.MODELS
TRACKS = b.TRACKS


def audit():
    """Verify the original freeze, including the documented pin-only delta."""
    original = b.load(PARENT)
    parent = dict(original)
    fingerprint = parent.pop("fingerprint")
    assert b.sha(json.dumps(parent, sort_keys=True)) == fingerprint
    current = b.code_hashes()
    expected = dict(original["working_tree_sha256"])
    if current["requirements.txt"] != expected["requirements.txt"]:
        with zipfile.ZipFile(HISTORICAL / "frozen_code.zip") as archive:
            before = archive.read("requirements.txt").decode().splitlines()
        after = (b.ROOT / "requirements.txt").read_text(encoding="utf8").splitlines()
        assert after == before + ["sentencepiece==0.2.3"], "Unexpected requirements delta"
        assert original["dependencies"]["sentencepiece"] == "0.2.3"
        expected["requirements.txt"] = current["requirements.txt"]
    assert current == expected, "Production or protocol code differs"
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    assert versions == original["dependencies"], "Installed dependencies differ"
    assert b.file_sha(b.INPUTS) == original["prepared_sources_sha256"]
    assert b.file_sha(b.ROOT / "reports/final_model_verification.json") == original["verification_sha256"]
    assert b.asdict(b.full.load_config()) == original["processing_config"]
    assert b.torch.cuda.is_available(), "CUDA required; CPU run requires a user decision"
    assert b.torch.cuda.get_device_name(0) == original["hardware"]
    rows = b.load(b.INPUTS)
    assert [r["row_idx"] for r in rows] == b.random.Random(42).sample(range(11490), 1000)
    assert b.load(b.ROOT / "reports/pilot50/results.json")["ids"] == [r["id"] for r in rows[:50]]
    for row in rows[:250]:
        for field in ("article", "reference", "common_input"):
            assert b.sha(row[field]) == row[field + "_sha256"]
    if CONFIG.exists():
        config = b.load(CONFIG)
        signed = dict(config)
        signature = signed.pop("fingerprint")
        assert b.sha(json.dumps(signed, sort_keys=True)) == signature
        assert config["parent_fingerprint"] == fingerprint
        assert config["working_tree_sha256"] == current
        assert config["runner_sha256"] == b.file_sha(Path(__file__))
    return original, rows[:250]


def check_record(record, source, track, model, fingerprint):
    supplied = source["common_input"] if track == "common_input" else source["article"]
    assert record["fingerprint"] == fingerprint
    assert record["id"] == source["id"] and record["row_idx"] == source["row_idx"]
    assert record["input_sha256"] == b.sha(supplied)
    assert record["reference_sha256"] == source["reference_sha256"]
    assert record["model"] == model and record["track"] == track
    assert record["status"] in {"completed", "failed"}
    assert record["scores"] == b.score(record["summary"], source["reference"])
    if record["status"] == "completed" and model != "Lead-3":
        assert record["device"] == "cuda"
        assert record["generation_policy"] == b.full.NATURAL_POLICY
        assert record["covered_characters"] == len(supplied)
        assert all(not t["input"].get("truncated", False) for t in record["trace"])
    if record["status"] == "failed":
        assert record["summary"] == "" and not any(record["scores"].values())


def prepare():
    original, rows = audit()
    WORK.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    if CONFIG.exists():
        for reused in b.load(REUSE):
            target = WORK / "predictions" / reused["track"] / reused["model"] / (reused["id"] + ".json")
            if target.exists():
                assert b.file_sha(target) == reused["sha256"], "Reused prediction changed"
                continue
            source = b.ROOT / reused["origin"]
            assert source.exists(), "Restore the historical raw prediction cache before resuming this frozen run"
            assert b.file_sha(source) == reused["sha256"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        print("Existing 250-article freeze verified.", flush=True)
        return
    # Verify every actual checkpoint even when its predictions can all be reused.
    for model in b.engine.MODEL_CHOICES:
        spec = original["model_verification"][model]
        if model in {"LSTM", "Transformer"}:
            assert b.file_sha(b.engine.SETUP / (model.lower() + "_inference.pt")) == spec["inference_file_sha256"]
        else:
            cp = spec["checkpoint"]
            folder = b.ROOT / ".cache/huggingface/hub" / ("models--" + cp["id"].replace("/", "--")) / "snapshots" / cp["revision"]
            assert {p.name: b.file_sha(p) for p in folder.iterdir() if p.is_file()} == spec["snapshot_sha256"]
    reused = []
    for track in TRACKS:
        for model in MODELS:
            for source in rows:
                path = HISTORICAL / "predictions" / track / model / (source["id"] + ".json")
                if path.exists():
                    check_record(b.load(path), source, track, model, original["fingerprint"])
                    target = WORK / "predictions" / track / model / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        assert target.read_bytes() == path.read_bytes()
                    else:
                        shutil.copyfile(path, target)
                    reused.append({"track": track, "model": model, "id": source["id"],
                                   "origin": path.relative_to(b.ROOT).as_posix(), "sha256": b.file_sha(path)})
    b.save(REUSE, reused)
    config = {**original, "parent_fingerprint": original["fingerprint"], "sample_size": 250,
              "sample": original["sample"][:250], "selection": "First 250 of existing frozen 1000 IDs; pilot50 is nested",
              "working_tree_sha256": b.code_hashes(), "runner_sha256": b.file_sha(Path(__file__)),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=b.ROOT, text=True).strip(),
              "requirements_delta": "Added sentencepiece==0.2.3 pin; already installed at original freeze; identical dependency inventory",
              "timing_limitation": "Mixed historical and new predictions; desktop, app and Epic overlay contexts. No controlled speed ranking.",
              "reused_count": len(reused), "disk_free_bytes": shutil.disk_usage(b.ROOT).free,
              "gpu_conditions": subprocess.check_output(["nvidia-smi"], text=True),
              "storage": "Raw traces in ignored test-results/benchmark250; compact predictions and original references in reports/benchmark250",
              "analysis_scope": "Descriptive mean ROUGE only; no bootstrap, human review or significance analysis was performed for benchmark250.",
              "parent_manifest_sha256": b.file_sha(PARENT)}
    for key in ("fingerprint", "review_ids", "review_seed", "blinding_seed", "bootstrap", "bertscore"):
        config.pop(key, None)
    config["fingerprint"] = b.sha(json.dumps(config, sort_keys=True))
    b.save(CONFIG, config)
    b.save(DEST / "article_ids.json", config["sample"])
    print("PREPARED", len(reused), "verified reused predictions", flush=True)


def status(emit=True):
    reused = {(r["track"], r["model"], r["id"]) for r in b.load(REUSE)}
    progress = {}
    for track in TRACKS:
        progress[track] = {}
        for model in MODELS:
            records = [b.load(p) for p in (WORK / "predictions" / track / model).glob("*.json")]
            new = [r for r in records if (track, model, r["id"]) not in reused]
            durations = [r["request_wall_seconds"] for r in new]
            count = len(records)
            progress[track][model] = {
                "completed": sum(r["status"] == "completed" for r in records),
                "failed": sum(r["status"] == "failed" for r in records),
                "pending": 250 - count, "reused": count - len(new), "newly_attempted": len(new),
                "measured_new_mean_request_seconds": statistics.mean(durations) if durations else None,
                "estimated_pending_seconds": (250-count)*statistics.mean(durations) if durations else None}
    b.save(WORK / "progress.json", progress)
    if emit:
        print(json.dumps(progress), flush=True)
    return progress


def run():
    prepare()
    lock = WORK / "run.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    stop = threading.Event()
    def monitor():
        announced = set()
        while not stop.wait(10):
            current = status(False)
            for track in TRACKS:
                for model in MODELS:
                    entry = current[track][model]
                    count = entry["newly_attempted"]
                    marker = (track, model, count // 20)
                    if count >= 20 and marker not in announced:
                        announced.add(marker)
                        print("MEASURED_PROGRESS", track, model, json.dumps(entry), flush=True)
    worker = threading.Thread(target=monitor, daemon=True)
    try:
        b.WORK = WORK
        b.validate = audit
        worker.start()
        b.run(250, list(MODELS))
    finally:
        stop.set()
        worker.join(timeout=11)
        lock.unlink()
        status()
    report()


def report():
    original, sources = audit()
    reuse = {(r["track"], r["model"], r["id"]): r for r in b.load(REUSE)}
    metrics = {}
    flat = []
    for track in TRACKS:
        metrics[track] = {}
        for model in MODELS:
            records = []
            for source in sources:
                path = WORK / "predictions" / track / model / (source["id"] + ".json")
                if not path.exists():
                    continue
                record = b.load(path)
                check_record(record, source, track, model, original["fingerprint"])
                records.append(record)
                reused = reuse.get((track, model, source["id"]))
                flat.append({k: record.get(k, "") for k in (
                    "track", "model", "id", "row_idx", "status", "summary", "error", "input_sha256",
                    "reference_sha256", "original_source_words", "supplied_words", "supplied_characters",
                    "supplied_source_character_coverage", "supplied_source_word_coverage", "covered_characters",
                    "output_words", "compression_original", "compression_supplied", "source_chunks",
                    "consolidation_passes", "model_calls", "generation_seconds", "request_wall_seconds",
                    "device", "generation_policy", "repeated_trigram_rate")}
                    | {"reference": source["reference"], "reused": bool(reused),
                       "reuse_origin": reused["origin"] if reused else "", **record["scores"]})
            values = {k: statistics.mean(r["scores"][k] for r in records) for k in b.METRICS} if records else {}
            metrics[track][model] = {
                "n": len(records), "completed": sum(r["status"] == "completed" for r in records),
                "failed": sum(r["status"] == "failed" for r in records), "pending": 250-len(records),
                "reused": sum((track, model, r["id"]) in reuse for r in records),
                **values, "mean_words": statistics.mean(r["output_words"] for r in records) if records else None}
    complete = all(a["pending"] == 0 for t in metrics.values() for a in t.values())
    b.save(DEST / "results.json", {"complete": complete, "expected": 2500,
           "fingerprint": b.load(CONFIG)["fingerprint"], "parent_fingerprint": original["fingerprint"],
           "ids": [s["id"] for s in sources], "results": metrics})
    if flat:
        with (DEST / "predictions.csv").open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    print("REPORT", "complete" if complete else "PARTIAL", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "status", "report"])
    command = parser.parse_args().command
    {"prepare": prepare, "run": run, "status": status, "report": report}[command]()
