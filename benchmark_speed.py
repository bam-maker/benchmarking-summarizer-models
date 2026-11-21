"""Same-device four-model speed/real-inference check, not a quality benchmark."""

import argparse
import hashlib
import json
import platform
import statistics
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from summarizer import MODEL_CHOICES, ROOT, SETUP, clean_summary, get_model, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Deliberately no auto: a partial CPU fallback must not pollute GPU rankings.
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--length", choices=["Short", "Medium", "Detailed"], default="Medium"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups nonnegative")
    article = clean_summary((SETUP / "article.txt").read_text().split())
    rows = []
    for name in MODEL_CHOICES:
        try:
            cold = summarize(article, name, args.length, device=args.device)
            assert cold.summary and cold.device == args.device
            model = get_model(name, args.device)
            identity = id(model)
            assert next(model.model.parameters()).device.type == args.device
            assert next(model.model.parameters()).dtype == torch.float32
            for _ in range(args.warmups):
                summarize(article, name, args.length, device=args.device)
            results = [
                summarize(article, name, args.length, device=args.device)
                for _ in range(args.repeats)
            ]
            assert all(
                result.summary and result.device == args.device for result in results
            )
            assert id(get_model(name, args.device)) == identity
            import summarizer

            assert (
                sum(backend.device == "cuda" for backend in summarizer._models.values())
                <= 1
            )
            latencies = [result.seconds for result in results]
            row = {
                "model": name,
                "status": "passed",
                "device": args.device,
                "dtype": "float32",
                "parameters": sum(p.numel() for p in model.model.parameters()),
                "learned_parameters": sum(
                    p.numel() for p in model.model.parameters() if p.requires_grad
                ),
                "checkpoint": model.spec
                if hasattr(model, "spec")
                else {
                    **json.loads((SETUP / "checkpoint_manifest.json").read_text())[
                        name.lower()
                    ],
                    "code_revision": "6db7ec1b3afae28c257a0416dd947022c3b12bc0",
                },
                "loading_info": getattr(model, "loading_info", {}),
                "cold_load_transfer_seconds": cold.load_seconds,
                "cold_generation_seconds": cold.seconds,
                "generation_median_seconds": statistics.median(latencies),
                "generation_p95_seconds": float(np.percentile(latencies, 95)),
                "summary": results[-1].summary,
                "word_count": results[-1].word_count,
                "warm_results": [asdict(result) for result in results],
            }
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "warm_results"}),
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 -- Persist failures, never omit a model.
            rows.append(
                {
                    "model": name,
                    "status": "failed",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            traceback.print_exc()
    report = {
        "kind": "synthetic fixture speed/inference check; NOT CNN/DailyMail quality evaluation",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": args.device,
        "hardware": torch.cuda.get_device_name(0)
        if args.device == "cuda" and torch.cuda.is_available()
        else platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "threads": 2,
        "batch_size": 1,
        "beam_size": 4,
        "precision": "float32; TF32 disabled",
        "length": args.length,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "article": article,
        "article_sha256": hashlib.sha256(article.encode()).hexdigest(),
        "timing": "CUDA synchronized before/after timed regions. Generation excludes load/transfer, tokenization, and queue waiting. Cold loading includes device transfers and disk/model initialization, including first-use download if not cached.",
        "results": rows,
    }
    path = args.output or ROOT / "reports" / f"speed_{args.device}.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)
    if any(row["status"] != "passed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
