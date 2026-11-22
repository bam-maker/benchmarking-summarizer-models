"""Small real-model verification, saved separately from adaptive experiments."""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SUMMARIZER_DEVICE"] = "cuda"

from diagnose_quality import Trace

import app
import article_summarizer as full
import summarizer as engine


def main():
    sources = json.loads(
        (ROOT / "data/validation_40/sources.json").read_text(encoding="utf-8")
    )
    inputs = {
        "tiny": "The council approved new buses.",
        "short": next(r["article"] for r in sources if r["row_idx"] == 21),
        "multichunk": next(r["article"] for r in sources if r["row_idx"] == 39),
    }
    report = {
        "policy": full.NATURAL_POLICY,
        "scope": "Execution/coverage checks, not factuality validation or benchmark",
        "cases": [],
        "explicit_failures": [],
    }
    for name in engine.MODEL_CHOICES:
        backend = engine.get_model(name, "cuda")
        identity = id(backend)
        for label, text in inputs.items():
            try:
                with Trace(backend) as trace:
                    r = full.summarize_article(
                        text, name, generation_policy=full.NATURAL_POLICY
                    )
            except full.ProcessingLimitError as error:
                if label != "tiny":
                    raise
                with patch.object(app, "summarize_article", side_effect=error):
                    displayed = app.generate_summary(text, name)
                    assert displayed[0] == "" and not displayed[3]["interactive"]
                report["explicit_failures"].append(
                    {
                        "model": name,
                        "case": label,
                        "input": text,
                        "error": str(error),
                        "trace": trace.calls,
                    }
                )
                print(name, label, "explicit empty-output rejection", flush=True)
                continue
            assert r.summary and r.coverage_complete and r.requested_words is None
            assert r.generation_policy == full.NATURAL_POLICY
            assert r.model_calls == 1 if label != "multichunk" else r.source_chunks > 1
            assert r.total_generated_tokens == sum(
                c["output_tokens"] for c in trace.calls
            )
            assert (
                abs(
                    r.generation_seconds
                    - sum(c["generation_seconds"] for c in trace.calls)
                )
                < 1e-8
            )
            assert all(c["eos"] for c in trace.calls)
            assert id(engine._models[name]) == identity
            with patch.object(app, "summarize_article", return_value=r) as route:
                displayed = app.generate_summary(text, name)
                assert (
                    route.call_args.kwargs["generation_policy"] == full.NATURAL_POLICY
                )
                assert displayed[0] == r.summary
            report["cases"].append(
                {
                    "case": label,
                    "input": text,
                    "result": asdict(r),
                    "trace": trace.calls,
                    "displayed_statistics": displayed[1],
                }
            )
            print(
                name, label, r.actual_words, r.source_chunks, r.model_calls, flush=True
            )
    path = ROOT / "test-results/natural_policy/verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {
        "policy": full.NATURAL_POLICY,
        "cases": [{"case": c["case"], **c["result"]} for c in report["cases"]],
        "explicit_failures": report["explicit_failures"],
        "checks": "Completed cases have full coverage/EOS and exact token/time accounting; same cached backend and natural UI routing. Tiny-input empty decodes explicitly rejected, no partial UI result. No factuality claim.",
    }
    (ROOT / "reports/natural_policy_verification.json").write_text(
        json.dumps(compact, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
