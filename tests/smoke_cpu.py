"""Original fixed-budget CPU regression, independent of the adaptive demo."""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import summarizer as engine


def main():
    os.environ["SUMMARIZER_DEVICE"] = "cpu"
    article = engine.clean_summary((engine.SETUP / "article.txt").read_text().split())
    rows = []
    with patch.object(
        engine, "OpenNMTSummarizer", wraps=engine.OpenNMTSummarizer
    ) as factory:
        for name in ("LSTM", "Transformer"):
            for length, budget in engine.LENGTH_PRESETS.items():
                result = engine.summarize(article, name, length, device="cpu")
                assert result.summary and result.device == "cpu"
                model = engine.get_model(name, "cpu")
                assert model.translator.copy_attn
                assert (model.translator.min_length, model.translator.max_length) == (
                    budget.min_tokens,
                    budget.max_tokens,
                )
                rows.append(asdict(result))
        assert factory.call_count == 2
        try:
            engine.summarize(article * 5, "LSTM", "Short", device="cpu")
        except engine.InputError:
            pass
        else:
            raise AssertionError(
                "The benchmark API still requires explicit prefix consent"
            )
    path = ROOT / "test-results" / "cpu_smoke.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(
            {"kind": "fixed-budget CPU regression; not adaptive demo", "checks": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("PASS: six original CPU backend cases and reuse.", flush=True)


if __name__ == "__main__":
    main()
