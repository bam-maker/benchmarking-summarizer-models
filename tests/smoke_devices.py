"""All four real models through the simple demo API."""

import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gradio_client import Client

import hf_summarizer
import summarizer as engine
from app import build_demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    os.environ["SUMMARIZER_DEVICE"] = args.device
    os.environ["SUMMARIZER_PROCESSING_MODE"] = "full"
    article = engine.clean_summary((engine.SETUP / "article.txt").read_text().split())
    rows = []
    demo = build_demo()
    try:
        with (
            patch.object(
                engine, "OpenNMTSummarizer", wraps=engine.OpenNMTSummarizer
            ) as old_factory,
            patch.object(
                hf_summarizer, "HFSummarizer", wraps=hf_summarizer.HFSummarizer
            ) as new_factory,
        ):
            _, url, _ = demo.launch(
                server_name="127.0.0.1",
                server_port=7861,
                share=False,
                show_api=False,
                prevent_thread_lock=True,
                quiet=True,
                inbrowser=False,
            )
            client = Client(url, verbose=False)
            for name in engine.MODEL_CHOICES:
                identity = None
                for length in ("Short", "Medium", "Detailed"):
                    outputs = client.predict(
                        article, name, length, api_name="/summarize"
                    )
                    summary = outputs[0]
                    assert isinstance(summary, str) and summary
                    backend = engine._models[name]
                    assert backend.device == args.device
                    identity = identity or id(backend)
                    assert identity == id(backend)
                    rows.append(
                        {
                            "model": name,
                            "length": length,
                            "device": args.device,
                            "summary": summary,
                        }
                    )
                    print(json.dumps(rows[-1]), flush=True)
            identities = {name: id(model) for name, model in engine._models.items()}
            assert client.predict(article, "LSTM", "Medium", api_name="/summarize")[0]
            assert identities == {
                name: id(model) for name, model in engine._models.items()
            }
            assert old_factory.call_count == new_factory.call_count == 2
            assert sum(model.device == "cuda" for model in engine._models.values()) <= 1
        path = ROOT / "test-results" / f"{args.device}_interface.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "kind": "adaptive demo check; not benchmark",
                    "cases": rows,
                    "models_reused": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "PASS: 12 real simplified-interface model/length requests and cached switching.",
            flush=True,
        )
    finally:
        demo.close()


if __name__ == "__main__":
    main()
