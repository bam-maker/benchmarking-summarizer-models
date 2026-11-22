"""Pre-test model integrity and synthetic Riverton diagnosis; no quality tuning."""
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SUMMARIZER_DEVICE"] = "cuda"
os.environ["SUMMARIZER_PROCESSING_MODE"] = "full"
import torch
import article_summarizer as full
import summarizer as engine
from diagnose_quality import Trace, direct_hf, _inspect_backend
from runtime import synchronize


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    article = (ROOT / "tests/fixtures/riverton.txt").read_text(encoding="utf-8").rstrip("\n")
    report = {"kind": "Synthetic diagnostic, not CNN/DailyMail or verified news", "article": article,
              "article_sha256": hashlib.sha256(article.encode()).hexdigest(),
              "words": len(article.split()),
              "previous_output_provenance": "User-reported; runtime configuration unrecorded",
              "previous_output": "The problem began early Tuesday when a control valve stopped working during routine maintenance. Residents were instructed to boil tap water before drinking it, preparing food, brushing their teeth, or making ice. Water could still be used for laundry and most cleaning activities.",
              "models": {}}
    originals = json.loads((engine.SETUP / "checkpoint_manifest.json").read_text())
    for name in engine.MODEL_CHOICES:
        synchronize("cuda")
        start = time.perf_counter()
        backend = engine.get_model(name, "cuda")
        synchronize("cuda")
        cold = time.perf_counter() - start
        assert backend.model_name == name and backend.device == "cuda"
        assert not any(m.training for m in backend.model.modules())
        assert {p.dtype for p in backend.model.parameters()} == {torch.float32}
        identity = id(backend)
        backend.source_limit = full.source_capacity(backend, full.load_config())
        info, _ = _inspect_backend(backend, article)
        info.update(parameters=sum(p.numel() for p in backend.model.parameters()),
                    cold_load_transfer_seconds=cold, all_modules_eval=True, dtype="float32")
        if hasattr(backend, "tokenizer"):
            info["loading_info"] = backend.loading_info
            assert not backend.loading_info["unexpected_keys"]
            assert not backend.loading_info["mismatched_keys"]
            assert not backend.loading_info["error_msgs"]
            embedding_rows = backend.model.get_input_embeddings().num_embeddings
            outside = {token: idx for token, idx in backend.tokenizer.get_vocab().items() if idx >= embedding_rows}
            # The released BART CNN checkpoint omits the pretraining-only mask
            # embedding. Ordinary news tokens must all remain in range.
            assert not outside or (name == "BART" and outside == {"<mask>": 50264})
            assert max(backend.tokenizer.encode(article)) < embedding_rows
            info["vocabulary_alignment"] = {"tokenizer_size": len(backend.tokenizer), "embedding_rows": embedding_rows,
                                             "checkpoint_excluded_pretraining_tokens": outside}
            assert backend.tokenizer.eos_token_id == backend.model.config.eos_token_id
            assert backend.tokenizer.pad_token_id == backend.model.config.pad_token_id
            folder = ROOT / ".cache/huggingface/hub" / ("models--" + backend.spec["id"].replace("/", "--")) / "snapshots" / backend.spec["revision"]
            info["checkpoint"] = backend.spec
            info["snapshot_sha256"] = {p.name: file_sha(p) for p in folder.iterdir() if p.is_file()}
            if name == "PEGASUS":
                for module in (backend.model.model.encoder, backend.model.model.decoder):
                    positions = module.embed_positions
                    expected = type(positions)(positions.num_embeddings, positions.embedding_dim).weight
                    assert torch.equal(positions.weight.detach().cpu(), expected)
                    assert not positions.weight.requires_grad
                info["missing_positions_verified"] = "Exact deterministic sinusoidal tables; fixed, not randomly initialized learned weights"
            original_generate = backend.model.generate
            observations = []
            def checked_generate(*args, **kwargs):
                assert torch.is_inference_mode_enabled() and not backend.model.training
                assert torch.equal(kwargs["attention_mask"], torch.ones_like(kwargs["input_ids"]))
                assert kwargs["input_ids"].shape[0] == 1
                observations.append({"inference_mode": True, "unpad_batch1_mask_all_ones": True})
                return original_generate(*args, **kwargs)
            native = backend.model.generation_config.to_dict()
            with patch.object(backend.model, "generate", checked_generate):
                with Trace(backend) as trace:
                    direct = direct_hf(backend, backend.prepare(article), trace)
                info["literal_checkpoint_direct"] = {"summary": direct, "trace": trace.calls}
                with Trace(backend) as trace:
                    result = full.summarize_article(article, name, generation_policy=full.NATURAL_POLICY)
                info["application"] = {"result": asdict(result), "trace": trace.calls}
            assert native == backend.model.generation_config.to_dict()
            info["inference_checks"] = observations
            info["generation_config_unchanged"] = True
        else:
            spec = originals[name.lower()]
            candidates = [p for p in engine.SETUP.glob("*.pt") if p.stat().st_size == spec["bytes"]]
            assert len(candidates) == 1 and file_sha(candidates[0]) == spec["sha256"]
            derived = engine.SETUP / f"{name.lower()}_inference.pt"
            # Verify every derived tensor/vocabulary against the verified original.
            # Historical optimizer metadata is never used for inference. Modern
            # optimizer __setstate__ cannot read the 2018 pickle; retain it as data.
            with patch.object(torch.optim.Optimizer, "__setstate__", lambda obj, state: obj.__dict__.update(state)), patch.object(torch.optim.Adagrad, "__setstate__", lambda obj, state: obj.__dict__.update(state)):
                original = torch.load(candidates[0], map_location="cpu", weights_only=False)
            stripped = torch.load(derived, map_location="cpu", weights_only=False)
            for group in ("model", "generator"):
                assert original[group].keys() == stripped[group].keys()
                assert all(torch.equal(v, stripped[group][k]) for k, v in original[group].items())
            for (field1, vocab1), (field2, vocab2) in zip(original["vocab"], stripped["vocab"]):
                assert field1 == field2 and vocab1.itos == vocab2.itos and vocab1.stoi == vocab2.stoi
            del original, stripped
            info.update(checkpoint=spec, inference_file_sha256=file_sha(derived),
                        derived_tensors_and_vocab_match_original=True,
                        loading_info="Strict model and generator load_state_dict in pinned model_builder.py:219-220")
            original_translate = backend.translator.translate_batch
            def checked_translate(*args, **kwargs):
                assert torch.is_inference_mode_enabled() and not backend.model.training
                return original_translate(*args, **kwargs)
            with patch.object(backend.translator, "translate_batch", checked_translate):
                with Trace(backend) as trace:
                    result = full.summarize_article(article, name, generation_policy=full.NATURAL_POLICY)
                info["application"] = {"result": asdict(result), "trace": trace.calls}
            info["inference_mode_checked"] = True
        assert result.coverage_complete and result.source_chunks == result.model_calls == 1
        assert info["application"]["trace"][0]["input"]["text"] == article
        assert id(engine.get_model(name, "cuda")) == identity
        info["cache_identity_verified"] = True
        report["models"][name] = info
        print(name, "verified; summary:", result.summary, flush=True)
    path = ROOT / "reports/final_model_verification.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", path, flush=True)


if __name__ == "__main__":
    main()
