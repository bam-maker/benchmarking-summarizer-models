"""One frozen causal audit; never modify production generation or old results.

Five development failures, direct documented decoding, one-factor length controls,
and explicitly partial 400/1024-token OpenNMT prefix diagnostics. No ROUGE tuning.
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["SUMMARIZER_DEVICE"] = "cuda"

import torch

import article_summarizer as full
import summarizer as engine
from runtime import synchronize

WORK = ROOT / "test-results/factual_audit"
PUBLIC = ROOT / "reports/factual_audit_manifest.json"
ROWS = {
    19: "reversed_match",
    35: "invented_author",
    21: "missed_bombing",
    30: "repeated_claim",
    39: "multi_chunk",
}
MODELS = ("LSTM", "Transformer", "BART", "PEGASUS")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = read(PUBLIC)
    for path, expected in manifest["preserved_sha256"].items():
        assert digest(ROOT / path) == expected, f"Protected evidence changed: {path}"
    return manifest


def stanford(text):
    tools = read(ROOT / ".cache/factual-audit-tools/provenance.json")
    command = [
        str(ROOT / tools["java"]),
        "-cp",
        str(ROOT / tools["jar"]),
        "edu.stanford.nlp.process.PTBTokenizer",
        "-preserveLines",
        "-encoding",
        "UTF-8",
    ]
    completed = subprocess.run(
        command,
        input=text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=30,
    )
    # Same order as the original data script: Stanford tokenization, then lowercase.
    # Original .story line boundaries are absent in the HF flattened article.
    return completed.stdout.lower().split()


def prepare():
    if PUBLIC.exists():
        verify()
        print("Existing frozen audit protocol verified; no replacement.")
        return
    old = read(ROOT / "reports/validation_40_manifest.json")
    sources = read(ROOT / "data/validation_40/sources.json")
    paths = list(old["generation_code_sha256"])
    paths += [
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in sorted((ROOT / "reports").glob("validation_40*"))
    ]
    paths += ["data/validation_40/sources.json", "reports/quality_regression.json"]
    paths += [
        f"test-results/validation_40/cases/{row:03d}_{name}.json"
        for row in ROWS
        for name in MODELS
    ]
    for path, expected in old["generation_code_sha256"].items():
        assert digest(ROOT / path) == expected
    protocol = {
        "kind": "Fixed factual-error causal audit; not a benchmark, tuning sweep or human rating",
        "cases": [
            {
                "row_idx": s["row_idx"],
                "id": s["id"],
                "label": ROWS[s["row_idx"]],
                "article_sha256": s["article_sha256"],
            }
            for s in sources
            if s["row_idx"] in ROWS
        ],
        "models": MODELS,
        "device": "RTX 4090 CUDA float32, same device for all probes",
        "preserved_sha256": {p: digest(ROOT / p) for p in paths},
        "preprocessing": read(ROOT / ".cache/factual-audit-tools/provenance.json"),
        "opennmt_document": "OpenNMT_Verified_Setup/legacy/docs/source/Summarization.md",
        "opennmt_document_sha256": digest(
            ROOT / "OpenNMT_Verified_Setup/legacy/docs/source/Summarization.md"
        ),
        "opennmt_recipe": {
            "beam": 5,
            "min_length": 35,
            "max_length": 100,
            "alpha": 0.9,
            "beta": 5.0,
            "length_penalty": "wu",
            "coverage_penalty": "summary",
            "stepwise_penalty": True,
            "block_ngram_repeat": 3,
            "ignore_when_blocking": [".", "</t>", "<t>"],
            "replace_unk": False,
            "fast": False,
        },
        "historical_ambiguity": "Pinned prose says alpha .8; explicit CNN/DM command says .9. Use command .9 matching app. Today's website uses beam 10; pinned checkpoint-era document uses 5. OpenNMT has no embedded HF-style generation config; use documented recipe plus pinned translator max_length default 100.",
        "hf_control_graph": "checkpoint_defaults -> EOS-unforced -> adaptive maximum only -> adaptive minimum too (app settings). Separate minimum-only branch from EOS-unforced. Each edge changes one decoding control; every edge uses identical checkpoint tokenizer IDs and source chunk.",
        "opennmt_control_graph": "Per exact app source chunk: recipe 35/100 -> adaptive maximum only -> adaptive minimum too. Also recipe -> adaptive minimum only. Independently swap NLTK approximation for Stanford 3.7 with fixed 35/100 and identical source text. Entire-source prefix tokens: 400 versus 1024, with preprocessing and decoding otherwise fixed; these are explicitly partial diagnostics whenever tokens are omitted.",
        "multi_chunk_policy": "Row39. Preserve app's original complete source chunks; compare each directly then same source-order exact-sentence dedup merge. Never describe a prefix as the full article. No further generative consolidation if app did none.",
        "tokenizer_limitations": "Original .story line boundaries and original OpenNMT prepared corpus are unavailable. Stanford3.7 reproduces the documented tokenization/lowercasing on this exact flattened HF source, not every historical caption-period repair or corpus packaging step. Source text is unchanged; target <t> tags are not inserted into source.",
        "no_tuning": "One predeclared matrix; no repeated parameter search. Separate validation rows42-47 are reserved for any evidence-supported proposed production change. No candidate is promoted merely because one failure disappears.",
        "template_anomalies": {
            "rows": [10, 14],
            "policy": "Retain all original 40 results and existing labeled post-hoc sensitivity; preserve their hashes. Inspect contents without regenerating scores or excluding bad model outputs.",
        },
    }
    save(PUBLIC, protocol)
    for source in sources:
        if source["row_idx"] in {*ROWS, 10, 14}:
            baselines = {
                name: read(
                    ROOT
                    / f"test-results/validation_40/cases/{source['row_idx']:03d}_{name}.json"
                )
                for name in MODELS
            }
            save(
                WORK / "sources" / f"{source['row_idx']:03d}.json",
                {"source": source, "application_saved": baselines},
            )
    print("Frozen five cases and immutable original validation evidence.")


def legacy_settings(t, minimum, maximum):
    return {
        "beam_size": t.beam_size,
        "min_length": minimum,
        "max_length": maximum,
        "stepwise_penalty": t.stepwise_penalty,
        "alpha": t.global_scorer.alpha,
        "beta": t.global_scorer.beta,
        "length_penalty": t.global_scorer.length_penalty.__name__,
        "coverage_penalty": t.global_scorer.cov_penalty.__name__,
        "block_ngram_repeat": t.block_ngram_repeat,
        "ignore_when_blocking": sorted(t.ignore_when_blocking),
        "replace_unk": t.replace_unk,
        "fast": t.fast,
    }


def legacy_direct(backend, tokens, minimum, maximum):
    from onmt.translate.beam import GNMTGlobalScorer

    t = backend.translator
    settings = {
        "beam_size": 5,
        "min_length": minimum,
        "max_length": maximum,
        "stepwise_penalty": True,
        "global_scorer": GNMTGlobalScorer(0.9, 5.0, "summary", "wu"),
        "block_ngram_repeat": 3,
        "ignore_when_blocking": {".", "</t>", "<t>"},
        "replace_unk": False,
        "fast": False,
        "out_file": io.StringIO(),
    }
    trace = {
        "route": "pinned Translator.translate, independent of app backend.generate",
        "prepared_tokens": tokens,
        "model_input_text": " ".join(tokens),
    }
    translate = t.translate_batch

    def raw(batch, data, **kwargs):
        assert tuple(data.examples[0].src) == tuple(tokens)
        trace["source_tensor_ids"] = batch.src[0][:, 0].tolist()
        trace["source_dynamic_vocabulary"] = list(data.src_vocabs[0].itos)
        result = translate(batch, data, **kwargs)
        trace["raw_output_ids"] = [int(i) for i in result["predictions"][0][0]]
        return result

    with backend.lock, torch.inference_mode(), ExitStack() as stack:
        for key, value in settings.items():
            stack.enter_context(patch.object(t, key, value))
        stack.enter_context(patch.object(t, "translate_batch", raw))
        trace["effective_settings"] = legacy_settings(t, minimum, maximum)
        synchronize(backend.device)
        started = time.perf_counter()
        _, predictions = t.translate(src_data_iter=[" ".join(tokens)], batch_size=1)
        synchronize(backend.device)
        trace["generation_seconds"] = time.perf_counter() - started
    raw_text = predictions[0][0]
    eos = t.fields["tgt"].vocab.stoi[t.fields["tgt"].eos_token]
    trace.update(
        raw_output=raw_text,
        displayed_output=engine.clean_summary(raw_text.split()),
        eos=trace["raw_output_ids"][-1] == eos,
        output_words=full.word_count(engine.clean_summary(raw_text.split())),
        output_tokens=len(trace["raw_output_ids"])
        - int(trace["raw_output_ids"][-1] == eos),
        input_tokens=len(tokens),
    )
    trace["termination"] = "eos" if trace["eos"] else "maximum_length"
    return trace


def hf_direct(backend, text, overrides):
    ids = backend.tokenizer(
        text, add_special_tokens=True, truncation=False, return_tensors="pt"
    )
    assert ids.input_ids.shape[1] <= backend.source_limit, (
        "Never truncate the fixed source chunk in a decoding control"
    )
    effective = backend.model.generation_config.to_dict()
    effective.update(do_sample=False, **overrides)
    with backend.lock, torch.inference_mode():
        synchronize(backend.device)
        started = time.perf_counter()
        output = backend.model.generate(
            **{k: v.to(backend.device) for k, v in ids.items()},
            do_sample=False,
            **overrides,
        )[0].tolist()
        synchronize(backend.device)
        seconds = time.perf_counter() - started
    raw = backend.tokenizer.decode(
        output[1:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    display = (
        backend.tokenizer.decode(
            output[1:], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        .replace("<n>", "\n")
        .strip()
    )
    ended = output[-1] == backend.tokenizer.eos_token_id
    cap = overrides.get("max_new_tokens", effective["max_length"] - 1)
    forced_at_cap = (
        ended
        and effective.get("forced_eos_token_id") is not None
        and len(output) - 1 >= cap
    )
    return {
        "route": "checkpoint model.generate with its own tokenizer, bypassing app generation/preparation",
        "model_input_text": text,
        "source_tensor_ids": ids.input_ids[0].tolist(),
        "input_tokens": ids.input_ids.shape[1],
        "explicit_overrides": overrides,
        "effective_settings": effective,
        "raw_output_ids": output,
        "raw_with_specials": backend.tokenizer.decode(
            output, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        "raw_output": raw,
        "displayed_output": display,
        "eos": ended,
        "termination": "forced_eos_at_cap"
        if forced_at_cap
        else "eos"
        if ended
        else "maximum_length",
        "output_words": full.word_count(display),
        "output_tokens": len(output) - 1 - int(ended),
        "generation_seconds": seconds,
    }


def run(models):
    verify()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Use the same CUDA device for every control; CPU substitution is forbidden in this audit."
        )
    for name in models:
        with engine._execution_lock:
            backend = engine.get_model(name, "cuda")
            for number, label in ROWS.items():
                path = WORK / "controls" / f"{number:03d}_{name}.json"
                if path.exists():
                    print(f"EXISTS {number} {name}; not rerunning.", flush=True)
                    continue
                case = read(WORK / "sources" / f"{number:03d}.json")
                source = case["source"]
                baseline = case["application_saved"][name]
                row = {
                    "row_idx": number,
                    "label": label,
                    "model": name,
                    "source": source,
                    "saved_application": baseline,
                    "chunk_controls": [],
                    "prefix_diagnostics": [],
                    "device": backend.device,
                }
                for index, call in enumerate(baseline["decode_trace"]):
                    text = call["input"]["text"]
                    budget = call["budget"]
                    control = {
                        "chunk": index,
                        "source_text": text,
                        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "app_raw": call,
                        "variants": {},
                    }
                    if hasattr(backend, "tokenizer"):
                        variants = {
                            "checkpoint_defaults": {},
                            "eos_unforced_only": {"forced_eos_token_id": None},
                            "adaptive_max_only": {
                                "forced_eos_token_id": None,
                                "max_new_tokens": budget["max_tokens"],
                            },
                            "adaptive_min_only": {
                                "forced_eos_token_id": None,
                                "min_new_tokens": max(0, budget["min_tokens"] - 1),
                            },
                            "direct_app_settings": {
                                "forced_eos_token_id": None,
                                "max_new_tokens": budget["max_tokens"],
                                "min_new_tokens": max(0, budget["min_tokens"] - 1),
                            },
                        }
                        for variant, overrides in variants.items():
                            control["variants"][variant] = hf_direct(
                                backend, text, overrides
                            )
                        control["input_ids_equal_application"] = (
                            control["variants"]["direct_app_settings"][
                                "source_tensor_ids"
                            ]
                            == call["model_input_ids"]
                        )
                    else:
                        approximate = list(call["input"]["prepared_tokens"])
                        original = stanford(text)
                        control.update(
                            stanford_tokens=original,
                            app_approximation_tokens=approximate,
                            tokenizer_tokens_equal=original == approximate,
                        )
                        variants = {
                            "recipe_current_preprocessing": (approximate, 35, 100),
                            "adaptive_max_only": (
                                approximate,
                                35,
                                budget["max_tokens"],
                            ),
                            "adaptive_min_only": (
                                approximate,
                                budget["min_tokens"],
                                100,
                            ),
                            "direct_app_settings": (
                                approximate,
                                budget["min_tokens"],
                                budget["max_tokens"],
                            ),
                            "recipe_stanford_preprocessing": (original, 35, 100),
                            "adaptive_stanford_preprocessing": (
                                original,
                                budget["min_tokens"],
                                budget["max_tokens"],
                            ),
                        }
                        for variant, (tokens, minimum, maximum) in variants.items():
                            assert len(tokens) <= backend.extended_source_limit
                            control["variants"][variant] = legacy_direct(
                                backend, tokens, minimum, maximum
                            )
                        control["input_ids_equal_application"] = (
                            approximate == call["input"]["prepared_tokens"]
                        )
                    direct = control["variants"]["direct_app_settings"]
                    control["direct_same_settings_raw_ids_equal_application"] = (
                        direct["raw_output_ids"] == call["raw_output_ids"]
                    )
                    control["direct_same_settings_display_equal_application"] = (
                        direct["displayed_output"] == call["postprocessed_output"]
                    )
                    row["chunk_controls"].append(control)
                row["direct_app_merge"] = full.deduplicate_sentences(
                    "\n".join(
                        c["variants"]["direct_app_settings"]["displayed_output"]
                        for c in row["chunk_controls"]
                    )
                )
                row["direct_app_merge_equal_saved_application"] = (
                    row["direct_app_merge"] == baseline["summary"]
                )
                if not hasattr(backend, "tokenizer"):
                    complete_approx = list(
                        engine.prepare_article(
                            source["article"], True, source_limit=100000
                        ).tokens
                    )
                    complete_stanford = stanford(source["article"])
                    row["preprocessing_inspection"] = {
                        "source_vocabulary_size": len(
                            backend.translator.fields["src"].vocab
                        ),
                        "target_vocabulary_size": len(
                            backend.translator.fields["tgt"].vocab
                        ),
                        "source_init_token": backend.translator.fields[
                            "src"
                        ].init_token,
                        "source_eos_token": backend.translator.fields["src"].eos_token,
                        "target_init_token": backend.translator.fields[
                            "tgt"
                        ].init_token,
                        "target_eos_token": backend.translator.fields["tgt"].eos_token,
                        "source_tensor_vocabulary": "Pinned checkpoint fields; copying uses each source's dynamic vocabulary",
                        "app_full_tokens": len(complete_approx),
                        "stanford_full_tokens": len(complete_stanford),
                        "app_tokens": complete_approx,
                        "stanford_tokens": complete_stanford,
                    }
                    for preprocessing, tokens in (
                        ("app_nltk", complete_approx),
                        ("stanford37", complete_stanford),
                    ):
                        for window in (400, 1024):
                            probe = legacy_direct(backend, tokens[:window], 35, 100)
                            probe.update(
                                preprocessing=preprocessing,
                                window=window,
                                complete_original_tokens=len(tokens),
                                omitted_tokens=max(0, len(tokens) - window),
                                token_coverage_fraction=min(1.0, window / len(tokens)),
                                is_full_article=len(tokens) <= window,
                                notice="Prefix-only diagnostic when tokens are omitted; NEVER a full-article summary in that case.",
                            )
                            row["prefix_diagnostics"].append(probe)
                save(path, row)
                print(
                    json.dumps(
                        {
                            "row": number,
                            "model": name,
                            "chunks": len(row["chunk_controls"]),
                            "direct_app_equal": row[
                                "direct_app_merge_equal_saved_application"
                            ],
                            "probes": sum(
                                len(c["variants"]) for c in row["chunk_controls"]
                            )
                            + len(row["prefix_diagnostics"]),
                        }
                    ),
                    flush=True,
                )
        verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        run(args.models)
