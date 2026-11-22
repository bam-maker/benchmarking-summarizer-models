"""Small, inspectable development regression; not a quality/speed benchmark.

Saves exact sources, tokenizer inputs, raw IDs/text and every decode setting.
No checkpoints/dependencies are downloaded or changed. --fetch-validation reads
only two fixed public validation rows with their existing reference highlights.
The user-supplied Huang text is unverified; nothing here fact-checks its claims.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

import app
import article_summarizer as full
import summarizer as engine
from runtime import synchronize

WORK = ROOT / "test-results" / "quality_regression"
SOURCE_NOTICE = "User-pasted Huang article is unverified source text, not verified news. Generated allegations are not endorsed."


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_validation():
    url = "https://datasets-server.huggingface.co/rows?dataset=abisee%2Fcnn_dailymail&config=3.0.0&split=validation&offset=0&length=2"
    with urlopen(url, timeout=30) as response:
        data = json.load(response)
    cases = []
    for entry in data["rows"]:
        row = entry["row"]
        cases.append(
            {
                "case": f"cnndm_validation_{entry['row_idx']}",
                "article": row["article"],
                "reference": row["highlights"],
                "source": {
                    "dataset": "abisee/cnn_dailymail",
                    "config": "3.0.0",
                    "split": "validation",
                    "row_idx": entry["row_idx"],
                    "id": row["id"],
                    "retrieved_from": url,
                },
                "sha256": fingerprint(row["article"]),
            }
        )
    short = "The Northbridge council said on Monday that the Riverside footbridge will remain closed for three weeks after inspectors found cracks in its supports. No injuries have been reported. Engineers have not yet determined whether recent flooding caused the damage. Pedestrians can use the station bridge instead. The council expects to publish a repair estimate on Friday; it has not approved a construction budget."
    cases.append(
        {
            "case": "synthetic_short_uncertainty",
            "article": short,
            "reference": None,
            "source": {
                "kind": "synthetic development fixture, not actual news",
                "fact_checklist": [
                    "footbridge closed for three weeks",
                    "no injuries reported",
                    "flooding cause not established",
                    "repair budget not approved",
                ],
            },
            "sha256": fingerprint(short),
        }
    )
    save(WORK / "validation_sources.json", cases)


class Trace:
    def __init__(self, backend):
        self.backend = backend
        self.inputs = {}
        self.calls = []
        self.current = None

    def __enter__(self):
        backend = self.backend
        self.stack = ExitStack()
        prepare, generate = backend.prepare, backend.generate

        def prepare_trace(text, allow_truncation=False):
            prepared = prepare(text, allow_truncation)
            self.inputs[id(prepared)] = {
                "text": text,
                "prepared_tokens": list(prepared.tokens),
                "total_tokens": prepared.total_tokens,
                "truncated": prepared.truncated,
            }
            return prepared

        def generate_trace(prepared, budget):
            row = {
                "input": self.inputs.get(
                    id(prepared), {"prepared_tokens": list(prepared.tokens)}
                ),
                "budget": asdict(budget),
            }
            self.calls.append(row)
            self.current = row
            output, generated, ended, seconds = generate(prepared, budget)
            row.update(
                postprocessed_output=output,
                output_words=full.word_count(output),
                output_tokens=generated,
                eos=ended,
                termination="eos" if ended else "maximum_length",
                generation_seconds=seconds,
            )
            return output, generated, ended, seconds

        self.stack.enter_context(patch.object(backend, "prepare", prepare_trace))
        self.stack.enter_context(patch.object(backend, "generate", generate_trace))
        if hasattr(backend, "tokenizer"):
            model_generate = backend.model.generate

            def raw_generate(*args, **kwargs):
                ids = model_generate(*args, **kwargs)
                row = self.current
                row["model_input_ids"] = kwargs["input_ids"][0].tolist()
                row["model_generation_kwargs"] = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in {"input_ids", "attention_mask"}
                }
                row["raw_output_ids"] = ids[0].tolist()
                row["raw_decoded_with_specials"] = backend.tokenizer.decode(
                    row["raw_output_ids"],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                row["decoded_before_cleanup"] = backend.tokenizer.decode(
                    row["raw_output_ids"][1:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                return ids

            self.stack.enter_context(
                patch.object(backend.model, "generate", raw_generate)
            )
        else:
            translate, cleanup = (
                backend.translator.translate_batch,
                engine.clean_summary,
            )

            def raw_translate(*args, **kwargs):
                decoded = translate(*args, **kwargs)
                self.current["raw_output_ids"] = [
                    int(i) for i in decoded["predictions"][0][0]
                ]
                t = backend.translator
                self.current["model_generation_kwargs"] = {
                    "beam_size": t.beam_size,
                    "min_length": t.min_length,
                    "max_length": t.max_length,
                    "stepwise_penalty": t.stepwise_penalty,
                    "alpha": t.global_scorer.alpha,
                    "beta": t.global_scorer.beta,
                    "length_penalty": t.global_scorer.length_penalty.__name__,
                    "coverage_penalty": t.global_scorer.cov_penalty.__name__,
                    "block_ngram_repeat": t.block_ngram_repeat,
                }
                return decoded

            def clean(tokens):
                self.current["raw_target_tokens"] = list(tokens)
                self.current["decoded_before_cleanup"] = " ".join(tokens)
                return cleanup(tokens)

            self.stack.enter_context(
                patch.object(backend.translator, "translate_batch", raw_translate)
            )
            self.stack.enter_context(patch.object(engine, "clean_summary", clean))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


@contextmanager
def documented_opennmt(backend):
    from onmt.translate.beam import GNMTGlobalScorer

    # The pinned 0.2.1 documentation uses beam 5 (current website uses 10).
    translator = backend.translator
    values = {
        "beam_size": 5,
        "stepwise_penalty": True,
        "global_scorer": GNMTGlobalScorer(0.9, 5.0, "summary", "wu"),
    }
    with ExitStack() as stack:
        for name, value in values.items():
            stack.enter_context(patch.object(translator, name, value))
        yield


def direct_hf(backend, prepared, trace):
    trace.current = {
        "input": trace.inputs[id(prepared)],
        "path": "checkpoint_generation_config",
    }
    trace.calls.append(trace.current)
    ids = torch.tensor([prepared.tokens], dtype=torch.long, device=backend.device)
    with torch.inference_mode():
        synchronize(backend.device)
        started = time.perf_counter()
        generated = backend.model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False
        )[0].tolist()
        synchronize(backend.device)
        seconds = time.perf_counter() - started
    text = (
        backend.tokenizer.decode(
            generated[1:], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        .replace("<n>", "\n")
        .strip()
    )
    at_cap = len(generated) >= backend.model.generation_config.max_length
    ended = generated[-1] == backend.tokenizer.eos_token_id
    trace.current.update(
        postprocessed_output=text,
        output_words=full.word_count(text),
        output_tokens=len(generated) - 1 - int(ended),
        eos=ended,
        termination="forced_eos_at_maximum"
        if at_cap and ended
        else "eos"
        if ended
        else "maximum_length",
        generation_seconds=seconds,
    )
    return text


def _inspect_backend(backend, article):
    prepared = backend.prepare(article, True)
    chunks = full.ChunkPlanner(backend).plan(
        article, full.load_config().max_source_chunks
    )
    info = {
        "original_words": full.word_count(article),
        "model_tokens": prepared.total_tokens,
        "source_limit": backend.source_limit,
        "fits_supported_single_pass": prepared.total_tokens <= backend.source_limit,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "start": c.start,
                "end": c.end,
                "text": c.text,
                "prepared_tokens": list(c.prepared.tokens),
            }
            for c in chunks
        ],
        "exact_source_reconstruction": "".join(c.text for c in chunks) == article,
    }
    if hasattr(backend, "tokenizer"):
        info.update(
            special_tokens=backend.tokenizer.special_tokens_map,
            bos=backend.tokenizer.bos_token_id,
            eos=backend.tokenizer.eos_token_id,
            pad=backend.tokenizer.pad_token_id,
            special_tokens_added=backend.tokenizer.num_special_tokens_to_add(),
            encoder_position_capacity=backend.model.config.max_position_embeddings,
            checkpoint_generation_config=backend.model.generation_config.to_dict(),
            case_preserved_in_source=True,
        )
    else:
        source = backend.translator.fields["src"]
        target = backend.translator.fields["tgt"]
        specials = int(source.init_token is not None) + int(
            source.eos_token is not None
        )
        info.update(
            model_tokens_including_specials=prepared.total_tokens + specials,
            source_special_tokens_added=specials,
            source_vocabulary=len(source.vocab),
            target_vocabulary=len(target.vocab),
            target_specials={
                "bos": target.init_token,
                "eos": target.eos_token,
                "pad": target.pad_token,
                "unk": target.unk_token,
            },
            case_lowered_as_required=True,
            source_policy="400-token training convention; app window recorded separately, encoder positional capacity respected",
            checkpoint_oov_source_tokens=sum(
                token not in source.vocab.stoi
                for token in engine.prepare_article(article, True).tokens
            ),
            model_embedded_vocab_matches=len(target.vocab)
            == backend.model.decoder.embeddings.word_lut.num_embeddings,
            positional_buffers={
                n: list(b.shape)
                for n, b in backend.model.named_buffers()
                if n.endswith("pe")
            },
        )
    return info, chunks


def inspect_backend(backend, article):
    with patch.object(
        backend, "source_limit", full.source_capacity(backend, full.load_config())
    ):
        return _inspect_backend(backend, article)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="recheck")
    parser.add_argument("--fetch-validation", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(engine.MODEL_CHOICES),
        choices=engine.MODEL_CHOICES,
    )
    parser.add_argument("--only-huang", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.fetch_validation:
        fetch_validation()
        return
    article = (WORK / "huang_article.txt").read_text(encoding="utf-8")
    cases = [
        {
            "case": "huang_unverified",
            "article": article,
            "reference": None,
            "source": SOURCE_NOTICE,
            "sha256": fingerprint(article),
        }
    ]
    if not args.only_huang:
        cases += json.loads(
            (WORK / "validation_sources.json").read_text(encoding="utf-8")
        )
    manifest = {
        "notice": SOURCE_NOTICE,
        "device": "cuda",
        "length": "Medium",
        "config": asdict(full.load_config()),
        "source_hashes": {c["case"]: c["sha256"] for c in cases},
        "code_hashes": {
            f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
            for f in [
                "summarizer.py",
                "hf_summarizer.py",
                "article_summarizer.py",
                "processing_config.json",
            ]
        },
    }
    save(WORK / args.label / "manifest.json", manifest)
    for name in args.models:
        with engine._execution_lock:
            backend = engine.get_model(name, "cuda")
            identity = id(backend)
            for case in cases:
                if (
                    args.skip_existing
                    and (WORK / args.label / f"{case['case']}_{name}.json").exists()
                ):
                    continue
                article = case["article"]
                info, chunks = inspect_backend(backend, article)
                row = {
                    "notice": SOURCE_NOTICE,
                    "case": case["case"],
                    "model": name,
                    "source": case,
                    "inspection": info,
                    "backend_identity": identity,
                    "direct": [],
                }
                with Trace(backend) as trace:
                    try:
                        result = full.summarize_article(article, name, "Medium")
                        row["pipeline"] = asdict(result)
                        displayed = app.generate_summary
                        with patch.object(
                            app, "summarize_article", return_value=result
                        ):
                            row["displayed_summary"] = displayed(
                                article, name, "Medium"
                            )[0]
                    except (ValueError, RuntimeError) as error:
                        row["pipeline_error"] = repr(error)
                        row["displayed_summary"] = ""
                    row["pipeline_decodes"] = trace.calls
                if not args.skip_direct:
                    # Identical complete content for models that fit; otherwise
                    # identical source chunks, explicitly NOT a full single pass.
                    for chunk in chunks:
                        with (
                            patch.object(backend, "source_limit", info["source_limit"]),
                            Trace(backend) as trace,
                        ):
                            prepared = backend.prepare(chunk.text, False)
                            try:
                                if hasattr(backend, "tokenizer"):
                                    direct_hf(backend, prepared, trace)
                                else:
                                    with documented_opennmt(backend):
                                        backend.generate(
                                            prepared, engine.Budget(35, 100)
                                        )
                            except (ValueError, RuntimeError) as error:
                                trace.calls.append({"error": repr(error)})
                            row["direct"].extend(trace.calls)
                assert id(engine._models[name]) == identity
                save(WORK / args.label / f"{case['case']}_{name}.json", row)
                print(
                    json.dumps(
                        {
                            "label": args.label,
                            "case": case["case"],
                            "model": name,
                            "words": info["original_words"],
                            "tokens": info["model_tokens"],
                            "capacity": info["source_limit"],
                            "chunks": info["chunk_count"],
                            "summary_words": row.get("pipeline", {}).get(
                                "actual_words"
                            ),
                            "summary": row["displayed_summary"],
                            "error": row.get("pipeline_error"),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
