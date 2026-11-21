"""Direct, revision-pinned BART and PEGASUS inference; no pipeline or training."""

import json
import threading
import time
from pathlib import Path

from runtime import synchronize

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "model_manifest.json").read_text(encoding="utf-8"))


class HFSummarizer:
    def __init__(self, name):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.spec = MANIFEST[name]
        options = {
            "revision": self.spec["revision"],
            "cache_dir": str(ROOT / ".cache" / "huggingface" / "hub"),
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec["id"], use_fast=False, **options
        )
        self.model, self.loading_info = AutoModelForSeq2SeqLM.from_pretrained(
            self.spec["id"],
            torch_dtype=torch.float32,
            use_safetensors=self.spec["weight_file"].endswith(".safetensors"),
            attn_implementation="eager",
            output_loading_info=True,
            **options,
        )
        allowed_missing = (
            {
                "model.encoder.embed_positions.weight",
                "model.decoder.embed_positions.weight",
            }
            if name == "PEGASUS"
            else set()
        )
        missing = set(self.loading_info["missing_keys"])
        if (
            missing - allowed_missing
            or self.loading_info["mismatched_keys"]
            or self.loading_info["error_msgs"]
        ):
            raise RuntimeError(
                f"Unexpected incomplete checkpoint load: {self.loading_info}"
            )
        parameters = dict(self.model.named_parameters())
        if any(parameters[key].requires_grad for key in missing):
            raise RuntimeError(
                "Missing parameters are not fixed positional embeddings."
            )
        self.model.eval()
        self.model_name = name
        self.source_limit = min(
            self.spec["source_limit"], self.model.config.max_position_embeddings
        )
        self.device = "cpu"
        self.lock = threading.Lock()

    def to_device(self, device):
        with self.lock:
            self.model.to(device)
            self.device = device

    @property
    def generation_limit(self):
        return self.model.config.max_position_embeddings - 1

    def prepare(self, article, allow_truncation=False):
        from summarizer import InputError, PreparedArticle

        ids = self.tokenizer.encode(article, add_special_tokens=True, truncation=False)
        total = len(ids)
        if total > self.source_limit:
            if not allow_truncation:
                raise InputError(
                    f"Article has {total} {self.model_name} input tokens; its limit is {self.source_limit}. "
                    "No text was summarized. Shorten it or explicitly allow supported-prefix truncation. "
                    "Your full article remains in the input textbox."
                )
            ids = self.tokenizer.encode(
                article,
                add_special_tokens=True,
                truncation=True,
                max_length=self.source_limit,
            )
        return PreparedArticle(tuple(ids), total)

    def generate(self, article, budget):
        import torch

        with self.lock, torch.inference_mode():
            synchronize(self.device)
            started = time.perf_counter()
            ids = torch.tensor([article.tokens], dtype=torch.long, device=self.device)
            generated = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                min_new_tokens=max(0, budget.min_tokens - 1),
                max_new_tokens=budget.max_tokens,
                num_beams=4,
                do_sample=False,
                no_repeat_ngram_size=3,
                early_stopping=True,
                # Otherwise HF can manufacture EOS at the cap, hiding truncation.
                forced_eos_token_id=None,
            )[0, 1:].tolist()  # Exclude the decoder-start token (BART uses EOS).
            ended = bool(generated) and generated[-1] == self.tokenizer.eos_token_id
            summary = (
                self.tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                .replace("<n>", "\n")
                .strip()
            )
            synchronize(self.device)
            seconds = time.perf_counter() - started
        return summary, len(generated) - int(ended), ended, seconds
