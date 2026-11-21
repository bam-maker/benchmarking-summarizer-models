"""Lazy CPU/CUDA adapters for the four pretrained summarizers."""

from __future__ import annotations

import argparse
import io
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer

from runtime import resolve_device, synchronize

ROOT = Path(__file__).resolve().parent
SETUP = ROOT / "OpenNMT_Verified_Setup"
SOURCE_LIMIT = 400
MODEL_CHOICES = ("LSTM", "Transformer", "BART", "PEGASUS")


@dataclass(frozen=True)
class Budget:
    min_tokens: int
    max_tokens: int


# These are the kit's tested decoding budgets, not word-count targets.
LENGTH_PRESETS = {
    "Short": Budget(10, 35),
    "Medium": Budget(30, 65),
    "Detailed": Budget(55, 100),
}


class InputError(ValueError):
    """An input that must be corrected or explicitly approved before decoding."""


@dataclass(frozen=True)
class PreparedArticle:
    tokens: tuple[str | int, ...]
    total_tokens: int

    @property
    def truncated(self) -> bool:
        return self.total_tokens > len(self.tokens)


@dataclass(frozen=True)
class SummaryResult:
    summary: str
    model: str
    length: str
    word_count: int
    seconds: float
    generated_tokens: int
    ended_with_eos: bool
    input_tokens: int
    used_tokens: int
    notices: str
    load_seconds: float = 0.0
    device: str = "cpu"


def prepare_article(article: str, allow_truncation: bool = False) -> PreparedArticle:
    if not isinstance(article, str) or not article.strip():
        raise InputError("Enter an English news article before summarizing.")
    if "\uffe8" in article:
        raise InputError(
            "Replace the reserved OpenNMT feature separator (U+FFE8) first."
        )
    normalized = article.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
            }
        )
    )
    # Treebank only splits periods at sentence ends. Apply it to simple sentence
    # segments so raw, multi-sentence articles also get separate period tokens.
    # This is a documented PTB-style approximation, not Stanford CoreNLP.
    tokenizer = TreebankWordTokenizer()
    tokens = [
        token.lower()
        for segment in re.split(r"(?<=[.!?])\s+", normalized)
        for token in tokenizer.tokenize(segment, convert_parentheses=True)
    ]
    if not tokens:
        raise InputError("The article has no usable tokens.")
    if len(tokens) > SOURCE_LIMIT and not allow_truncation:
        raise InputError(
            f"Article has {len(tokens)} model-input tokens; the limit is {SOURCE_LIMIT}. "
            "No text was summarized. Shorten it or explicitly allow first-400-token "
            "truncation. Your full article remains in the input textbox."
        )
    return PreparedArticle(tuple(tokens[:SOURCE_LIMIT]), len(tokens))


def clean_summary(raw_tokens: list[str]) -> str:
    tokens = [token for token in raw_tokens if token not in {"<t>", "</t>"}]
    brackets = {
        "-lrb-": "(",
        "-rrb-": ")",
        "-lsb-": "[",
        "-rsb-": "]",
        "-lcb-": "{",
        "-rcb-": "}",
    }
    tokens = [brackets.get(token, token) for token in tokens]
    text = TreebankWordDetokenizer().detokenize(tokens)
    return re.sub(r"\s+([.,!?;:%])", r"\1", text).strip()


def output_notices(summary: str, ended_with_eos: bool, budget: Budget) -> list[str]:
    notices = []
    if not ended_with_eos:
        notices.append(
            f"OUTPUT LIMIT: decoding reached the {budget.max_tokens}-token cap "
            "without an end-of-sequence token. The output may be unfinished."
        )
    if not summary:
        notices.append("EMPTY OUTPUT: the model did not produce readable summary text.")
    elif not re.search(r"[.!?][\"'\u201d\u2019)\]]*$", summary):
        notices.append(
            "POSSIBLY UNFINISHED: the output has no final sentence punctuation. "
            "This is a heuristic, not a grammatical completeness check."
        )
    return notices


class OpenNMTSummarizer:
    """One resident translator, its checkpoint vocabulary, and a decoding lock."""

    def __init__(self, model_name: str):
        checkpoint = SETUP / f"{model_name.lower()}_inference.pt"
        legacy = SETUP / "legacy"
        if not checkpoint.is_file() or not (legacy / "onmt").is_dir():
            raise FileNotFoundError(
                "Missing verified model files. Run "
                r".\.venv\Scripts\python.exe OpenNMT_Verified_Setup\prepare.py first."
            )
        if str(legacy) not in sys.path:
            sys.path.insert(0, str(legacy))
        import onmt.opts
        import torch
        from onmt.translate.translator import build_translator

        torch.set_num_threads(2)
        parser = argparse.ArgumentParser()
        onmt.opts.translate_opts(parser)
        options = parser.parse_args(
            [
                "-model",
                str(checkpoint),
                "-src",
                "unused",
                "-gpu",
                "-1",
                "-beam_size",
                "4",
                "-batch_size",
                "1",
                "-n_best",
                "1",
                "-block_ngram_repeat",
                "3",
                "-ignore_when_blocking",
                ".",
                "</t>",
                "<t>",
            ]
        )
        self.translator = build_translator(
            options, report_score=False, out_file=io.StringIO()
        )
        self.model_name = model_name
        self.lock = threading.Lock()
        self.device = "cpu"
        self.source_limit = SOURCE_LIMIT

    @property
    def model(self):
        return self.translator.model

    @property
    def generation_limit(self):
        limits = [
            buffer.shape[0] - 1
            for name, buffer in self.model.decoder.embeddings.named_buffers()
            if name.split(".")[-1] == "pe"
        ]
        return min(limits) if limits else None

    def to_device(self, device):
        with self.lock:
            self.model.to(device)
            self.translator.gpu = 0 if device == "cuda" else -1
            self.translator.cuda = device == "cuda"
            self.device = device

    def prepare(self, article, allow_truncation=False):
        return prepare_article(article, allow_truncation)

    def generate(
        self, article: PreparedArticle, budget: Budget
    ) -> tuple[str, int, bool, float]:
        import torch
        from onmt import inputters
        from onmt.translate import TranslationBuilder

        # Length settings and legacy decoder state are mutable. Requests for the
        # same resident model must not interleave, even outside the Gradio queue.
        with self.lock, torch.inference_mode():
            synchronize(self.device)
            started = time.perf_counter()
            translator = self.translator
            translator.min_length = budget.min_tokens
            translator.max_length = budget.max_tokens
            data = inputters.build_dataset(
                translator.fields,
                "text",
                src_data_iter=[" ".join(article.tokens)],
                use_filter_pred=False,
            )
            # No hidden filtering, additional source truncation, or line splitting.
            if len(data.examples) != 1 or tuple(data.examples[0].src) != article.tokens:
                raise RuntimeError("OpenNMT altered the prepared source unexpectedly.")
            batches = inputters.OrderedIterator(
                dataset=data,
                device=self.device,
                batch_size=1,
                train=False,
                sort=False,
                sort_within_batch=True,
                shuffle=False,
            )
            batch = next(iter(batches))
            # The normal translate() wrapper discards EOS. Use its same batch
            # decoder/builder directly so termination notices can inspect real IDs.
            decoded = translator.translate_batch(batch, data, fast=False)
            hypothesis = decoded["predictions"][0][0]
            eos_id = translator.fields["tgt"].vocab.stoi[inputters.EOS_WORD]
            ended_with_eos = bool(hypothesis) and int(hypothesis[-1]) == eos_id
            generated_tokens = len(hypothesis) - int(ended_with_eos)
            translation = TranslationBuilder(data, translator.fields).from_batch(
                decoded
            )[0]
            summary = clean_summary(translation.pred_sents[0])
            synchronize(self.device)
            seconds = time.perf_counter() - started
        return summary, generated_tokens, ended_with_eos, seconds


_models: dict[str, object] = {}
_models_lock = threading.Lock()
# Device transfers and requests must not overlap while inactive models are moved
# off the GPU. The browser queue alone would not protect direct Python callers.
_execution_lock = threading.RLock()


def get_model(model_name: str, device: str = "auto"):
    if model_name not in MODEL_CHOICES:
        raise InputError(f"Choose one of: {', '.join(MODEL_CHOICES)}.")
    # A lock around construction avoids duplicate first loads on concurrent calls.
    with _models_lock:
        target = resolve_device(device)
        import torch

        torch.set_num_threads(2)
        if target == "cuda":
            # Use FP32 for every backend in the timing comparison.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        for name, inactive in _models.items():
            if name != model_name and inactive.device == "cuda":
                inactive.to_device("cpu")
        if target == "cuda":
            torch.cuda.empty_cache()
        if model_name not in _models:
            if model_name in {"LSTM", "Transformer"}:
                _models[model_name] = OpenNMTSummarizer(model_name)
            else:
                from hf_summarizer import HFSummarizer

                _models[model_name] = HFSummarizer(model_name)
        if _models[model_name].device != target:
            try:
                _models[model_name].to_device(target)
            except torch.cuda.OutOfMemoryError:
                # A failed Module.to() can leave a partially moved model. Restore
                # one consistent device before a later request or CPU retry.
                _models[model_name].to_device("cpu")
                torch.cuda.empty_cache()
                raise
        return _models[model_name]


def summarize(
    article: str,
    model_name: str = "LSTM",
    length: str = "Medium",
    allow_truncation: bool = False,
    device: str = "auto",
) -> SummaryResult:
    if length not in LENGTH_PRESETS:
        raise InputError(f"Choose one of: {', '.join(LENGTH_PRESETS)}.")
    if model_name not in MODEL_CHOICES:
        raise InputError(f"Choose one of: {', '.join(MODEL_CHOICES)}.")
    if not isinstance(article, str) or not article.strip():
        raise InputError("Enter an English news article before summarizing.")
    # Reject invalid legacy input before loading any weights.
    prepared = (
        prepare_article(article, allow_truncation)
        if model_name in {"LSTM", "Transformer"}
        else None
    )
    budget = LENGTH_PRESETS[length]
    fallback = ""
    import torch

    with _execution_lock:
        target = resolve_device(device)
        synchronize(target)
        load_started = time.perf_counter()
        try:
            model = get_model(model_name, target)
        except torch.cuda.OutOfMemoryError:
            if device.lower() != "auto":
                raise
            target = "cpu"
            model = get_model(model_name, "cpu")
            fallback = "CUDA model loading ran out of memory; using explicitly reported CPU fallback."
        synchronize(target)
        load_seconds = time.perf_counter() - load_started
        if prepared is None:
            prepared = model.prepare(article, allow_truncation)
        attempt_started = time.perf_counter()
        try:
            summary, generated, ended, seconds = model.generate(prepared, budget)
        except torch.cuda.OutOfMemoryError:
            if device.lower() != "auto":
                raise
            failed_seconds = time.perf_counter() - attempt_started
            transfer_started = time.perf_counter()
            model.to_device("cpu")
            torch.cuda.empty_cache()
            load_seconds += time.perf_counter() - transfer_started
            fallback = f"CUDA ran out of memory; explicitly reported CPU fallback. Failed CUDA attempt: {failed_seconds:.3f} seconds (separate from successful CPU generation latency)."
            target = "cpu"
            summary, generated, ended, seconds = model.generate(prepared, budget)
    notices = [
        (
            f"{model_name} | {target.upper()} | {length}: approximate {budget.min_tokens}-"
            f"{budget.max_tokens} decoding-token budget (not a word-count promise)."
        ),
        (
            f"Input: {len(prepared.tokens)} of {prepared.total_tokens} tokens used. "
            f"Output: {generated} backend tokens, excluding EOS; subwords, punctuation, and OpenNMT sentence tags count."
        ),
    ]
    if prepared.truncated:
        notices.append(
            f"INPUT TRUNCATED WITH YOUR PERMISSION: only the first {len(prepared.tokens)} "
            f"tokens were summarized; {prepared.total_tokens - len(prepared.tokens)} trailing "
            "tokens were excluded. Your full article remains in the input textbox."
        )
    notices.extend(output_notices(summary, ended, budget))
    if ended:
        notices.append("The model emitted an end-of-sequence token.")
    notices.append(
        f"Model loading / device transfer: {load_seconds:.3f} seconds (separate from generation)."
    )
    if fallback:
        notices.append(fallback)
    if device.lower() == "auto" and target == "cpu" and not fallback:
        notices.append(
            "Auto selected CPU fallback because CUDA is unavailable or CPU was configured."
        )
    notices.append("Generation time excludes model loading and input tokenization.")
    return SummaryResult(
        summary,
        model_name,
        length,
        len(summary.split()),
        seconds,
        generated,
        ended,
        prepared.total_tokens,
        len(prepared.tokens),
        "\n".join(notices),
        load_seconds,
        target,
    )
