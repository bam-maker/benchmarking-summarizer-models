"""Coverage-checked, bounded map/reduce summarization for the demo.

The existing fixed-budget summarizer/benchmark path is intentionally unchanged.
Source coverage means every character was presented to a model, not that the
abstractive result contains every fact. No partial result escapes on failure.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

import summarizer as engine
from runtime import resolve_device, synchronize

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
LENGTH_CHOICES = ("Short", "Medium", "Detailed")


class ProcessingLimitError(engine.InputError):
    """The full job cannot complete; do not display its intermediate summaries."""


@dataclass(frozen=True)
class ProcessingConfig:
    processing_mode: str = "full"
    ratios: dict = field(
        default_factory=lambda: {"Short": 0.10, "Medium": 0.15, "Detailed": 0.22}
    )
    tokens_per_word: dict = field(
        default_factory=lambda: {
            "LSTM": 1.6,
            "Transformer": 1.6,
            "BART": 1.5,
            "PEGASUS": 1.6,
        }
    )
    max_output_tokens: dict = field(
        default_factory=lambda: {
            "LSTM": 256,
            "Transformer": 256,
            "BART": 512,
            "PEGASUS": 512,
        }
    )
    word_budget_slack: float = 1.35
    intermediate_ratio: float = 0.35
    max_article_characters: int = 120000
    max_article_words: int = 12000
    max_source_chunks: int = 64
    max_model_calls: int = 96
    max_total_input_tokens: int = 64000
    max_consolidation_passes: int = 4
    max_runtime_seconds: float = 180
    concise_input_words: int = 12

    def __post_init__(self):
        if self.processing_mode not in {"full", "prefix"}:
            raise ValueError("processing_mode must be full or prefix")
        if set(self.ratios) != set(LENGTH_CHOICES) or not all(
            0 < self.ratios[k] <= 1 for k in LENGTH_CHOICES
        ):
            raise ValueError(
                "ratios must define Short, Medium and Detailed between 0 and 1"
            )
        if not self.ratios["Short"] <= self.ratios["Medium"] <= self.ratios["Detailed"]:
            raise ValueError("length ratios must be nondecreasing")
        for name in engine.MODEL_CHOICES:
            if self.tokens_per_word[name] < 1 or self.max_output_tokens[name] < 8:
                raise ValueError("token factors must be >= 1 and output caps >= 8")
        for name in (
            "max_article_characters",
            "max_article_words",
            "max_source_chunks",
            "max_model_calls",
            "max_total_input_tokens",
            "max_runtime_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_consolidation_passes < 0 or self.concise_input_words < 0:
            raise ValueError("depth and concise-input threshold must be nonnegative")
        if (
            not 0 < self.intermediate_ratio < 0.7
            or not 1 <= self.word_budget_slack <= 2
        ):
            raise ValueError("invalid intermediate ratio or word-budget slack")


def load_config() -> ProcessingConfig:
    path = Path(
        os.environ.get("SUMMARIZER_CONFIG", str(ROOT / "processing_config.json"))
    )
    try:
        config = ProcessingConfig(**json.loads(path.read_text(encoding="utf-8")))
        mode = os.environ.get("SUMMARIZER_PROCESSING_MODE")
        return replace(config, processing_mode=mode) if mode else config
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise engine.InputError(f"Invalid processing configuration: {error}") from error


def word_count(text: str) -> int:
    return len(text.split())


def requested_words(article_words: int, length: str, config: ProcessingConfig) -> int:
    if length not in LENGTH_CHOICES:
        raise engine.InputError("Choose Short, Medium or Detailed.")
    return min(article_words, max(1, int(article_words * config.ratios[length] + 0.5)))


_punkt_params = PunktParameters()
_punkt_params.abbrev_types = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "jr",
    "sr",
    "u.s",
    "u.k",
    "e.g",
    "i.e",
    "vs",
    "inc",
}
_sentences = PunktSentenceTokenizer(_punkt_params)  # No corpus/model download.


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Contiguous spans, including all original whitespace and trailing text."""
    ends = [end for _, end in _sentences.span_tokenize(text)]
    if not ends:
        return [(0, len(text))] if text else []
    ends[-1] = len(text)
    starts = [0, *ends[:-1]]
    return list(zip(starts, ends))


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    text: str
    prepared: engine.PreparedArticle


class ChunkPlanner:
    def __init__(self, backend, check=lambda: None):
        self.backend = backend
        self.check = check

    def count(self, text: str) -> int:
        self.check()
        if not text.strip():
            return 0
        # Truncation is used ONLY to measure the full token count, never as the
        # source fed to generate(). Every actual chunk is prepared strictly.
        return self.backend.prepare(text, True).total_tokens

    def fits(self, text: str) -> bool:
        return self.count(text) <= self.backend.source_limit

    def prefix_end(self, text: str) -> int:
        """Find a verified fitting word boundary, or Unicode codepoint boundary."""
        if self.fits(text):
            return len(text)
        boundaries = [match.end() for match in re.finditer(r"\S+\s*", text)]
        low, high, best = 0, len(boundaries) - 1, 0
        while low <= high:
            middle = (low + high) // 2
            if self.fits(text[: boundaries[middle]]):
                best = boundaries[middle]
                low = middle + 1
            else:
                high = middle - 1
        if best:
            return best
        # A single enormous word/URL can exceed the encoder limit. Slice the
        # original Python string, never decode partial bytes or drop characters.
        low, high = 1, len(text)
        while low <= high:
            middle = (low + high) // 2
            if self.fits(text[:middle]):
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if not best:
            raise ProcessingLimitError(
                "A source character cannot fit this model's input. No final summary was returned."
            )
        return best

    def plan(self, text: str, max_chunks: int) -> list[Chunk]:
        spans = []
        pending_start = pending_end = 0
        for start, end in sentence_spans(text):
            while not self.fits(text[start:end]):
                if pending_end > pending_start:
                    spans.append((pending_start, pending_end))
                    pending_start = pending_end
                split = start + self.prefix_end(text[start:end])
                spans.append((start, split))
                start = split
                pending_start = pending_end = split
                if len(spans) >= max_chunks and start < len(text):
                    raise ProcessingLimitError(
                        "This article needs more chunks than the configured limit. No final summary was returned."
                    )
            if not text[start:end].strip():
                pending_end = end
                continue
            if pending_end > pending_start and not self.fits(text[pending_start:end]):
                spans.append((pending_start, pending_end))
                pending_start = start
            pending_end = end
            if len(spans) >= max_chunks:
                raise ProcessingLimitError(
                    "This article needs more chunks than the configured limit. No final summary was returned."
                )
        if pending_end > pending_start:
            spans.append((pending_start, pending_end))
        if len(spans) > max_chunks:
            raise ProcessingLimitError(
                "This article needs too many chunks. No final summary was returned."
            )
        chunks = [
            Chunk(
                start,
                end,
                text[start:end],
                self.backend.prepare(text[start:end], False),
            )
            for start, end in spans
        ]
        cursor = 0
        for chunk in chunks:
            if (
                chunk.start != cursor
                or chunk.end <= chunk.start
                or chunk.prepared.truncated
            ):
                raise RuntimeError("Chunk planner lost or overlapped source content.")
            cursor = chunk.end
        if cursor != len(text) or "".join(chunk.text for chunk in chunks) != text:
            raise RuntimeError("Chunk planner did not cover the complete source.")
        return chunks


def deduplicate_sentences(text: str) -> str:
    """Remove only exact complete generated-sentence duplicates, in source order."""
    kept, seen = [], set()
    for start, end in sentence_spans(text):
        sentence = text[start:end].strip()
        key = " ".join(sentence.split()).casefold()
        complete = re.search(r"[.!?][\"'\u201d\u2019)\]]*$", sentence)
        if not complete or key not in seen:
            kept.append(sentence)
            seen.add(key)
    return " ".join(kept)


def token_budget(
    backend, chunk: Chunk, target: int, config: ProcessingConfig
) -> tuple[engine.Budget, bool]:
    words = max(1, word_count(chunk.text))
    target = min(words, max(1, target))
    factor = max(
        config.tokens_per_word[backend.model_name],
        min(4.0, len(chunk.prepared.tokens) / words),
    )
    desired_max = max(8, math.ceil(target * factor * config.word_budget_slack) + 6)
    # The application work cap is configurable; positional decoder capacity is
    # not. The recurrent LSTM has no positional-table limit.
    ceiling = (
        getattr(backend, "generation_limit", None)
        or config.max_output_tokens[backend.model_name]
    )
    maximum = min(config.max_output_tokens[backend.model_name], ceiling, desired_max)
    minimum = 0 if words < 40 else min(12, int(target * factor * 0.2), maximum - 2)
    return engine.Budget(minimum, maximum), desired_max > maximum


@dataclass(frozen=True)
class Stage:
    phase: str
    chunks: int
    input_words: int
    output_words: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ArticleResult:
    summary: str
    model: str
    length: str
    requested_words: int
    source_words: int
    actual_words: int
    source_characters: int
    covered_characters: int
    source_chunks: int
    chunk_spans: tuple
    consolidation_passes: int
    model_calls: int
    total_input_tokens: int
    generation_seconds: float
    load_seconds: float
    device: str
    warnings: tuple[str, ...]
    stages: tuple[Stage, ...]
    processing_mode: str
    returned_input: bool = False
    retained_fragments: int = 0

    @property
    def coverage_complete(self):
        return self.covered_characters == self.source_characters


def summarize_article(
    article: str,
    model_name="LSTM",
    length="Medium",
    *,
    config=None,
    progress_callback=None,
) -> ArticleResult:
    config = config or load_config()
    if model_name not in engine.MODEL_CHOICES:
        raise engine.InputError("Choose LSTM, Transformer, BART or PEGASUS.")
    if not isinstance(article, str) or not article.strip():
        raise engine.InputError("Enter an article before summarizing.")
    words = word_count(article)
    target_words = requested_words(words, length, config)
    if len(article) > config.max_article_characters or words > config.max_article_words:
        raise ProcessingLimitError(
            "This article exceeds the configured processing limit. Shorten it or raise the local limit; no final summary was returned."
        )
    notify = progress_callback or (lambda completed, total, phase: None)
    if words <= config.concise_input_words and len(article) <= 240:
        warning = (
            "Already concise: returned the original text without model generation."
        )
        result = ArticleResult(
            article.strip(),
            model_name,
            length,
            target_words,
            words,
            words,
            len(article),
            len(article),
            0,
            ((0, len(article)),),
            0,
            0,
            0,
            0.0,
            0.0,
            "none",
            (warning,),
            (),
            config.processing_mode,
            True,
        )
        diagnostics = asdict(result)
        diagnostics.pop("summary")
        logger.info("article_summary %s", json.dumps(diagnostics))
        notify(1, 1, "Complete")
        return result

    import torch

    with engine._execution_lock:
        started = time.perf_counter()
        calls = input_tokens = source_completed = 0
        generation_seconds = 0.0
        warnings = []
        stages = []
        retained_fragments = 0

        def check():
            if time.perf_counter() - started > config.max_runtime_seconds:
                raise ProcessingLimitError(
                    "Processing reached the configured time limit. No final summary was returned."
                )

        notify(0, 1, "Preparing article")
        device = resolve_device("auto")
        load_started = time.perf_counter()
        try:
            backend = engine.get_model(model_name, device)
        except torch.cuda.OutOfMemoryError:
            device = "cpu"
            backend = engine.get_model(model_name, "cpu")
            logger.warning("CUDA activation OOM; continuing on CPU")
        synchronize(device)
        load_seconds = time.perf_counter() - load_started
        planner = ChunkPlanner(backend, check)

        def decode_once(chunk, budget, wanted, capped, intermediate):
            nonlocal calls, input_tokens, generation_seconds
            nonlocal device, load_seconds, retained_fragments
            check()
            if (
                calls >= config.max_model_calls
                or input_tokens + len(chunk.prepared.tokens)
                > config.max_total_input_tokens
            ):
                raise ProcessingLimitError(
                    "Processing reached the configured workload limit. No final summary was returned."
                )
            calls += 1
            input_tokens += len(chunk.prepared.tokens)
            try:
                output, generated, ended, seconds = backend.generate(
                    chunk.prepared, budget
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "CUDA decoding OOM; retrying the same complete chunk on CPU"
                )
                transfer_started = time.perf_counter()
                backend.to_device("cpu")
                torch.cuda.empty_cache()
                device = "cpu"
                load_seconds += time.perf_counter() - transfer_started
                check()
                if (
                    calls >= config.max_model_calls
                    or input_tokens + len(chunk.prepared.tokens)
                    > config.max_total_input_tokens
                ):
                    raise ProcessingLimitError(
                        "Processing reached the workload limit during recovery. No final summary was returned."
                    )
                calls += 1
                input_tokens += len(chunk.prepared.tokens)
                output, generated, ended, seconds = backend.generate(
                    chunk.prepared, budget
                )
            generation_seconds += seconds
            logger.info(
                "decode %s",
                json.dumps(
                    {
                        "model": model_name,
                        "device": device,
                        "target_words": wanted,
                        "min_tokens": budget.min_tokens,
                        "max_tokens": budget.max_tokens,
                        "capped": capped,
                        "input_tokens": len(chunk.prepared.tokens),
                        "generated_tokens": generated,
                        "eos": ended,
                    }
                ),
            )
            check()
            if not output.strip():
                if (
                    intermediate
                    and word_count(chunk.text) <= config.concise_input_words
                    and len(chunk.text) <= 240
                ):
                    # The original source was still fed to the model. On an empty
                    # tiny-fragment decode, pass its exact text into consolidation
                    # rather than silently omit it or pretend it was summarized.
                    retained_fragments += 1
                    logger.info(
                        "Empty concise-fragment decode; retaining source span=%s:%s",
                        chunk.start,
                        chunk.end,
                    )
                    return chunk.text.strip(), True
                raise ProcessingLimitError(
                    "The model returned an empty chunk summary. Full-article processing could not finish; no final summary was returned."
                )
            return output, ended

        def generate(chunk, wanted, intermediate=False):
            budget, capped = token_budget(backend, chunk, wanted, config)
            hard_limit = min(
                config.max_output_tokens[model_name],
                getattr(backend, "generation_limit", None)
                or config.max_output_tokens[model_name],
            )
            while True:
                output, ended = decode_once(chunk, budget, wanted, capped, intermediate)
                if ended:
                    return output, capped
                if budget.max_tokens >= hard_limit:
                    raise ProcessingLimitError(
                        "The model reached its safe output limit before finishing. No final summary was returned."
                    )
                # Word-derived budgets are soft preferences. Retry with room for
                # natural EOS rather than returning a mid-sentence word-count cut.
                maximum = min(
                    hard_limit,
                    max(budget.max_tokens + 1, math.ceil(budget.max_tokens * 1.75)),
                )
                logger.info(
                    "Extending soft decoding budget for EOS: %s -> %s",
                    budget.max_tokens,
                    maximum,
                )
                budget = replace(budget, max_tokens=maximum)

        try:
            if config.processing_mode == "prefix":
                end = planner.prefix_end(article)
                processing_text = article[:end]
                if end < len(article):
                    warnings.append(
                        f"Prefix mode: only the first {word_count(processing_text)} of {words} article words were processed."
                    )
            else:
                processing_text = article
            chunks = planner.plan(processing_text, config.max_source_chunks)
            original_spans = tuple((chunk.start, chunk.end) for chunk in chunks)
            source_chunks = len(chunks)
            current = processing_text
            passes = 0
            while True:
                final = len(chunks) == 1
                phase = (
                    "Article chunks" if passes == 0 else f"Consolidating pass {passes}"
                )
                notify(0, len(chunks), phase)
                if (
                    calls + len(chunks) > config.max_model_calls
                    or input_tokens
                    + sum(len(chunk.prepared.tokens) for chunk in chunks)
                    > config.max_total_input_tokens
                ):
                    raise ProcessingLimitError(
                        "This stage exceeds the configured workload limit. No final summary was returned."
                    )
                before_words, before_tokens = (
                    word_count(current),
                    planner.count(current),
                )
                outputs = []
                for i, chunk in enumerate(chunks, 1):
                    chunk_words = word_count(chunk.text)
                    if final:
                        wanted = min(target_words, chunk_words)
                        if passes:
                            wanted = min(
                                wanted,
                                max(
                                    1, int(chunk_words * (0.5 + config.ratios[length]))
                                ),
                            )
                    else:
                        share = target_words * chunk_words / max(1, before_words)
                        wanted = max(
                            1,
                            min(
                                int(chunk_words * 0.6),
                                math.ceil(
                                    max(
                                        chunk_words * config.intermediate_ratio,
                                        share * 1.2,
                                    )
                                ),
                            ),
                        )
                    output, capped = generate(chunk, wanted, intermediate=not final)
                    outputs.append(output)
                    if passes == 0:
                        source_completed += chunk.end - chunk.start
                    notify(i, len(chunks), phase)
                combined = deduplicate_sentences("\n".join(outputs))
                after_words, after_tokens = (
                    word_count(combined),
                    planner.count(combined),
                )
                stages.append(
                    Stage(
                        phase,
                        len(chunks),
                        before_words,
                        after_words,
                        before_tokens,
                        after_tokens,
                    )
                )
                # Every reduction, including the final consolidation, must shrink
                # both words and tokenizer size. No arbitrary sentence clipping.
                if (not final or passes) and (
                    after_words >= before_words or after_tokens >= before_tokens
                ):
                    raise ProcessingLimitError(
                        "The model could not shorten the intermediate text safely. No final summary was returned."
                    )
                if final:
                    if capped:
                        warnings.append(
                            "The requested length exceeds this model's final-pass budget; the summary may be shorter."
                        )
                    break
                if passes >= config.max_consolidation_passes:
                    raise ProcessingLimitError(
                        "Processing reached the consolidation-depth limit. No final summary was returned."
                    )
                current = combined
                chunks = planner.plan(current, config.max_source_chunks)
                if calls + len(chunks) > config.max_model_calls:
                    raise ProcessingLimitError(
                        "Processing needs more work than the configured limit. No final summary was returned."
                    )
                passes += 1

            check()
            if source_completed != len(processing_text):
                raise RuntimeError("Not every original source chunk was processed.")
            if not re.search(r"[.!?][\"'\u201d\u2019)\]]*$", combined):
                warnings.append(
                    "Some generated text may be unfinished; review the summary."
                )
            if after_words < target_words * 0.5 or after_words > target_words * 2:
                warnings.append(
                    f"Length is approximate: requested about {target_words} words; produced {after_words}."
                )
            result = ArticleResult(
                combined,
                model_name,
                length,
                target_words,
                words,
                after_words,
                len(article),
                source_completed,
                source_chunks,
                original_spans,
                passes,
                calls,
                input_tokens,
                generation_seconds,
                load_seconds,
                device,
                tuple(warnings),
                tuple(stages),
                config.processing_mode,
                retained_fragments=retained_fragments,
            )
            # Detailed diagnostics stay out of the demo. No article body is logged.
            diagnostics = asdict(result)
            diagnostics.pop("summary")
            logger.info("article_summary %s", json.dumps(diagnostics))
            return result
        except Exception:
            logger.exception(
                "Article processing aborted: model=%s calls=%s covered_characters=%s/%s stages=%s",
                model_name,
                calls,
                source_completed,
                len(article),
                stages,
            )
            raise
