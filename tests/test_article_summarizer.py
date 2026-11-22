"""Coverage/termination checks use deterministic fake decoders, not fake results."""

import unittest
from dataclasses import replace
from itertools import pairwise
from unittest.mock import Mock, patch

import torch

import article_summarizer as full
import summarizer as engine


class Backend:
    model_name = "LSTM"
    source_limit = 40
    device = "cpu"

    def __init__(self):
        self.calls = []

    def prepare(self, text, allow_truncation=False):
        tokens = text.split()
        if not tokens:
            raise engine.InputError("empty")
        if len(tokens) > self.source_limit and not allow_truncation:
            raise engine.InputError("too long")
        return engine.PreparedArticle(tuple(tokens[: self.source_limit]), len(tokens))

    def generate(self, prepared, budget):
        self.calls.append((prepared, budget))
        count = max(1, min(len(prepared.tokens) // 4, budget.max_tokens - 1))
        text = " ".join(prepared.tokens[:count]).rstrip(".") + "."
        return text, count + 1, True, 0.1


class CharacterBackend(Backend):
    def prepare(self, text, allow_truncation=False):
        if len(text) > self.source_limit and not allow_truncation:
            raise engine.InputError("too long")
        return engine.PreparedArticle(tuple(text[: self.source_limit]), len(text))


class PlanningTests(unittest.TestCase):
    def test_sentence_boundaries_and_every_character_are_preserved(self):
        article = "  Dr. Liu approved the plan.\n\nA second sentence follows!  Last sentence? \n"
        pieces = full.sentence_spans(article)
        self.assertEqual("".join(article[a:b] for a, b in pieces), article)
        self.assertIn(
            "Dr. Liu approved the plan.", article[pieces[0][0] : pieces[0][1]]
        )

    def test_greedy_chunks_are_contiguous_and_not_truncated(self):
        article = (
            "\n"
            + " ".join(
                f"Section {i} describes a different local transport plan."
                for i in range(25)
            )
            + "  \n"
        )
        chunks = full.ChunkPlanner(Backend()).plan(article, 64)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(c.text for c in chunks), article)
        self.assertTrue(
            all(
                len(c.prepared.tokens) <= 40 and not c.prepared.truncated
                for c in chunks
            )
        )
        self.assertTrue(all(a.end == b.start for a, b in pairwise(chunks)))

    def test_sentence_larger_than_encoder_is_split_without_loss(self):
        article = " ".join(f"word{i}" for i in range(123)) + "."
        chunks = full.ChunkPlanner(Backend()).plan(article, 64)
        self.assertEqual(len(chunks), 4)
        self.assertEqual("".join(c.text for c in chunks), article)

    def test_one_enormous_unicode_word_is_split_by_codepoint(self):
        article = " \n" + "漢🙂é" * 70
        chunks = full.ChunkPlanner(CharacterBackend()).plan(article, 64)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(c.text for c in chunks), article)
        self.assertTrue(all(len(c.prepared.tokens) <= 40 for c in chunks))

    def test_only_generated_exact_complete_sentences_are_deduplicated(self):
        self.assertEqual(
            full.deduplicate_sentences("One fact. Two facts. one fact. Unfinished"),
            "One fact. Two facts. Unfinished",
        )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.config = full.ProcessingConfig()
        self.loader = patch.object(engine, "get_model", return_value=self.backend)
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.device = patch.object(full, "resolve_device", return_value="cpu")
        self.device.start()
        self.addCleanup(self.device.stop)
        self.article = " ".join(
            f"Place{i} introduced buses and new routes in September." for i in range(40)
        )

    def run_article(self, **kwargs):
        return full.summarize_article(self.article, config=self.config, **kwargs)

    def test_all_source_is_processed_and_multiple_reductions_terminate(self):
        updates = []
        result = self.run_article(progress_callback=lambda *args: updates.append(args))
        self.assertTrue(result.coverage_complete)
        self.assertEqual(result.covered_characters, len(self.article))
        self.assertGreater(result.source_chunks, 1)
        self.assertGreaterEqual(result.consolidation_passes, 1)
        self.assertLessEqual(
            result.consolidation_passes, self.config.max_consolidation_passes
        )
        self.assertEqual(
            result.requested_words,
            round(len(self.article.split()) * self.config.ratios["Medium"]),
        )
        for stage in result.stages:
            self.assertLess(stage.output_words, stage.input_words)
            self.assertLess(stage.output_tokens, stage.input_tokens)
        initial = [u for u in updates if u[2] == "Article chunks"]
        self.assertEqual(initial[-1][0], result.source_chunks)
        self.assertEqual(result.model_calls, len(self.backend.calls))
        expected_tokens = sum(
            max(1, min(len(prepared.tokens) // 4, budget.max_tokens - 1)) + 1
            for prepared, budget in self.backend.calls
        )
        self.assertEqual(result.total_generated_tokens, expected_tokens)
        self.assertAlmostEqual(result.generation_seconds, len(self.backend.calls) * 0.1)

    def test_preferences_use_full_article_and_model_specific_budgets(self):
        article = " ".join(f"word{i}" for i in range(100)) + "."
        self.backend.source_limit = 400
        results, budgets = [], []
        for label in full.LENGTH_CHOICES:
            results.append(
                full.summarize_article(article, length=label, config=self.config)
            )
            budgets.append(self.backend.calls[-1][1].max_tokens)
        self.assertEqual([r.requested_words for r in results], [15, 25, 35])
        self.assertLess(budgets[0], budgets[1])
        self.assertLess(budgets[1], budgets[2])
        self.assertNotEqual(budgets, [15, 25, 35])

    def test_single_fitting_article_is_decoded_once(self):
        self.backend.source_limit = 400
        result = full.summarize_article(self.article, config=self.config)
        self.assertEqual(result.source_chunks, 1)
        self.assertEqual(result.consolidation_passes, 0)
        self.assertEqual(result.model_calls, 1)
        self.assertTrue(result.coverage_complete)

    def test_already_short_chunk_summaries_are_not_compressed_again(self):
        article = " ".join(f"word{i}" for i in range(80)) + "."
        result = full.summarize_article(article, config=self.config)
        self.assertEqual(result.source_chunks, 2)
        self.assertTrue(result.coverage_complete)
        self.assertLessEqual(result.actual_words, result.requested_words)
        self.assertEqual(result.consolidation_passes, 0)
        self.assertEqual(result.model_calls, result.source_chunks)
        # Both chunk outputs survive; do not assert a particular generated sentence.
        self.assertIn("word0", result.summary)
        self.assertIn("word40", result.summary)

    def test_checkpoint_floor_is_bounded_for_short_inputs(self):
        self.backend.source_limit = 400
        self.backend.summary_min_length = 56
        text = " ".join(f"word{i}" for i in range(20))
        chunk = full.ChunkPlanner(self.backend).plan(text, 2)[0]
        budget, _ = full.token_budget(self.backend, chunk, 5, self.config)
        self.assertEqual(budget.profile, "checkpoint")
        self.assertLessEqual(budget.min_tokens, len(chunk.prepared.tokens) // 4)
        self.assertLess(budget.min_tokens, budget.max_tokens)

    def test_short_input_is_an_explicit_non_model_result(self):
        with patch.object(engine, "get_model") as loader:
            result = full.summarize_article(
                "Council approves new buses.", config=self.config
            )
            loader.assert_not_called()
        self.assertTrue(result.returned_input)
        self.assertEqual(result.total_generated_tokens, 0)
        self.assertIn("original text", result.warnings[0])
        self.assertLessEqual(result.requested_words, result.source_words)

    def test_prefix_is_only_enabled_in_configuration_and_is_labeled(self):
        result = full.summarize_article(
            self.article, config=replace(self.config, processing_mode="prefix")
        )
        self.assertFalse(result.coverage_complete)
        self.assertEqual(result.source_chunks, 1)
        self.assertIn("Prefix mode", result.warnings[0])

    def test_echoing_model_cannot_loop_or_claim_completion(self):
        def echo(prepared, budget):
            self.backend.calls.append((prepared, budget))
            return " ".join(prepared.tokens), len(prepared.tokens), True, 0.1

        self.backend.generate = echo
        with (
            self.assertLogs(full.logger, level="ERROR"),
            self.assertRaisesRegex(full.ProcessingLimitError, "could not shorten"),
        ):
            self.run_article()
        self.assertLess(len(self.backend.calls), 20)

    def test_limits_never_return_a_partial_summary(self):
        configurations = [
            replace(self.config, max_article_characters=10),
            replace(self.config, max_source_chunks=1),
            replace(self.config, max_model_calls=1),
            replace(self.config, max_total_input_tokens=1),
            replace(self.config, max_consolidation_passes=0),
            replace(self.config, max_runtime_seconds=1e-9),
        ]
        for config in configurations:
            with (
                self.subTest(config=config),
                self.assertRaises(full.ProcessingLimitError),
                patch.object(full.logger, "exception"),
            ):
                full.summarize_article(self.article, config=config)

    def test_empty_tiny_tail_is_fed_to_model_then_retained_for_consolidation(self):
        original = self.backend.generate

        def decode(prepared, budget):
            if len(prepared.tokens) <= 3:
                self.backend.calls.append((prepared, budget))
                return "", 0, True, 0.1
            return original(prepared, budget)

        self.backend.generate = decode
        result = full.summarize_article(
            self.article + " Last unique fact.", config=self.config
        )
        self.assertTrue(result.coverage_complete)
        self.assertGreaterEqual(result.retained_fragments, 1)

    def test_empty_or_invalid_choices_are_rejected(self):
        for article, model, length in [
            (" ", "LSTM", "Short"),
            ("a b c", "wrong", "Short"),
            ("a b c", "LSTM", "wrong"),
        ]:
            with self.assertRaises(engine.InputError):
                full.summarize_article(article, model, length, config=self.config)

    def test_cuda_decode_recovery_keeps_full_chunk_and_counts_retry_work(self):
        self.backend.source_limit = 400
        self.backend.to_device = Mock()
        self.backend.generate = Mock(
            side_effect=[
                torch.cuda.OutOfMemoryError("simulated"),
                ("A completed CPU summary.", 5, True, 0.25),
            ]
        )
        with (
            patch.object(full, "resolve_device", return_value="cuda"),
            patch.object(full, "synchronize"),
            patch.object(torch.cuda, "empty_cache"),
            self.assertLogs(full.logger, level="WARNING"),
        ):
            result = full.summarize_article(
                " ".join(f"word{i}" for i in range(80)), config=self.config
            )
        self.assertEqual(result.device, "cpu")
        self.assertTrue(result.coverage_complete)
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(result.total_input_tokens, 160)
        self.assertIsNone(result.total_generated_tokens)
        self.assertEqual(result.generation_seconds, 0.25)
        self.backend.to_device.assert_called_once_with("cpu")
        first, second = self.backend.generate.call_args_list
        self.assertEqual(first.args[0], second.args[0])

    def test_cuda_activation_recovery_uses_existing_cpu_backend(self):
        self.backend.source_limit = 400
        with (
            patch.object(full, "resolve_device", return_value="cuda"),
            patch.object(full, "synchronize"),
            patch.object(
                engine,
                "get_model",
                side_effect=[torch.cuda.OutOfMemoryError("simulated"), self.backend],
            ) as loader,
            self.assertLogs(full.logger, level="WARNING"),
        ):
            result = full.summarize_article(
                " ".join(f"word{i}" for i in range(80)), config=self.config
            )
        self.assertEqual(result.device, "cpu")
        self.assertTrue(result.coverage_complete)
        self.assertEqual(loader.call_args.args, ("LSTM", "cpu"))

    def test_soft_budget_extends_to_eos_instead_of_cutting_a_sentence(self):
        self.backend.source_limit = 400
        original = self.backend.generate

        def decode(prepared, budget):
            output, count, _ended, seconds = original(prepared, budget)
            return output, count, len(self.backend.calls) > 1, seconds

        self.backend.generate = decode
        result = full.summarize_article(
            " ".join(f"word{i}" for i in range(80)), length="Short", config=self.config
        )
        self.assertEqual(result.model_calls, 2)
        self.assertGreater(result.total_generated_tokens, len(result.summary.split()))
        self.assertGreater(
            self.backend.calls[1][1].max_tokens, self.backend.calls[0][1].max_tokens
        )
        self.assertTrue(result.summary.endswith("."))

    def test_no_eos_at_hard_limit_does_not_return_unfinished_text(self):
        self.backend.source_limit = 400
        self.backend.generate = lambda prepared, budget: (
            "unfinished text",
            2,
            False,
            0.1,
        )
        caps = {name: 8 for name in engine.MODEL_CHOICES}
        with (
            patch.object(full.logger, "exception"),
            self.assertRaisesRegex(full.ProcessingLimitError, "before finishing"),
        ):
            full.summarize_article(
                " ".join(f"word{i}" for i in range(80)),
                config=replace(self.config, max_output_tokens=caps),
            )


if __name__ == "__main__":
    unittest.main()
