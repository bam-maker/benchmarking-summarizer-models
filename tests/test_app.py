"""Focused checks for routing, budgets, input handling, and model reuse."""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import app
import summarizer as engine


class InputAndOutputTests(unittest.TestCase):
    def test_empty_input(self):
        for text in ("", " \n\t"):
            with self.assertRaises(engine.InputError):
                engine.prepare_article(text)

    def test_multiline_preprocessing_keeps_both_sentences(self):
        article = engine.prepare_article(
            "The council approved buses.\nThey'll arrive (soon)."
        )
        self.assertEqual(
            article.tokens,
            (
                "the",
                "council",
                "approved",
                "buses",
                ".",
                "they",
                "'ll",
                "arrive",
                "-lrb-",
                "soon",
                "-rrb-",
                ".",
            ),
        )

    def test_long_article_requires_permission_and_counts_exclusions(self):
        text = "word " * 401
        with self.assertRaisesRegex(engine.InputError, "401.*400"):
            engine.prepare_article(text)
        approved = engine.prepare_article(text, True)
        self.assertEqual(approved.total_tokens, 401)
        self.assertEqual(len(approved.tokens), 400)
        self.assertTrue(approved.truncated)
        self.assertFalse(engine.prepare_article("word " * 400).truncated)

    def test_legacy_feature_separator_is_not_silently_dropped(self):
        with self.assertRaises(engine.InputError):
            engine.prepare_article("word\uffe8feature")

    def test_detokenization(self):
        self.assertEqual(
            engine.clean_summary(
                [
                    "<t>",
                    "they",
                    "ca",
                    "n't",
                    "wait",
                    ".",
                    "</t>",
                    "<t>",
                    "-lrb-",
                    "really",
                    "-rrb-",
                    ".",
                    "</t>",
                ]
            ),
            "they can't wait. (really).",
        )

    def test_real_eos_not_punctuation_determines_limit_notice(self):
        budget = engine.LENGTH_PRESETS["Short"]
        self.assertTrue(
            any(
                "OUTPUT LIMIT" in n
                for n in engine.output_notices("It ended.", False, budget)
            )
        )
        self.assertFalse(
            any(
                "OUTPUT LIMIT" in n
                for n in engine.output_notices("It ended.", True, budget)
            )
        )
        self.assertTrue(
            any(
                "POSSIBLY UNFINISHED" in n
                for n in engine.output_notices("It did not", True, budget)
            )
        )


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"SUMMARIZER_DEVICE": "cpu"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_fixed_budget_benchmark_api_still_reaches_the_decoder(self):
        for name in engine.MODEL_CHOICES:
            for length, budget in engine.LENGTH_PRESETS.items():
                with self.subTest(model=name, length=length):
                    model = SimpleNamespace(
                        prepare=engine.prepare_article,
                        generate=lambda article, actual: (
                            "a real sentence.",
                            5,
                            True,
                            0.5,
                        ),
                    )
                    with (
                        patch.object(engine, "get_model", return_value=model) as loader,
                        patch.object(
                            model, "generate", wraps=model.generate
                        ) as decoder,
                    ):
                        result = engine.summarize("Article text.", name, length)
                    loader.assert_called_once_with(name, "cpu")
                    self.assertEqual(decoder.call_args.args[1], budget)
                    self.assertEqual(
                        (result.summary, result.word_count, result.seconds),
                        ("a real sentence.", 3, 0.5),
                    )
                    self.assertIn(name, result.notices)
                    self.assertIn(length, result.notices)
                    self.assertGreaterEqual(result.load_seconds, 0)

    def test_invalid_and_unapproved_input_never_loads_a_model(self):
        with patch.object(engine, "get_model") as loader:
            with self.assertRaisesRegex(engine.InputError, "No text was summarized"):
                engine.summarize("word " * 401, "LSTM", "Short", False)
            loader.assert_not_called()

    def test_approved_truncation_is_visible(self):
        model = SimpleNamespace(
            generate=lambda article, budget: ("a summary.", 4, True, 0.1)
        )
        with patch.object(engine, "get_model", return_value=model):
            result = engine.summarize("word " * 405, "Transformer", "Detailed", True)
        self.assertIn("INPUT TRUNCATED WITH YOUR PERMISSION", result.notices)
        self.assertIn("5 trailing tokens", result.notices)

    def test_model_factory_loads_each_model_once_even_with_concurrent_calls(self):
        with (
            patch.object(engine, "_models", {}),
            patch.object(engine, "OpenNMTSummarizer") as factory,
        ):
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(engine.get_model, ["LSTM"] * 8))
            factory.assert_called_once_with("LSTM")
            self.assertTrue(all(model is results[0] for model in results))

    def test_gradio_button_wires_the_expected_inputs_and_handler(self):
        demo = app.build_demo()
        dependencies = [
            d for d in demo.config["dependencies"] if d["api_name"] == "summarize"
        ]
        self.assertEqual(len(dependencies), 1)
        dependency = dependencies[0]
        self.assertEqual(len(dependency["inputs"]), 3)
        self.assertEqual(len(dependency["outputs"]), 4)
        self.assertTrue(any(fn.fn is app.generate_summary for fn in demo.fns.values()))


if __name__ == "__main__":
    unittest.main()
