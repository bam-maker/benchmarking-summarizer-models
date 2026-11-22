"""Regression checks for demonstrated decoding-policy and overcompression bugs."""

import sys
import threading
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

import article_summarizer as full
import summarizer as engine
from hf_summarizer import HFSummarizer


class QualityPolicyTests(unittest.TestCase):
    def test_hf_profile_preserves_checkpoint_search_and_does_not_mutate_it(self):
        backend = HFSummarizer.__new__(HFSummarizer)
        backend.device = "cpu"
        backend.lock = threading.Lock()
        backend.tokenizer = SimpleNamespace(
            eos_token_id=1, decode=Mock(return_value="A coherent summary.")
        )
        config = SimpleNamespace(
            num_beams=8,
            no_repeat_ngram_size=0,
            early_stopping=False,
            min_length=32,
            max_length=128,
        )
        backend.model = SimpleNamespace(
            generation_config=config,
            generate=Mock(return_value=torch.tensor([[0, 8, 9, 1]])),
        )
        source = engine.PreparedArticle((5, 6, 7), 3)
        _, count, ended, _ = backend.generate(
            source, engine.Budget(10, 127, "checkpoint")
        )
        kwargs = backend.model.generate.call_args.kwargs
        self.assertFalse(kwargs["do_sample"])
        self.assertIsNone(kwargs["forced_eos_token_id"])
        for setting in (
            "num_beams",
            "no_repeat_ngram_size",
            "early_stopping",
            "length_penalty",
        ):
            self.assertNotIn(setting, kwargs)
        self.assertEqual((count, ended), (2, True))
        self.assertEqual(
            (config.num_beams, config.no_repeat_ngram_size, config.early_stopping),
            (8, 0, False),
        )
        backend.generate(source, engine.Budget(10, 65))
        self.assertEqual(backend.model.generate.call_args.kwargs["num_beams"], 4)
        self.assertEqual(backend.summary_max_length, 127)

    def test_native_maximum_is_room_for_eos_not_a_forced_minimum(self):
        backend = SimpleNamespace(
            model_name="BART",
            generation_limit=1023,
            summary_min_length=56,
            summary_max_length=141,
        )
        text = "word " * 20
        chunk = full.Chunk(
            0, len(text), text, engine.PreparedArticle(tuple(range(24)), 24)
        )
        budget, _ = full.token_budget(backend, chunk, 5, full.ProcessingConfig())
        self.assertGreaterEqual(budget.max_tokens, 141)
        self.assertLessEqual(budget.min_tokens, 6)

    def test_opennmt_profile_restores_cached_state_even_on_failure(self):
        module = ModuleType("onmt.translate.beam")
        scorer = object()
        module.GNMTGlobalScorer = Mock(return_value=scorer)
        backend = engine.OpenNMTSummarizer.__new__(engine.OpenNMTSummarizer)
        original_scorer = object()
        backend.translator = SimpleNamespace(
            beam_size=4,
            stepwise_penalty=False,
            global_scorer=original_scorer,
            min_length=30,
            max_length=65,
        )
        modules = {
            "onmt": ModuleType("onmt"),
            "onmt.translate": ModuleType("onmt.translate"),
            "onmt.translate.beam": module,
        }
        with (
            patch.dict(sys.modules, modules),
            self.assertRaisesRegex(RuntimeError, "test failure"),
            backend.decoding_profile(engine.Budget(35, 100, "checkpoint")),
        ):
            self.assertEqual(backend.translator.beam_size, 5)
            self.assertTrue(backend.translator.stepwise_penalty)
            self.assertIs(backend.translator.global_scorer, scorer)
            raise RuntimeError("test failure")
        self.assertEqual(backend.translator.beam_size, 4)
        self.assertFalse(backend.translator.stepwise_penalty)
        self.assertIs(backend.translator.global_scorer, original_scorer)
        self.assertEqual(
            (backend.translator.min_length, backend.translator.max_length), (30, 65)
        )
        module.GNMTGlobalScorer.assert_called_once_with(0.9, 5.0, "summary", "wu")

    def test_hf_preprocessing_preserves_case_and_source_specials(self):
        backend = HFSummarizer.__new__(HFSummarizer)
        backend.model_name = "BART"
        backend.source_limit = 1024
        backend.tokenizer = SimpleNamespace(encode=Mock(return_value=[0, 12, 13, 2]))
        article = "Nvidia said Jensen Huang may attend."
        source = backend.prepare(article)
        backend.tokenizer.encode.assert_called_once_with(
            article, add_special_tokens=True, truncation=False
        )
        self.assertEqual(source.tokens, (0, 12, 13, 2))
        self.assertFalse(source.truncated)

    def test_legacy_training_window_is_not_a_hard_encoder_capacity(self):
        backend = engine.OpenNMTSummarizer.__new__(engine.OpenNMTSummarizer)
        backend.source_limit = 400
        encoder = SimpleNamespace(embeddings=SimpleNamespace(named_buffers=list))
        backend.translator = SimpleNamespace(model=SimpleNamespace(encoder=encoder))
        self.assertEqual(full.source_capacity(backend, full.ProcessingConfig()), 1024)
        backend.source_limit = 1024
        prepared = backend.prepare("word " * 580)
        self.assertEqual(len(prepared.tokens), 580)
        self.assertFalse(prepared.truncated)
        # The fixed-budget public preprocessing API remains at its old policy.
        with self.assertRaises(engine.InputError):
            engine.prepare_article("word " * 580)
        encoder.embeddings.named_buffers = lambda: [("pe", torch.zeros(512, 1, 1))]
        self.assertEqual(full.source_capacity(backend, full.ProcessingConfig()), 512)

    def test_extended_app_window_restores_cached_source_policy_on_error(self):
        backend = SimpleNamespace(
            model_name="LSTM", source_limit=400, extended_source_limit=5000
        )
        backend.prepare = lambda text, allow_truncation=False: engine.prepare_article(
            text, allow_truncation, backend.source_limit
        )
        backend.generate = Mock(
            return_value=("A coherent model summary.", 5, True, 0.1)
        )
        with (
            patch.object(engine, "get_model", return_value=backend),
            patch.object(full, "resolve_device", return_value="cpu"),
            patch.object(full, "synchronize"),
        ):
            result = full.summarize_article("word " * 580)
            self.assertEqual(result.source_chunks, 1)
            self.assertEqual(result.model_calls, 1)
            self.assertTrue(result.coverage_complete)
            self.assertEqual(backend.source_limit, 400)
            backend.generate.side_effect = RuntimeError("decode failed")
            with (
                patch.object(full.logger, "exception"),
                self.assertRaises(RuntimeError),
            ):
                full.summarize_article("word " * 580)
            self.assertEqual(backend.source_limit, 400)


if __name__ == "__main__":
    unittest.main()
