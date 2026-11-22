import unittest
from dataclasses import replace
from unittest.mock import patch

from test_article_summarizer import Backend

import article_summarizer as full


class NaturalTests(unittest.TestCase):
    def run_case(self, text, backend=None, config=None):
        backend = backend or Backend()
        with (
            patch.object(full.engine, "get_model", return_value=backend),
            patch.object(full, "resolve_device", return_value="cpu"),
            patch.object(
                full,
                "requested_words",
                side_effect=AssertionError("percentage target invoked"),
            ),
            patch.object(
                full, "token_budget", side_effect=AssertionError("word budget invoked")
            ),
        ):
            return full.summarize_article(
                text,
                generation_policy=full.NATURAL_POLICY,
                config=config or full.ProcessingConfig(),
            )

    def test_short_uses_model_once_without_percentage_target(self):
        backend = Backend()
        r = self.run_case("The council approved new buses.", backend)
        self.assertEqual(r.model_calls, 1)
        self.assertFalse(r.returned_input)
        self.assertIsNone(r.requested_words)
        self.assertEqual(r.generation_policy, full.NATURAL_POLICY)
        self.assertEqual(backend.calls[0][1].min_tokens, 0)

    def test_multichunk_coverage_and_reduction(self):
        text = " ".join(
            f"District {i} approved new buses for local routes." for i in range(50)
        )
        r = self.run_case(text)
        self.assertTrue(r.coverage_complete)
        self.assertGreater(r.source_chunks, 1)
        self.assertTrue(all(s.output_tokens < s.input_tokens for s in r.stages))
        self.assertIsNone(r.requested_words)

    def test_work_limit_returns_no_partial(self):
        with self.assertRaises(full.ProcessingLimitError):
            self.run_case(
                "word " * 100,
                config=replace(full.ProcessingConfig(), max_model_calls=1),
            )

    def test_nonprogress_terminates(self):
        backend = Backend()
        backend.generate = lambda p, b: (" ".join(p.tokens), len(p.tokens), True, 0.1)
        with self.assertRaises(full.ProcessingLimitError):
            self.run_case("word " * 100, backend)

    def test_unfinished_at_hard_cap_is_rejected(self):
        backend = Backend()
        backend.generate = lambda p, b: ("Unfinished", b.max_tokens, False, 0.1)
        with self.assertRaises(full.ProcessingLimitError):
            self.run_case("A short article about the council and buses.", backend)

    def test_model_native_maximum_and_short_source_floor(self):
        backend = Backend()
        backend.summary_max_length = 141
        backend.summary_min_length = 56
        self.run_case("word " * 32, backend)
        self.assertEqual(backend.calls[0][1].max_tokens, 141)
        self.assertEqual(backend.calls[0][1].min_tokens, 4)


if __name__ == "__main__":
    unittest.main()
