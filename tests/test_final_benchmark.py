"""Metric/accounting tests; no final predictions used for parameter selection."""
import unittest
import final_benchmark as bench


class FinalProtocolTests(unittest.TestCase):
    def test_failed_empty_scores_zero(self):
        self.assertEqual(bench.score("", "The council approved new buses."),
                         dict.fromkeys(bench.METRICS, 0.0))

    def test_identical_scores_and_sentence_segmentation(self):
        text = "The council approved buses. They arrive on Friday."
        self.assertEqual(bench.score(text, text), dict.fromkeys(bench.METRICS, 100.0))
        self.assertEqual(bench.segmented(text), "The council approved buses.\nThey arrive on Friday.")

    def test_trigram_rate(self):
        self.assertEqual(bench.repetition("a b c a b c"), .25)
        self.assertEqual(bench.repetition("short input"), 0)

    def test_common_prefix_keeps_original_text(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [0] * (len(text.split()) + 2)
        text = "Opening fact. " + "Another sentence. " * 300
        prefix, counts = bench.common_prefix(text, {"BART": Tokenizer(), "PEGASUS": Tokenizer()})
        self.assertTrue(text.startswith(prefix))
        self.assertTrue(prefix.startswith("Opening fact."))
        self.assertLessEqual(counts["LSTM"], 400)
        self.assertLess(len(prefix), len(text))


if __name__ == "__main__":
    unittest.main()
