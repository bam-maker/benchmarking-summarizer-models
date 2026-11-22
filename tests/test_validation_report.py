"""Audit the saved validation report without loading models or rerunning inference."""

import csv
import hashlib
import json
import statistics
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    (ROOT / "reports/validation_40_results.json").exists(), "Saved report unavailable"
)
class SavedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(
            (ROOT / "reports/validation_40_results.json").read_text(encoding="utf-8")
        )
        cls.manifest = json.loads(
            (ROOT / "reports/validation_40_manifest.json").read_text(encoding="utf-8")
        )

    def test_frozen_generation_files_and_same_validation_ids(self):
        for name, digest in self.manifest["generation_code_sha256"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest
            )
        self.assertEqual(self.manifest["split"], "validation")
        expected = [a["id"] for a in self.manifest["articles"]]
        self.assertEqual(len(set(expected)), 40)
        for name in self.results["models"]:
            cases = [c for c in self.results["cases"] if c["model"] == name]
            self.assertEqual([c["article_id"] for c in cases], expected)
            self.assertTrue(all(c["pipeline"]["device"] == "cuda" for c in cases))

    def test_aggregate_matches_saved_per_case_scores(self):
        for name, stats in self.results["models"].items():
            cases = [c for c in self.results["cases"] if c["model"] == name]
            for metric, mean in stats["rouge_f1_mean_all_articles_percent"].items():
                self.assertAlmostEqual(
                    mean, statistics.mean(c["rouge_f1"][metric] * 100 for c in cases)
                )
            self.assertAlmostEqual(
                stats["generation_success_mean_seconds"],
                statistics.mean(c["generation_seconds"] for c in cases),
            )

    def test_coverage_eos_and_actual_settings_are_recorded(self):
        self.assertEqual(len(self.results["cases"]), 160)
        for case in self.results["cases"]:
            self.assertEqual(case["status"], "completed")
            self.assertEqual(case["input_presented_fraction"], 1)
            self.assertTrue(case["completed_full_article_summary"])
            self.assertTrue(case["final_attempt_eos"])
            self.assertEqual(case["pipeline"]["consolidation_passes"], 0)
            self.assertEqual(len(case["calls"]), case["planned_chunks"])
            for call in case["calls"]:
                self.assertFalse(call["truncated"])
                self.assertLessEqual(call["input_tokens"], case["input_capacity"])
                self.assertIn("min_tokens", call["budget"])
                self.assertIn("max_tokens", call["budget"])
                self.assertTrue(call["model_generation_kwargs"])

    def test_ai_notes_are_separate_from_uncollected_human_ratings(self):
        with (ROOT / "reports/validation_40_review.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 160)
        for row in rows:
            self.assertEqual(row["human_review_status"], "not_reviewed")
            self.assertTrue(row["ai_reading_note"])
            for key, value in row.items():
                if key.startswith("human_") and key != "human_review_status":
                    self.assertEqual(value, "")
        self.assertEqual(self.results["ai_review"]["human_ratings"], 0)
        self.assertEqual(self.results["ai_review"]["notes"], 160)

    def test_template_sensitivity_preserves_primary_sample(self):
        self.assertEqual(
            [s["row_idx"] for s in self.results["reference_audit"]["template_rows"]],
            [10, 14],
        )
        self.assertTrue(
            all(m["articles"] == 40 for m in self.results["models"].values())
        )


class MetricDefinitionTests(unittest.TestCase):
    def test_rouge_identity_empty_and_stemming(self):
        sys.path.insert(0, str(ROOT / ".cache/eval-metrics"))
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            self.skipTest("Optional isolated evaluation metric package unavailable")
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.assertTrue(
            all(
                s.fmeasure == 1
                for s in scorer.score("A cat sat.", "A cat sat.").values()
            )
        )
        self.assertTrue(
            all(s.fmeasure == 0 for s in scorer.score("A cat sat.", "").values())
        )
        self.assertEqual(scorer.score("cats", "cat")["rouge1"].fmeasure, 1)
        self.assertAlmostEqual(
            scorer.score("alpha beta gamma", "alpha gamma beta")["rougeL"].fmeasure,
            2 / 3,
        )


if __name__ == "__main__":
    unittest.main()
