"""Check experimental controls and saved evidence, without neural output assertions."""

import unittest

from audit_factual_failures import MODELS, ROOT, ROWS, WORK, read, verify


@unittest.skipUnless(
    (ROOT / "reports/factual_audit_results.json").exists()
    and all(
        (WORK / "controls" / f"{row:03d}_{name}.json").exists()
        for row in ROWS
        for name in MODELS
    ),
    "Saved causal audit or ignored local raw traces unavailable",
)
class FactualAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [
            read(WORK / "controls" / f"{row:03d}_{name}.json")
            for row in ROWS
            for name in MODELS
        ]

    def test_production_and_original_validation_evidence_unchanged(self):
        manifest = verify()
        self.assertIn("app.py", manifest["preserved_sha256"])
        self.assertIn(
            "reports/validation_40_results.json", manifest["preserved_sha256"]
        )
        self.assertIn("reports/validation_40_review.csv", manifest["preserved_sha256"])

    def test_direct_replay_matches_raw_inputs_and_outputs(self):
        self.assertEqual(len(self.rows), 20)
        for row in self.rows:
            self.assertEqual(row["device"], "cuda")
            self.assertTrue(row["direct_app_merge_equal_saved_application"])
            self.assertEqual(
                "".join(c["source_text"] for c in row["chunk_controls"]),
                row["source"]["article"],
            )
            for chunk in row["chunk_controls"]:
                self.assertTrue(chunk["input_ids_equal_application"])
                self.assertTrue(chunk["direct_same_settings_raw_ids_equal_application"])
                self.assertTrue(chunk["direct_same_settings_display_equal_application"])

    def test_each_decoding_edge_changes_only_one_factor(self):
        for row in self.rows:
            for chunk in row["chunk_controls"]:
                variants = chunk["variants"]
                if row["model"] in {"BART", "PEGASUS"}:
                    edges = [
                        (
                            "checkpoint_defaults",
                            "eos_unforced_only",
                            "forced_eos_token_id",
                        ),
                        ("eos_unforced_only", "adaptive_max_only", "max_new_tokens"),
                        ("adaptive_max_only", "direct_app_settings", "min_new_tokens"),
                        ("eos_unforced_only", "adaptive_min_only", "min_new_tokens"),
                    ]
                else:
                    edges = [
                        (
                            "recipe_current_preprocessing",
                            "adaptive_max_only",
                            "max_length",
                        ),
                        ("adaptive_max_only", "direct_app_settings", "min_length"),
                        (
                            "recipe_current_preprocessing",
                            "adaptive_min_only",
                            "min_length",
                        ),
                    ]
                for left, right, allowed in edges:
                    a, b = variants[left], variants[right]
                    self.assertEqual(a["source_tensor_ids"], b["source_tensor_ids"])
                    before, after = a["effective_settings"], b["effective_settings"]
                    changed = {
                        key
                        for key in before.keys() | after.keys()
                        if before.get(key) != after.get(key)
                    }
                    self.assertLessEqual(changed, {allowed})

    def test_preprocessing_swap_keeps_source_and_decoding_fixed(self):
        for row in self.rows:
            if row["model"] not in {"LSTM", "Transformer"}:
                continue
            for chunk in row["chunk_controls"]:
                a = chunk["variants"]["recipe_current_preprocessing"]
                b = chunk["variants"]["recipe_stanford_preprocessing"]
                self.assertEqual(a["effective_settings"], b["effective_settings"])
                self.assertEqual(
                    a["prepared_tokens"], chunk["app_approximation_tokens"]
                )
                self.assertEqual(b["prepared_tokens"], chunk["stanford_tokens"])

    def test_window_comparisons_share_token_prefix_and_label_missing_content(self):
        for row in self.rows:
            for preprocessing in {
                p["preprocessing"] for p in row["prefix_diagnostics"]
            }:
                pair = {
                    p["window"]: p
                    for p in row["prefix_diagnostics"]
                    if p["preprocessing"] == preprocessing
                }
                self.assertEqual(
                    pair[400]["prepared_tokens"], pair[1024]["prepared_tokens"][:400]
                )
                self.assertEqual(
                    pair[400]["effective_settings"], pair[1024]["effective_settings"]
                )
                for window, probe in pair.items():
                    self.assertEqual(
                        probe["omitted_tokens"],
                        max(0, probe["complete_original_tokens"] - window),
                    )
                    self.assertEqual(
                        probe["is_full_article"], probe["omitted_tokens"] == 0
                    )

    def test_template_rows_retained_and_no_human_ratings_fabricated(self):
        sources = [
            read(WORK / "sources" / f"{row:03d}.json")["source"] for row in (10, 14)
        ]
        self.assertNotEqual(sources[0]["article"], sources[1]["article"])
        self.assertEqual(sources[0]["reference"], sources[1]["reference"])
        result = read(ROOT / "reports/factual_audit_results.json")
        self.assertFalse(result["human_ratings_collected"])
        self.assertIn("ai_assisted_source_comparisons", result)
        original = read(ROOT / "reports/validation_40_results.json")
        self.assertTrue(all(m["articles"] == 40 for m in original["models"].values()))
        self.assertEqual(
            result["template_reference_audit"]["preserved_posthoc_38_non_template"],
            original["reference_audit"]["non_template_rouge_f1_mean_percent"],
        )


if __name__ == "__main__":
    unittest.main()
