"""UI-only selection, measurement and state checks; no pretrained weights loaded."""

import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app
from article_summarizer import ProcessingLimitError


def result(**overrides):
    values = {
        "summary": "A useful short summary.",
        "warnings": (),
        "coverage_complete": True,
        "processing_mode": "full",
        "model_calls": 1,
        "consolidation_passes": 0,
        "total_generated_tokens": 20,
        "generation_seconds": 0.5,
        "load_seconds": 9.0,
        "returned_input": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def values(markup):
    return re.findall(r"<dd>(.*?)</dd>", markup)


class UITests(unittest.TestCase):
    def test_every_model_routes_to_natural_policy(self):
        for name in app.MODEL_CHOICES:
            with (
                self.subTest(model=name),
                patch.object(
                    app, "summarize_article", return_value=result()
                ) as summarize,
            ):
                outputs = app.generate_summary("Article text.", name)
                self.assertEqual(outputs[0], "A useful short summary.")
                self.assertEqual(summarize.call_args.args, ("Article text.", name))
                self.assertEqual(
                    summarize.call_args.kwargs["generation_policy"], app.NATURAL_POLICY
                )
                self.assertTrue(
                    callable(summarize.call_args.kwargs["progress_callback"])
                )

    def test_statistics_use_actual_tokens_not_words_and_exclude_loading(self):
        markup = app.render_statistics(result(), "word " * 20, 11)
        self.assertEqual(values(markup), ["4", "80.0%", "11.00 s", "40.0"])
        self.assertIn("Generation tok/s", markup)
        self.assertNotIn("Pipeline generation tok/s", markup)
        self.assertIn("Excludes loading", markup)
        self.assertIn("Model tokenizers differ", markup)

    def test_pipeline_rate_uses_all_decode_tokens_over_all_generation_time(self):
        markup = app.render_statistics(
            result(
                total_generated_tokens=180,
                generation_seconds=3,
                model_calls=5,
                consolidation_passes=2,
            ),
            "word " * 100,
            10,
        )
        self.assertEqual(values(markup), ["4", "96.0%", "10.00 s", "60.0"])
        self.assertIn("Pipeline generation tok/s", markup)
        self.assertIn("chunks, consolidation and completion retries", markup)

    def test_missing_zero_or_failed_measurements_are_not_fabricated(self):
        for changes in (
            {"total_generated_tokens": None},
            {"total_generated_tokens": 0},
            {"generation_seconds": 0},
            {"generation_seconds": -1},
            {"generation_seconds": float("nan")},
            {"generation_seconds": float("inf")},
            {"returned_input": True},
        ):
            with self.subTest(changes=changes):
                self.assertEqual(
                    values(app.render_statistics(result(**changes), "", None)),
                    ["4", "\u2014", "\u2014", "\u2014"],
                )
        self.assertEqual(values(app.render_statistics()), ["\u2014"] * 4)
        self.assertIn(
            "&lt;0.01 s", app.render_statistics(result(), "word " * 20, 0.001)
        )
        self.assertEqual(
            values(app.render_statistics(result(), "one", 0))[1], "-300.0%"
        )

    def test_request_processing_time_is_independent_of_generation_time(self):
        with (
            patch.object(app, "summarize_article", return_value=result()),
            patch.object(app.time, "perf_counter", side_effect=[100, 112]),
        ):
            outputs = app.generate_summary("word " * 20, "LSTM")
        self.assertEqual(values(outputs[1])[2:], ["12.00 s", "40.0"])

    def test_processing_limit_clears_text_and_statistics_and_disables_copy(self):
        with patch.object(
            app,
            "summarize_article",
            side_effect=ProcessingLimitError("Processing limit."),
        ):
            outputs = app.generate_summary("Article", "BART")
        self.assertEqual(outputs[0], "")
        self.assertEqual(values(outputs[1]), ["\u2014"] * 4)
        self.assertIn('role="alert"', outputs[2])
        self.assertFalse(outputs[3]["interactive"])

    def test_unexpected_error_and_incomplete_results_never_expose_partial_output(self):
        for failure in (RuntimeError("private diagnostics"), None):
            with (
                self.subTest(failure=failure),
                patch.object(
                    app,
                    "summarize_article",
                    side_effect=failure,
                    return_value=result(coverage_complete=False),
                ),
                patch.object(app.logger, "exception"),
            ):
                outputs = app.generate_summary("Article", "BART")
                self.assertEqual(outputs[0], "")
                self.assertNotIn("private", outputs[2])
                self.assertFalse(outputs[3]["interactive"])
        with patch.object(app, "summarize_article", return_value=result(summary=" ")):
            self.assertEqual(app.generate_summary("Article", "LSTM", "Short")[0], "")

    def test_notices_are_escaped_and_prefix_remains_explicit(self):
        with patch.object(
            app,
            "summarize_article",
            return_value=result(
                warnings=("<script>not markup</script>",),
                processing_mode="prefix",
                coverage_complete=False,
            ),
        ):
            outputs = app.generate_summary("Article", "BART")
        self.assertIn("&lt;script&gt;", outputs[2])
        self.assertNotIn("<script>", outputs[2])

    def test_start_finish_clear_and_input_change_reset_all_result_state(self):
        started = app.begin_request()
        self.assertEqual(started[0], "")
        self.assertEqual(values(started[1]), ["\u2014"] * 4)
        self.assertFalse(started[3]["interactive"])
        self.assertTrue(all(not update["interactive"] for update in started[4:8]))
        self.assertTrue(all(update["interactive"] for update in app.finish_request()))
        cleared = app.clear_form()
        self.assertEqual(cleared[0], "")
        self.assertIn("0 words", cleared[1])
        self.assertEqual(cleared[2], "")
        self.assertEqual(cleared[4], "")
        self.assertFalse(cleared[5]["interactive"])
        self.assertEqual(cleared[6], "")
        self.assertIn("3 words", app.article_changed("one\n two  three")[0])
        self.assertIn("1 word<", app.input_words("one"))
        self.assertEqual(app.invalidate_result()[0], "")

    def test_native_dropdowns_actions_and_request_chain(self):
        demo = app.build_demo()
        try:
            components = demo.config["components"]
            dropdowns = [c["props"] for c in components if c["type"] == "dropdown"]
            self.assertEqual([d["value"] for d in dropdowns], ["LSTM"])
            self.assertEqual(
                [v for _, v in dropdowns[0]["choices"]], list(app.MODEL_CHOICES)
            )
            self.assertTrue(
                all(
                    not d["filterable"] and not d["allow_custom_value"]
                    for d in dropdowns
                )
            )
            self.assertFalse(
                any(c["type"] in {"slider", "radio", "tabs"} for c in components)
            )
            buttons = {
                c["props"]["value"]: c["props"]
                for c in components
                if c["type"] == "button"
            }
            self.assertEqual(set(buttons), {"Summarize", "Clear", "Copy summary"})
            self.assertFalse(buttons["Copy summary"]["interactive"])
            summary = next(
                c["props"]
                for c in components
                if c["props"].get("elem_id") == "summary-output"
            )
            self.assertFalse(summary["interactive"])
            deps = demo.config["dependencies"]
            request = next(d for d in deps if d["api_name"] == "summarize")
            self.assertEqual((len(request["inputs"]), len(request["outputs"])), (2, 4))
            self.assertEqual(request["trigger_mode"], "once")
            self.assertIsNotNone(request["trigger_after"])
            copy = next(d for d in deps if d.get("js") == app.COPY_JS)
            self.assertFalse(copy["backend_fn"])
            self.assertIn("navigator.clipboard.writeText(summary)", copy["js"])
            self.assertTrue(any(fn.fn is app.clear_form for fn in demo.fns.values()))
            self.assertTrue(
                any(fn.fn is app.finish_request for fn in demo.fns.values())
            )
        finally:
            demo.close()


if __name__ == "__main__":
    unittest.main()
