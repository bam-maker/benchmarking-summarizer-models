"""Simple demo wiring and error clearing; no real weights required."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app
from article_summarizer import ProcessingLimitError


class UITests(unittest.TestCase):
    def test_every_model_and_slider_position_routes_correctly(self):
        for name in app.MODEL_CHOICES:
            for position, label in enumerate(app.LENGTH_CHOICES):
                with (
                    self.subTest(model=name, length=label),
                    patch.object(
                        app,
                        "summarize_article",
                        return_value=SimpleNamespace(
                            summary="Actual summary.", warnings=()
                        ),
                    ) as summarize,
                ):
                    self.assertEqual(
                        app.generate_summary("Article text.", name, position),
                        "Actual summary.",
                    )
                    self.assertEqual(
                        summarize.call_args.args, ("Article text.", name, label)
                    )
                    self.assertTrue(
                        callable(summarize.call_args.kwargs["progress_callback"])
                    )

    def test_processing_limit_clears_summary_and_displays_concise_notice(self):
        with (
            patch.object(
                app,
                "summarize_article",
                side_effect=ProcessingLimitError(
                    "Processing limit; no final summary returned."
                ),
            ),
            patch.object(app.gr, "Warning") as notice,
        ):
            self.assertEqual(app.generate_summary("Article", "BART", 1), "")
            notice.assert_called_once()

    def test_unexpected_error_does_not_expose_partial_or_technical_result(self):
        with (
            patch.object(
                app,
                "summarize_article",
                side_effect=RuntimeError("private diagnostics"),
            ),
            patch.object(app.gr, "Warning") as notice,
            patch.object(app.logger, "exception"),
        ):
            self.assertEqual(app.generate_summary("Article", "BART", 1), "")
            self.assertNotIn("private", notice.call_args.args[0])

    def test_only_five_main_elements_and_one_summary_output(self):
        demo = app.build_demo()
        components = demo.config["components"]
        controls = [
            c
            for c in components
            if c["type"]
            in {
                "textbox",
                "radio",
                "slider",
                "button",
                "checkbox",
                "number",
                "markdown",
                "tabs",
            }
        ]
        self.assertEqual(
            [c["type"] for c in controls],
            ["textbox", "radio", "slider", "button", "textbox"],
        )
        slider = next(c for c in controls if c["type"] == "slider")["props"]
        self.assertEqual(
            (slider["minimum"], slider["maximum"], slider["step"], slider["value"]),
            (0, 2, 1, 1),
        )
        self.assertIn("Medium", slider["label"])
        self.assertFalse(controls[-1]["props"]["interactive"])
        event = next(
            d for d in demo.config["dependencies"] if d["api_name"] == "summarize"
        )
        self.assertEqual((len(event["inputs"]), len(event["outputs"])), (3, 1))


if __name__ == "__main__":
    unittest.main()
