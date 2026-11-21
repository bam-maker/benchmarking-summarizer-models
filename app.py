"""Single-process, local Gradio demo. Launch with the project venv's Python."""

import argparse
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Keep generated runtime files local and ignored; do not enable public sharing.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".cache" / "gradio"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import gradio as gr

from article_summarizer import LENGTH_CHOICES, summarize_article
from summarizer import MODEL_CHOICES, InputError

logger = logging.getLogger(__name__)
PROGRESS = gr.Progress()


def selected_length(position):
    value = float(position)
    if not value.is_integer() or not 0 <= value <= 2:
        raise InputError("Choose Short, Medium or Detailed.")
    return LENGTH_CHOICES[int(value)]


def generate_summary(article, model_name, length_position=1, progress=PROGRESS):
    """One visible output; diagnostics stay in logs and warnings are concise."""

    def report(completed, total, phase):
        progress((completed, total), desc=f"{phase}: {completed}/{total}")

    try:
        result = summarize_article(
            article,
            model_name,
            selected_length(length_position),
            progress_callback=report,
        )
    except InputError as error:
        logger.warning("Article request not completed: %s", error)
        gr.Warning(str(error))
        return ""
    except Exception:
        logger.exception("Full-article summarization failed")
        gr.Warning(
            "The article could not be processed. No final summary was returned; see the local log for details."
        )
        return ""
    if result.warnings:
        gr.Warning(" ".join(result.warnings[:2]))
    return result.summary


CSS = """
.gradio-container { max-width: 940px !important; margin: auto; padding: 24px 18px !important; }
footer { display: none !important; }
#model-selector .wrap { display: flex; flex-wrap: nowrap; gap: 0; }
#model-selector .wrap > label { flex: 1; justify-content: center; text-align: center;
  border-radius: 0; margin: 0; min-width: 0; padding: 12px 8px; }
#model-selector .wrap > label:first-child { border-radius: 8px 0 0 8px; }
#model-selector .wrap > label:last-child { border-radius: 0 8px 8px 0; }
#model-selector label:has(input:checked) { background: #2563eb; color: white; border-color: #2563eb; }
#model-selector label:focus-within { outline: 2px solid #2563eb; outline-offset: 2px; }
#model-selector input[type=radio] { position: absolute; opacity: 0; width: 1px; height: 1px; }
#summary-length input[type=number] { display: none; }
#summary-length .min_value, #summary-length .max_value { display: none; }
#summary-length .info { text-align: center; }
@media (max-width: 520px) {
  .gradio-container { padding: 14px 10px !important; }
  #model-selector .wrap > label { padding: 12px 4px; font-size: 12px; }
}
"""


def build_demo():
    with gr.Blocks(
        title="Article summarizer", analytics_enabled=False, css=CSS
    ) as demo:
        article = gr.Textbox(
            label="Article", lines=14, placeholder="Paste an article..."
        )
        model = gr.Radio(
            list(MODEL_CHOICES), value="LSTM", label="Model", elem_id="model-selector"
        )
        length = gr.Slider(
            0,
            2,
            value=1,
            step=1,
            precision=0,
            label="Summary length: Medium",
            info="Short · Medium · Detailed",
            show_reset_button=False,
            elem_id="summary-length",
        )
        length.change(
            lambda value: gr.update(label=f"Summary length: {selected_length(value)}"),
            inputs=length,
            outputs=length,
            queue=False,
            api_name=False,
        )
        submit = gr.Button("Summarize", variant="primary")
        summary = gr.Textbox(label="Summary", lines=12, interactive=False)
        submit.click(
            generate_summary,
            inputs=[article, model, length],
            outputs=summary,
            api_name="summarize",
            concurrency_limit=1,
        )
    return demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processing-mode", choices=["full", "prefix"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.processing_mode:
        os.environ["SUMMARIZER_PROCESSING_MODE"] = args.processing_mode
    if args.config:
        os.environ["SUMMARIZER_CONFIG"] = str(args.config)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    build_demo().launch(
        server_name="127.0.0.1", server_port=args.port, share=False, show_api=False
    )
