"""Local Gradio UI. Model decoding and fixed-budget benchmarks are separate."""

import argparse
import html
import logging
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".cache" / "gradio"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import gradio as gr

from article_summarizer import NATURAL_POLICY, summarize_article, word_count
from summarizer import MODEL_CHOICES, InputError

logger = logging.getLogger(__name__)
PROGRESS = gr.Progress()


def status_html(message, error=False):
    role = "alert" if error else "status"
    return f'<p class="request-status {"error" if error else ""}" role="{role}">{html.escape(message)}</p>'


def input_words(article):
    count = word_count(article or "")
    return f'<span class="input-count" aria-live="polite">{count:,} {"word" if count == 1 else "words"}</span>'


def finite_number(value, positive=False):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def render_statistics(result=None, article="", processing_seconds=None):
    """Never estimate decoded token counts from words or final-summary tokens."""
    output_words = word_count(result.summary) if result else None
    original_words = word_count(article)
    compression = (
        100 * (1 - output_words / original_words)
        if result and original_words > 0
        else None
    )
    seconds = processing_seconds
    processing = (
        ("<0.01 s" if 0 < seconds < 0.01 else f"{seconds:.2f} s")
        if finite_number(seconds)
        else "\u2014"
    )
    calls = getattr(result, "model_calls", 0)
    pipeline = calls > 1 or getattr(result, "consolidation_passes", 0) > 0
    rate_label = "Pipeline generation tok/s" if pipeline else "Generation tok/s"
    tokens = getattr(result, "total_generated_tokens", None)
    generation = getattr(result, "generation_seconds", None)
    rate = (
        tokens / generation
        if result
        and not getattr(result, "returned_input", False)
        and isinstance(tokens, int)
        and not isinstance(tokens, bool)
        and tokens > 0
        and finite_number(generation, positive=True)
        else None
    )
    scope = (
        "All decoded output tokens across chunks, consolidation and completion retries"
        if pipeline
        else "Decoded output tokens from the single generation pass"
    )
    rate_help = (
        f"{scope}, divided by summed generation time. Excludes loading and start/EOS IDs. "
        "Model tokenizers differ; rates are not directly equivalent across models. "
        "Unavailable when measurement is incomplete or no generation occurred."
    )
    items = [
        (
            "Output words",
            f"{output_words:,}" if result else "\u2014",
            "Whitespace-separated words in this summary.",
        ),
        (
            "Shorter by",
            f"{compression:.1f}%" if compression is not None else "\u2014",
            "100 \u00d7 (1 \u2212 summary words / complete article words). Negative means the summary is longer.",
        ),
        (
            "Processing time",
            processing if result else "\u2014",
            "Server processing from request execution through the final result, including loading, preprocessing and consolidation. Excludes queue wait and browser/network time.",
        ),
        (rate_label, f"{rate:.1f}" if rate is not None else "\u2014", rate_help),
    ]
    cells = "".join(
        f'<div class="metric"><dt tabindex="0" title="{html.escape(tip, quote=True)}" '
        f'aria-label="{html.escape(label + ". " + tip, quote=True)}">{label}</dt>'
        f"<dd>{html.escape(value)}</dd></div>"
        for label, value, tip in items
    )
    return f'<dl class="statistics" aria-label="Summary statistics">{cells}</dl>'


def invalidate_result():
    return "", render_statistics(), "", gr.update(interactive=False), ""


def article_changed(article):
    return (input_words(article), *invalidate_result())


def clear_form():
    return ("", input_words(""), *invalidate_result())


def begin_request():
    controls = [gr.update(interactive=False) for _ in range(4)]
    controls[2] = gr.update(value="Summarizing\u2026", interactive=False)
    return (
        "",
        render_statistics(),
        status_html("Preparing your summary\u2026"),
        gr.update(interactive=False),
        *controls,
        "",
    )


def finish_request():
    controls = [gr.update(interactive=True) for _ in range(4)]
    controls[2] = gr.update(value="Summarize", interactive=True)
    return tuple(controls)


def generate_summary(article, model_name, progress=PROGRESS):
    """Only completed results get text, statistics and an enabled Copy action."""
    started = time.perf_counter()

    def report(completed, total, phase):
        progress((completed, total), desc=f"{phase}: {completed}/{total}")

    try:
        result = summarize_article(
            article,
            model_name,
            generation_policy=NATURAL_POLICY,
            progress_callback=report,
        )
        if getattr(result, "processing_mode", "full") != "prefix" and not getattr(
            result, "coverage_complete", False
        ):
            raise InputError(
                "Processing could not cover the complete article. No final summary was returned."
            )
        if not result.summary.strip():
            raise InputError("No summary was produced. Please try a different model.")
        processing_seconds = time.perf_counter() - started
        statistics = render_statistics(result, article, processing_seconds)
    except InputError as error:
        logger.warning("Article request not completed: %s", error)
        return (
            "",
            render_statistics(),
            status_html(str(error), error=True),
            gr.update(interactive=False),
        )
    except Exception:
        logger.exception("Full-article summarization failed")
        return (
            "",
            render_statistics(),
            status_html(
                "The article could not be processed. Your text is safe; please try again.",
                error=True,
            ),
            gr.update(interactive=False),
        )
    message = " ".join(result.warnings[:2]) if result.warnings else "Summary ready."
    logger.info(
        "UI request completed: model=%s policy=%s processing_seconds=%.4f",
        model_name,
        NATURAL_POLICY,
        processing_seconds,
    )
    return result.summary, statistics, status_html(message), gr.update(interactive=True)


COPY_JS = """
async (summary) => {
    if (!summary || !summary.trim()) return "";
    try {
        await navigator.clipboard.writeText(summary);
        return '<span role="status">Copied to clipboard.</span>';
    } catch (_) {
        return '<span role="alert">Copy unavailable. Select the summary and copy it manually.</span>';
    }
}
"""

# Gradio 5 opens a readonly dropdown on focus, so clicking again after Escape
# needs to re-enter its native keyboard handler while focus stays accessible.
DROPDOWN_JS = """
() => {
    document.addEventListener('pointerdown', (event) => {
        const input = event.target;
        if (input instanceof HTMLInputElement &&
            input.matches('#selectors input[readonly]') && !input.disabled &&
            document.activeElement === input &&
            input.getAttribute('aria-expanded') === 'false') {
            input.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'ArrowDown', code: 'ArrowDown', bubbles: true
            }));
        }
    }, true);
}
"""

CSS = """
.gradio-container {
  width: 100% !important; max-width: 1240px !important; margin: auto; padding: 40px 28px !important;
  color-scheme: light; font-family: "Segoe UI", system-ui, sans-serif !important;
  --body-background-fill: #f4f7f6; --body-text-color: #183330;
  --block-background-fill: #fff; --block-border-color: #dce5e2;
  --block-label-text-color: #405d56; --input-background-fill: #fff;
  --block-title-background-fill: transparent; --block-title-text-color: #405d56;
  --input-border-color: #d0ded9; --input-text-color: #183330;
  --input-placeholder-color: #647971;
  --button-secondary-background-fill: #fff; --button-secondary-text-color: #244c42;
  --button-secondary-border-color: #cadbd4;
}
body, .dark { background: #f4f7f6 !important; }
.gradio-container *, .gradio-container *::before, .gradio-container *::after { box-sizing: border-box; }
footer { display: none !important; }
.gradio-container main, .gradio-container .main { padding: 0 !important; }
#heading { margin-bottom: 20px; }
#heading h1 { color: #153c32; font-size: clamp(28px, 4vw, 38px); font-weight: 700; letter-spacing: -.035em; margin: 0 0 10px; line-height: 1.2; }
#heading p { color: #50675f; font-size: 16px; margin: 0; line-height: 1.6; }
#selectors { max-width: 480px; gap: 16px; margin-bottom: 12px; }
#selectors > div { min-width: 0 !important; }
#selectors .wrap-inner { background: #fff; border: 1px solid #c8d9d1; border-radius: 10px; min-height: 46px; }
#selectors input { color: #183d31; font-weight: 600; cursor: pointer; min-height: 42px; }
#selectors .wrap-inner:focus-within { outline: 2px solid #0f766e; outline-offset: 2px; }
#selectors .icon-wrap { color: #456b5c; }
#editors { gap: 24px; align-items: stretch; }
#editors > .text-panel { min-width: 0 !important; padding: 20px; border: 1px solid #dce5e2; border-radius: 16px; background: #fff; box-shadow: 0 3px 16px #163c2d05; gap: 14px; }
.text-panel textarea { min-height: 360px !important; height: 360px !important; font-size: 15px !important; line-height: 1.7 !important; color: #263f35 !important; background: #fff !important; }
.text-panel textarea:disabled { opacity: 1 !important; -webkit-text-fill-color: #263f35; cursor: text; }
.text-panel textarea:focus-visible { outline: 2px solid #0f766e; outline-offset: -2px; }
.input-count { color: #526a60; font-size: 13px; }
#article-count { min-height: 24px; }
.actions { margin-top: auto !important; gap: 12px; align-items: center; }
.gradio-container button { min-height: 44px; border-radius: 9px !important; font-weight: 600 !important; transition: background .15s, border-color .15s; }
.gradio-container button:focus-visible, .metric dt:focus-visible { outline: 2px solid #0f766e; outline-offset: 3px; }
#summarize-button { background: #0f766e !important; border-color: #0f766e !important; color: #fff !important; }
#summarize-button:hover:not(:disabled) { background: #115e59 !important; }
#summarize-button:disabled { background: #58736d !important; border-color: #58736d !important; cursor: wait; opacity: 1; }
#copy-button:disabled { color: #5c6d65 !important; background: #f0f4f1 !important; opacity: 1; }
#clear-button:hover:not(:disabled), #copy-button:hover:not(:disabled) { background: #edf5f1 !important; }
.statistics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0; padding: 2px 0 6px; }
.metric { min-width: 0; }
.metric dt { color: #526a60; font-size: 12px; line-height: 1.5; min-height: 36px; cursor: help; }
.metric dd { color: #204a3b; font-size: 19px; font-weight: 650; line-height: 1.4; margin: 5px 0 0; }
#request-status { min-height: 30px; }
.request-status { color: #405f52; font-size: 13px; line-height: 1.6; margin: 0; }
.request-status.error { color: #a52d2d; }
#copy-feedback { color: #405f52; font-size: 12px; }
@media (max-width: 760px) {
  .gradio-container { padding: 26px 16px !important; }
  #editors { flex-direction: column; gap: 18px; }
  #editors > .text-panel { width: 100% !important; flex-basis: auto !important; padding: 16px; }
  .text-panel textarea { min-height: 280px !important; height: 280px !important; }
}
@media (max-width: 420px) {
  .gradio-container { padding: 22px 10px !important; }
  #selectors { gap: 10px; }
  .statistics { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px 20px; }
  .metric dt { min-height: 0; font-size: 12px; }
}
@media (prefers-reduced-motion: reduce) { .gradio-container button { transition: none; } }
"""


def build_demo():
    theme = gr.themes.Soft(
        primary_hue="teal",
        neutral_hue="slate",
        font=["Segoe UI", "system-ui", "sans-serif"],
    )
    with gr.Blocks(
        title="News Summarizer",
        analytics_enabled=False,
        theme=theme,
        css=CSS,
        js=DROPDOWN_JS,
        fill_width=True,
    ) as demo:
        gr.HTML(
            "<header><h1>News Summarizer</h1><p>Choose a model and turn an article into a shorter summary.</p></header>",
            elem_id="heading",
        )
        with gr.Row(elem_id="selectors"):
            model = gr.Dropdown(
                list(MODEL_CHOICES),
                value="LSTM",
                label="Model",
                filterable=False,
                interactive=True,
                min_width=0,
                elem_id="model-dropdown",
            )
        with gr.Row(elem_id="editors", equal_height=True):
            with gr.Column(
                scale=1,
                min_width=0,
                elem_id="article-panel",
                elem_classes=["text-panel"],
            ):
                article = gr.Textbox(
                    label="Article",
                    lines=16,
                    max_lines=16,
                    placeholder="Paste your news article here...",
                    elem_id="article-input",
                )
                count = gr.HTML(input_words(""), elem_id="article-count")
                with gr.Row(elem_classes=["actions"]):
                    submit = gr.Button(
                        "Summarize",
                        variant="primary",
                        scale=1,
                        min_width=140,
                        elem_id="summarize-button",
                    )
                    clear = gr.Button(
                        "Clear", scale=0, min_width=84, elem_id="clear-button"
                    )
            with gr.Column(
                scale=1,
                min_width=0,
                elem_id="summary-panel",
                elem_classes=["text-panel"],
            ):
                summary = gr.Textbox(
                    label="Summary",
                    lines=16,
                    max_lines=16,
                    interactive=False,
                    placeholder="Your summary will appear here.",
                    elem_id="summary-output",
                )
                statistics = gr.HTML(render_statistics(), elem_id="result-statistics")
                with gr.Row(elem_classes=["actions"]):
                    copy = gr.Button(
                        "Copy summary",
                        interactive=False,
                        scale=0,
                        min_width=144,
                        elem_id="copy-button",
                    )
                    copy_feedback = gr.HTML("", elem_id="copy-feedback")
        status = gr.HTML("", elem_id="request-status")
        results = [summary, statistics, status, copy]
        controls = [article, model, submit, clear]
        start = submit.click(
            begin_request,
            outputs=[*results, *controls, copy_feedback],
            queue=False,
            show_progress="hidden",
            api_name=False,
            trigger_mode="once",
        )
        request = start.then(
            generate_summary,
            inputs=[article, model],
            outputs=results,
            api_name="summarize",
            concurrency_limit=1,
            concurrency_id="summarization",
            show_progress="full",
            show_progress_on=[summary],
            trigger_mode="once",
        )
        request.then(
            finish_request,
            outputs=controls,
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        article.input(
            article_changed,
            inputs=article,
            outputs=[count, *results, copy_feedback],
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        for dropdown in (model,):
            dropdown.input(
                invalidate_result,
                outputs=[*results, copy_feedback],
                queue=False,
                show_progress="hidden",
                api_name=False,
            )
        clear.click(
            clear_form,
            outputs=[article, count, *results, copy_feedback],
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        copy.click(
            None,
            inputs=summary,
            outputs=copy_feedback,
            js=COPY_JS,
            queue=False,
            show_progress="hidden",
            api_name=False,
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
