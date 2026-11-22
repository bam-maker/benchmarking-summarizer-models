"""Rendered UI checks with installed Chrome and deterministic test-only decoders.

Uses the existing Gradio/websockets dependencies. No model weights are loaded.
The fixture server is temporary and never replaces the real application.
"""

import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import websockets

import app


class Browser:
    def __init__(self, websocket):
        self.websocket = websocket
        self.sequence = 0
        self.errors = []

    async def call(self, method, params=None):
        self.sequence += 1
        sequence = self.sequence
        await self.websocket.send(
            json.dumps({"id": sequence, "method": method, "params": params or {}})
        )
        while True:
            response = json.loads(
                await asyncio.wait_for(self.websocket.recv(), timeout=15)
            )
            if response.get("method") == "Runtime.exceptionThrown":
                self.errors.append(response["params"])
            if response.get("id") == sequence:
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response.get("result", {})

    async def evaluate(self, expression):
        response = await self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        if "exceptionDetails" in response:
            raise RuntimeError(response["exceptionDetails"])
        return response["result"].get("value")

    async def wait(self, expression, timeout=12):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if await self.evaluate(expression):
                return
            await asyncio.sleep(0.05)
        raise AssertionError("Browser condition timed out: " + expression)

    async def click(self, selector):
        bounds = await self.evaluate(
            "(() => {const e=document.querySelector("
            + json.dumps(selector)
            + "); e.scrollIntoView({block:'center'}); const r=e.getBoundingClientRect(); return {x:r.x+r.width/2,y:r.y+r.height/2};})()"
        )
        for kind in ("mousePressed", "mouseReleased"):
            await self.call(
                "Input.dispatchMouseEvent",
                {"type": kind, **bounds, "button": "left", "clickCount": 1},
            )

    async def fill(self, selector, value):
        await self.evaluate(
            "(() => {const e=document.querySelector("
            + json.dumps(selector)
            + "); e.focus(); e.select();})()"
        )
        await self.call("Input.insertText", {"text": value})

    async def key(self, key):
        code = {"ArrowDown": 40, "ArrowUp": 38, "Enter": 13, "Escape": 27, "Tab": 9}[
            key
        ]
        for kind in ("keyDown", "keyUp"):
            await self.call(
                "Input.dispatchKeyEvent",
                {"type": kind, "key": key, "code": key, "windowsVirtualKeyCode": code},
            )

    async def choose(self, dropdown, label):
        await self.click(f"#{dropdown} input")
        await self.wait(
            f"document.querySelector('#{dropdown} input').getAttribute('aria-expanded') === 'true'"
        )
        for _ in range(6):
            active = await self.evaluate(
                f"document.querySelector('#{dropdown} .active[role=option]')?.getAttribute('aria-label')"
            )
            if active == label:
                break
            await self.key("ArrowDown")
        else:
            raise AssertionError("Keyboard option missing: " + label)
        await self.key("Enter")
        await self.wait(
            f"document.querySelector('#{dropdown} input').value === {json.dumps(label)}"
        )

    async def screenshot(self, path):
        image = await self.call(
            "Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True}
        )
        path.write_bytes(base64.b64decode(image["data"]))


async def check_rendered(browser, url, calls, output):
    await browser.call("Runtime.enable")
    await browser.call("Page.enable")
    await browser.call("Page.navigate", {"url": url})
    await browser.wait("document.querySelector('#model-dropdown input') !== null")
    await browser.call(
        "Emulation.setDeviceMetricsOverride",
        {"width": 1280, "height": 1000, "deviceScaleFactor": 1, "mobile": False},
    )
    await browser.wait("document.querySelector('#summarize-button') !== null")
    assert await browser.evaluate("document.title") == "News Summarizer"
    assert (
        await browser.evaluate("document.querySelector('#model-dropdown input').value")
        == "LSTM"
    )
    assert await browser.evaluate("document.querySelector('#length-dropdown') === null")
    assert await browser.evaluate("document.querySelector('#copy-button').disabled")

    await browser.click("#model-dropdown input")
    assert (
        await browser.evaluate(
            "getComputedStyle(document.querySelector('#model-dropdown .wrap-inner')).outlineWidth"
        )
        == "2px"
    )
    await browser.key("Escape")
    await browser.wait(
        "document.querySelector('#model-dropdown input').getAttribute('aria-expanded') === 'false'"
    )
    await browser.choose("model-dropdown", "BART")
    article = "pipeline " + "news " * 99
    await browser.fill("#article-input textarea", article)
    await browser.wait(
        "document.querySelector('#article-count').textContent.includes('100 words')"
    )
    await browser.click("#summarize-button")
    await browser.wait("document.querySelector('#summarize-button').disabled")
    assert await browser.evaluate(
        "document.querySelector('#article-input textarea').disabled && document.querySelector('#model-dropdown input').disabled && document.querySelector('#clear-button').disabled"
    )
    assert await browser.evaluate(
        "document.querySelector('#summary-output textarea').value === '' && document.querySelector('#copy-button').disabled"
    )
    # A second physical click on the disabled button must not enqueue another job.
    await browser.click("#summarize-button")
    progress = set()
    deadline = time.perf_counter() + 12
    while time.perf_counter() < deadline:
        text = await browser.evaluate("document.body.innerText")
        for phase in ("Loading test backend", "Article chunks", "Consolidating pass"):
            if phase in text:
                progress.add(phase)
        if await browser.evaluate(
            "!document.querySelector('#summarize-button').disabled && document.querySelector('#summary-output textarea').value.length > 0"
        ):
            break
        await asyncio.sleep(0.05)
    assert (
        await browser.evaluate(
            "document.querySelector('#summary-output textarea').value"
        )
        == "A BART natural summary."
    )
    assert (
        len(calls) == 1
        and calls[-1]["model"] == "BART"
        and calls[-1]["policy"] == app.NATURAL_POLICY
    )
    assert "Article chunks" in progress and "Consolidating pass" in progress, progress
    assert await browser.evaluate("!document.querySelector('#copy-button').disabled")
    metrics = await browser.evaluate(
        "Array.from(document.querySelectorAll('.metric dd'), e=>e.textContent)"
    )
    assert metrics[:2] == ["4", "96.0%"], metrics
    assert (
        abs(float(metrics[3]) - calls[-1]["tokens"] / calls[-1]["generation_seconds"])
        < 0.051
    )
    assert await browser.evaluate(
        "document.querySelector('#result-statistics').textContent.includes('Pipeline generation tok/s')"
    )
    await browser.call(
        "Browser.grantPermissions",
        {
            "origin": url.rstrip("/"),
            "permissions": ["clipboardReadWrite", "clipboardSanitizedWrite"],
        },
    )
    await browser.call("Page.bringToFront")
    await browser.click("#copy-button")
    await browser.wait(
        "document.querySelector('#copy-feedback').textContent.includes('Copied to clipboard')"
    )
    assert (
        await browser.evaluate("navigator.clipboard.readText()")
        == "A BART natural summary."
    )

    layouts = []
    for width in (1280, 760, 390, 320):
        await browser.call(
            "Emulation.setDeviceMetricsOverride",
            {"width": width, "height": 1000, "deviceScaleFactor": 1, "mobile": False},
        )
        await asyncio.sleep(0.25)
        layout = await browser.evaluate(
            "(() => {const a=document.querySelector('#article-panel').getBoundingClientRect(),s=document.querySelector('#summary-panel').getBoundingClientRect(); return {width:innerWidth,scroll:document.documentElement.scrollWidth,ax:a.x,ay:a.y,sx:s.x,sy:s.y,aw:a.width,sw:s.width};})()"
        )
        assert layout["scroll"] <= width, layout
        if width > 760:
            assert (
                abs(layout["ay"] - layout["sy"]) < 2
                and abs(layout["aw"] - layout["sw"]) < 2
            ), layout
        else:
            assert (
                layout["sy"] > layout["ay"] and abs(layout["ax"] - layout["sx"]) < 2
            ), layout
        layouts.append(layout)
        # A page may hide overflow; check actual action bounds as well.
        clipped = await browser.evaluate(
            "Array.from(document.querySelectorAll('#editors button')).filter(e => {const b=e.getBoundingClientRect(), p=e.closest('.text-panel').getBoundingClientRect();return b.x < p.x || b.right > p.right+1;}).map(e=>e.textContent)"
        )
        assert not clipped, {"width": width, "clipped": clipped}
        if width in (1280, 390):
            await browser.evaluate("scrollTo(0,0)")
            await browser.screenshot(output / f"ui-{width}.png")

    # Choosing another model must invalidate the previous text and statistics.
    await browser.choose("model-dropdown", "PEGASUS")
    await browser.wait(
        "document.querySelector('#summary-output textarea').value === '' && document.querySelector('#copy-button').disabled"
    )
    assert await browser.evaluate(
        "Array.from(document.querySelectorAll('.metric dd')).every(e=>e.textContent === '\u2014')"
    )
    await browser.fill("#article-input textarea", "FAIL keep my original article.")
    await browser.click("#summarize-button")
    await browser.wait(
        "document.querySelector('#request-status [role=alert]') !== null"
    )
    await browser.wait("!document.querySelector('#summarize-button').disabled")
    assert (
        await browser.evaluate(
            "document.querySelector('#article-input textarea').value"
        )
        == "FAIL keep my original article."
    )
    assert await browser.evaluate(
        "document.querySelector('#summary-output textarea').value === '' && document.querySelector('#copy-button').disabled"
    )
    await browser.click("#clear-button")
    await browser.wait("document.querySelector('#article-input textarea').value === ''")
    assert await browser.evaluate(
        "document.querySelector('#article-count').textContent.includes('0 words') && document.querySelector('#request-status').textContent.trim() === '' && document.querySelector('#copy-feedback').textContent.trim() === ''"
    )
    assert not browser.errors, browser.errors
    return {
        "kind": "UI-only deterministic fixtures; NOT model benchmark results",
        "progress_seen": sorted(progress),
        "layouts": layouts,
        "clipboard_verified": True,
        "duplicate_submission_prevented": True,
        "error_preserves_article": True,
        "clear_resets_all": True,
        "keyboard_dropdowns": True,
        "javascript_errors": browser.errors,
    }


def read_targets(port):
    with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=0.3) as response:
        return json.load(response)


async def main():
    output = ROOT / "test-results/natural_ui"
    output.mkdir(exist_ok=True)
    chrome = (
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
        / "Google/Chrome/Application/chrome.exe"
    )
    if not chrome.exists():
        raise RuntimeError(
            "Install-free browser check requires the existing Chrome installation."
        )
    calls = []

    def fixture(article, model, generation_policy, progress_callback):
        if article.startswith("FAIL"):
            raise app.InputError(
                "This test request could not complete. Your article is preserved."
            )
        record = {"model": model, "policy": generation_policy}
        calls.append(record)
        progress_callback(0, 1, "Loading test backend")
        time.sleep(0.3)
        generation_seconds, generated = 0.0, 0
        for phase, done, total in (
            ("Article chunks", 1, 2),
            ("Article chunks", 2, 2),
            ("Consolidating pass 1", 1, 1),
        ):
            progress_callback(done - 1, total, phase)
            started = time.perf_counter()
            token_ids = list(
                range(20)
            )  # Test fixture IDs, never pretrained-model outputs.
            time.sleep(0.3)
            generation_seconds += time.perf_counter() - started
            generated += len(token_ids)
            progress_callback(done, total, phase)
        record.update(tokens=generated, generation_seconds=generation_seconds)
        return SimpleNamespace(
            summary=f"A {model} natural summary.",
            warnings=(),
            coverage_complete=True,
            processing_mode="full",
            model_calls=3,
            consolidation_passes=1,
            total_generated_tokens=generated,
            generation_seconds=generation_seconds,
            returned_input=False,
        )

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        debug_port = sock.getsockname()[1]
    profile = Path(tempfile.mkdtemp(prefix="ui-browser-", dir=ROOT / ".cache"))
    process = await asyncio.to_thread(
        subprocess.Popen,
        [
            str(chrome),
            "--headless=new",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-extensions",
            f"--remote-debugging-port={debug_port}",
            f"--user-data-dir={profile}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    demo = None
    browser = None
    try:
        for _ in range(30):
            try:
                targets = await asyncio.to_thread(read_targets, debug_port)
                websocket_url = next(
                    t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page"
                )
                break
            except (OSError, StopIteration):
                await asyncio.sleep(0.1)
        else:
            raise RuntimeError("Headless Chrome did not start")
        with patch.object(app, "summarize_article", side_effect=fixture):
            demo = app.build_demo()
            _, url, _ = demo.launch(
                server_name="127.0.0.1",
                server_port=7861,
                share=False,
                prevent_thread_lock=True,
                quiet=True,
                show_api=False,
            )
            async with websockets.connect(
                websocket_url, proxy=None, max_size=10_000_000
            ) as websocket:
                browser = Browser(websocket)
                try:
                    report = await check_rendered(browser, url, calls, output)
                except Exception:
                    await browser.screenshot(output / "ui-failure.png")
                    print(
                        await browser.evaluate(
                            "JSON.stringify({body:document.body.innerText,buttons:Array.from(document.querySelectorAll('button'), e=>({text:e.textContent,disabled:e.disabled}))})"
                        )
                    )
                    raise
                (output / "ui_browser.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
                print(
                    "PASS: rendered dropdowns, keyboard/Escape/focus, metrics, Copy/Clear, busy/error states and responsive layouts."
                )
    finally:
        if demo:
            demo.close()
        if browser:
            # Close the real browser, not only Chrome's short-lived launcher.
            try:
                async with websockets.connect(websocket_url, proxy=None) as cleanup:
                    await cleanup.send(json.dumps({"id": 1, "method": "Browser.close"}))
            except (OSError, websockets.ConnectionClosed):
                pass
        if process.poll() is None:
            process.terminate()
        await asyncio.to_thread(process.wait, timeout=10)


if __name__ == "__main__":
    asyncio.run(main())
