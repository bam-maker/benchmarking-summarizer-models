# Browser deck validation

Verified on the Windows PC on November 22, 2026, using installed Chrome 157.0.8103.79 through Playwright. The static server bound to `127.0.0.1:8008`. Browser checks and PDF inspection used isolated presentation tooling.

| Check | Observed result |
|---|---|
| Audience view | Opened all 30 slides at 1280 × 720 and captured 52 initial and intermediate/final states. Reviewed the screenshots, text boundaries, labels and connector placement. Moved the LSTM state legend clear of the attention lines. |
| Navigation | Clicks and Right advances traversed fragment groups exactly once. Left reversed a step. Replay reset every stepped slide. Overview opened by button and keyboard, and selecting a thumbnail opened the chosen slide. Fullscreen worked by button and F. |
| Motion | P and the button paused and resumed computed animation playback. Inactive cover nodes stopped animating. With reduced motion emulated, the cover was still and the expanded chart’s final labels remained visible. |
| Offline operation | Blocked non-loopback browser requests before first load. The page requested only HTML, local JS, CSS and fonts. Observed zero external requests, HTTP failures or page errors. Navigation also worked after the browser context went offline. |
| Evidence | Compared every score on both main charts and both complete tables with the frozen pilot50 JSON rounded to two decimals. Checked mean output length, article count and failures in both tables. Matched all four displayed diagnostic strings to the saved complete outputs. |
| PDF | Exported 30 landscape pages at 960 × 540 PDF points (16:9). Rendered all pages, reviewed the contact sheets, and checked final diagrams, full chart labels and the demo transition. Motion and fragment timing are exclusive to the browser deck. |
| Separation | Slides contain audience-facing text and citations. Checked the slide source and rendered sections for notes and directions. The separate script runs continuously from 00:00 to 15:00 and includes the demo operations. Gradio responded on port 7860; the presentation made no request to it. |
| Existing project | Confirmed the original PowerPoint remained intact. No training, inference, benchmark execution, model installation or application-environment change was performed for the browser deck. |

The Windows server command and browser were exercised locally. The Ubuntu command uses Python 3’s standard-library static server with the same options. This PC had no available Ubuntu WSL distribution. Ubuntu browser rendering, Intel Iris Xe performance and physical projector readability remain untested. The full-slide contact sheet is available for a venue check. [Preview](contact-sheet.jpg) · [Launch and export instructions](README.md)
