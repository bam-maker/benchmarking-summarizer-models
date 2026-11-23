# Browser deck validation

Verified on the Windows PC on November 22, 2026, using installed Chrome 157.0.8103.79 through Playwright. The static server bound to `127.0.0.1:8008`. Browser checks and PDF inspection used isolated presentation tooling.

| Check | Observed result |
|---|---|
| Audience view | Opened all 31 slides at 1280 × 720 and captured 58 initial and intermediate/final states. Reviewed changed charts, the final result table, conclusions and historical backup labels. |
| Navigation | Clicks and Right advances traversed fragment groups exactly once. Left reversed a step. Replay reset every stepped slide. Overview opened by button and keyboard, and selecting a thumbnail opened the chosen slide. Fullscreen worked by button and F. |
| Motion | P and the button paused and resumed computed animation playback. Inactive cover nodes stopped animating. With reduced motion emulated, the cover was still and the expanded chart’s final labels remained visible. |
| Offline operation | Blocked non-loopback browser requests before first load. The page requested only HTML, local JS, CSS and fonts. Observed zero external requests, HTTP failures or page errors. Navigation also worked after the browser context went offline. |
| Evidence | Compared both main charts and the final table against all supplied values in reported_scores.json. Verified mathematically that PEGASUS leads every metric and all four neural models exceed Lead-3 on every metric. The user confirmed the same 250 full articles. Earlier full-article and shared-prefix backup tables match the saved benchmark250 JSON. Training predictions and logs for the supplied scores were not provided. |
| PDF | Exported 31 landscape pages at 960 × 540 PDF points (16:9). Rendered the pages and checked changed chart labels, the final table and the findings. Motion and fragment timing are exclusive to the browser deck. |
| Separation | Slide bottoms contain only consecutive numbers 01–31. The final scores and earlier repository measurements have distinct table titles. Checked source and rendered sections for notes and directions. The separate script runs continuously from 00:00 to 15:00 and includes demo operations with updated fallback slide numbers. |
| Existing project | This update changed presentation content and documentation. The earlier benchmark files, application, model weights and environments retain their saved state. |

The Windows server command and browser were exercised locally. The Ubuntu command uses Python 3’s standard-library static server with the same options. This PC had no available Ubuntu WSL distribution. Ubuntu browser rendering, Intel Iris Xe performance and physical projector readability remain untested. The full-slide contact sheet is available for a venue check. [Preview](contact-sheet.jpg) · [Launch and export instructions](README.md)
