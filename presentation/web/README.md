# Comparing Neural Models for News Summarization

An offline Reveal.js presentation by Biruk Anley and Daniela Ortega. The 16-slide talk takes about 15 minutes including the live-demo switch. Fourteen appendix slides hold the two benchmark tracks, checkpoint settings, design decisions, complete reproduced outputs and references. Main charts and both backup tables use the completed 250-article benchmark.

## Start the slide server

Run one command **from the repository root**. The server uses only Python 3’s standard library. Keep that terminal running during the talk. The entire `presentation/web` directory can also be copied to another machine; when running from inside that directory, omit `--directory presentation/web`.

**Windows PowerShell**

```powershell
python -m http.server 8008 --bind 127.0.0.1 --directory presentation/web
```

**Ubuntu**

```bash
python3 -m http.server 8008 --bind 127.0.0.1 --directory presentation/web
```

Open the slide address in a browser. The classroom app has its own process and port. Only the presenter switches screens or submits a request. The speaking script includes the offline app checklist for both systems. [Open slides](http://127.0.0.1:8008/) · [Speaking script and demo checklist](SPEAKING_SCRIPT.md)

## Controls

| Action | Control |
|---|---|
| Next diagram stage, then next slide | Click the slide canvas, press Right, or press Space |
| Previous stage or slide | Left arrow |
| Pause / resume motion on the current and subsequent slides | P or **Pause motion** / **Resume motion** |
| Replay a slide | R or **Replay** resets its stages; advance them again at your pace |
| Open the overview, then select a slide | O or **Overview**; Esc returns to the deck |
| Enter fullscreen | F or **Fullscreen**; Esc exits |
| Jump back after the demo | Open `http://127.0.0.1:8008/#/conclusion` |

The deck honors the operating system’s reduced-motion preference. Reveals still follow the presenter’s clicks. Slide 4 distinguishes historical preparation, current input handling and evaluation. Its word IDs are labeled illustrative; its lowercase words and BART ID prefix come from a saved trace. The architecture diagrams label their token activity and attention links as illustrative.

## Static backup and export

The PDF contains one 16:9 final state per slide. All diagram stages, labels and bars appear together. PDF does not preserve animation. [Download the 30-page PDF](News_Summarization.pdf) · [Preview all slides](contact-sheet.jpg) · [Original editable PowerPoint](../Comparing_Neural_Models_for_News_Summarization.pptx)

To export again, open the deck from the local server in Chrome or Chromium, wait for the cover to appear, then press **Ctrl+P**. Choose **Save as PDF**, enable background graphics, disable headers and footers, and keep the default scale. The print layout supplies the landscape page size. The preview should contain 30 pages even when the browser is on an intermediate diagram stage.

## Files and sources

- Edit audience-facing slides and inline SVG in [index.html](index.html). The [stylesheet](deck.css) controls typography, motion and print layout. The [JavaScript](deck.js) handles Reveal initialization, controls and printing.
- The 315-word fictional article and four reproduced outputs are available locally for demo fallback. [Article body](riverton_article.txt) · [Saved outputs](saved_demo_outputs.md)
- All runtime libraries and fonts are in `vendor/`. Source Sans 3 is used for slide text and IBM Plex Mono for labels. The viewer needs no package manager, application environment, model cache or network connection beyond loopback. [Third-party notices](THIRD_PARTY.md)
- Scores come from the two completed 250-article tracks, displayed to two decimals. The 50-article pilot remains historical evidence within the larger sample. Riverton examples remain in backup material. [Benchmark report and resume commands](../../reports/benchmark250/README.md) · [Frozen metrics](../../reports/benchmark250/results.json) · [Historical pilot](../../reports/pilot50/results.json) · [Citations and evidence map](REFERENCES.md)
- Validation covers the Windows browser and the PDF. [Validation record](VALIDATION.md)
