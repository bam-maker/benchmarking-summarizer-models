# Presentation package

**Comparing Neural Models for News Summarization** by Biruk Anley and Daniela
Ortega. The redesigned deck contains 31 slides. Slides 1–16 form the 15-minute
talk, slides 17–18 provide references, and slides 19–31 provide backups.
The talk includes 100 seconds for the diagnostic and brief demo on slide 14.

- [Revised editable PowerPoint](Comparing_Neural_Models_for_News_Summarization.pptx)
- [PDF export](Comparing_Neural_Models_for_News_Summarization.pdf)
- [Original v1 PowerPoint](Comparing_Neural_Models_for_News_Summarization_v1.pptx)
- [Original v1 PDF](Comparing_Neural_Models_for_News_Summarization_v1.pdf)
- [Speaking script and Ubuntu demo checklist](SPEAKING_SCRIPT.md)
- [Rendered slides](rendered/) and [contact sheet](rendered/contact_sheet.png)
- [Validation notes](VALIDATION.md) and [primary references](REFERENCES.md)

The warm-white design uses Calibri, restrained model colors, editable shapes,
connectors and six native charts with embedded data workbooks. Body text is
generally 24–32 pt. Diagram annotations, table headings, numeric chart labels,
citations and page numbers use smaller type. The app screenshot is a raster
capture of the current interface, taken without submitting a request. Its
onscreen controls retain their original text. No generated output is pictured.

The teaching order places preprocessing before separate architecture slides.
Core results appear before BART and PEGASUS are introduced. All chart and table
values come from the existing frozen 50-article comparison. The latest fictional
Riverton example is separate. Saved fallback outputs occupy slides 25–28.
The earlier duplicate display remains unresolved, as explained in backup notes.

No course guideline was located in the repository, supplied attachment, or
top-level Documents PDF search. The supplied redesign brief governs this deck.
The metric explanation and core results use separate slides for readability,
giving 16 main slides within the requested approximate count.

## Rebuilding

Use a separate tooling environment. The application environment and dependency
pins were preserved. The build reads saved repository evidence and writes into
this folder. The source includes the spoken script and detailed notes.

```powershell
python -m venv .cache/presentation-tools
.cache/presentation-tools/Scripts/python.exe -m pip install python-pptx==1.0.2 Pillow==11.3.0 pymupdf==1.26.6 pywin32==311
.cache/presentation-tools/Scripts/python.exe presentation/build_presentation.py
.cache/presentation-tools/Scripts/python.exe presentation/export_presentation.py
.cache/presentation-tools/Scripts/python.exe presentation/validate_presentation.py
```

[build_presentation.py](build_presentation.py) uses the committed screenshot and
saved dataset example, plus repository reports and manifests. It produces the
PowerPoint, script, speaker-note JSON, slide manifest, CSV values and estimated
text audit. The script and note JSON are generated artifacts. Edit the readable
content in the builder before rebuilding them.

[export_presentation.py](export_presentation.py) uses installed Windows PowerPoint
via COM, opens the PPTX read-only, exports the PDF and all 1600 × 900 PNG previews,
and records actual text bounds. It also creates the contact sheet. PowerPoint
requires an interactive Windows logon session with Office access. In this session
COM export required execution outside the restricted sandbox. It succeeded.

Calibri is installed on this PC. Linux builds can use the compatible Carlito
fonts if present. Export on the target PowerPoint setup and inspect the result
when fonts or renderers change. The PDF supplies a fixed visual copy.

The [evidence manifest](assets/evidence_manifest.json) records source hashes and
the byte-exact v1 archives. [validate_presentation.py](validate_presentation.py)
checks all chart caches, embedded workbook values, tables, exact fallback text,
note durations, text bounds, preview counts and preserved evidence.

## Demo assets

The [exact latest article body](assets/riverton_article.txt) and
[saved model outputs](assets/saved_demo_outputs.md) support the classroom demo.
Use the documented Ubuntu launch command in the speaking script. Submit at most
one live request during the timed talk, with a roughly 30-second visible wait.
The redesign ran no inference, training or benchmark.
