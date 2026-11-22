# Validation record

Prepared November 22, 2026 for the presentation deliverables. No model inference,
training, benchmark run or app environment change was performed for this package.

## Artifact checks

- PowerPoint opens with 18 slides in 16:9 format: **eight main slides**, two
  references slides and eight backups.
- Main-slide speaker notes are embedded and match `speaker_notes.json`; the
  allocations sum to **900 seconds (15:00)**, including the short demo, transitions
  and CPU wait. The separate Markdown script includes the offline command and
  complete saved fallback evidence.
- Three **native editable PowerPoint charts**, with three embedded XLSX data
  sheets, show full-article ROUGE-1, ROUGE-2 and ROUGE-Lsum. Every raw chart value
  matches `reports/pilot50/results.json`. All three axes run from zero to 50.
  Full and shared-prefix backup values are read from the same saved JSON.
- Native editable shapes/text implement the two architecture diagrams, app
  schematic, pipeline diagram, model cards and tables.
- The latest demo article body is **315 words / 2,202 characters** and has SHA-256
  `648c2a202c8912f6d1fc5bbe468162d2670be52508916bbbf4ac7fd7e5457fa8`.
  Complete saved outputs in the PDF match the diagnosis. The LSTM fallback is
  the distinct reproduced summary, not the originally duplicated display.
- Every slide has notes, and shape bounds fall within the slide. Local Markdown
  links resolve. No tracked application or saved benchmark file changed.

## Rendering and visual inspection

The actual `.pptx` was exported with **LibreOffice 26.8.2.2** to an 18-page PDF.
Every page was rendered to a 1,600 × 900 PNG with Poppler. The contact sheet and
individual dense pages were inspected for text overflow, alignment, diagram
connections, legibility and chart accuracy. Corrections included chart-label
alignment, reference colors/wrapping, the LSTM decoder query connection, and
space for the full BART fallback. The final deck needs no animations to explain
its figures, and the PDF preserves the visible slide content.

The final export includes the corrected illustrative data example, explicit
bounded-consolidation label and sufficient panel space for the complete saved
LSTM fallback. Its visible content is covered by the final page renders.

Run only these lightweight artifact checks with the isolated presentation tools:

```bash
/tmp/summarizer-presentation-tools/bin/python presentation/validate_presentation.py
```

This checks package structure, native charts/data, notes/timings, exact source
hash, full saved outputs in PDF, rendered-page count, bounds and relative links.
It reads saved results; it does not recalculate ROUGE or load any summarizer.

## Limits and unresolved evidence

- No presentation-specific guideline PDF was located. The supplied presentation
  brief was used. No required content placeholders remain.
- No current app screenshot was available. Slide 4 is explicitly labeled as an
  editable **interface schematic**, based on the actual current implementation.
- The original duplicate-display cause remains unresolved, as the diagnosis
  states; fresh direct and Gradio LSTM requests had a distinct output. No confirmed
  UI or routing bug is asserted.
- The font is Lato, installed on this laptop. Other computers may substitute a
  font in the editable PowerPoint; use the PDF for a stable visual fallback.
- Presentation software versions can render native charts slightly differently.
  The inspected PDF is the export from this laptop, not a Microsoft PowerPoint
  rendering on a different machine.
