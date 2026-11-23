# Presentation validation

The revised presentation contains 31 slides, with 16 main slides timed to 900
seconds, two references slides and 13 backups. Biruk Anley and Daniela Ortega
share the talk. Slide 14 includes the brief demo and CPU wait.

## Evidence and scope

- Reviewed the original 18-slide PowerPoint and PDF, script and notes, project
  documentation, preprocessing and full-article pipeline, Hugging Face adapter,
  pinned OpenNMT encoder/decoder/copy code, manifests, saved model verification,
  diagnosis reports and frozen 50-article results.
- Archived the original PPTX and PDF byte-for-byte with `_v1` filenames. Their
  hashes appear in [the evidence manifest](assets/evidence_manifest.json).
- Used exact full-article and shared-prefix values from the saved results JSON.
  Six native charts contain embedded workbooks. Both complete tables preserve
  ROUGE-L and ROUGE-Lsum as separate metrics. Timing results are omitted.
- Used an actual saved CNN/DailyMail test-row excerpt with ID and source hashes.
  Dataset counts were checked against configuration 3.0.0 of the primary dataset
  card. The preprocessing slide uses exact saved OpenNMT tokens from the earlier
  fictional Riverton fixture, identified on the slide and in notes.
- Captured the current running app page with Chrome without submitting a model
  request. The screenshot shows the current controls and empty output field.
- The latest Riverton body and all four complete reproduced display outputs
  match the saved diagnosis. The original duplicate display remains unresolved.
- No application, model, decoding, benchmark or application-environment changes
  were made. No inference, training or benchmark ran during the redesign.

## Rendering and visual inspection

Exported the actual PPTX with installed Microsoft PowerPoint 16.0 to a 31-page
PDF and 31 PNG previews at 1600 × 900. Reviewed every main, reference and backup
slide using large four-slide review sheets, with individual architecture and
chart checks. The [contact sheet](rendered/contact_sheet.png) covers the full deck.

The review corrected wrapping in native chart values, increased room for text
in the introduction and conclusion, clarified architecture arrows, and adjusted
source text and full-output boxes. Word embeddings stays on one line. Transformer
model labels stay on one line. Source-event text stays within its banner, and
conclusion text stays within its assigned space.

The automated validator checks estimated text heights and actual PowerPoint text
bounds. It also checks shape bounds, notes, exact chart/workbook values, both
tables, preserved evidence hashes, full saved outputs in the PDF, local links,
slide counts and unwrapped chart values. The final export passed all artifact
checks. All 94 protected source/archive hashes matched. No estimated or measured
text-bound violations remained, and all six charts and embedded workbooks matched
the saved values. Git whitespace checks also passed.

## Limitations

No course guideline was located in the available repository, supplied attachment
or top-level Documents PDF search. Compliance with an unavailable course guideline
has not been assessed. The supplied redesign brief was used.

PowerPoint export succeeded on this PC after using Office through the Windows
user session outside the restricted sandbox. Re-exporting on another machine
requires PowerPoint or another renderer and installed compatible fonts. Native
objects remain editable, while the app screenshot is a raster image. Preview
images and the PDF preserve the rendered appearance for review.
