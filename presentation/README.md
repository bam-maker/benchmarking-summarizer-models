# Presentation package

**Comparing Neural Models for News Summarization** — Biruk Anley and Daniela
Ortega, CIS 634. The deck has eight main slides, two references slides, and eight
backup slides. Main-slide notes total 15:00, including the short demo and CPU wait.

## Deliverables

- [Editable PowerPoint](Comparing_Neural_Models_for_News_Summarization.pptx)
- [PDF export](Comparing_Neural_Models_for_News_Summarization.pdf)
- [Speaking script and demo checklist](SPEAKING_SCRIPT.md)
- [Primary references and evidence mapping](REFERENCES.md)
- [Exact demo body](assets/riverton_article.txt) and [saved fallback outputs](assets/saved_demo_outputs.md)
- [Rendered slide images](rendered/) and [visual contact sheet](rendered/contact_sheet.png)
- [Validation record](VALIDATION.md)

The PowerPoint includes speaker notes for all slides. Diagrams, interface
schematic, text and benchmark charts are editable native PowerPoint objects.
The charts contain embedded data sheets populated directly from the saved
benchmark JSON. Slide 4 is explicitly a **schematic**, because no current saved
screenshot was located. Slide 7 quotes exact saved output excerpts; full outputs
are in backups B5/B6 and the script. The source is the latest Riverton diagnostic,
not the older fixture. The original duplicate display remains unresolved.

Use only slides 1–8 for the timed talk. References and B1–B8 are for questions
and fallback. Run the demo checklist before class, reuse the cached environment,
and allow a maximum visible wait of about 30 seconds before using saved outputs.
No inference was run during presentation creation.

## Rebuild without using the app environment

The builder is [build_presentation.py](build_presentation.py), with editable
speaker notes in [speaker_notes.json](speaker_notes.json). It reads repository
evidence and writes only into this folder. Dependencies used were Python 3.12,
`python-pptx==1.0.2`, and `Pillow==11.3.0`, installed in an isolated temporary
environment. The app `.venv` and dependency pins were not altered.

For a later rebuild with `uv` and internet available, create isolated tooling:

```bash
UV_CACHE_DIR=/tmp/summarizer-presentation-uv uv venv --python 3.12 /tmp/summarizer-presentation-tools
UV_CACHE_DIR=/tmp/summarizer-presentation-uv uv pip install --python /tmp/summarizer-presentation-tools/bin/python python-pptx==1.0.2 pillow==11.3.0
/tmp/summarizer-presentation-tools/bin/python presentation/build_presentation.py
```

The layout uses the installed open-source **Lato** font. PDF export preserves the
rendered appearance; another machine's PowerPoint may substitute fonts if Lato
is unavailable. Font size is mostly 22–26 pt body / 34 pt title; diagram labels,
citations and dense backup metadata are smaller. No animations are needed.

Export the real PowerPoint with installed LibreOffice (the actual PDF was made
with LibreOffice 26.8.2.2), then render every PDF page using Poppler:

```bash
libreoffice -env:UserInstallation=file:///tmp/summarizer-presentation-lo --headless --convert-to pdf --outdir presentation presentation/Comparing_Neural_Models_for_News_Summarization.pptx
pdftoppm -scale-to 1600 -png presentation/Comparing_Neural_Models_for_News_Summarization.pdf presentation/rendered/slide
```

The exact app command for the **later classroom demo**, from the repository root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python app.py --port 7860
```

Open <http://127.0.0.1:7860/>. Stop any existing app on that port before relaunch.
This slide-generation task did not start an app or submit a summarization request.
No presentation guideline PDF was located, so this package follows the supplied
brief. There are no content placeholders requiring replacement.
