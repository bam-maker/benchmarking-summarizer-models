"""Lightweight artifact checks only: no model imports, inference or benchmarks."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from pptx import Presentation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PPTX = HERE / "Comparing_Neural_Models_for_News_Summarization.pptx"
PDF = PPTX.with_suffix(".pdf")
NS = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
ORDER = ["LSTM", "Transformer", "BART", "PEGASUS", "Lead-3"]


def main():
    deck = Presentation(PPTX)
    results = json.loads((ROOT / "reports/pilot50/results.json").read_text())["results"]
    notes = json.loads((HERE / "speaker_notes.json").read_text())
    manifest = json.loads((HERE / "assets/slide_manifest.json").read_text())
    assert len(deck.slides) == 18
    assert sum(s["main"] for s in manifest) == 8
    assert abs(deck.slide_width / deck.slide_height - 16 / 9) < .0001
    assert len(notes) == 8 and sum(n["duration_seconds"] for n in notes) == 900
    for i, slide in enumerate(deck.slides):
        embedded = slide.notes_slide.notes_text_frame.text
        assert len(embedded) > 100, (i + 1, "missing notes")
        if i < 8:
            assert notes[i]["notes"] in embedded
        for sh in slide.shapes:
            assert sh.left >= 0 and sh.top >= 0
            assert sh.left + sh.width <= deck.slide_width + 1500, (i + 1, "off-slide shape")
            assert sh.top + sh.height <= deck.slide_height + 1500, (i + 1, "off-slide shape")

    with zipfile.ZipFile(PPTX) as z:
        charts = sorted(n for n in z.namelist() if re.fullmatch(r"ppt/charts/chart\d+\.xml", n))
        assert len(charts) == 3
        assert len([n for n in z.namelist() if n.startswith("ppt/embeddings/") and n.endswith(".xlsx")]) == 3
        for path, key in zip(charts, ["rouge1", "rouge2", "rougeLsum"]):
            root = ET.fromstring(z.read(path))
            series = root.find(".//c:ser", NS)
            cats = [x.text for x in series.findall(".//c:cat//c:pt/c:v", NS)]
            values = [float(x.text) for x in series.findall(".//c:val//c:pt/c:v", NS)]
            assert cats == ORDER
            assert len(values) == 5
            for name, value in zip(ORDER, values):
                assert abs(value - results["full_article"][name][key]) < 1e-10
            assert root.find(".//c:valAx/c:scaling/c:min", NS).attrib["val"] == "0.0"
            assert root.find(".//c:valAx/c:scaling/c:max", NS).attrib["val"] == "50.0"

    with (HERE / "assets/benchmark_values.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10
    for row in rows:
        expected = results[row["track"]][row["system"]]
        for key in ["n", "failed", "rouge1", "rouge2", "rougeL", "rougeLsum", "mean_words"]:
            assert float(row[key]) == expected[key]

    body = (HERE / "assets/riverton_article.txt").read_text()
    assert len(body) == 2202 and len(body.split()) == 315
    assert hashlib.sha256(body.encode()).hexdigest() == "648c2a202c8912f6d1fc5bbe468162d2670be52508916bbbf4ac7fd7e5457fa8"
    pdfinfo = subprocess.check_output(["pdfinfo", str(PDF)], text=True)
    assert re.search(r"Pages:\s+18\b", pdfinfo)
    pdftext = subprocess.check_output(["pdftotext", str(PDF), "-"], text=True)
    normalize = lambda s: re.sub(r"\s+", " ", s).strip()
    assert "Article: The valve was repaired." in normalize(pdftext)
    assert "Bounded consolidation if needed" in normalize(pdftext)
    report = (ROOT / "reports/LAPTOP_MODEL_DIAGNOSIS.md").read_text()
    for value in re.findall(r"^> ([^\n]+)$", report, re.M)[-4:]:
        # Exact complete reproduced LSTM, BART and PEGASUS appear in backups;
        # the fourth captured quote is the shared Transformer string.
        assert normalize(value) in normalize(pdftext), value[:50]
    rendered = sorted((HERE / "rendered").glob("slide-*.png"))
    assert len(rendered) == 18
    assert not any(not v["fits_estimate"] for v in json.loads((HERE / "assets/text_layout_audit.json").read_text()))
    for fn in ["README.md", "SPEAKING_SCRIPT.md", "REFERENCES.md"]:
        path = HERE / fn
        for link in re.findall(r"(?<!!)\[[^]]+\]\(([^)]+)\)", path.read_text()):
            if link.startswith(("http:", "https:", "#")):
                continue
            relative = link.split("#", 1)[0]
            assert not relative or (path.parent / relative).exists(), (fn, link)
    assert not subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).strip()
    print("PASS: 18 slides; 8 main; notes 900 seconds; 16:9; 3 native charts + workbooks.")
    print("PASS: native chart values and both track exports equal saved benchmark JSON.")
    print("PASS: exact demo-body hash; complete saved outputs in PDF; 18 rendered pages.")
    print("PASS: notes embedded; shapes within slide bounds; local links; tracked app/evidence unchanged.")


if __name__ == "__main__":
    main()
