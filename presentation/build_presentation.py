"""Build the editable deck from saved repository evidence; performs no inference.

Tooling: isolated Python with python-pptx==1.0.2 and Pillow==11.3.0.
Run from any directory. See README.md for export and validation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

from PIL import ImageFont
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_DATA_LABEL_POSITION, XL_TICK_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
W, H = 13.333333, 7.5
BG = "0B1528"
PANEL = "14253D"
PANEL2 = "1B304B"
WHITE = "F5F8FD"
MUTED = "B5C5DA"
TEAL = "45D6BD"
AMBER = "FFC465"
BLUE = "81B2FF"
PURPLE = "C2ABFF"
GRAY = "A6B5C8"
MODEL_COLORS = {"LSTM": TEAL, "Transformer": BLUE, "BART": AMBER, "PEGASUS": PURPLE, "Lead-3": GRAY}
FONT = "Lato"
FONT_FILE = "/usr/share/fonts/truetype/lato/Lato-Regular.ttf"
BOLD_FILE = "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"
metrics = json.loads((ROOT / "reports/pilot50/results.json").read_text())
manifest = json.loads((ROOT / "model_manifest.json").read_text())
diagnosis = (ROOT / "reports/LAPTOP_MODEL_DIAGNOSIS.md").read_text()
ORDER = ["LSTM", "Transformer", "BART", "PEGASUS", "Lead-3"]


def report_quote(label, next_label=None):
    chunk = diagnosis.split(label, 1)[1]
    if next_label:
        chunk = chunk.split(next_label, 1)[0]
    else:
        chunk = chunk.split("\n\nThe raw OpenNMT", 1)[0]
    return "\n\n".join(line[2:] for line in chunk.splitlines() if line.startswith("> "))


full_article = report_quote("**Supplied article**", "**Reported displayed LSTM")
body = full_article.split("\n\n", 1)[1]
outputs = {
    "LSTM": report_quote("**Reproduced LSTM direct, pipeline and fresh Gradio route:**", "**Reported and reproduced BART:**"),
    "Transformer": report_quote("**Reported displayed LSTM and Transformer, exactly identical:**", "**Reproduced LSTM direct"),
    "BART": report_quote("**Reported and reproduced BART:**", "**Reported and reproduced PEGASUS display:**"),
    "PEGASUS": report_quote("**Reported and reproduced PEGASUS display:**"),
}
assert len(body) == 2202 and len(body.split()) == 315
assert hashlib.sha256(body.encode()).hexdigest() == "648c2a202c8912f6d1fc5bbe468162d2670be52508916bbbf4ac7fd7e5457fa8"

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
prs.core_properties.title = "Comparing Neural Models for News Summarization"
prs.core_properties.subject = "CIS 634 • saved model integration and evaluation evidence"
prs.core_properties.author = "Biruk Anley and Daniela Ortega"
prs.core_properties.keywords = "news summarization, LSTM, Transformer, CNN/DailyMail, BART, PEGASUS"
text_audit = []
slide_meta = []


def rgb(v):
    return RGBColor.from_string(v)


def shape(slide, x, y, w, h, fill=PANEL, line=None, radius=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1.3)
    else:
        s.line.fill.background()
    if radius:
        s.adjustments[0] = 0.12
    return s


def text(slide, value, x, y, w, h, size=24, color=WHITE, bold=False, align=None,
         margin=0, valign=MSO_ANCHOR.TOP, font=FONT, check=True, spacing=1.08):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    for idx, line in enumerate(value.split("\n")):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = font
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = rgb(color)
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        p.line_spacing = spacing
        if align is not None:
            p.alignment = align
    if check:
        # Conservative line wrapping estimate; exported PDF is inspected too.
        ft = ImageFont.truetype(BOLD_FILE if bold else FONT_FILE, size=max(1, round(size * 2)))
        maxpx = max(1, (w - 2 * margin) * 72 * 2)
        count = 0
        for para in value.split("\n"):
            words = para.split()
            line = ""
            n = 1
            for word in words:
                candidate = (line + " " + word).strip()
                if ft.getlength(candidate) > maxpx and line:
                    n += 1
                    line = word
                else:
                    line = candidate
            count += n
        estimated = count * size * 1.10 / 72
        text_audit.append({"slide": len(prs.slides), "text": value[:70], "font_pt": size,
                           "height_inches": h, "estimated_text_height": round(estimated, 3),
                           "fits_estimate": estimated <= h + 0.05})
    return box


def line(slide, x1, y1, x2, y2, color=MUTED, width=1.8, arrow=False, dash=False):
    s = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    s.line.color.rgb = rgb(color)
    s.line.width = Pt(width)
    if arrow:
        ln = s._element.spPr.get_or_add_ln()
        end = OxmlElement("a:tailEnd")
        end.set("type", "triangle")
        ln.append(end)
    if dash:
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        s.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return s


def route(slide, points, color=MUTED, width=1.8, dash=False):
    for i, (a, b) in enumerate(zip(points, points[1:])):
        line(slide, *a, *b, color=color, width=width, arrow=i == len(points) - 2, dash=dash)


def pill(slide, value, x, y, w, color=TEAL, size=16, h=.34):
    shape(slide, x, y, w, h, fill=PANEL2)
    text(slide, value, x + .10, y + .035, w - .20, h - .04, size, color, bold=True)


def add_notes(slide, value):
    slide.notes_slide.notes_text_frame.text = value


def base(title, section, n, source, main=True, subtitle=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(BG)
    shape(slide, .56, .35, .10, .33, TEAL, radius=False)
    text(slide, section.upper(), .79, .35, 11.8, .32, 14, TEAL, bold=True)
    title_h = 1.10 if "\n" in title else .60
    text(slide, title, .58, .86, 12.15, title_h, 34, bold=True)
    if subtitle:
        text(slide, subtitle, .60, 1.53 if title_h == .60 else 1.99, 12.1, .4, 20, MUTED)
    line(slide, .6, 7.02, 12.73, 7.02, color=PANEL2, width=1)
    text(slide, source, .6, 7.12, 11.6, .23, 11.5, MUTED, check=False)
    text(slide, f"{n:02d}" if main else str(n), 12.12, 7.10, .6, .25, 13, TEAL,
         align=PP_ALIGN.RIGHT, check=False)
    slide_meta.append({"slide": len(prs.slides), "title": title.replace("\n", " "), "main": main, "source": source})
    return slide


def model_label(slide, name, x, y, size=21, w=2):
    shape(slide, x, y+.05, .10, .28, MODEL_COLORS[name], radius=False)
    text(slide, name, x+.20, y, w-.2, .4, size, MODEL_COLORS[name], bold=True)


def panel_title(slide, label, x, y, w, color=TEAL):
    text(slide, label, x, y, w, .42, 22, color, bold=True)


# 01 — Problem, goals and project identity.
s = base("Comparing Neural Models\nfor News Summarization", "CIS 634 • project overview", 1,
         "Biruk Anley + Daniela Ortega • Repository: PROJECT_STATUS.md")
text(s, "Biruk Anley  •  Daniela Ortega", .63, 2.02, 11.9, .40, 22, MUTED)
shape(s, .62, 2.78, 4.02, 2.22)
panel_title(s, "NEWS ARTICLE", .86, 3.00, 3.6)
for i, ww in enumerate([3.18, 3.35, 2.90, 3.27, 2.15]):
    shape(s, .88, 3.59+i*.21, ww, .065, MUTED if i != 0 else TEAL, radius=False)
shape(s, 5.04, 3.54, 1.09, .52, TEAL, radius=False)
text(s, "→", 5.12, 3.47, .9, .68, 36, BG, bold=True, align=PP_ALIGN.CENTER)
shape(s, 6.51, 2.78, 6.17, 2.22)
panel_title(s, "SHORT SUMMARY", 6.78, 3.0, 5.5)
text(s, "Keep the main event.\nKeep the facts connected.", 6.78, 3.63, 5.55, 1.0, 29, bold=True)
text(s, "OUR GOAL", .67, 5.54, 1.6, .35, 16, TEAL, bold=True)
text(s, "Compare models + build a usable local demo", .67, 6.01, 11.9, .50, 26)

# 02 — Core models and the paired data idea.
s = base("Two core models learn from article–summary pairs", "Core models + data", 2,
         "Sources: OpenNMT summarization recipe; Nallapati et al. (2016); dataset card; manifests")
for name, xx, desc, size in [("LSTM",.65,"Recurrent encoder–decoder\nwith attention and copying","43M"),
                            ("Transformer",6.82,"Encoder–decoder attention\nwith copying","81M")]:
    shape(s,xx,1.74,5.86,1.42)
    model_label(s,name,xx+.22,1.94,25,w=3.8)
    text(s,size,xx+4.30,1.92,1.25,.51,29,MODEL_COLORS[name],True,PP_ALIGN.RIGHT)
    text(s,desc,xx+.22,2.43,5.35,.70,20,MUTED)
text(s,"CNN/DAILYMAIL",.66,3.55,5,.4,22,TEAL,True)
pill(s,"ILLUSTRATIVE PAIR • not a dataset row",6.5,3.54,6.15,size=15)
shape(s,.65,4.12,5.57,1.35)
text(s,"Article: The valve was repaired.\nTwo clear tests followed.\nOfficials lifted the advisory.",.85,4.28,5.16,1.15,22)
line(s,6.35,4.71,6.85,4.71,TEAL,2.2,True)
shape(s,7.0,4.12,5.65,1.35)
text(s,"Reference: Officials lifted the advisory\nafter repairs and clear tests.",7.23,4.32,5.18,.83,22)
text(s,"Released model training",.72,5.76,5.4,.4,22,TEAL,True)
text(s,"CNN/DM summarization provenance",.72,6.21,5.7,.4,21,MUTED)
text(s,"Our evaluation",7.04,5.76,5.3,.4,22,AMBER,True)
text(s,"50 held-out test articles • version 3.0.0",7.04,6.21,5.7,.4,21,MUTED)

# 03 — Editable architecture diagrams, specific to these checkpoints.
s = base("Two routes from article to summary", "Actual OpenNMT checkpoints", 3,
         "Sources: checkpoint metadata; legacy/onmt encoders, decoders and CopyGenerator; LSTM / Transformer papers")
model_label(s,"LSTM",.66,1.62,24,2.3)
text(s,"≈43M parameters",10.0,1.64,2.64,.4,20,TEAL,align=PP_ALIGN.RIGHT)
for x,y,w,h,label in [(.65,2.30,.95,.88,"Words"),(1.85,2.30,1.45,.88,"Word\nembeddings"),
                      (3.40,2.30,1.72,.88,"Bidirectional\nLSTM encoder"),(5.43,2.30,1.40,.88,"Attention"),
                      (7.16,2.30,1.75,.88,"Recurrent\nLSTM decoder"),(9.27,2.30,1.60,.88,"Vocabulary /\ncopy mixture"),
                      (11.21,2.30,1.46,.88,"Summary")]:
    shape(s,x,y,w,h,line=TEAL if x in [3.4,7.16] else None)
    text(s,label,x+.07,y+.13,w-.14,h-.15,17.5,align=PP_ALIGN.CENTER,valign=MSO_ANCHOR.MIDDLE)
for a,b in [(1.60,1.85),(3.30,3.40),(5.12,5.43),(6.83,7.16),(8.91,9.27),(10.87,11.21)]:
    line(s,a,2.74,b,2.74,TEAL,2,True)
route(s,[(7.34,2.30),(7.34,2.02),(6.12,2.02),(6.12,2.30)],TEAL,1.5)
text(s,"decoder query",5.38,1.76,1.96,.25,13,TEAL)
route(s,[(8.44,2.30),(8.44,1.99),(7.58,1.99),(7.58,2.30)],TEAL,1.5)
text(s,"state",8.67,1.97,.7,.28,13,TEAL)
route(s,[(6.13,3.18),(6.13,3.55),(10.05,3.55),(10.05,3.18)],TEAL,1.4,True)
text(s,"attention weights → source copying",6.48,3.59,4.9,.30,15,TEAL)
model_label(s,"Transformer",.66,4.15,24,3.9)
text(s,"≈81M parameters",10.0,4.17,2.64,.4,20,BLUE,align=PP_ALIGN.RIGHT)
for x,y,w,h,label in [(.65,5.07,.95,.97,"Words"),(1.85,5.07,1.65,.97,"Embeddings\n+ positions"),
                      (3.84,5.07,2.02,.97,"Encoder\nself-attention"),(9.27,5.07,1.60,.97,"Vocabulary /\ncopy mixture"),
                      (11.21,5.07,1.46,.97,"Summary")]:
    shape(s,x,y,w,h,line=BLUE if x==3.84 else None)
    text(s,label,x+.07,y+.13,w-.14,h-.15,18,align=PP_ALIGN.CENTER,valign=MSO_ANCHOR.MIDDLE)
shape(s,6.30,4.89,2.54,1.40,line=BLUE)
text(s,"DECODER",6.43,4.98,2.28,.27,13,BLUE,True,PP_ALIGN.CENTER)
text(s,"Masked self-attn\nCross-attention",6.43,5.35,2.28,.77,18.5,align=PP_ALIGN.CENTER)
for a,b in [(1.6,1.85),(3.50,3.84),(5.86,6.30),(8.84,9.27),(10.87,11.21)]:
    line(s,a,5.67,b,5.67,BLUE,2,True)
route(s,[(7.55,6.29),(7.55,6.65),(10.05,6.65),(10.05,6.04)],BLUE,1.4,True)
text(s,"cross-attention weights → copy",8.04,6.64,4.5,.30,14.5,BLUE)
route(s,[(6.30,4.94),(5.95,4.94),(5.95,4.69),(6.70,4.69),(6.70,4.89)],BLUE,1.4)
text(s,"previous summary tokens",3.6,4.50,2.9,.30,13.5,MUTED)

# 04 — Honest editable schematic and actual branching logic.
s = base("A local app handles short and long articles", "Application", 4,
         "Sources: app.py; article_summarizer.py; PROJECT_STATUS.md • Interface is an editable schematic")
pill(s,"INTERFACE SCHEMATIC • based on current app",.65,1.68,7.27,size=15)
shape(s,.65,2.20,7.28,3.57,fill="EEF4F7")
text(s,"News Summarizer",.91,2.42,5,.40,23,BG,True)
shape(s,.92,3.01,3.15,.45,fill="FFFFFF",line="C1D0DA")
text(s,"Model:  LSTM   ▾",1.04,3.08,2.9,.31,17,BG)
for xx,label in [( .92,"Article"),(4.42,"Summary")]:
    text(s,label,xx,3.67,3.05,.33,19,BG,True)
    shape(s,xx,4.09,3.19,.78,fill="FFFFFF",line="C1D0DA")
    for i,ww in enumerate([2.69,2.39,1.91]):
        shape(s,xx+.15,4.27+i*.14,ww,.045,"8B9CA9",radius=False)
shape(s,.92,5.08,2.0,.40,fill="0B7767")
text(s,"Summarize",1.04,5.14,1.76,.29,16,"FFFFFF",True,PP_ALIGN.CENTER)
text(s,"Clear",3.23,5.14,.9,.3,16,BG)
text(s,"Copy",6.63,5.14,.9,.3,16,BG)
text(s,"Output words  •  Shorter by  •  Time  •  tok/s",4.42,4.94,3.18,.25,10.2,"506572",check=False)
text(s,"No length selector",.92,5.98,6.7,.44,24,TEAL,True)
shape(s,8.52,1.82,3.55,.61)
text(s,"Fits the input window?",8.69,1.96,3.2,.36,21,align=PP_ALIGN.CENTER)
route(s,[(9.10,2.43),(9.10,2.84),(8.57,2.84)],TEAL,1.7)
text(s,"YES",8.37,2.49,.8,.3,14,TEAL,True)
shape(s,8.24,2.94,1.84,.88)
text(s,"Direct\nmodel pass",8.37,3.07,1.58,.61,20,align=PP_ALIGN.CENTER)
route(s,[(11.55,2.43),(11.55,2.84),(11.90,2.84)],AMBER,1.7)
text(s,"NO",11.42,2.49,.7,.3,14,AMBER,True)
shape(s,10.62,2.94,2.06,.88)
text(s,"Sentence-aware\nchunks",10.72,3.07,1.86,.61,18.5,align=PP_ALIGN.CENTER)
line(s,11.65,3.82,11.65,4.10,AMBER,1.7,True)
shape(s,10.62,4.12,2.06,.91)
text(s,"Merge +\nconsolidation",10.72,4.27,1.86,.67,17.5,align=PP_ALIGN.CENTER)
route(s,[(9.16,3.82),(9.16,5.40),(10.45,5.40)],TEAL,1.7)
route(s,[(11.65,5.03),(11.65,5.40),(10.45,5.40)],TEAL,1.7)
text(s,"Completed summary",8.55,5.68,4.15,.39,22,TEAL,True)
text(s,"Bounded consolidation if needed",8.45,6.14,4.24,.30,15,MUTED)
text(s,"Offline CPU checks passed • Complete input ≠ complete facts",.68,6.49,12,.39,22,MUTED)

# 05 — Comparison systems, tracks and metric language.
s = base("Five systems, two evaluation tracks", "Evaluation design", 5,
         "Sources: reports/pilot50/results.json; BENCHMARK_RESULTS.md; BART / PEGASUS cards; Lin (2004)")
text(s,"50 CNN/DailyMail 3.0.0 test articles  •  fixed seed 42",.67,1.67,12,.43,24,TEAL,True)
for name,x,subtitle,desc in [("BART",.65,"406M learned parameters","Broad denoising pretraining\n→ CNN/DM fine-tuning"),
                            ("PEGASUS",4.78,"569M learned parameters","Gap-sentence pretraining\n→ CNN/DM fine-tuning"),
                            ("Lead-3",8.90,"Extractive baseline","First three source sentences\n→ no model generation")]:
    shape(s,x,2.35,3.78,1.78)
    model_label(s,name,x+.20,2.52,24,w=3.39)
    text(s,subtitle,x+.20,3.01,3.36,.33,18,MODEL_COLORS[name])
    text(s,desc,x+.20,3.45,3.36,.59,18.5,MUTED)
panel_title(s,"FULL ARTICLE",.69,4.57,5.6,TEAL)
text(s,"Each app processes the full source.",.69,5.05,5.7,.67,23)
panel_title(s,"SHARED PREFIX",.69,5.83,5.6,BLUE)
text(s,"Same bounded text; original full references.",.69,6.30,5.7,.50,21,MUTED)
for label,xx,yy,desc in [("ROUGE-1",7.0,4.61,"Shared words"),("ROUGE-2",9.92,4.61,"Word pairs"),
                         ("ROUGE-L",7.0,5.65,"Ordered sequence"),("ROUGE-Lsum",9.92,5.65,"Sentence-aware LCS")]:
    text(s,label,xx,yy,2.75,.35,21,AMBER,True)
    text(s,desc,xx,yy+.43,2.75,.35,18.5,MUTED)
text(s,"Overlap ≠ factual accuracy",7.0,6.53,5.72,.35,21,WHITE,True)

# 06 — Three native editable charts, honest zero-based identical axes.
s = base("BART leads most overlap metrics in this sample", "Saved benchmark • full-article track", 6,
         "Source: reports/pilot50/results.json • F1 × 100, Porter stemming • n=50 per system")
text(s,"Full-article track • higher reference overlap is better",.68,1.66,12,.4,23,MUTED)
for row,name in enumerate(ORDER):
    model_label(s,name,.68,3.23+row*.646,20,w=1.84)
for j,(key,label) in enumerate([("rouge1","ROUGE-1"),("rouge2","ROUGE-2"),("rougeLsum","ROUGE-Lsum")]):
    x=2.40+j*3.47
    text(s,label,x+.05,2.30,3.32,.40,23,WHITE,True,PP_ALIGN.CENTER)
    data=CategoryChartData()
    data.categories=ORDER
    data.add_series(label,[metrics['results']['full_article'][n][key] for n in ORDER])
    chart=s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED,Inches(x),Inches(2.76),Inches(3.34),Inches(3.65),data).chart
    chart.has_legend=False
    chart.chart_style=2
    chart.chart_title.has_text_frame=False
    chart.has_title=False
    chart.font.name=FONT
    chart.font.size=Pt(17)
    chart.font.color.rgb=rgb(MUTED)
    chart.chart_style=2
    # Transparent chart and plot backgrounds retain navy in PowerPoint and PDF.
    for parent in [chart._chartSpace, chart._chartSpace.chart.plotArea]:
        spPr=parent.find('{http://schemas.openxmlformats.org/drawingml/2006/chart}spPr')
        if spPr is None:
            spPr=OxmlElement('c:spPr'); parent.append(spPr)
        nofill=OxmlElement('a:noFill'); spPr.append(nofill)
        ln=OxmlElement('a:ln'); ln.append(OxmlElement('a:noFill')); spPr.append(ln)
    cat=chart.category_axis
    cat.reverse_order=True
    cat.tick_label_position=XL_TICK_LABEL_POSITION.NONE
    cat.major_tick_mark=XL_TICK_MARK.NONE
    cat.minor_tick_mark=XL_TICK_MARK.NONE
    cat.format.line.fill.background()
    val=chart.value_axis
    val.minimum_scale=0
    val.maximum_scale=50
    val.major_unit=25
    val.has_major_gridlines=True
    val.major_gridlines.format.line.color.rgb=rgb(PANEL2)
    val.major_gridlines.format.line.width=Pt(.7)
    val.major_tick_mark=XL_TICK_MARK.NONE
    val.minor_tick_mark=XL_TICK_MARK.NONE
    val.format.line.fill.background()
    val.tick_labels.font.name=FONT
    val.tick_labels.font.size=Pt(14)
    val.tick_labels.font.color.rgb=rgb(MUTED)
    plot=chart.plots[0]
    plot.gap_width=55
    plot.has_data_labels=True
    dl=plot.data_labels
    dl.position=XL_DATA_LABEL_POSITION.OUTSIDE_END
    dl.number_format='0.00'
    dl.font.name=FONT
    dl.font.size=Pt(16)
    dl.font.bold=True
    dl.font.color.rgb=rgb(WHITE)
    for i,name in enumerate(ORDER):
        point=chart.series[0].points[i]
        point.format.fill.solid()
        point.format.fill.fore_color.rgb=rgb(MODEL_COLORS[name])
        point.format.line.fill.background()
text(s,"Small sample. No significance claim or factual-quality winner.",.68,6.57,12,.35,22,AMBER)

# 07 — Exact saved case comparison and live-demo bridge.
s=base("A good score can still miss the headline", "Fictional Riverton diagnostic + live demo",7,
       "Source: reports/LAPTOP_MODEL_DIAGNOSIS.md • separate from the benchmark • saved outputs")
shape(s,.66,1.70,12.01,.83,fill=PANEL2)
text(s,"SOURCE MAIN EVENT",.85,1.85,2.39,.25,14,TEAL,True)
text(s,"“Riverton officials lifted a boil-water advisory Friday morning”",3.0,1.86,9.35,.42,23,TEAL,True)
demo_excerpts={
    'BART': 'Officials said the advisory was precautionary; contamination had not been confirmed. Two consecutive rounds of testing found no harmful bacteria.',
    'PEGASUS': 'Riverton officials lifted a boil-water advisory Friday morning. The advisory had affected approximately 18,000 residents since Tuesday.',
}
for name,x in [('BART',.66),('PEGASUS',6.83)]:
    model_label(s,name,x,2.84,25,w=5.8)
    shape(s,x,3.36,5.84,2.48)
    text(s,"EXACT SAVED OUTPUT EXCERPT",x+.22,3.55,5.40,.29,14,MODEL_COLORS[name],True)
    text(s,demo_excerpts[name],x+.22,3.98,5.40,1.80,22,spacing=1.0)
text(s,"Supported facts; resolution omitted",.72,6.05,5.8,.45,21,AMBER,True)
text(s,"Lifting captured; repairs/tests omitted",6.88,6.05,5.8,.45,21,PURPLE,True)
text(s,"“All the details… except the headline.”",.72,6.56,9.4,.32,19,MUTED)
pill(s,"LIVE • ONE MODEL",10.15,6.52,2.45,TEAL,size=14)

# 08 — Conclusions grounded in limits and implemented work.
s=base("We built a usable demo; factual review comes next", "Limitations + conclusion",8,
       "Sources: MODEL_CAPABILITIES.md; PROJECT_STATUS.md; BENCHMARK_RESULTS.md; laptop diagnosis")
for x,num,title,desc in [(.68,'01','Content selection','LSTM and Transformer can omit\nthe main event or confuse relationships.'),
                         (.68,'02','Long articles','400-token training convention;\nlong chunks can lose connections.')]:
    yy=1.92 if num=='01' else 4.04
    text(s,num,x,yy,1.0,.6,35,TEAL,True)
    text(s,title,x+1.13,yy+.02,4.6,.45,26,WHITE,True)
    text(s,desc,x+1.13,yy+.65,5.00,1.08,23,MUTED)
shape(s,7.31,1.91,5.32,4.65)
panel_title(s,"WHAT WE DELIVERED",7.61,2.20,4.72,TEAL)
text(s,"A four-model local app\nA reproducible 50-article comparison\nAn offline Ubuntu demo",7.61,2.85,4.72,1.72,25)
panel_title(s,"NEXT: HUMAN REVIEW",7.61,4.91,4.72,AMBER)
text(s,"Check main events, support and\nrelationships against each source.",7.61,5.49,4.72,.85,22,MUTED)
text(s,"All four can miss facts. Different complete systems → no architecture-only verdict.",.70,6.64,12,.31,18.5,MUTED)

# References — outside the eight main slides.
refs1=[
    ("LSTM", "Hochreiter & Schmidhuber (1997)\nLong Short-Term Memory · Neural Computation 9(8), 1735–1780", "https://doi.org/10.1162/neco.1997.9.8.1735"),
    ("TRANSFORMER", "Vaswani et al. (2017)\nAttention Is All You Need · NeurIPS 30", "https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html"),
    ("COPY MIXTURE", "See, Liu & Manning (2017)\nGet To The Point · ACL, 1073–1083", "https://aclanthology.org/P17-1099/"),
    ("BART", "Lewis et al. (2020)\nBART: Denoising Sequence-to-Sequence Pre-training · ACL", "https://aclanthology.org/2020.acl-main.703/"),
    ("PEGASUS", "Zhang et al. (2020)\nPEGASUS: Pre-training with Extracted Gap-sentences · ICML", "https://proceedings.mlr.press/v119/zhang20ae.html"),
]
s=base("References • model foundations", "References • outside timed deck",'R1',
       "Primary publications • complete URLs in speaker notes and REFERENCES.md",main=False)
for i,(tag,cite,url) in enumerate(refs1):
    yy=1.75+i*.97
    pill(s,tag,.68,yy,2.12,size=14)
    tb=text(s,cite,3.06,yy-.01,9.54,.76,20.5)
    for p in tb.text_frame.paragraphs:
        for r in p.runs:
            r.font.color.rgb=rgb(WHITE)
            r.font.underline=False
add_notes(s,"Reference slide, outside the 15-minute talk.\n\n"+'\n\n'.join(f'{a}: {b}\n{c}' for a,b,c in refs1)+"\n\nSee et al. is related copy-mixture literature, not a claim that these OpenNMT checkpoints implement their exact learned coverage model.")
refs2=[
    ("DATASET", "Hermann et al. (2015) · Nallapati et al. (2016)\nDataset origins, summarization task and version 3.0.0 card", "https://huggingface.co/datasets/abisee/cnn_dailymail"),
    ("METRICS", "Lin (2004) · ROUGE\nGoogle Research implementation: flat L vs sentence-aware Lsum", "https://aclanthology.org/W04-1013/"),
    ("CHECKPOINTS", "Official OpenNMT summarization releases\nfacebook/bart-large-cnn · google/pegasus-cnn_dailymail", "https://opennmt.net/OpenNMT-py/examples/summary/Summarization.html"),
    ("OUR EVIDENCE", "PROJECT_STATUS.md · reports/pilot50/results.json\nreports/LAPTOP_MODEL_DIAGNOSIS.md · manifests and adapter code", "https://github.com/bam-maker/benchmarking-summarizer-models/tree/0a549a7197fab099b7ef8711e28e358c4b62a4b1"),
]
s=base("References • data, metrics and project evidence", "References • outside timed deck",'R2',
       "Original sources and frozen repository evidence • details in REFERENCES.md",main=False)
for i,(tag,cite,url) in enumerate(refs2):
    yy=1.85+i*1.17
    pill(s,tag,.68,yy,2.12,size=14)
    tb=text(s,cite,3.06,yy-.01,9.54,1.03,21)
    for p in tb.text_frame.paragraphs:
        for r in p.runs:
            r.font.color.rgb=rgb(WHITE)
            r.font.underline=False
add_notes(s,"References, outside the timed talk.\n\n"+'\n\n'.join(f'{a}: {b}\n{c}' for a,b,c in refs2)+"\n\nDataset origins: https://proceedings.neurips.cc/paper_files/paper/2015/hash/afdec7005cc9f14302cd0474fd0f3c96-Abstract.html\nSummarization task: https://aclanthology.org/K16-1028/\nROUGE implementation: https://github.com/google-research/google-research/tree/master/rouge")


def results_slide(track,title,num):
    s=base(title,"Backup • complete metrics",num,"Source: reports/pilot50/results.json • F1 × 100, Porter stemming • each n=50, failed=0",main=False)
    headers=["System","R-1","R-2","R-L","R-Lsum","Mean words"]
    xpos=[.82,3.55,5.26,6.97,8.68,10.4]
    widths=[2.6,1.5,1.5,1.5,1.5,2.0]
    shape(s,.67,1.86,12.0,.69,fill=PANEL2,radius=False)
    for x,w,h in zip(xpos,widths,headers): text(s,h,x,2.03,w,.4,21,WHITE,True)
    for i,name in enumerate(ORDER):
        yy=2.84+i*.66
        shape(s,.67,yy-.10,12,.60,fill=PANEL if i%2==0 else BG,radius=False)
        model_label(s,name,xpos[0],yy,22,w=2.6)
        row=metrics['results'][track][name]
        for x,w,k in zip(xpos[1:],widths[1:],["rouge1","rouge2","rougeL","rougeLsum","mean_words"]):
            v=f"{row[k]:.2f}" if k!='mean_words' else f"{row[k]:.1f}"
            text(s,v,x,yy,w,.4,23)
    caution="PEGASUS narrowly leads flat R-L; BART leads R-1, R-2 and R-Lsum." if track=='full_article' else "Full references remain the target; some reference facts can lie beyond the prefix."
    text(s,caution,.80,6.49,11.9,.4,21,AMBER)
    add_notes(s,"Backup table. Values are loaded directly from reports/pilot50/results.json, without recomputation of ROUGE.\nAll five systems use the same 50 articles. Track: "+track+".\nNo significance test or human factuality ratings were measured.\nThe interrupted 1,000-article study remains incomplete. No benchmark timings are compared because the saved GPU context was contaminated by desktop/game activity.")
    return s

results_slide('full_article','Full-article benchmark • all four metrics','B1')
results_slide('common_input','Shared-prefix benchmark • separate track','B2')

# Checkpoint identity backup.
s=base("Which released checkpoints did we integrate?","Backup • provenance",'B3',
       "Sources: checkpoint_manifest.json; model_manifest.json; MODEL_CAPABILITIES.md",main=False)
cards=[('LSTM','42,789,205 learned','ada6_bridge_oldcopy_tagged_\nacc_54.17_ppl_11.17_e20','OpenNMT CNN/DM summarization'),
       ('Transformer','81,468,757 learned','sum_transformer_model_\nacc_57.25_ppl_9.22_e16','OpenNMT CNN/DM summarization'),
       ('BART','406,290,432 learned','facebook/bart-large-cnn','rev 37f520fa929c961707657b28798b30c003dd100b'),
       ('PEGASUS','568,699,904 learned','google/pegasus-cnn_dailymail','rev 40d588fdab0cc077b80d950b300bf66ad3c75b92')]
for i,(name,count,ck,rev) in enumerate(cards):
    x=.67+(i%2)*6.17; y=1.77+(i//2)*2.44
    shape(s,x,y,5.83,2.18)
    model_label(s,name,x+.21,y+.18,25,w=5.4)
    text(s,count,x+.22,y+.67,5.35,.36,20,MODEL_COLORS[name])
    text(s,ck,x+.22,y+1.15,5.35,.70,19)
    text(s,rev,x+.22,y+1.91,5.35,.22,11.7,MUTED,check=False)
text(s,"PEGASUS: 570,797,056 instantiated; 2,097,152 fixed positional parameters.",.77,6.72,12,.24,17,MUTED)
add_notes(s,"All four are released trained checkpoints integrated into this app. We did not train or fine-tune them.\nThe two OpenNMT model names refer to official releases with SHA-256 verification in OpenNMT_Verified_Setup/checkpoint_manifest.json.\nBART is denoising-pretrained then CNN/DM fine-tuned.\nPEGASUS's released card describes updated Mixed & Stochastic pretraining on C4 and HugeNews, followed by CNN/DM fine-tuning. Its fixed sinusoidal position tables account for 2,097,152 instantiated but non-learned parameters.\nExact full revisions are shown for recovery, not intended to be read aloud.")

# Decoding backup.
s=base("The demo preserves each checkpoint's search policy","Backup • decoding and input windows",'B4',
       "Sources: current summarizer.py; hf_summarizer.py; article_summarizer.py; processing_config.json",main=False)
headers=["System","Beam","Initial cap","Search details"]
xs=[.81,3.12,4.52,7.23]; ws=[2.1,1.2,2.45,5.1]
shape(s,.67,1.80,12,.7,fill=PANEL2,radius=False)
for x,w,h in zip(xs,ws,headers): text(s,h,x,1.99,w,.4,22,WHITE,True)
rows=[('LSTM','5','100','α=.9; coverage β=5; trigram block'),('Transformer','5','100','Same OpenNMT search recipe'),
      ('BART','4','141 new','Length penalty 2; no-repeat 3'),('PEGASUS','8','127 new','Length penalty .8; no-repeat 0')]
for i,row in enumerate(rows):
    yy=2.87+i*.7
    model_label(s,row[0],xs[0],yy,22,w=2.1)
    for x,w,val in zip(xs[1:],ws[1:],row[1:]): text(s,val,x,yy,w,.47,20.5)
text(s,"Natural EOS • deterministic search • bounded retries if capped",.80,6.01,11.6,.4,23,TEAL,True)
text(s,"App work window: 1,024 tokens per chunk; historical OpenNMT training: 400.",.80,6.52,11.6,.4,20,MUTED)
add_notes(s,"Current natural-checkpoint-v1 only. OpenNMT initial minimum is min(35, input tokens//8, cap-2). Native HF floor is56 for BART and32 for PEGASUS, bounded by input tokens//8; the adapter subtracts1 for min_new_tokens.\nOpenNMT beam5, stepwise penalty, Wu length alpha .9, summary coverage beta5, trigram repeat block with punctuation/sentence tags exempt. This is search coverage, not learned coverage attention.\nBART early_stopping=True. PEGASUS early_stopping=False. do_sample=False and forced_eos_token_id=None for HF. Hard caps256 OpenNMT and512 HF.\nFor latest body:359/359/405/368 input tokens; generated54/45/59/46 excluding EOS, all natural EOS before caps.\nHF special tokens count in their1024 positions. Historical OpenNMT400 training truncation is not a recurrent architecture ceiling. Transformer saved positions5000 are not an empirically validated quality range.")

# Saved core outputs: truthful fallback and mismatch context.
s=base("Saved fallback • distinct core-model outputs","Backup • exact latest Riverton outputs",'B5',
       "Source: reports/LAPTOP_MODEL_DIAGNOSIS.md • body-only input • no new generation",main=False)
for name,x in [('LSTM',.67),('Transformer',6.84)]:
    model_label(s,name,x,1.85,25,w=5.8)
    shape(s,x,2.46,5.83,3.85)
    text(s,outputs[name],x+.22,2.69,5.39,3.55,22)
text(s,"Original duplicate display: unresolved. Fresh direct and Gradio LSTM differed.",.77,6.32,11.85,.55,22,AMBER)
add_notes(s,"These are the reproduced distinct outputs from the latest article, not the old fixture.\nThe user-supplied original LSTM and Transformer display strings were exactly identical. Only Transformer reproduced that shared string. Fresh LSTM direct, pipeline and live Gradio route produced the schools sentence shown here.\nModel dispatch, checkpoint classes and cache keys checked correctly; the original browser selection and response were not retained. Do not claim a confirmed routing bug, nor a proven user error.\nBoth omit the advisory lifting. Exact case and raw cleanup evidence are in reports/LAPTOP_MODEL_DIAGNOSIS.md.")

s=base("Saved fallback • BART and PEGASUS","Backup • exact latest Riverton outputs",'B6',
       "Source: reports/LAPTOP_MODEL_DIAGNOSIS.md • fictional diagnostic; separate from benchmark",main=False)
for name,x in [('BART',.67),('PEGASUS',6.84)]:
    model_label(s,name,x,1.85,25,w=5.8)
    shape(s,x,2.46,5.83,4.04)
    text(s,outputs[name],x+.22,2.69,5.39,3.66,21,spacing=1.0)
text(s,"Raw BART omits lifting; PEGASUS includes it. One example does not rank general quality.",.77,6.66,11.85,.30,19,AMBER)
add_notes(s,"Exact saved displayed outputs, suitable if live inference is slow. BART facts are supported, but advisory lifted and completed repairs are absent. It includes two consecutive test rounds with no harmful bacteria, and does not explicitly say the warning remains active.\nPEGASUS clearly includes lifted Friday, affected residents and valve failure; it omits completed repair and testing. Neither output has an identified unsupported claim, confused entity or repetition in this case.\nAll input was processed in one chunk; BART generated59 and PEGASUS46 tokens then natural EOS. Cleanup did not remove the lifting sentence from BART.")

s=base("The source resolves the advisory explicitly","Backup • source facts for the demo",'B7',
       "Source: exact latest article in reports/LAPTOP_MODEL_DIAGNOSIS.md; full body in assets/riverton_article.txt",main=False)
for i,(tag,quote) in enumerate([
    ('MAIN EVENT',body.split('\n\n')[0].split('. The advisory')[0]+'.'),
    ('REPAIR', 'Repair crews replaced the damaged valve Tuesday evening.'),
    ('TESTING','Two consecutive rounds of testing found no harmful bacteria, allowing the town to lift the advisory.')]):
    y=1.85+i*1.52
    pill(s,tag,.70,y,2.12,size=14)
    text(s,quote,3.08,y-.01,9.42,1.22,25,TEAL if i==0 else WHITE)
text(s,"315-word body • title inclusion originally uncertain; log counts match body only.",.77,6.63,11.9,.3,19,MUTED)
add_notes(s,"These are exact source sentences from the latest user-supplied fictional Riverton article. Main event, repair completion and two clear testing rounds are separate factual checkpoints.\nThe log had315words and2202characters, exactly matching the supplied body; adding the title gives321words and2249characters. Title exclusion is strongly inferred, not text captured by the original log.\nThe full exact input body is included in presentation/assets/riverton_article.txt for the later live demo. No model was run during slide creation.")

s=base("Live-demo checklist • stay within the time box","Backup • Ubuntu offline demo",'B8',
       "Sources: README.md; PROJECT_STATUS.md; laptop diagnosis • all preparation below is for the later class demo",main=False)
text(s,"From the repository root, with the cached setup:",.77,1.82,11.8,.47,24,TEAL,True)
shape(s,.70,2.49,11.95,1.04)
text(s,"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n.venv/bin/python app.py --port 7860",.91,2.70,11.55,.72,21,font="DejaVu Sans Mono")
text(s,"One shell line; open http://127.0.0.1:7860/",.80,3.75,11.6,.46,23,MUTED)
for i,txt in enumerate(["Paste the saved body; identify the lifted-advisory sentence.","Submit at most one model; narrate while the CPU works.","After ~30 seconds, use B5/B6 saved outputs and continue."]):
    yy=4.49+i*.65
    text(s,str(i+1),.81,yy,.44,.41,25,TEAL,True)
    text(s,txt,1.48,yy,10.83,.46,23)
add_notes(s,"The displayed command is visually wrapped into two lines for readability; execute it as ONE shell line exactly:\nHF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python app.py --port 7860\n\nStop an existing app on7860 before relaunch. Do not reinstall or download during class. Prepared laptop resources were already checked offline.\nUse assets/riverton_article.txt without the title, preserve punctuation/case. If useful, prepare a result before class and clearly label it saved. Do not promise the current session is already running.\nDuring the15-minute talk generate at most one model if desired; cap the visible wait at about30seconds and switch to saved exact outputs B5/B6. Laptop cold loading and other RAM use can add delay. Do not interpret speed from the old GPU benchmark.")

# Embed the full main-script notes supplied by the evidence author.
notes_path=HERE/'speaker_notes.json'
if not notes_path.exists():
    raise RuntimeError('speaker_notes.json must be written before final deck build')
notes=json.loads(notes_path.read_text())
assert len(notes)==8 and sum(n['duration_seconds'] for n in notes)==900
for i,entry in enumerate(notes):
    add_notes(prs.slides[i],f"MAIN SLIDE {i+1} / 8\nPresenter: {entry['presenter']}\nTime allocation: {entry['duration_seconds']} seconds (including transitions/demo).\n\n{entry['notes']}")

HERE.mkdir(exist_ok=True)
(HERE/'assets').mkdir(exist_ok=True)
(HERE/'assets/riverton_article.txt').write_text(body,encoding='utf-8')
(HERE/'assets/saved_demo_outputs.md').write_text('# Saved latest Riverton outputs\n\nSource: [laptop diagnosis](../../reports/LAPTOP_MODEL_DIAGNOSIS.md). LSTM is the reproduced distinct output.\n\n'+'\n\n'.join(f'## {name}\n\n{outputs[name]}' for name in ORDER[:4])+'\n',encoding='utf-8')
with (HERE/'assets/benchmark_values.csv').open('w',newline='') as f:
    writer=csv.writer(f, lineterminator="\n")
    writer.writerow(['track','system','n','failed','rouge1','rouge2','rougeL','rougeLsum','mean_words'])
    for track in ['full_article','common_input']:
        for name in ORDER:
            row=metrics['results'][track][name]
            writer.writerow([track,name,*[row[k] for k in ['n','failed','rouge1','rouge2','rougeL','rougeLsum','mean_words']]])
outfile=HERE/'Comparing_Neural_Models_for_News_Summarization.pptx'
prs.save(outfile)
(HERE/'assets/slide_manifest.json').write_text(json.dumps(slide_meta,indent=2),encoding='utf-8')
(HERE/'assets/text_layout_audit.json').write_text(json.dumps(text_audit,indent=2),encoding='utf-8')
print(f'Wrote {outfile.name}: {len(prs.slides)} slides; eight main, two reference, eight backup.')
issues=[v for v in text_audit if not v['fits_estimate']]
print('Potential text height issues:',json.dumps(issues,indent=2))
