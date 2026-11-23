"""Build an editable deck from saved evidence. Never imports a model backend.

Tooling dependencies are python-pptx 1.0.2 and Pillow 11.3.0, installed separately
from the application. See README.md for rendering and validation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
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
STEM = 'Comparing_Neural_Models_for_News_Summarization'
W, H = 13.333333, 7.5
BG, INK, MUTED, RULE = 'FAF7F1', '25312F', '626D68', 'D8DDD6'
TEAL, BLUE, ORANGE, PLUM, GRAY = '246B5C', '3C648F', 'AE612F', '79587E', '7B8076'
PALE, PALEBLUE = 'E7EEE6', 'E6ECF2'
COLORS = dict(zip(['LSTM', 'Transformer', 'BART', 'PEGASUS', 'Lead-3'], [TEAL, BLUE, ORANGE, PLUM, GRAY]))
FONT = 'Calibri'
FONT_DIR = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
REGULAR, BOLD = FONT_DIR / 'calibri.ttf', FONT_DIR / 'calibrib.ttf'
if not REGULAR.exists():
    REGULAR = Path('/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf')
    BOLD = Path('/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf')
METRICS = json.loads((ROOT / 'reports/pilot50/results.json').read_text(encoding='utf-8'))
DATASET = json.loads((HERE / 'assets/dataset_example.json').read_text(encoding='utf-8'))
DIAGNOSIS = (ROOT / 'reports/LAPTOP_MODEL_DIAGNOSIS.md').read_text(encoding='utf-8')
VERIFICATION = json.loads((ROOT / 'reports/final_model_verification.json').read_text(encoding='utf-8'))
SAVED_TOKENS = VERIFICATION['models']['LSTM']['chunks'][0]['prepared_tokens'][:7]
assert SAVED_TOKENS == ['riverton', 'officials', 'lifted', 'a', 'three-day', 'boil-water', 'advisory']
ORDER = list(COLORS)


def quote_between(label, end):
    section = DIAGNOSIS.split(label, 1)[1].split(end, 1)[0]
    return '\n\n'.join(line[2:] for line in section.splitlines() if line.startswith('> '))


ARTICLE = quote_between('**Supplied article**', '**Reported displayed LSTM')
BODY = ARTICLE.split('\n\n', 1)[1]
OUTPUTS = {
    'LSTM': quote_between('**Reproduced LSTM direct, pipeline and fresh Gradio route:**', '**Reported and reproduced BART:**'),
    'Transformer': quote_between('**Reported displayed LSTM and Transformer, exactly identical:**', '**Reproduced LSTM direct'),
    'BART': quote_between('**Reported and reproduced BART:**', '**Reported and reproduced PEGASUS display:**'),
    'PEGASUS': quote_between('**Reported and reproduced PEGASUS display:**', '\n\nThe raw OpenNMT'),
}
assert hashlib.sha256(BODY.encode()).hexdigest() == '648c2a202c8912f6d1fc5bbe468162d2670be52508916bbbf4ac7fd7e5457fa8'
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
prs.core_properties.title = 'Comparing Neural Models for News Summarization'
prs.core_properties.author = 'Biruk Anley and Daniela Ortega'
prs.core_properties.subject = 'Released checkpoint integration and frozen 50-article evaluation'
slides, audit, talks = [], [], []


def rgb(value):
    return RGBColor.from_string(value)


def rect(s, x, y, w, h, fill=PALE):
    item = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    item.fill.solid()
    item.fill.fore_color.rgb = rgb(fill)
    item.line.fill.background()
    item._element.spPr.append(OxmlElement('a:effectLst'))
    return item


def text(s, value, x, y, w, h, size=24, color=INK, bold=False, align=None, font=FONT):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, paragraph in enumerate(value.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = paragraph
        p.font.name, p.font.size, p.font.bold = font, Pt(size), bold
        p.font.color.rgb = rgb(color)
        p.line_spacing = 1.06
        p.space_after = p.space_before = Pt(0)
        if align is not None:
            p.alignment = align
    ft = ImageFont.truetype(str(BOLD if bold else REGULAR), size=round(size * 2))
    lines = 0
    for paragraph in value.split('\n'):
        current = ''
        lines += 1
        for word in paragraph.split():
            candidate = (current + ' ' + word).strip()
            if ft.getlength(candidate) > w * 144 and current:
                lines += 1
                current = word
            else:
                current = candidate
    estimate = lines * size * 1.09 / 72
    audit.append(dict(slide=len(prs.slides), text=value, font_pt=size, height=h,
                      estimated_height=round(estimate, 3), fits_estimate=estimate <= h + .04))
    return box


def line(s, x1, y1, x2, y2, color=RULE, width=1.6, arrow=False, dashed=False):
    item = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    item.line.color.rgb, item.line.width = rgb(color), Pt(width)
    item._element.spPr.append(OxmlElement('a:effectLst'))
    if arrow:
        end = OxmlElement('a:tailEnd')
        end.set('type', 'triangle')
        item._element.spPr.get_or_add_ln().append(end)
    if dashed:
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        item.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return item


def route(s, points, color=TEAL, dashed=False):
    for i, (start, end) in enumerate(zip(points, points[1:])):
        line(s, *start, *end, color=color, arrow=i == len(points) - 2, dashed=dashed)


def base(title, citation='', main=True):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(BG)
    text(s, title, .65, .55, 12.05, 1.34 if '\n' in title else .70, 36, bold=True)
    text(s, citation, .67, 7.09, 11.4, .26, 12, MUTED)
    text(s, str(len(prs.slides)), 12.1, 7.07, .55, .28, 14, MUTED, align=PP_ALIGN.RIGHT)
    slides.append(dict(slide=len(prs.slides), title=title.replace('\n', ' '), main=main, source=citation))
    return s


def label(s, name, x, y, width=3, size=26):
    text(s, name, x, y, width, .48, size, COLORS[name], True)


def node(s, value, x, y, w, h=.85, color=TEAL, fill=PALE, size=24):
    rect(s, x, y, w, h, fill)
    rect(s, x, y, .055, h, color)
    box = text(s, value, x + .16, y + .12, w - .30, h - .15, size, color, True, PP_ALIGN.CENTER)
    box.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE


def talk(s, seconds, presenter, speech, detail='', cue=''):
    entry = dict(slide=len(prs.slides), title=slides[-1]['title'], duration_seconds=seconds,
                 presenter=presenter, speech=speech, detail=detail, stage_directions=cue)
    talks.append(entry)
    s.notes_slide.notes_text_frame.text = (
        f'{presenter}\n{seconds} seconds including transitions and demo\n\n{speech}\n\n'
        f'Stage directions\n{cue}\n\nEvidence and technical detail\n{detail}')


def backup_notes(s, value):
    s.notes_slide.notes_text_frame.text = value


def results_chart(s, models, y=2.35, h=3.45):
    """Three editable charts, common zero-based axes, aligned model rows."""
    for i, name in enumerate(models):
        yy = y + h * .18 + h * .80 * (i + .5) / len(models) - .19
        label(s, name, .70, yy, 2.35, 24)
    for j, (metric, heading) in enumerate([('rouge1', 'ROUGE-1'), ('rouge2', 'ROUGE-2'), ('rougeLsum', 'ROUGE-Lsum')]):
        x = 3.05 + j * 3.12
        text(s, heading, x, y - .52, 2.9, .43, 24, bold=True, align=PP_ALIGN.CENTER)
        data = CategoryChartData()
        data.categories = models
        data.add_series(heading, [METRICS['results']['full_article'][m][metric] for m in models])
        chart = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(2.92), Inches(h), data).chart
        chart.has_legend = chart.has_title = False
        chart.font.name, chart.font.size = FONT, Pt(18)
        chart.font.color.rgb = rgb(MUTED)
        layout = chart._chartSpace.chart.plotArea.find('{http://schemas.openxmlformats.org/drawingml/2006/chart}layout')
        if layout is None:
            layout = OxmlElement('c:layout')
            chart._chartSpace.chart.plotArea.insert(0, layout)
        manual = OxmlElement('c:manualLayout')
        for tag, value in [('layoutTarget', 'inner'), ('xMode', 'factor'), ('yMode', 'factor'), ('wMode', 'factor'), ('hMode', 'factor'), ('x', '.02'), ('y', '.08'), ('w', '.70'), ('h', '.80')]:
            child = OxmlElement('c:' + tag)
            child.set('val', value)
            manual.append(child)
        layout.append(manual)
        for parent in [chart._chartSpace, chart._chartSpace.chart.plotArea]:
            sp = OxmlElement('c:spPr')
            sp.append(OxmlElement('a:noFill'))
            ln = OxmlElement('a:ln')
            ln.append(OxmlElement('a:noFill'))
            sp.append(ln)
            parent.append(sp)
        cat, val = chart.category_axis, chart.value_axis
        cat.reverse_order = True
        cat.tick_label_position = XL_TICK_LABEL_POSITION.NONE
        cat.major_tick_mark = cat.minor_tick_mark = XL_TICK_MARK.NONE
        cat.format.line.fill.background()
        val.minimum_scale, val.maximum_scale, val.major_unit = 0, 50, 25
        val.tick_labels.font.size = Pt(18)
        val.major_tick_mark = val.minor_tick_mark = XL_TICK_MARK.NONE
        val.format.line.fill.background()
        val.has_major_gridlines = True
        val.major_gridlines.format.line.color.rgb = rgb(RULE)
        plot = chart.plots[0]
        plot.gap_width = 65
        plot.has_data_labels = True
        plot.data_labels.position = XL_DATA_LABEL_POSITION.OUTSIDE_END
        plot.data_labels.number_format = '0.00'
        plot.data_labels.font.name, plot.data_labels.font.size = FONT, Pt(20)
        plot.data_labels.font.color.rgb = rgb(INK)
        # PowerPoint can wrap a numeric label despite ample chart whitespace.
        # Keep the complete two-decimal value on one line in the native chart.
        label_body = plot._element.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}bodyPr')
        label_body.set('wrap', 'none')
        for point, model in zip(chart.series[0].points, models):
            point.format.fill.solid()
            point.format.fill.fore_color.rgb = rgb(COLORS[model])
            point.format.line.fill.background()


s = base('Comparing Neural Models\nfor News Summarization', 'Fictional Riverton diagnostic • condensed teaching example')
text(s, 'Biruk Anley and Daniela Ortega', .69, 1.91, 11.8, .43, 24, MUTED)
text(s, 'ARTICLE', .7, 2.85, 5.2, .4, 18, TEAL, True)
text(s, 'A valve failed. Crews repaired it.\nTwo clear tests followed.\nOfficials lifted the advisory.', .7, 3.42, 5.25, 1.95, 28)
line(s, 6.08, 4.19, 6.78, 4.19, TEAL, 2.5, True)
text(s, 'SUMMARY', 7.15, 2.85, 5.3, .4, 18, TEAL, True)
text(s, 'Officials lifted the advisory\nafter repairs and clear tests.', 7.15, 3.42, 5.35, 1.8, 30, TEAL, True)
line(s, .7, 5.82, 12.6, 5.82)
text(s, 'The summary needs to preserve what happened.', .7, 6.13, 11.9, .55, 28)
talk(s, 45, 'Biruk Anley',
     'Imagine opening a news summary to find out whether you can use your tap water again. A summary might correctly mention a broken valve and testing, yet leave you unsure whether the warning ended. Our example compresses a fictional Riverton article into its main event and the evidence behind it. That is the problem we explored. We are Biruk Anley and Daniela Ortega, and our project compares neural news summarizers while building a working interface for complete articles. We will look at how two smaller models work, how we evaluated them, and why reading the source still matters.',
     'Latest laptop diagnosis. Exact source preserved in assets/riverton_article.txt. Displayed introduction is a condensed teaching example.', 'Point to the lifted advisory before introducing model names.')

s = base('Our target and goals', 'Project documentation • released checkpoint manifests')
for yy, heading, description in [(1.85, 'Compare two core models', 'LSTM and Transformer summarization systems'), (3.25, 'Build a working interface', 'Paste an article, choose a model, inspect a summary'), (4.65, 'Evaluate the comparison', 'Reference summaries and a simple Lead-3 baseline')]:
    line(s, .7, yy + .12, 1.3, yy + .12, TEAL, 4)
    text(s, heading, 1.6, yy - .04, 10.8, .48, 30, bold=True)
    text(s, description, 1.6, yy + .55, 10.8, .48, 24, MUTED)
text(s, 'Our contribution was integration, engineering and evaluation.', .7, 6.4, 12, .45, 24, TEAL)
talk(s, 40, 'Biruk Anley',
     'We had three goals. First, compare an LSTM system with an encoder–decoder Transformer. Second, make them usable through an article-summarization interface. Third, evaluate returned summaries against saved reference highlights and a simple baseline. Lead-3 takes the first three sentences of an article, which is a useful standard for news. We integrated released pretrained checkpoints. We did the application engineering and evaluation, and we did not train or fine-tune these checkpoint weights. Later we will introduce two larger released systems to put the core comparison in context.',
     'README.md, PROJECT_STATUS.md, OpenNMT_Verified_Setup/checkpoint_manifest.json, model_manifest.json.')

s = base('CNN/DailyMail pairs articles with highlights', '[1] Dataset card, configuration 3.0.0 • exact saved test-row excerpts')
for x, value, name in [(.7, '287,113', 'Training'), (4.7, '13,368', 'Validation'), (8.7, '11,490', 'Test')]:
    text(s, value, x, 1.65, 3.8, .68, 38, TEAL, True)
    text(s, name, x, 2.39, 3.8, .43, 24, MUTED)
line(s, .7, 3.13, 12.6, 3.13)
text(s, 'SAVED ARTICLE EXCERPT', .7, 3.48, 5.65, .35, 18, MUTED, True)
text(s, DATASET['article_excerpt'], .7, 4.02, 5.52, 1.8, 26)
text(s, 'REFERENCE HIGHLIGHT EXCERPT', 7.0, 3.48, 5.65, .35, 18, MUTED, True)
text(s, DATASET['reference_excerpt'], 7.0, 4.02, 5.57, 1.8, 28, TEAL)
text(s, 'Our fixed evaluation used 50 test articles, selected with seed 42.', .7, 6.28, 11.95, .48, 24, bold=True)
talk(s, 55, 'Biruk Anley',
     'CNN/DailyMail pairs a news article with short reference highlights. The card for version 3.0.0 lists these training, validation and test splits. Training is used to fit a model, validation helps development, and the test split supports evaluation. The example below is an actual saved row from our fixed test sample. The article reports why Brady missed the White House visit, and the reference captures that point. We evaluated the same 50 test articles for every system, using the frozen seed-42 selection. The released checkpoints have different histories. The OpenNMT releases were trained for CNN/DailyMail summarization. BART and PEGASUS also had broad pretraining before task fine-tuning. These counts describe our dataset version. They do not establish identical historical training data or preprocessing for all checkpoints.',
     'Counts checked against abisee/cnn_dailymail README configuration3.0.0 on2026-11-22. Row ID ' + DATASET['id'] + '. Hashes and provenance in assets/dataset_example.json. Previous validation rows are development evidence, separate from the fixed test sample.')

s = base('From article text to model input', 'Inspected OpenNMT adapter and article planner • [2] original recipe')
text(s, 'Check the input', .7, 1.72, 3.6, .48, 28, TEAL, True)
text(s, 'Nonempty text, valid model,\nsize and workload limits', .7, 2.3, 4.15, 1.0, 24)
line(s, 4.9, 2.55, 5.5, 2.55, TEAL, 2, True)
text(s, 'Prepare OpenNMT words', 5.85, 1.72, 6.7, .48, 28, TEAL, True)
text(s, 'Normalize curly quotes → split sentences\nTreebank-style word tokens → lowercase', 5.85, 2.3, 6.7, 1.0, 24)
rect(s, .7, 3.72, 11.9, 1.36, PALE)
text(s, 'Saved tokenizer trace • earlier fictional Riverton fixture', .94, 3.88, 10.9, .36, 18, MUTED)
text(s, ' | '.join(SAVED_TOKENS), .94, 4.34, 11.15, .51, 25, TEAL)
text(s, 'Saved vocabulary + source dictionary for copying', .7, 5.5, 11.9, .48, 26, bold=True)
text(s, 'Later comparison path    BART / PEGASUS use case-preserving subwords.', .7, 6.25, 12, .47, 24, MUTED)
talk(s, 60, 'Biruk Anley',
     'Before either architecture runs, the software checks that the input is usable and within application workload limits. For OpenNMT, our adapter normalizes curly quotes, splits simple sentence segments, applies a Treebank-style word tokenizer and lowercases its tokens. The line shown contains actual saved tokens from an earlier fictional Riverton fixture. The source starts with Riverton officials lifted a three-day boil-water advisory. The saved vocabulary maps words to model IDs. A source-specific dictionary also lets the generator copy words from the article. Summary decoding starts with a beginning token and can end with an end-of-sequence token. Training used an older Stanford preprocessing convention. Our current NLTK approximation is a documented difference. Sentence boundaries also help plan chunks for long articles. BART and PEGASUS later use their own case-preserving subword tokenizers. Porter stemming will appear in ROUGE scoring.',
     'Exact first seven prepared_tokens from reports/final_model_verification.json models.LSTM.chunks[0]. This trace belongs to the earlier fictional fixture, separate from the latest classroom diagnostic. prepare_article rejects U+FFE8 and uses convert_parentheses=True. OpenNMT fields supply PAD, UNK, BOS, EOS and dynamic source maps. HF prepare uses add_special_tokens=True, truncation=False. No vocabulary fitted, input stopword removal, stemming, lemmatization or dataset deduplication. ChunkPlanner uses Punkt spans. ROUGE stemming is evaluation only.')

s = base('LSTM encoder–decoder with attention', '[2,3] OpenNMT release and verified architecture • approximately 43M parameters')
text(s, '≈43M', 10.2, 1.43, 2.2, .49, 28, TEAL, True, PP_ALIGN.RIGHT)
node(s, 'Word embeddings', .7, 2.1, 3.05, 1.0)
node(s, 'Bidirectional\nLSTM encoder', 4.35, 2.1, 3.35, 1.0)
node(s, 'Attention', 9.0, 2.1, 3.45, 1.0)
line(s, 3.75, 2.6, 4.35, 2.6, TEAL, 2, True)
line(s, 7.7, 2.6, 9.0, 2.6, TEAL, 2, True)
text(s, 'encoder outputs', 7.64, 1.65, 2.2, .35, 20, MUTED)
node(s, 'Recurrent LSTM\ndecoder', 4.35, 4.72, 3.35, 1.08)
node(s, 'Vocabulary + copy\nlearned mixture gate', 9.0, 4.72, 3.45, 1.08)
route(s, [(4.65, 3.1), (4.65, 4.72)], TEAL)
text(s, 'initial state', 2.45, 3.75, 2.05, .40, 22, MUTED)
route(s, [(9.0, 2.88), (8.25, 2.88), (8.25, 4.38), (6.1, 4.38), (6.1, 4.72)], TEAL)
text(s, 'context', 6.2, 3.86, 1.65, .36, 22, TEAL)
route(s, [(7.7, 5.0), (8.62, 5.0), (8.62, 3.65), (10.0, 3.65), (10.0, 3.1)], BLUE)
text(s, 'decoder query', 9.9, 3.75, 2.53, .39, 22, BLUE)
route(s, [(11.9, 3.1), (11.9, 4.72)], TEAL, True)
text(s, 'copy\nweights', 12.0, 3.28, 1.0, .75, 18, TEAL)
line(s, 7.7, 5.5, 9.0, 5.5, TEAL, 2, True)
route(s, [(4.7, 5.8), (4.7, 6.25), (5.9, 6.25), (5.9, 5.8)], TEAL)
text(s, 'state across steps', 2.05, 6.27, 4.5, .39, 22, MUTED)
text(s, 'Previous\nsummary word', .7, 4.8, 3.0, .95, 24, MUTED)
line(s, 3.7, 5.28, 4.35, 5.28, TEAL, 2, True)
text(s, 'Next word → summary', 9.0, 6.23, 3.55, .48, 24, TEAL, True)
line(s, 10.7, 5.8, 10.7, 6.13, TEAL, 2, True)
talk(s, 65, 'Daniela Ortega',
     'An LSTM carries a memory state as it reads or writes a sequence. Here word embeddings turn input words into vectors. The bidirectional encoder reads the article in both directions, so each encoded word can include left and right context. The decoder writes the summary in one direction, carrying its recurrent state from one generated word to the next. Its state queries attention over encoder outputs. Attention supplies a weighted source context. The generator combines vocabulary prediction with copying, using source attention and a learned gate. For example, it may generate lifted from its vocabulary and copy Riverton from the article. Copying can help reproduce names, while deciding which event matters remains the model’s task. This released checkpoint has about 43 million parameters. The arrows show dependencies within a decoding step, and the recurrent loop carries information into the next step.',
     '42,789,205 parameters. One bidirectional encoder layer has 512 total hidden dimensions. Word embeddings have 128 dimensions. One input-feeding LSTM decoder has 512 hidden dimensions, MLP/global attention and reused copy attention. The encoder bridge initializes decoder hidden and cell states. Decoder input also includes the previous attended output. CopyGenerator mixes vocabulary probabilities with a dynamic source-attention distribution using a learned sigmoid gate. Learned coverage attention is absent. The search coverage penalty belongs to beam scoring. Verified against local rnn_encoder.py, decoder.py and copy_generator.py.', 'Trace source memory, decoder query, context and copy weights.')

s = base('What the LSTM gives us', 'Architecture tendencies and separate saved observations • latest Riverton diagnostic')
text(s, '43M', .7, 1.8, 4.0, 1.3, 78, TEAL, True)
text(s, 'parameters', .75, 3.18, 3.9, .5, 28, TEAL)
text(s, 'Compact weights\nAttention to the source\nCopying of source words', 5.2, 1.95, 7.25, 1.9, 30)
line(s, .7, 4.15, 12.6, 4.15)
text(s, 'RECURRENT COMPUTATION', .7, 4.5, 5.4, .35, 18, MUTED, True)
text(s, 'State updates happen in sequence.\nLong-distance selection can be difficult.', .7, 5.02, 5.6, 1.2, 24)
text(s, 'OBSERVED ON RIVERTON', 7.0, 4.5, 5.4, .35, 18, MUTED, True)
text(s, 'It retained background details\nand omitted the lifted advisory.', 7.0, 5.02, 5.55, 1.2, 26, TEAL)
talk(s, 40, 'Daniela Ortega',
     'The LSTM is our smallest checkpoint, making its weight storage relatively modest. Its attention and copy mechanism give it a direct route back to source words, useful for names and numbers. Recurrent computation still proceeds step by step, and carrying context across a long article can be difficult. Those are architectural tendencies. Our saved Riverton observation is more specific. The reproduced LSTM summary retained the population, pressure risk and school details, while omitting that the advisory ended. Copying supported words did not guarantee selecting the important event. One diagnostic example cannot tell us how often that failure occurs.',
     'FP32 parameter storage floor is about 171 MB before activations and framework overhead. The exact reproduced output is in backup. This example supplies no measured factuality rate and cannot isolate architecture as the cause.')

s = base('Our encoder–decoder Transformer', '[2,4] Four encoder and four decoder layers • approximately 81M parameters')
text(s, '≈81M', 10.2, 1.72, 2.2, .49, 28, BLUE, True, PP_ALIGN.RIGHT)
node(s, 'Token embeddings\n+ positions', .7, 1.53, 3.6, 1.0, BLUE, PALEBLUE)
node(s, 'Summary embeddings\n+ positions', 5.2, 1.53, 4.0, 1.0, BLUE, PALEBLUE)
text(s, 'ENCODER × 4', .9, 2.66, 3.2, .38, 20, BLUE, True)
text(s, 'DECODER × 4', 5.4, 2.66, 3.6, .38, 20, BLUE, True)
rect(s, .7, 3.15, 3.6, 2.57, PALEBLUE)
rect(s, 5.2, 3.15, 4.0, 2.57, PALEBLUE)
text(s, 'Self-attention', .92, 3.52, 3.16, .47, 26, BLUE, True, PP_ALIGN.CENTER)
text(s, 'Feed-forward', .92, 4.76, 3.16, .47, 26, BLUE, True, PP_ALIGN.CENTER)
text(s, 'Masked self-attention', 5.43, 3.34, 3.54, .47, 24, BLUE, True, PP_ALIGN.CENTER)
text(s, 'Cross-attention', 5.43, 4.25, 3.54, .47, 26, BLUE, True, PP_ALIGN.CENTER)
text(s, 'Feed-forward', 5.43, 5.13, 3.54, .47, 24, BLUE, True, PP_ALIGN.CENTER)
line(s, 3.94, 2.53, 3.94, 3.15, BLUE, 2, True)
line(s, 8.83, 2.53, 8.83, 3.15, BLUE, 2, True)
line(s, 2.5, 4.06, 2.5, 4.64, BLUE, 2, True)
line(s, 7.2, 3.85, 7.2, 4.13, BLUE, 2, True)
line(s, 7.2, 4.77, 7.2, 5.03, BLUE, 2, True)
line(s, 4.3, 4.48, 5.2, 4.48, BLUE, 3, True)
text(s, 'Source memory', .82, 6.02, 3.4, .44, 24, BLUE)
node(s, 'Vocabulary\n+ copy gate', 10.03, 4.08, 2.6, 1.2, BLUE, PALEBLUE)
line(s, 9.2, 4.69, 10.03, 4.69, BLUE, 2, True)
route(s, [(8.8, 4.78), (8.8, 6.17), (11.3, 6.17), (11.3, 5.28)], BLUE, True)
text(s, 'Cross-attention weights for copying', 5.25, 6.47, 7.38, .39, 22, BLUE)
text(s, 'Next word', 10.1, 2.93, 2.55, .44, 24, BLUE, True)
route(s, [(11.3, 4.08), (11.3, 3.55)], BLUE)
talk(s, 65, 'Daniela Ortega',
     'Our Transformer also has an encoder and a decoder, with four layers in each. Embeddings represent tokens, and positions provide order information. Encoder self-attention lets source tokens consult other source tokens, followed by a feed-forward transformation. In the decoder, masked self-attention sees only the summary words available so far. Cross-attention connects that decoder representation to the encoded article. Think of a query as what the decoder needs, keys as source addresses to compare it with, and values as the information it retrieves. The large horizontal arrow is that source-to-decoder connection. This particular OpenNMT checkpoint also has a vocabulary and copy mixture. It uses its final cross-attention weights for copying. The resulting distribution chooses the next summary word, then generation continues. It has about 81 million parameters.',
     '81,468,757 parameters. Four encoder and four decoder layers, model dimension 512, eight attention heads and feed-forward dimension 2048. Positional encodings supply order information. Residual connections and layer normalization are omitted from the diagram for readability. Legacy TransformerDecoder uses the final standard cross-attention map for copying, despite constructing a separate copy-attention module. Encoder memory supplies cross-attention keys and values. The decoder supplies its query. Configuration and modules were checked against the pinned local OpenNMT checkout.', 'Emphasize the source arrow entering cross-attention.')

s = base('Connections help, but selection still matters', '[4] Transformer architecture • saved checkpoint observations')
for word, x, width in [('Officials', .8, 2.8), ('lifted', 4.1, 1.8), ('the', 6.45, 1.1), ('advisory', 8.2, 3.1)]:
    text(s, word, x, 2.7, width, .66, 38, BLUE, True)
route(s, [(1.2, 2.47), (1.2, 1.85), (9.5, 1.85), (9.5, 2.47)], BLUE)
text(s, 'Self-attention can connect distant source positions.', .8, 3.74, 11.7, .53, 28)
line(s, .7, 4.55, 12.6, 4.55)
text(s, 'Parallel across training positions\nAttention memory grows with input length', .8, 4.97, 5.7, 1.3, 24)
text(s, 'Summary inference is sequential\nThis checkpoint still missed the lifting', 7.0, 4.97, 5.55, 1.3, 24, BLUE)
talk(s, 45, 'Daniela Ortega',
     'Self-attention gives words a way to connect across the source. That helps represent context, and during training many token positions can be computed in parallel. The cost is attention memory that grows with input length. Summary inference still generates one new token after another. These are general architectural properties. Our particular released checkpoint still omitted the lifting in Riverton, and separate saved development reviews found relationship and entity errors. A connection mechanism gives the model access to information. It does not ensure that it selects the central event or preserves who did what. Our comparisons also differ in training, tokenizer and decoding, so they do not isolate architecture alone.',
     'Dense source attention storage scales quadratically in length. Relationship errors are evidence-linked AI-assisted development observations in MODEL_CAPABILITIES.md and factual_audit_review.html without human-rated prevalence. The illustrated word connection is schematic.')

s = base('How the complete application works', 'Current app screenshot • implementation inspected without generation')
s.shapes.add_picture(str(HERE / 'assets/current_app.png'), Inches(.7), Inches(1.75), width=Inches(5.2))
text(s, 'Current local interface', .7, 5.57, 5.2, .45, 24, MUTED)
text(s, 'Complete input still needs\na factual coverage check.', .7, 6.16, 5.25, .91, 25, TEAL, True)
node(s, 'Article + chosen model', 6.5, 1.65, 6.0, .66)
line(s, 9.5, 2.31, 9.5, 2.60, TEAL, 2, True)
text(s, 'Token-capacity check', 6.6, 2.64, 5.8, .45, 26, TEAL, True, PP_ALIGN.CENTER)
node(s, 'Fits → direct', 6.5, 3.32, 2.7, .66, size=22)
node(s, 'Sentence-aware\nchunks', 9.8, 3.20, 2.7, .91, size=22)
route(s, [(8.15, 3.1), (8.15, 3.32)])
route(s, [(11.1, 3.1), (11.1, 3.20)])
route(s, [(7.85, 3.98), (7.85, 4.28), (9.5, 4.28)])
route(s, [(11.15, 4.11), (11.15, 4.28), (9.5, 4.28)])
line(s, 9.5, 4.28, 9.5, 4.47, TEAL, 2, True)
text(s, 'Generate → source-order merge', 6.5, 4.53, 6.0, .45, 25, bold=True, align=PP_ALIGN.CENTER)
text(s, 'Exact-sentence deduplication', 6.5, 5.16, 6.0, .45, 24, align=PP_ALIGN.CENTER)
text(s, 'Bounded consolidation if needed', 6.5, 5.78, 6.0, .45, 24, align=PP_ALIGN.CENTER)
text(s, 'Summary + statistics', 6.5, 6.4, 6.0, .45, 26, TEAL, True, PP_ALIGN.CENTER)
talk(s, 70, 'Biruk Anley',
     'This is a screenshot of our current local interface. The user supplies an article and chooses a model. The system checks the article using that model’s tokenizer. If the input fits, it gets one direct source pass. Otherwise the planner groups text near sentence boundaries, splitting an oversized sentence safely when needed. Each chunk is summarized. The app joins those outputs in source order and removes exact complete-sentence duplicates. If the merged result fits the model window, it can be returned directly. If further reduction is necessary, the app uses bounded consolidation with limits on passes and total work. These choices try to avoid preventable information loss from unnecessary chunking or repeated compression. Statistics describe the request, including output words and time. Complete input processing leaves a separate question. Did the summary preserve the main event and its evidence? The current interface uses natural checkpoint generation and has a model selector without a summary-length selector.',
     'Screenshot captured from localhost without submitting a request. app.py uses NATURAL_POLICY. Work windows contain 1024 selected-model tokens. ChunkPlanner covers exact source spans using Punkt boundaries. Deduplication compares whitespace-normalized, casefolded complete sentences. Natural-policy consolidation is needed only when the merge exceeds the source window. Limits include four consolidation passes, 96 calls, 64000 total input tokens and 120000 characters or 12000 words per article. The runtime guard is cooperative and cannot interrupt an in-flight model call. Implementation is in article_summarizer.py and processing_config.json.')

s = base('How we measured reference overlap', '[5] ROUGE and Google Research implementation • frozen 50-article protocol')
text(s, '50 fixed CNN/DailyMail 3.0.0 test articles • seed 42', .7, 1.6, 11.9, .48, 26, TEAL, True)
text(s, 'Reference    crews lifted the advisory', .7, 2.55, 11.8, .58, 32)
text(s, 'Candidate    officials lifted the advisory', .7, 3.26, 11.8, .58, 32, TEAL)
line(s, .7, 4.16, 12.6, 4.16)
for x, y, title, desc in [(.7, 4.54, 'ROUGE-1', 'Shared individual words'), (7.0, 4.54, 'ROUGE-2', 'Shared adjacent word pairs'), (.7, 5.6, 'ROUGE-L', 'Longest ordered word sequence'), (7.0, 5.6, 'ROUGE-Lsum', 'Sentence-aware union of sequences')]:
    text(s, title, x, y, 5.6, .43, 26, TEAL, True)
    text(s, desc, x, y + .49, 5.6, .43, 24)
text(s, 'Reported scores are F1 × 100 with Porter stemming.', .7, 6.61, 12, .39, 22, MUTED)
talk(s, 55, 'Biruk Anley',
     'We used the same frozen 50 test articles and original highlights for every system. ROUGE compares the model summary with a reference. In this illustrative sentence pair, lifted the advisory appears in both texts. ROUGE-1 counts individual word overlap, while ROUGE-2 compares adjacent pairs. ROUGE-L looks for a matching sequence in order, allowing gaps. ROUGE-Lsum uses sentence-aware matching across the summary. We report F1, which balances precision and recall, multiplied by 100. Porter stemming reduces words to related forms during evaluation. A summary can earn overlap while missing the main event, and a valid paraphrase can receive less overlap. We therefore use ROUGE as one measured comparison, alongside source-grounded reading. The full-article track is our main narrative because it evaluates the application’s handling of complete source articles.',
     'The sentence pair illustrates overlap without assigning a numerical score. Sample fingerprint dd071ed3cffed23cb5a566d3db9e7aa8c5b435bd87e130e041a619429c18f4de. tests/final_benchmark.py uses use_stemmer=True and consistent sentence segmentation for ROUGE-Lsum. Both tracks retain full reference highlights. The shared-prefix protocol and its separate results are in backup.')

s = base('Core models and a simple baseline', 'Saved full-article results • ROUGE F1 × 100 • n = 50 per system')
results_chart(s, ['LSTM', 'Transformer', 'Lead-3'], 2.45, 3.45)
text(s, 'Lead-3 takes the first three source sentences.', .7, 6.13, 11.9, .47, 26, bold=True)
text(s, 'LSTM led this Transformer checkpoint on these overlap measures.', .7, 6.65, 11.9, .38, 22, MUTED)
talk(s, 45, 'Biruk Anley',
     'Here are the core results from the saved full-article track. LSTM scored 40.95 on ROUGE-1, Transformer 38.75, and Lead-3 40.36. LSTM led this Transformer checkpoint on the three measures shown. Lead-3 is a serious comparison because news often puts central facts early. It exceeded Transformer on these measures while requiring no generated summary. This describes whole systems on a small sample. It does not show that recurrent models are generally better than Transformers. The full four-metric table is in backup, including the separate ROUGE-L values. These are the existing benchmark results, with no new evaluation for this presentation.',
     'Native charts read exact unrounded values from results.full_article. Visible values round to two decimals, with identical axes from 0 to 50. ROUGE-L values are 28.70856 for LSTM, 26.98862 for Transformer and 24.90873 for Lead-3. Historical training and decoding differ across systems.')

s = base('Larger released comparison systems', '[6,7] Papers and released checkpoint cards • learned parameter counts')
for i, (name, millions) in enumerate([('LSTM', 43), ('Transformer', 81), ('BART', 406), ('PEGASUS', 569)]):
    y = 1.98 + i * 1.10
    label(s, name, .7, y, 2.5, 24)
    rect(s, 3.18, y + .08, millions / 569 * 2.65, .30, COLORS[name])
    text(s, f'{millions}M', 3.18, y + .46, 2.7, .40, 24, COLORS[name], True)
line(s, 6.35, 1.75, 6.35, 6.65)
label(s, 'BART', 6.95, 1.94, 5.3, 30)
text(s, 'Reconstruct corrupted text\n→ CNN/DailyMail fine-tuning', 6.95, 2.58, 5.5, 1.05, 26)
label(s, 'PEGASUS', 6.95, 4.11, 5.3, 30)
text(s, 'Generate removed sentences\n→ CNN/DailyMail fine-tuning', 6.95, 4.75, 5.5, 1.05, 26)
text(s, 'Broad pretraining adds context.', 6.95, 6.24, 5.5, .65, 24, MUTED)
talk(s, 55, 'Biruk Anley',
     'We also integrated two larger released comparison systems. BART-large-CNN has about 406 million learned parameters. Its denoising pretraining teaches it to reconstruct corrupted text, followed by fine-tuning for CNN/DailyMail summarization. PEGASUS-CNN/DailyMail has about 569 million learned parameters. Its gap-sentence objective removes selected sentences and asks the model to generate them. The released card documents mixed pretraining on C4 and HugeNews before task fine-tuning. The size bars use a common scale. These systems bring different training histories as well as more parameters. BART uses byte-level subwords and PEGASUS uses SentencePiece. Both preserve case. They generate through vocabulary distributions, while the OpenNMT checkpoints have an explicit copy gate. We are adding comparison systems to the story using results already saved.',
     'Exact learned parameter counts are 42,789,205 for LSTM, 81,468,757 for Transformer, 406,290,432 for BART and 568,699,904 for PEGASUS. Bars use rounded millions. PEGASUS has 570,797,056 instantiated parameters, including 2,097,152 fixed positional entries. Checkpoint IDs are facebook/bart-large-cnn and google/pegasus-cnn_dailymail, pinned in model_manifest.json.')

s = base('The expanded comparison', 'Same saved full-article benchmark • ROUGE F1 × 100 • n = 50 per system')
results_chart(s, ORDER, 2.30, 3.55)
text(s, 'BART led ROUGE-1, ROUGE-2 and ROUGE-Lsum on this sample.', .7, 6.06, 11.95, .48, 26, ORANGE, True)
text(s, 'PEGASUS narrowly led ROUGE-L, 30.92 to BART’s 30.88.', .7, 6.61, 11.95, .40, 24, MUTED)
talk(s, 55, 'Daniela Ortega',
     'This reveals the remaining systems from the same frozen comparison. We used the same 50 articles, references and saved results. BART led ROUGE-1, ROUGE-2 and ROUGE-Lsum on this sample. Its ROUGE-1 was 45.21, followed by PEGASUS at 43.81. PEGASUS narrowly led the separate flat ROUGE-L metric, 30.92 to BART’s 30.88. All five systems completed 50 requests in each of the two tracks, giving 500 completed evaluations and zero failed requests. Completion means an output was returned under the protocol. These measurements do not establish a general factual-quality winner or statistical significance. We also leave speed out because desktop and game activity contaminated the saved GPU timing context. Next, a diagnostic example shows why an aggregate overlap score cannot answer every useful question.',
     'Charts read the same results.full_article object as the core slide. All four metrics and the common_input track appear in backup. No benchmark was rerun or tuned for the slides. The interrupted 1000-article study remains incomplete. Human factuality ratings and significance tests have not been measured for this comparison.')

s = base('What did the summary leave out?', 'Exact saved excerpts • latest fictional Riverton diagnostic • separate from benchmark')
rect(s, .7, 1.57, 11.9, 1.04, PALE)
text(s, 'Source event     Officials lifted the advisory Friday morning.', .94, 1.87, 11.40, .49, 27, TEAL, True)
label(s, 'BART', .7, 3.05, 5.4, 28)
text(s, '“Two consecutive rounds of testing found no harmful bacteria.”', .7, 3.73, 5.45, 1.58, 28)
text(s, 'Supported background.\nFull output omits the lifting.', .7, 5.41, 5.5, .95, 24, ORANGE)
label(s, 'PEGASUS', 7.05, 3.05, 5.4, 28)
text(s, '“Riverton officials lifted a boil-water advisory Friday morning.”', 7.05, 3.73, 5.5, 1.58, 28)
text(s, 'Captures the lifting.\nRepair and testing evidence is absent.', 7.05, 5.41, 5.5, .95, 24, PLUM)
text(s, 'All the details… except the headline.', .7, 6.61, 11.9, .42, 24, MUTED)
talk(s, 100, 'Daniela Ortega',
     'The main event in our latest fictional Riverton article is that officials lifted the boil-water advisory. Repairs, flushing and two clear testing rounds support that decision. The saved BART output includes supported background, including the testing sentence shown, but its full summary omits that the advisory ended. PEGASUS clearly captures the lifting, while omitting the repair and testing evidence. BART gives us a small all-the-details-except-the-headline moment. More precisely, it selected true secondary facts and missed the resolution. Both outputs ended naturally after processing the entire article in one pass. Cleanup or chunking did not remove the lifting from BART. For the brief demo, we will paste the saved article body, check the selected model and submit at most one request. While it runs, compare the returned summary with the source event. If the CPU takes about 30 seconds, we will continue with the complete saved outputs in backup. Check event, resolution and evidence separately. This example does not establish a general model ranking.',
     'Exact saved excerpts. BART generated 59 tokens and PEGASUS 46 tokens before natural EOS. Each used one chunk and one call, processing the complete 2202-character, 315-word body. The banner is a shortened source-event paraphrase. The original duplicated LSTM/Transformer display remains unresolved. The reproduced LSTM output differs. Full exact outputs appear in backup and the saved assets.',
     'Reserve about 30 seconds for CPU wait within this 100-second allocation. Submit at most one live request in class. If it is slow, use the saved output slides.')

s = base('Experiments and lessons', 'Saved development evidence • current policy and earlier experiments')
xs, widths = [.7, 4.65, 8.65], [3.55, 3.55, 3.95]
for x, title in zip(xs, ['Attempt', 'Observation', 'Decision']):
    text(s, title, x, 1.65, 3.6, .43, 26, TEAL, True)
rows = [
    ('Length presets and\npercentage targets', 'Early stops and\nsimilar outputs', 'Selector removed\nAPI remains experimental'),
    ('Chunking and\nconsolidation', 'Repeated compression\ncould lose useful detail', 'Direct pass when fitting\nConsolidate when needed'),
    ('Model-specific\ndecoding restoration', 'One development case\nrecovered its main claim', 'Retained checkpoint\nsearch profiles'),
    ('Input-window\nassumptions', '400 was a training\nconvention', '1024-token app window\nwithin capacity guards'),
]
for i, row in enumerate(rows):
    y = 2.43 + i * 1.11
    line(s, .7, y - .18, 12.6, y - .18)
    for j, (x, width, value) in enumerate(zip(xs, widths, row)):
        text(s, value, x, y, width, .88, 24, TEAL if j == 2 else INK)
text(s, 'Length control was implementable. Extra words did not reliably add useful facts.', .7, 6.81, 11.9, .32, 20, MUTED)
talk(s, 65, 'Daniela Ortega',
     'We tried several engineering choices before settling on the current demo. Short, Medium and Detailed preferences, including percentage targets, were technically implementable. Actual lengths were unreliable. Models could stop early, different choices could produce similar summaries, and forcing more words did not reliably improve useful content. We removed the selector for a simpler, more predictable interface. The older API remains available for experiments. We also changed chunking and consolidation so a fitting article takes one source pass and a fitting merge avoids another compression step. Restoring model-specific search settings recovered the main claim in one saved development case, while other errors remained. Finally, we revisited an input-window assumption. The OpenNMT recipe’s 400 tokens was a training convention. The current application uses a guarded 1024-token work window, with no claim that this extends a learned quality boundary. These are documented decisions and observations, rather than a measured improvement on the frozen test sample.',
     'Evidence includes the current README policy, FULL_ARTICLE.md length and quality diagnosis, MODEL_CAPABILITIES.md controls, quality_regression.json and natural_policy_verification.json. Targets of 15%, 25% and 35% remain experimental. Restored PEGASUS search recovered the main claim on the earlier unverified Huang development text. Raising only the minimum length did not. Saved direct checks processed that 580-OpenNMT-token source. The fixed-budget API keeps its 400-token convention. These observations support no aggregate causal improvement claim.')

s = base('What we learned', 'Completed integration and frozen comparison • proposed next step')
for y, heading, body in [(1.70, 'Built', 'A working article interface with four released summarizers.'), (2.95, 'Measured', 'BART led most reference-overlap metrics on 50 test articles.'), (4.20, 'Limited', 'Main events and supporting facts can still disappear.')]:
    text(s, heading, .7, y, 2.15, .49, 28, TEAL, True)
    text(s, body, 3.15, y, 9.3, 1.02, 28)
line(s, .7, 5.65, 12.6, 5.65, TEAL, 1.5)
text(s, 'Next step', .7, 6.03, 2.2, .5, 28, TEAL, True)
text(s, 'Structured human review of events, relationships and evidence.', 3.15, 5.98, 9.3, 1.02, 28, bold=True)
talk(s, 40, 'Daniela Ortega',
     'We built an article interface and compared four released summarizers with Lead-3 on the same fixed articles. BART led most overlap metrics in this small sample. The diagnostic showed that supported details can survive while the main event disappears. Full input processing and fluent output still leave factual coverage to check. Our next useful step is structured human review of events, relationships and evidence on a separate set of articles. That remains proposed work. Thank you. The backups contain full results, checkpoint details and saved demo outputs.',
     'The 16 main slides total 900 seconds, including one optional demo request. This redesign performed no benchmark, training, inference or production changes.')
assert len(talks) == 16 and sum(t['duration_seconds'] for t in talks) == 900

references = [
    ('References for models', [
        ('[2] OpenNMT summarization releases', 'Official recipe and pinned local checkpoint metadata', 'https://opennmt.net/OpenNMT-py/examples/summary/Summarization.html'),
        ('[3] Hochreiter & Schmidhuber, 1997', 'Long Short-Term Memory', 'https://doi.org/10.1162/neco.1997.9.8.1735'),
        ('[4] Vaswani et al., 2017', 'Attention Is All You Need', 'https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html'),
        ('Copy mixture implementation', 'Local OpenNMT CopyGenerator and decoder modules', 'https://github.com/OpenNMT/OpenNMT-py/tree/6db7ec1b3afae28c257a0416dd947022c3b12bc0/onmt'),
    ]),
    ('References for data and comparison', [
        ('[1] CNN/DailyMail dataset card', 'Configuration 3.0.0 and article/highlight fields', 'https://huggingface.co/datasets/abisee/cnn_dailymail'),
        ('[5] Lin, 2004 and Google Research ROUGE', 'ROUGE F1, stemming and the two LCS variants', 'https://github.com/google-research/google-research/tree/master/rouge'),
        ('[6] Lewis et al., 2020', 'BART denoising pretraining and released CNN checkpoint', 'https://aclanthology.org/2020.acl-main.703/'),
        ('[7] Zhang et al., 2020', 'PEGASUS gap-sentence pretraining and released checkpoint', 'https://proceedings.mlr.press/v119/zhang20ae.html'),
    ]),
]
for title, items in references:
    s = base(title, 'Primary sources • full links and evidence mapping in REFERENCES.md', False)
    for i, (heading, description, url) in enumerate(items):
        y = 1.70 + i * 1.22
        box = text(s, heading, .7, y, 11.7, .44, 26, TEAL, True)
        box.text_frame.paragraphs[0].runs[0].hyperlink.address = url
        run = box.text_frame.paragraphs[0].runs[0]
        run.font.color.rgb = rgb(TEAL)
        run.font.underline = False
        text(s, description, .7, y + .53, 11.7, .45, 24)
    backup_notes(s, '\n\n'.join(f'{a}\n{b}\n{c}' for a, b, c in items) + '\n\nRelated sources include Nallapati et al. (2016), Bahdanau et al. (2015) and See et al. (2017), listed in REFERENCES.md. Local checkpoint code establishes the actual implementation. Frozen project scores come from the saved benchmark results.')


def table(s, headers, rows, xs, widths, y=1.9, step=.77, size=24):
    for x, w, value in zip(xs, widths, headers):
        text(s, value, x, y, w, .43, 22, TEAL, True)
    line(s, .7, y + .6, 12.6, y + .6)
    for i, row in enumerate(rows):
        yy = y + .86 + i * step
        for j, (x, w, value) in enumerate(zip(xs, widths, row)):
            text(s, str(value), x, yy, w, .57, size, COLORS.get(str(value), INK), j == 0)


for track, title in [('full_article', 'Full-article results'), ('common_input', 'Shared-prefix results')]:
    s = base(title, 'Frozen 50-article comparison • each system n = 50, failed = 0 • ROUGE F1 × 100', False)
    rows = []
    for model in ORDER:
        v = METRICS['results'][track][model]
        rows.append([model] + [f'{v[k]:.2f}' for k in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']] + [f'{v["mean_words"]:.1f}'])
    table(s, ['System', 'R-1', 'R-2', 'R-L', 'R-Lsum', 'Words'], rows, [.7, 3.5, 5.2, 6.9, 8.6, 10.7], [2.6, 1.5, 1.5, 1.5, 1.9, 1.7], y=1.7)
    message = 'BART led R-1, R-2 and R-Lsum. PEGASUS narrowly led R-L.' if track == 'full_article' else 'Identical bounded source prefix. References remain the full highlights.'
    text(s, message, .7, 6.62, 11.9, .43, 24, MUTED)
    backup_notes(s, 'Exact source is results.json, track ' + track + '. Values are rounded only for display. Both tracks use original references, Porter stemming, seed 42 and the same 50 IDs. The shared prefix fits 400 OpenNMT tokens and the Hugging Face tokenizer capacities. Reference facts may lie beyond that prefix. LSTM and Transformer change relative order across tracks. This presentation performed no new experiment or significance test. Saved GPU timings include competing desktop activity.')

s = base('The two OpenNMT checkpoints', '[2] Official releases • verified metadata and tensor/vocabulary equivalence', False)
for name, x, count, config in [
    ('LSTM', .7, '42,789,205 learned parameters', '1 bidirectional encoder layer\n1 recurrent decoder layer\n128-dimensional embeddings\n512-dimensional hidden states\nMLP attention + copy mixture'),
    ('Transformer', 7.0, '81,468,757 learned parameters', '4 encoder / 4 decoder layers\n512-dimensional model states\n8 attention heads\n2048-dimensional feed-forward\nPositions + copy mixture')]:
    label(s, name, x, 1.75, 5.5, 30)
    text(s, count, x, 2.4, 5.5, .46, 24, COLORS[name])
    text(s, config, x, 3.25, 5.5, 2.6, 26)
text(s, 'CNN/DailyMail summarization training is documented for both releases.', .7, 6.42, 11.9, .49, 24, MUTED)
backup_notes(s, 'Official LSTM filename is ada6_bridge_oldcopy_tagged_acc_54.17_ppl_11.17_e20.pt. Transformer filename is sum_transformer_model_acc_57.25_ppl_9.22_e16.pt. Official URLs and SHA-256 hashes are in checkpoint_manifest.json. The compatibility checkout is OpenNMT-py 0.2.1 at commit 6db7ec1b3afae28c257a0416dd947022c3b12bc0. Inference copies have matching tensors and vocabularies to the verified downloads. Broad pretraining is undocumented. Learned coverage attention is absent. The LSTM encoder has 512 total hidden dimensions, 256 per direction. Its decoder is unidirectional with 512 hidden dimensions. Transformer final cross-attention supplies copy probabilities. The training convention was first 400 source tokens and target length 100. We integrated these releases without training them.')

s = base('BART and PEGASUS checkpoint details', '[6,7] Released cards, pinned revisions and local verification', False)
for name, x, content in [
    ('BART', .7, '406,290,432 learned parameters\n12 encoder / 12 decoder layers\n1024-dimensional states, 16 heads\nByte-level BPE, case preserved\nLearned positional embeddings'),
    ('PEGASUS', 7.0, '568,699,904 learned parameters\n16 encoder / 16 decoder layers\n1024-dimensional states, 16 heads\nSentencePiece, case preserved\nFixed sinusoidal positions')]:
    label(s, name, x, 1.75, 5.5, 30)
    text(s, content, x, 2.65, 5.6, 2.85, 24)
text(s, 'facebook/bart-large-cnn', .7, 6.2, 5.6, .55, 24, ORANGE)
text(s, 'google/pegasus-cnn_dailymail', 7.0, 6.2, 5.6, .55, 24, PLUM)
backup_notes(s, 'BART uses facebook/bart-large-cnn at revision 37f520fa929c961707657b28798b30c003dd100b, with model.safetensors. PEGASUS uses google/pegasus-cnn_dailymail at revision 40d588fdab0cc077b80d950b300bf66ad3c75b92, with pytorch_model.bin. Identities are pinned in model_manifest.json. PEGASUS has 570,797,056 instantiated parameters. Subtracting 2,097,152 fixed positional entries gives 568,699,904 learned parameters. Missing positional tables match deterministic sinusoids in the saved verification. The release documents Mixed & Stochastic pretraining on C4 and HugeNews. BART uses denoising pretraining. Both were subsequently fine-tuned on CNN/DailyMail. They use standard vocabulary generation. This project integrated the released weights without training or fine-tuning them.')

s = base('Current decoding settings', 'Inspected natural-checkpoint-v1 policy • deterministic search and natural EOS', False)
table(s, ['System', 'Beam', 'Initial cap', 'Other search settings'], [
    ['LSTM', '5', '100', 'Wu α .9 / coverage β 5'],
    ['Transformer', '5', '100', 'Same OpenNMT recipe'],
    ['BART', '4', '141 new', 'Length penalty 2 / no-repeat 3'],
    ['PEGASUS', '8', '127 new', 'Length penalty .8 / no-repeat 0'],
], [.7, 3.4, 4.8, 7.3], [2.5, 1.1, 2.2, 5.1], step=.83, size=23)
text(s, 'Natural EOS can stop before the cap. Bounded retries handle capped output.', .7, 6.21, 11.9, .80, 26, TEAL)
backup_notes(s, 'OpenNMT uses beam 5, min_length=min(35, input_tokens//8, cap-2), initial maximum 100, stepwise scoring, Wu length alpha 0.9 and summary search coverage beta 5. Trigram blocking exempts punctuation and sentence tags. The coverage setting belongs to search scoring. BART checkpoint min_length is 56 including decoder start, with early_stopping=True. PEGASUS min_length is 32 including start, with early_stopping=False. Natural-policy floors are bounded by input_tokens//8. The HF adapter subtracts one for min_new_tokens. Hugging Face calls use do_sample=False and forced_eos_token_id=None. Hard output caps are 256 tokens for OpenNMT and 512 for HF. Capped outputs may retry within workload guards. Exact logic is in the adapters and article_summarizer.py.')

s = base('Training conventions and input windows', 'Checkpoint provenance and inspected application capacity guards', False)
for y, heading, desc in [(1.75, 'Historical OpenNMT training', 'First 400 source tokens, target length 100'), (3.14, 'Current OpenNMT app work window', '1024 tokens per chunk, bounded by positional capacity'), (4.53, 'BART and PEGASUS input capacity', '1024 positions, including each tokenizer’s special tokens')]:
    text(s, heading, .7, y, 11.9, .48, 28, TEAL, True)
    text(s, desc, .7, y + .59, 11.9, .51, 26)
text(s, 'A capacity setting does not establish factual quality at that length.', .7, 6.39, 11.9, .51, 26, MUTED)
backup_notes(s, 'The OpenNMT training convention was 400 source tokens and 100 target tokens. LSTM recurrence has no learned absolute-position ceiling. Transformer saved positional buffers have length 5000, with an adapter generation ceiling of 4999. Quality at 5000 tokens has not been validated. The app uses a 1024-token source window and a 256-token output cap for OpenNMT. Hugging Face configurations allow 1024 input positions, an adapter ceiling of 1023 new output tokens and an app output cap of 512. BART includes two special source tokens and PEGASUS includes EOS. The historical fixed-budget API retains 400 tokens and explicit truncation permission. The separate shared-prefix track is bounded at 400 OpenNMT tokens.')

for name in ORDER[:4]:
    s = base(f'Saved Riverton output · {name}', 'Exact reproduced display text • latest fictional diagnostic • no new generation', False)
    label(s, name, .7, 1.7, 11.7, 30)
    rect(s, .7, 2.47, .07, 2.8, COLORS[name])
    text(s, OUTPUTS[name], 1.12, 2.53, 11.05, 3.2, 28)
    observation = {'LSTM': 'Omits lifting, completed repairs and testing.', 'Transformer': 'Omits lifting, completed repairs and testing.', 'BART': 'Includes negative tests. Omits lifting and completed repairs.', 'PEGASUS': 'Includes lifting. Omits completed repairs and testing.'}[name]
    text(s, observation, .7, 6.1, 11.9, .83, 26, COLORS[name], True)
    backup_notes(s, 'Exact display from the latest diagnosis. ' + observation + '\nAll models received the same 315-word, 2202-character body in one chunk and one call. Input token counts were 359 for LSTM, 359 for Transformer, 405 for BART and 368 for PEGASUS. Generated token counts before natural EOS were 54, 45, 59 and 46, respectively. The original identical reported LSTM and Transformer strings match Transformer. Reproduced direct, pipeline and fresh Gradio LSTM outputs differ, including the schools sentence. Dispatch and caching checked correctly. The original browser response and selected value were not captured, leaving its cause unresolved. A routing bug has not been demonstrated. This single diagnostic cannot rank general quality.')

s = base('The source event and its evidence', 'Exact latest Riverton sentences • body-only input preserved in demo asset', False)
facts = [('EVENT', BODY.split('\n\n')[0].split('. The advisory')[0] + '.'), ('REPAIR', 'Repair crews replaced the damaged valve Tuesday evening.'), ('TESTING', 'Two consecutive rounds of testing found no harmful bacteria, allowing the town to lift the advisory.')]
for i, (heading, value) in enumerate(facts):
    y = 1.72 + i * 1.63
    text(s, heading, .7, y, 2.1, .44, 22, TEAL, True)
    text(s, value, 3.05, y - .02, 9.5, 1.43, 26)
backup_notes(s, 'These are exact source sentences. The full body is in assets/riverton_article.txt, with SHA-256 648c2a202c8912f6d1fc5bbe468162d2670be52508916bbbf4ac7fd7e5457fa8. Omitting the title matches the original log counts of 315 words and 2202 characters. Title exclusion is inferred from those counts because the original log did not retain the source. This latest case is separate from the older fixture and RIVERTON_DIAGNOSIS.md.')

s = base('Offline demo and fallback checklist', 'Documented Ubuntu command • prepared cached environment', False)
text(s, 'From the repository root', .7, 1.64, 11.9, .47, 28, TEAL, True)
rect(s, .7, 2.37, 11.9, 1.31, PALE)
text(s, 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n.venv/bin/python app.py --port 7860', .96, 2.62, 11.36, .86, 24, font='Consolas')
text(s, 'Execute as one shell line. Open http://127.0.0.1:7860/', .7, 3.96, 11.9, .47, 24, MUTED)
text(s, 'Paste the saved body and identify the lifted-advisory sentence.\nCheck the model name, submit one request, then read its result.\nAfter about 30 seconds, continue with the saved output slides.', .7, 4.82, 11.9, 1.55, 26)
backup_notes(s, 'Execute this command as one shell line.\nHF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python app.py --port 7860\n\nStop an existing app on port 7860 before relaunching. Use the existing cached Ubuntu CPU setup. Cold loading can be slow. The input is assets/riverton_article.txt and full saved outputs are in assets/saved_demo_outputs.md. Read the model name with its response. Submit at most one live request and wait about 30 seconds within the slide 14 allocation. Identify saved results clearly. The redesign captured the app page without running inference, training or a benchmark.')

s = base('Attention and the copy mixture', 'Local OpenNMT CopyGenerator and decoder implementation', False)
text(s, 'Context = weighted sum of source representations', .7, 1.78, 11.9, .59, 30, TEAL, True)
text(s, 'Attention weights depend on the current decoder query.', .7, 2.62, 11.9, .52, 26)
line(s, .7, 3.48, 12.6, 3.48)
text(s, 'p(word) = (1 − g) p_vocab(word) + g p_source(word)', .7, 3.97, 11.9, .68, 29, TEAL, True)
text(s, 'g is a learned sigmoid gate. Source probability sums attention\nonto matching source words in the dynamic dictionary.', .7, 4.98, 11.9, 1.1, 26)
text(s, 'Copying a word preserves its spelling. Selecting its role still needs checking.', .7, 6.42, 11.9, .52, 24, MUTED)
backup_notes(s, 'CopyGenerator computes sigmoid(linear_copy(hidden)). It applies vocabulary softmax excluding PAD, scales vocabulary probabilities by 1-p_copy and source attention by p_copy, then projects through src_map and concatenates the distributions. Repeated source words accumulate through the source map. Prediction collapse handles words present in both vocabularies. LSTM uses MLP global attention. Transformer uses final cross-attention. Learned coverage attention is absent from both checkpoints. The beta 5 setting is beam-search coverage scoring.')

prs.save(HERE / (STEM + '.pptx'))
(HERE / 'assets/riverton_article.txt').write_text(BODY, encoding='utf-8')
(HERE / 'assets/saved_demo_outputs.md').write_text('# Saved latest Riverton outputs\n\nExact reproduced display text from the [latest diagnosis](../../reports/LAPTOP_MODEL_DIAGNOSIS.md).\n\n' + '\n\n'.join(f'## {name}\n\n{OUTPUTS[name]}' for name in ORDER[:4]) + '\n', encoding='utf-8')
(HERE / 'speaker_notes.json').write_text(json.dumps(talks, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
(HERE / 'assets/slide_manifest.json').write_text(json.dumps(slides, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
(HERE / 'assets/text_layout_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
with (HERE / 'assets/benchmark_values.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, lineterminator='\n')
    keys = ['n', 'failed', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'mean_words']
    writer.writerow(['track', 'system'] + keys)
    for track in ['full_article', 'common_input']:
        for name in ORDER:
            writer.writerow([track, name] + [METRICS['results'][track][name][key] for key in keys])

script = ['# Speaking script', '', 'Biruk Anley and Daniela Ortega. Sixteen main slides, 15:00 including transitions and the brief demo.', '', 'References and backups follow the timed talk. Slide 14 receives 100 seconds, including up to about 30 seconds of CPU wait.', '']
elapsed = 0
for entry in talks:
    end = elapsed + entry['duration_seconds']
    start_time = f'{elapsed // 60:02d}:{elapsed % 60:02d}'
    end_time = f'{end // 60:02d}:{end % 60:02d}'
    script += [f"## Slide {entry['slide']} · {entry['title']}", '', f"{entry['presenter']} · {start_time}–{end_time} · {entry['duration_seconds']} seconds", '', entry['speech'], '']
    if entry['stage_directions']:
        script += ['**Stage direction**', '', entry['stage_directions'], '']
    elapsed = end
script += ['## Demo checklist for Ubuntu', '', 'Use the existing cached setup. Stop an existing app on port 7860 before relaunching. From the repository root, run this as one shell line.', '', '```bash', 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python app.py --port 7860', '```', '', 'Open http://127.0.0.1:7860/. Paste [the saved body](assets/riverton_article.txt), without the title. Read the model name with its result. Submit at most one live request during the talk. After about 30 seconds, use the complete saved outputs on slides 25–28 and continue. Exact fallback text is also in [saved outputs](assets/saved_demo_outputs.md).', '', 'The earlier duplicate LSTM/Transformer display remains unresolved. Fresh direct, pipeline and Gradio checks produced a distinct LSTM output. Describe the saved reproduced outputs. Detailed evidence is in slide notes and [references](REFERENCES.md).', '', '## Guideline and preparation scope', '', 'No course guideline was located in the repository or supplied attachment. The deck follows the supplied redesign brief. This redesign captured the current app page without submitting an article. It ran no inference, training or benchmark.', '']
(HERE / 'SPEAKING_SCRIPT.md').write_text('\n'.join(script), encoding='utf-8')
print(f'Built {len(prs.slides)} slides, {len(talks)} main, {elapsed} seconds.')
print('Text estimates requiring review', json.dumps([a for a in audit if not a['fits_estimate']], ensure_ascii=True))
