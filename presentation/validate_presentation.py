"""Validate presentation artifacts against saved evidence, without inference."""
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import fitz
from pptx import Presentation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STEM = 'Comparing_Neural_Models_for_News_Summarization'
NS = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart',
      's': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
ORDER = ['LSTM', 'Transformer', 'BART', 'PEGASUS', 'Lead-3']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def normalize(value):
    return re.sub(r'\s+', ' ', value).strip()


def main():
    deck = Presentation(HERE / (STEM + '.pptx'))
    metrics = read_json(ROOT / 'reports/pilot50/results.json')['results']
    notes = read_json(HERE / 'speaker_notes.json')
    manifest = read_json(HERE / 'assets/slide_manifest.json')
    assert len(deck.slides) == len(manifest) == 31
    assert sum(s['main'] for s in manifest) == 16
    assert len(notes) == 16 and sum(n['duration_seconds'] for n in notes) == 900
    assert abs(deck.slide_width / deck.slide_height - 16 / 9) < .00001
    for i, slide in enumerate(deck.slides):
        embedded = slide.notes_slide.notes_text_frame.text
        assert len(embedded) > 100, (i + 1, 'missing notes')
        if i < 16:
            assert notes[i]['speech'] in embedded
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0
            assert shape.left + shape.width <= deck.slide_width + 1000, (i + 1, 'off-slide width')
            assert shape.top + shape.height <= deck.slide_height + 1000, (i + 1, 'off-slide height')

    with zipfile.ZipFile(HERE / (STEM + '.pptx')) as package:
        charts = sorted((n for n in package.namelist() if re.fullmatch(r'ppt/charts/chart\d+\.xml', n)),
                        key=lambda n: int(re.search(r'(\d+)\.xml', n)[1]))
        workbooks = sorted(n for n in package.namelist() if n.startswith('ppt/embeddings/') and n.endswith('.xlsx'))
        assert len(charts) == len(workbooks) == 6
        for i, chart_path in enumerate(charts):
            metric = ['rouge1', 'rouge2', 'rougeLsum'][i % 3]
            expected_models = ['LSTM', 'Transformer', 'Lead-3'] if i < 3 else ORDER
            chart = ET.fromstring(package.read(chart_path))
            series = chart.find('.//c:ser', NS)
            categories = [p.text for p in series.findall('.//c:cat//c:pt/c:v', NS)]
            values = [float(p.text) for p in series.findall('.//c:val//c:pt/c:v', NS)]
            assert categories == expected_models
            expected_values = [metrics['full_article'][name][metric] for name in expected_models]
            assert all(abs(a - b) < 1e-10 for a, b in zip(values, expected_values))
            assert float(chart.find('.//c:valAx/c:scaling/c:min', NS).get('val')) == 0
            assert float(chart.find('.//c:valAx/c:scaling/c:max', NS).get('val')) == 50
            workbook_path = next(n for n in workbooks if n.endswith(f'Sheet{i + 1}.xlsx'))
            with zipfile.ZipFile(io.BytesIO(package.read(workbook_path))) as workbook:
                sheet = ET.fromstring(workbook.read('xl/worksheets/sheet1.xml'))
                cells = {cell.get('r'): cell.find('s:v', NS).text for cell in sheet.findall('.//s:c', NS)}
                assert all(abs(float(cells[f'B{j + 2}']) - value) < 1e-10 for j, value in enumerate(expected_values))

    with (HERE / 'assets/benchmark_values.csv').open(encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 10
    for row in rows:
        expected = metrics[row['track']][row['system']]
        for key in ['n', 'failed', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'mean_words']:
            assert float(row[key]) == expected[key]

    body = (HERE / 'assets/riverton_article.txt').read_text(encoding='utf-8')
    assert len(body) == 2202 and len(body.split()) == 315
    assert hashlib.sha256(body.encode()).hexdigest() == '648c2a202c8912f6d1fc5bbe468162d2670be52508916bbbf4ac7fd7e5457fa8'
    pdf = fitz.open(HERE / (STEM + '.pdf'))
    assert len(pdf) == 31
    pages = [normalize(page.get_text()) for page in pdf]
    all_text = ' '.join(pages)
    report = (ROOT / 'reports/LAPTOP_MODEL_DIAGNOSIS.md').read_text(encoding='utf-8')
    for value in re.findall(r'^> ([^\n]+)$', report, re.M)[-4:]:
        assert normalize(value) in all_text, ('missing complete saved output', value[:45])
    for index, models in [(10, ['LSTM', 'Transformer', 'Lead-3']), (12, ORDER)]:
        for model in models:
            assert model in pages[index]
            for metric in ['rouge1', 'rouge2', 'rougeLsum']:
                assert f'{metrics["full_article"][model][metric]:.2f}' in pages[index], ('wrapped chart number', index + 1, model, metric)
    for index, track in [(18, 'full_article'), (19, 'common_input')]:
        for model in ORDER:
            for metric in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
                assert f'{metrics[track][model][metric]:.2f}' in pages[index]
    assert 'Word embeddings' in pages[4]
    assert 'Cross-attention' in pages[6]
    dataset = read_json(HERE / 'assets/dataset_example.json')
    assert normalize(dataset['article_excerpt']) in pages[2]
    assert normalize(dataset['reference_excerpt']) in pages[2]
    assert dataset['id'] in read_json(ROOT / 'reports/pilot50/results.json')['ids']
    assert len(list((HERE / 'rendered').glob('slide-*.png'))) == 31
    assert all(row['fits_estimate'] for row in read_json(HERE / 'assets/text_layout_audit.json'))
    assert all(row['fits'] for row in read_json(HERE / 'assets/powerpoint_text_audit.json'))
    for relative, expected in read_json(HERE / 'assets/evidence_manifest.json')['sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, ('evidence changed', relative)
    for filename in ['README.md', 'SPEAKING_SCRIPT.md', 'REFERENCES.md', 'VALIDATION.md']:
        path = HERE / filename
        for link in re.findall(r'(?<!!)\[[^]]+\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if not link.startswith(('http:', 'https:', '#')):
                relative = link.split('#', 1)[0]
                assert not relative or (path.parent / relative).exists(), (filename, link)
    print('PASS — 31 slides, 16 main, 900 seconds, embedded notes and on-slide bounds.')
    print('PASS — 6 editable charts and embedded workbooks, exact saved metrics and both tables.')
    print('PASS — 31 PDF pages/previews, unwrapped chart values, complete exact saved outputs.')
    print('PASS — PowerPoint text bounds, preserved v1 hashes, app and frozen evidence hashes, links.')


if __name__ == '__main__':
    main()
