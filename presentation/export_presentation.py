"""Export with installed Windows PowerPoint, then create contact sheets.

Requires pywin32 311, Pillow 11.3.0 and PyMuPDF 1.26.6 in presentation tooling.
The source PPTX is opened read-only. No model or application code is imported.
"""
import json
import math
from pathlib import Path

import fitz
import win32com.client
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
STEM = 'Comparing_Neural_Models_for_News_Summarization'


def main():
    rendered = HERE / 'rendered'
    rendered.mkdir(exist_ok=True)
    app = win32com.client.DispatchEx('PowerPoint.Application')
    deck = None
    measurements = []
    try:
        app.DisplayAlerts = 1
        deck = app.Presentations.Open(str(HERE / (STEM + '.pptx')), ReadOnly=True, WithWindow=False)
        for number in range(1, deck.Slides.Count + 1):
            slide = deck.Slides(number)
            for shape in slide.Shapes:
                if shape.HasTextFrame and shape.TextFrame.HasText:
                    frame = shape.TextFrame2
                    bounds = frame.TextRange
                    measurements.append({
                        'slide': number, 'text': shape.TextFrame.TextRange.Text,
                        'box_width': shape.Width, 'box_height': shape.Height,
                        'text_width': bounds.BoundWidth, 'text_height': bounds.BoundHeight,
                        'fits': bounds.BoundWidth <= shape.Width + 2 and bounds.BoundHeight <= shape.Height + 2,
                    })
            slide.Export(str(rendered / f'slide-{number:02d}.png'), 'PNG', 1600, 900)
        deck.SaveAs(str(HERE / (STEM + '.pdf')), 32)
    finally:
        if deck is not None:
            deck.Close()
        app.Quit()
    (HERE / 'assets/powerpoint_text_audit.json').write_text(json.dumps(measurements, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    document = fitz.open(HERE / (STEM + '.pdf'))
    (HERE.parent / '.cache/presentation_pdf_text.json').write_text(json.dumps([page.get_text() for page in document], indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    paths = sorted(rendered.glob('slide-*.png'))
    make_sheet(paths, rendered / 'contact_sheet.png', columns=4, width=480)
    # Larger sheets are temporary inspection aids, kept in the ignored cache.
    cache = HERE.parent / '.cache'
    for start in range(0, len(paths), 4):
        make_sheet(paths[start:start + 4], cache / f'presentation-review-{start // 4 + 1:02d}.png', columns=2, width=1000)
    issues = [row for row in measurements if not row['fits']]
    print(f'Exported {len(document)} PDF pages and {len(paths)} PNG previews.')
    print('PowerPoint text bounds requiring review', json.dumps(issues, ensure_ascii=True))


def make_sheet(paths, destination, columns, width):
    height = round(width * 9 / 16)
    gutter, caption = 16, 27
    canvas = Image.new('RGB', (columns * (width + gutter) + gutter,
                              math.ceil(len(paths) / columns) * (height + caption + gutter) + gutter), '#E6E8E2')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/calibri.ttf', 20)
    for i, path in enumerate(paths):
        x, y = gutter + (i % columns) * (width + gutter), gutter + (i // columns) * (height + caption + gutter)
        with Image.open(path) as source:
            canvas.paste(source.resize((width, height), Image.Resampling.LANCZOS), (x, y))
        draw.text((x, y + height + 3), f'Slide {int(path.stem.split("-")[1])}', font=font, fill='#25312F')
    canvas.save(destination, optimize=True)


if __name__ == '__main__':
    main()
