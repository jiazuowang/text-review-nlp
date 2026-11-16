"""Portable English result exports with one content source for Word and PDF."""
import csv
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

SCOPE = ('English text is accepted without an input word-count limit. '
         'The models were evaluated on English Beemo documents containing 50 to 1,000 words; '
         'results outside that range have not been validated. '
         'Scores describe the model response to the document. Performance varies with topic and generator distribution.')
LOCAL = ('Passage impact is the original document score minus the score after removing the passage. '
         'Positive values support the AI-origin prediction; negative values support the human-origin prediction. '
         'Context scores use overlapping 60-word windows. Local authorship validation used synthetic mixed-source blocks; '
         'authentic sentence-level authorship requires further validation.')

def clean(text):
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', str(text))

def sections(report, title='Text Analysis Report'):
    content = [('title', title), ('body', SCOPE)]
    items = report.get('documents', [report])
    for item in items:
        if item.get('name'):
            content.append(('heading', item['name']))
        if item.get('error'):
            content.append(('body', 'Input status: '+item['error']))
        for r in item['results']:
            content.extend([('heading', r['model']), ('body',
                f"Prediction: {r['prediction']}. AI-origin score: {r['score']:.4f}. "
                f"Threshold: {r['threshold']:.4f} ({r['threshold_mode']}). Word count: {r['word_count']}.")])
            if r.get('sentences'):
                content.append(('body', LOCAL))
                measured = [s for s in r['sentences'] if s['deletion_impact'] is not None]
                for s in sorted(measured, key=lambda s: abs(s['deletion_impact']), reverse=True)[:10]:
                    content.append(('body', f"Impact {s['deletion_impact']:+.4f}; context score {s['context_score']:.4f}: {s['text'].strip()}"))
            if r.get('suggestions'):
                content.append(('heading', 'Writing Suggestions'))
                for tip in r['suggestions']:
                    content.append(('body', f"{tip['evidence']}\n{tip['suggestion']}"))
        content.append(('heading', 'Analyzed Text'))
        content.extend(('body', p) for p in item['text'].splitlines() if p.strip())
    return [(style, clean(text)) for style, text in content]

def write_docx(path, content):
    paragraphs = []
    for style, text in content:
        sid = {'title': 'Title', 'heading': 'Heading1', 'body': 'Normal'}[style]
        runs = '<w:br/>'.join(f'<w:t xml:space="preserve">{escape(line)}</w:t>' for line in text.split('\n'))
        paragraphs.append(f'<w:p><w:pPr><w:pStyle w:val="{sid}"/></w:pPr><w:r>{runs}</w:r></w:p>')
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        + ''.join(paragraphs) + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body></w:document>')
    styles = '''<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/><w:color w:val="000000"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="300" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="180" w:after="120"/></w:pPr><w:rPr><w:b/></w:rPr></w:style></w:styles>'''
    with ZipFile(path, 'w', ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
        z.writestr('_rels/.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        z.writestr('word/_rels/document.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
        z.writestr('word/document.xml', document)
        z.writestr('word/styles.xml', styles)

def write_pdf(path, content):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    fonts = Path(__file__).resolve().parents[2] / 'resources/fonts'
    import os
    windows_fonts = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    regular = windows_fonts/'times.ttf'
    bold = windows_fonts/'timesbd.ttf'
    if not regular.exists() or not bold.exists():
        regular, bold = fonts/'DejaVuSerif.ttf', fonts/'DejaVuSerif-Bold.ttf'
    if 'ReviewSerif' not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont('ReviewSerif', str(regular)))
        pdfmetrics.registerFont(TTFont('ReviewSerifBold', str(bold)))
    story = []
    for kind, text in content:
        style = ParagraphStyle(kind, fontName='ReviewSerif' if kind == 'body' else 'ReviewSerifBold',
                               fontSize=12, leading=15, spaceAfter=8, spaceBefore=8 if kind == 'heading' else 0,
                               keepWithNext=kind != 'body', splitLongWords=True)
        story.append(Paragraph(escape(text).replace('\n', '<br/>'), style))
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72,
                      title='Text Analysis Report', author='').build(story)

def export_report(report, path):
    path = Path(path)
    if path.suffix.lower() not in {'.pdf', '.docx', '.json'}:
        raise ValueError('Choose PDF, Word (.docx) or JSON.')
    with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as f:
        tmp = Path(f.name)
    try:
        if path.suffix.lower() == '.json':
            tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        elif path.suffix.lower() == '.docx':
            write_docx(tmp, sections(report))
        else:
            write_pdf(tmp, sections(report))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)

def safe_cell(value):
    value = str(value)
    return "'" + value if value.lstrip().startswith(('=', '+', '-', '@')) else value

def export_batch(rows, path):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'model', 'prediction', 'score', 'threshold', 'status', 'error'])
        for row in rows:
            for r in row['results'] or [{}]:
                writer.writerow([safe_cell(row['id']), r.get('model', ''), r.get('prediction', ''),
                                 r.get('score', ''), r.get('threshold', ''), row['status'], safe_cell(row.get('error', ''))])
