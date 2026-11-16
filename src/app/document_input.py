"""Read document text for the desktop application."""
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from xml.etree import ElementTree as ET

def read_document(path: Path) -> str:
    path = Path(path)
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError('Choose a document smaller than 20 MB.')
    suffix = path.suffix.lower()
    if suffix == '.txt':
        text = path.read_text(encoding='utf-8-sig')
    elif suffix == '.docx':
        try:
            with ZipFile(path) as archive:
                if archive.getinfo('word/document.xml').file_size > 20 * 1024 * 1024:
                    raise ValueError('The document text exceeds the 20 MB import limit.')
                root = ET.fromstring(archive.read('word/document.xml'))
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            text = '\n'.join(
                ''.join((n.text or '') if n.tag.endswith('}t') else '\t' if n.tag.endswith('}tab') else '\n'
                        for n in p.iter() if n.tag.rsplit('}',1)[-1] in {'t','tab','br','cr'})
                for p in root.findall('.//w:p', ns))
        except (BadZipFile, KeyError, ET.ParseError) as exc:
            raise ValueError('The Word document could not be read.') from exc
    elif suffix == '.pdf':
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError('PDF import requires pypdf. PDF support is pending installation.') from exc
        try:
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise ValueError('Save an unlocked copy of the PDF before importing.')
            if len(reader.pages) > 100:
                raise ValueError('Choose a PDF with 100 pages or fewer.')
            pages = [page.extract_text() or '' for page in reader.pages]
            if any(not page.strip() for page in pages):
                raise ValueError('A PDF page has no readable text. Run OCR or remove blank pages before importing.')
            text = '\n\n'.join(pages)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError('The PDF document could not be read.') from exc
    else:
        raise ValueError('Choose a TXT, DOCX or PDF document.')
    if not text.strip():
        raise ValueError('No readable text was found. For scanned documents, run OCR before importing.')
    return text
