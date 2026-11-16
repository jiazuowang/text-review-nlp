"""Integration checks for input scope, frozen inference, exports and cancellation."""
import csv
import json
from pathlib import Path
import tempfile
import threading
import unittest
from zipfile import ZipFile
from xml.etree import ElementTree as ET
from src.app.service import Detector, Cancelled, validate_text, read_batch
from src.app.document_input import read_document
from src.app.export import export_report, export_batch, write_docx, write_pdf
import gc

TEXT = ('The team collected measurements from the sensor during the afternoon trial. '
        'Each reading was compared with a reference value recorded under controlled conditions. '
        'Several observations changed after the temperature increased, so the researchers repeated the experiment. '
        'They documented the differences and checked the equipment before beginning another series of measurements. '
        'The final report includes the procedure, the observed variation, and the conditions for repeating the experiment.')

class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector=Detector()
        cls.report=cls.detector.run(TEXT,'Compare All')

    def test_all_models_and_thresholds(self):
        self.assertEqual(len(self.report['results']),3)
        low=self.detector.run(TEXT,'Compare All','Low False Positive',local=False)
        for a,b in zip(self.report['results'],low['results']):
            self.assertAlmostEqual(a['score'],b['score'],places=12)
            self.assertGreater(b['threshold'],a['threshold'])

    def test_local_offsets_and_english(self):
        for r in self.report['results']:
            for s in r['sentences']:
                self.assertEqual(s['text'],TEXT[s['start']:s['end']])
            self.assertFalse(any('\u4e00'<=c<='\u9fff' for c in json.dumps(r,ensure_ascii=False)))

    def test_scope(self):
        for text in ['', '   ', '123 !!!', TEXT+' 中文']:
            with self.assertRaises(ValueError):validate_text(text)
        for count in [1, 49, 50, 1000, 1001, 5000]:
            self.assertEqual(validate_text('word '*count),count)

    def test_inference_without_word_count_limits(self):
        for text,count in [('Hello.',1),('word '*49,49),('word '*1001,1001)]:
            for local in [True,False]:
                with self.subTest(words=count,local=local):
                    report=self.detector.run(text,'Compare All',local=local)
                    for result in report['results']:
                        self.assertEqual(result['status'],'ok')
                        self.assertEqual(result['word_count'],count)
                        self.assertGreaterEqual(result['score'],0)
                        self.assertLessEqual(result['score'],1)

    def test_short_sentence_impacts_and_empty_analysis(self):
        from src.app.analysis import analyze
        single=self.detector.run('Hello.')['results'][0]
        self.assertIsNone(single['sentences'][0]['deletion_impact'])
        self.assertEqual(single['sentences'][0]['impact_status'],'no_remaining_words')
        pair=self.detector.run('Hello world. Another sentence.')['results'][0]
        self.assertTrue(all(s['impact_status']=='measured' for s in pair['sentences']))
        self.assertTrue(all(s['deletion_impact'] is not None for s in pair['sentences']))
        with self.assertRaisesRegex(ValueError,'non-empty'):
            analyze('123 !!!',self.detector.load('X'))

    def test_batch_accepts_short_and_long_documents(self):
        rows=self.detector.batch([{'id':'short','text':'Hello.'},{'id':'long','text':'word '*1001}],
                                 'Compare All','Default')
        self.assertEqual([r['status'] for r in rows],['Complete','Complete'])
        self.assertTrue(all(len(r['results'])==3 for r in rows))

    def test_cancel_before_loading(self):
        cancel=threading.Event();cancel.set()
        with self.assertRaises(Cancelled):self.detector.run(TEXT,cancel=cancel)

    def test_batch_invalid_row_and_csv_escape(self):
        rows=self.detector.batch([{'id':'=1+1','text':TEXT},{'id':'empty','text':''}],'Lexical Features','Default')
        self.assertEqual([r['status'] for r in rows],['Complete','Invalid input'])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'results.csv';export_batch(rows,p)
            with p.open(encoding='utf-8-sig',newline='') as f:actual=list(csv.DictReader(f))
            self.assertEqual(actual[0]['id'],"'=1+1")
            self.assertEqual(actual[1]['score'],'')

    def test_csv_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.csv';p.write_text('id,text\na,one\na,two',encoding='utf-8')
            with self.assertRaises(ValueError):read_batch(p)

    def test_docx_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report.docx';export_report(self.report,p)
            text=read_document(p)
            self.assertIn(TEXT,text)
            for r in self.report['results']:self.assertIn(f"{r['score']:.4f}",text)
            with ZipFile(p) as z:
                for name in z.namelist():ET.fromstring(z.read(name))

    def test_pdf_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report.pdf';export_report(self.report,p)
            text=read_document(p)
            self.assertIn('Text Analysis Report',text)
            for r in self.report['results']:self.assertIn(f"{r['score']:.4f}",text)

    def test_encrypted_and_scanned_pdf(self):
        from pypdf import PdfWriter
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'blank.pdf';w=PdfWriter();w.add_blank_page(width=100,height=100);w.write(p)
            with self.assertRaisesRegex(ValueError,'OCR'):read_document(p)
            p=Path(d)/'locked.pdf';w.encrypt('secret');w.write(p)
            with self.assertRaisesRegex(ValueError,'unlocked'):read_document(p)

    def test_bad_docx(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.docx';p.write_bytes(b'invalid')
            with self.assertRaises(ValueError):read_document(p)

    def test_empty_and_unicode_documents(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'text.txt';p.write_text('',encoding='utf-8')
            with self.assertRaises(ValueError):read_document(p)
            p=Path(d)/'unicode.docx';write_docx(p,[('body','Temperature α and β\nNext line & details')])
            self.assertIn('α and β\nNext line & details',read_document(p))

    def test_hash_guard(self):
        detector=Detector();detector.config['models']['B']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'integrity'):detector.run(TEXT,'Lexical Features')

    def test_gui_result_invalidation(self):
        from src.app.desktop import Application
        app=Application();app.withdraw();app.update()
        try:
            app.single.insert('1.0',TEXT);app.update()
            app.saved['Text Analysis']=self.report
            app.snapshots['Text Analysis']=app.signature('Text Analysis')
            app.display('Text Analysis',self.report)
            self.assertIn('AI-origin score',app.result_boxes['Overview'].get('1.0','end'))
            app.single.insert('end',' Changed.');app.update()
            self.assertNotIn('Text Analysis',app.saved)
            self.assertEqual(app.result_boxes['Overview'].get('1.0','end').strip(),'')
        finally:
            app.close()
            del app
            gc.collect()

if __name__=='__main__':unittest.main()
