"""Run the same frozen models and exports from the command line."""
import argparse
from pathlib import Path
from src.app.service import Detector, read_batch, NAMES
from src.app.document_input import read_document
from src.app.export import export_report, export_batch

def main():
    parser=argparse.ArgumentParser(description='Analyze English documents locally.')
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--model',choices=[*NAMES,'Compare All'],default='Combined Features')
    parser.add_argument('--threshold',choices=['Default','Low False Positive'],default='Default')
    parser.add_argument('--batch',action='store_true')
    args=parser.parse_args()
    detector=Detector()
    if args.batch:
        result=detector.batch(read_batch(args.input),args.model,args.threshold)
        if args.output.suffix.lower()=='.csv':export_batch(result,args.output)
        else:export_report({'documents':[{'name':r['id'],**r} for r in result]},args.output)
    else:
        result=detector.run(read_document(args.input),args.model,args.threshold)
        export_report(result,args.output)
    print('Saved '+str(args.output))

if __name__=='__main__':main()
