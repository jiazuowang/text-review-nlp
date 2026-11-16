"""English desktop application with asynchronous local inference."""
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from src.app.ui_preview import Preview, BG, WHITE, INK, GREEN, PALE, FONT
from src.app.service import Detector, Cancelled, read_batch, check_cancel
from src.app.document_input import read_document
from src.app.export import export_report, export_batch, LOCAL

class Application(Preview):
    def __init__(self):
        self.detector = Detector()
        self.events = queue.Queue()
        self.cancel_event = threading.Event()
        self.busy = False
        self.job = 0
        self.snapshots = {}
        self.saved = {}
        self.batch_rows = []
        self.action_buttons = []
        super().__init__()
        self.title('Text Review')
        self.geometry('1240x820')
        self.minsize(1120,740)
        self.model.bind('<<ComboboxSelected>>', self.invalidate_all)
        self.threshold.bind('<<ComboboxSelected>>', self.invalidate_all)
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.after(100, self.poll)

    def output_box(self, parent):
        frame = tk.Frame(parent, bg=WHITE)
        frame.pack(fill='both', expand=True, padx=16, pady=12)
        text = tk.Text(frame, wrap='word', font=FONT, bg=WHITE, fg=INK, relief='flat',
                       padx=8, pady=8, width=1, height=1, spacing3=8, state='disabled')
        scroll = ttk.Scrollbar(frame, command=text.yview)
        scroll.pack(side='right', fill='y')
        text.configure(yscrollcommand=scroll.set)
        text.pack(fill='both', expand=True)
        return text

    def set_output(self, widget, value):
        widget.configure(state='normal')
        widget.delete('1.0','end')
        widget.insert('1.0',value)
        widget.configure(state='disabled')

    def editor_watch(self, editor, page):
        def changed(event=None):
            if editor.edit_modified():
                editor.edit_modified(False)
                self.invalidate(page)
        editor.bind('<<Modified>>', changed)
        editor.edit_modified(False)

    def footer(self, parent, page, action):
        bar = tk.Frame(parent, bg=BG)
        bar.pack(fill='x', pady=(14,0))
        progress = tk.StringVar(value='')
        tk.Label(bar, textvariable=progress, bg=BG, fg=INK, font=FONT).pack(side='left')
        run = self.button(bar, 'Analyze' if page != 'Compare Drafts' else 'Compare', action, True)
        run.pack(side='right')
        self.action_buttons.append(run)
        cancel = self.button(bar, 'Cancel', self.cancel)
        export = self.button(bar, 'Export', lambda:self.export(page))
        self.controls[page] = (progress, run, cancel, export)

    def make_single(self):
        self.controls = {}
        page = tk.Frame(self.area, bg=BG); self.pages['Text Analysis'] = page
        grid = tk.Frame(page, bg=BG); grid.pack(fill='both',expand=True)
        grid.columnconfigure(0,weight=1);grid.columnconfigure(1,weight=1);grid.rowconfigure(0,weight=1)
        left = self.card(grid);left.grid(row=0,column=0,sticky='nsew',padx=(0,16))
        right = self.card(grid);right.grid(row=0,column=1,sticky='nsew')
        head = tk.Frame(left,bg=WHITE);head.pack(fill='x',padx=20,pady=16)
        self.label(head,'Input').pack(side='left')
        self.single = self.editor(left)
        self.button(head,'Import Document',lambda:self.import_document(self.single)).pack(side='right')
        self.editor_watch(self.single,'Text Analysis')
        tabs=ttk.Notebook(right);tabs.pack(fill='both',expand=True,padx=12,pady=16)
        self.result_boxes={}
        for title in ['Overview','Passages','Suggestions']:
            pane=tk.Frame(tabs,bg=WHITE);tabs.add(pane,text=title)
            self.result_boxes[title]=self.output_box(pane)
        self.footer(page,'Text Analysis',self.run_single)

    def make_pair(self):
        page=tk.Frame(self.area,bg=BG);self.pages['Compare Drafts']=page
        grid=tk.Frame(page,bg=BG);grid.pack(fill='both',expand=True)
        grid.columnconfigure(0,weight=1);grid.columnconfigure(1,weight=1);grid.rowconfigure(0,weight=1)
        self.pair_editors=[]
        for i,name in enumerate(['Original','Revised']):
            card=self.card(grid);card.grid(row=0,column=i,sticky='nsew',padx=(0,8) if i==0 else (8,0))
            head=tk.Frame(card,bg=WHITE);head.pack(fill='x',padx=16,pady=12)
            self.label(head,name).pack(side='left')
            editor=self.editor(card);self.pair_editors.append(editor)
            self.button(head,'Import Document',lambda e=editor:self.import_document(e)).pack(side='right')
            self.editor_watch(editor,'Compare Drafts')
        result=self.card(page);result.pack(fill='x',pady=(16,0))
        self.pair_output=tk.Text(result,height=5,wrap='word',font=FONT,bg=WHITE,fg=INK,relief='flat',padx=16,pady=12,state='disabled')
        self.pair_output.pack(fill='x')
        self.footer(page,'Compare Drafts',self.run_pair)

    def make_batch(self):
        page=tk.Frame(self.area,bg=BG);self.pages['Batch Analysis']=page
        card=self.card(page);card.pack(fill='both',expand=True)
        head=tk.Frame(card,bg=WHITE);head.pack(fill='x',padx=20,pady=16)
        self.filename=tk.StringVar(value='')
        tk.Label(head,textvariable=self.filename,bg=WHITE,fg=INK,font=FONT).pack(side='left')
        self.button(head,'Import CSV',self.import_batch).pack(side='right')
        area=tk.Frame(card,bg=WHITE);area.pack(fill='both',expand=True,padx=16,pady=16)
        self.tree=ttk.Treeview(area,columns=('id','model','prediction','score','status'),show='headings')
        for key,title,width in [('id','ID',110),('model','Model',170),('prediction','Prediction',130),('score','Score',80),('status','Status',230)]:
            self.tree.heading(key,text=title);self.tree.column(key,width=width,minwidth=70,stretch=True)
        scroll=ttk.Scrollbar(area,command=self.tree.yview);scroll.pack(side='right',fill='y')
        self.tree.configure(yscrollcommand=scroll.set);self.tree.pack(fill='both',expand=True)
        self.footer(page,'Batch Analysis',self.run_batch)

    def signature(self, page):
        inputs = (self.single.get('1.0','end-1c'),) if page=='Text Analysis' else (
            tuple(e.get('1.0','end-1c') for e in self.pair_editors) if page=='Compare Drafts' else
            tuple((r['id'],r['text']) for r in self.batch_rows))
        return (inputs,self.model.get(),self.threshold.get())

    def invalidate(self, page):
        self.saved.pop(page,None)
        if page in getattr(self,'controls',{}):
            self.controls[page][3].pack_forget()
        if page=='Text Analysis':
            for box in getattr(self,'result_boxes',{}).values():self.set_output(box,'')
        elif page=='Compare Drafts' and hasattr(self,'pair_output'):self.set_output(self.pair_output,'')
        elif page=='Batch Analysis' and hasattr(self,'tree'):
            for item in self.tree.get_children():self.tree.delete(item)
            for row in self.batch_rows:self.tree.insert('','end',values=(row['id'],'','','','Pending'))

    def invalidate_all(self,event=None):
        for page in self.pages:self.invalidate(page)

    def import_document(self, editor):
        name=filedialog.askopenfilename(title='Import Document',filetypes=[('Documents','*.txt *.docx *.pdf'),('Word document','*.docx'),('PDF document','*.pdf'),('Text document','*.txt')])
        if not name:return
        try:text=read_document(Path(name))
        except Exception as exc:messagebox.showerror('Import failed',str(exc));return
        editor.delete('1.0','end');editor.insert('1.0',text);editor.focus_set()

    def import_batch(self):
        name=filedialog.askopenfilename(title='Import CSV',filetypes=[('CSV files','*.csv')])
        if not name:return
        try:rows=read_batch(name)
        except Exception as exc:messagebox.showerror('Import failed',str(exc));return
        self.batch_rows=rows;self.filename.set(Path(name).name);self.invalidate('Batch Analysis')

    def start(self,page,work):
        if self.busy:return
        self.invalidate(page)
        self.snapshots[page]=self.signature(page)
        self.busy=True;self.job+=1;job=self.job
        self.cancel_event=threading.Event();cancel=self.cancel_event
        for b in self.action_buttons:b.configure(state='disabled')
        self.controls[page][0].set('Analyzing...')
        self.controls[page][2].pack(side='right',padx=10)
        def worker():
            try:
                result=work(cancel)
                check_cancel(cancel)
                self.events.put((job,page,'done',result))
            except Cancelled:self.events.put((job,page,'cancelled',None))
            except Exception as exc:self.events.put((job,page,'error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def run_single(self):
        text=self.single.get('1.0','end-1c');selection=self.model.get();mode=self.threshold.get()
        self.start('Text Analysis',lambda c:self.detector.run(text,selection,mode,cancel=c))

    def run_pair(self):
        texts=[e.get('1.0','end-1c') for e in self.pair_editors];selection=self.model.get();mode=self.threshold.get()
        def work(cancel):
            docs=[]
            for name,text in zip(['Original','Revised'],texts):
                docs.append({'name':name,**self.detector.run(text,selection,mode,local=False,cancel=cancel)})
            return {'documents':docs}
        self.start('Compare Drafts',work)

    def run_batch(self):
        if not self.batch_rows:messagebox.showinfo('Import CSV','Import a CSV containing id and text columns.');return
        rows=list(self.batch_rows);selection=self.model.get();mode=self.threshold.get();job=self.job+1
        self.start('Batch Analysis',lambda c:self.detector.batch(rows,selection,mode,cancel=c,
            progress=lambda i,n:self.events.put((job,'Batch Analysis','progress',f'{i} / {n}'))))

    def cancel(self):
        self.cancel_event.set()

    def poll(self):
        try:
            while True:
                job,page,kind,value=self.events.get_nowait()
                if job!=self.job:continue
                progress,run,cancel,export=self.controls[page]
                if kind=='progress':progress.set(value);continue
                self.busy=False;cancel.pack_forget()
                for b in self.action_buttons:b.configure(state='normal')
                if kind=='error':progress.set('');messagebox.showerror('Analysis failed',value)
                elif kind=='cancelled':progress.set('Cancelled')
                elif self.signature(page)!=self.snapshots[page]:progress.set('Input changed. Run analysis again.')
                else:
                    progress.set('');self.saved[page]=value;self.display(page,value);export.pack(side='right',padx=10)
        except queue.Empty:pass
        self.after(100,self.poll)

    def display(self,page,report):
        if page=='Batch Analysis':
            for item in self.tree.get_children():self.tree.delete(item)
            for row in report:
                for r in row['results'] or [{}]:self.tree.insert('','end',values=(row['id'],r.get('model',''),r.get('prediction',''),f"{r['score']:.4f}" if 'score' in r else '',row.get('error',row['status'])))
            return
        if page=='Compare Drafts':
            original,revised=report['documents']
            text='\n'.join(f"{a['model']}: {a['score']:.4f} ({a['prediction']}) → {b['score']:.4f} ({b['prediction']}); change {b['score']-a['score']:+.4f}"
                           for a,b in zip(original['results'],revised['results']))
            self.set_output(self.pair_output,text);return
        overview=[];tips=[]
        for r in report['results']:
            overview.append(f"{r['model']}\n{r['prediction']}\nAI-origin score: {r['score']:.4f}\nThreshold: {r['threshold']:.4f}\nWords: {r['word_count']}")
        self.set_output(self.result_boxes['Overview'],'\n\n'.join(overview)+'\n\nScope: English text. No input word-count limit. The models were evaluated on 50–1,000-word documents; results outside that range have not been validated. The score describes the model response to this document.')
        r=report['results'][0]
        box=self.result_boxes['Passages'];box.configure(state='normal');box.delete('1.0','end')
        box.insert('end',r['model']+'\n\n')
        box.insert('end','Orange: supports AI origin. Blue: supports human origin.\nImpact measures the score change after removing a passage.\n\n')
        for i,s in enumerate(r['sentences']):
            start=box.index('end-1c');box.insert('end',s['text']);end=box.index('end-1c');impact=s['deletion_impact']
            if impact is not None and abs(impact)>=0.005:
                tag=f'impact{i}';box.tag_add(tag,start,end);box.tag_configure(tag,background='#f8dfc8' if impact>0 else '#d4e5f8',foreground=INK)
        box.insert('end','\n\nPassage measurements\n')
        for s in sorted(r['sentences'],key=lambda v:abs(v['deletion_impact'] or 0),reverse=True)[:10]:
            if s['deletion_impact'] is not None:box.insert('end',f"\n{s['deletion_impact']:+.4f}: {s['text'].strip()}\n")
        box.insert('end','\n'+LOCAL);box.configure(state='disabled')
        for tip in r['suggestions']:tips.append(tip['evidence']+'\n'+tip['suggestion'])
        self.set_output(self.result_boxes['Suggestions'],'\n\n'.join(tips) or 'No repeated sentences, sentences over 40 words, or targeted vague evaluations were found.')

    def export(self,page):
        if page not in self.saved or self.signature(page)!=self.snapshots[page]:
            self.invalidate(page);return
        is_batch=page=='Batch Analysis'
        types=([('CSV results','*.csv')] if is_batch else []) + [('PDF report','*.pdf'),('Word report','*.docx'),('JSON results','*.json')]
        name=filedialog.asksaveasfilename(title='Export Results',defaultextension='.csv' if is_batch else '.pdf',filetypes=types)
        if not name:return
        try:
            if is_batch and Path(name).suffix.lower()=='.csv':export_batch(self.saved[page],name)
            elif is_batch:export_report({'documents':[{'name':row['id'],**row} for row in self.saved[page]]},name)
            else:export_report(self.saved[page],name)
        except Exception as exc:messagebox.showerror('Export failed',str(exc));return
        messagebox.showinfo('Export complete',Path(name).name)

    def close(self):
        self.cancel_event.set();self.destroy()

if __name__=='__main__':
    Application().mainloop()
