"""Interactive layout preview. Model inference will be connected in module F."""
import csv
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk,filedialog,messagebox

BG='#f4f7fb';WHITE='#ffffff';INK='#22324a';MUTED='#43546b';LINE='#dce4ef';GREEN='#376caa';PALE='#e7effa'
FONT=('Segoe UI',11)

class Preview(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Text Review')
        self.geometry('1180x780');self.minsize(1020,690);self.configure(bg=BG)
        self.option_add('*Font',FONT)
        style=ttk.Style(self);style.theme_use('clam')
        style.configure('TCombobox',fieldbackground=WHITE,background=WHITE,foreground=INK,bordercolor=LINE,padding=7,arrowsize=12)
        style.map('TCombobox',fieldbackground=[('readonly',WHITE)],selectbackground=[('readonly',WHITE)],selectforeground=[('readonly',INK)])
        style.configure('TNotebook',background=WHITE,borderwidth=0)
        style.configure('TNotebook.Tab',background=WHITE,foreground=INK,padding=(16,10),borderwidth=0)
        style.map('TNotebook.Tab',background=[('selected',PALE)],foreground=[('selected',GREEN)])
        style.configure('Treeview',background=WHITE,fieldbackground=WHITE,foreground=INK,rowheight=36,borderwidth=0)
        style.configure('Treeview.Heading',background='#eef3fa',foreground=INK,font=FONT,padding=8)
        style.configure('Horizontal.TProgressbar',background=GREEN,troughcolor=LINE,borderwidth=0,thickness=5)
        side=tk.Frame(self,bg=WHITE,width=200);side.pack(side='left',fill='y');side.pack_propagate(False)
        tk.Label(side,text='Text Review',bg=WHITE,fg=INK,font=('Segoe UI',14,'bold')).pack(anchor='w',padx=20,pady=(34,32))
        self.nav={}
        for name in ['Text Analysis','Compare Drafts','Batch Analysis']:
            b=tk.Button(side,text=name,anchor='w',bg=WHITE,fg=INK,activebackground=PALE,activeforeground=GREEN,relief='flat',bd=0,padx=18,pady=14,cursor='hand2',command=lambda n=name:self.show(n))
            b.pack(fill='x',padx=12,pady=3);self.nav[name]=b
        body=tk.Frame(self,bg=BG);body.pack(side='left',fill='both',expand=True,padx=30,pady=26)
        header=tk.Frame(body,bg=BG);header.pack(fill='x')
        self.heading=tk.Label(header,text='',bg=BG,fg=INK,font=('Segoe UI',21,'bold'));self.heading.pack(side='left')
        options=tk.Frame(body,bg=BG);options.pack(fill='x',pady=(24,24))
        tk.Label(options,text='Model',bg=BG,fg=INK).pack(side='left',padx=(0,10))
        self.model=ttk.Combobox(options,state='readonly',values=['Combined Features','Lexical Features','Writing Style','Compare All'],width=22);self.model.current(0);self.model.pack(side='left')
        tk.Label(options,text='Threshold',bg=BG,fg=INK).pack(side='left',padx=(24,10))
        self.threshold=ttk.Combobox(options,state='readonly',values=['Default','Low False Positive'],width=20);self.threshold.current(0);self.threshold.pack(side='left')
        self.area=tk.Frame(body,bg=BG);self.area.pack(fill='both',expand=True)
        self.pages={};self.make_single();self.make_pair();self.make_batch()
        self.status=tk.StringVar(value='Interface preview')
        self.show('Text Analysis')

    def label(self,parent,text,small=False):
        return tk.Label(parent,text=text,bg=WHITE,fg=INK,font=FONT)

    def button(self,parent,text,command,primary=False):
        return tk.Button(parent,text=text,command=command,bg=GREEN if primary else WHITE,fg=WHITE if primary else INK,
                         activebackground='#285789' if primary else PALE,activeforeground=WHITE if primary else INK,
                         relief='flat',bd=0,padx=18,pady=9,cursor='hand2',highlightthickness=0)

    def card(self,parent):return tk.Frame(parent,bg=WHITE,highlightbackground=LINE,highlightthickness=1)

    def editor(self,parent,hint=''):
        frame=tk.Frame(parent,bg=WHITE);frame.pack(fill='both',expand=True,padx=18,pady=(8,16))
        editor=tk.Text(frame,wrap='word',bg=WHITE,fg=INK,insertbackground=GREEN,relief='flat',bd=0,font=('Segoe UI',11),undo=True,padx=4,pady=6,spacing3=5,width=1,height=1)
        scroll=ttk.Scrollbar(frame,command=editor.yview);editor.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y');editor.pack(side='left',fill='both',expand=True)
        return editor

    def preview_action(self):messagebox.showinfo('Preview','Detection will be connected in the next stage.')

    def import_txt(self,editor):
        name=filedialog.askopenfilename(title='Import Document',filetypes=[('Documents','*.txt *.docx *.pdf'),('Word document','*.docx'),('PDF document','*.pdf'),('Text document','*.txt')])
        if not name:return
        try:
            from src.app.document_input import read_document
            text=read_document(Path(name))
        except (UnicodeError,OSError,ValueError,ImportError) as exc:messagebox.showerror('Import failed',str(exc));return
        editor.delete('1.0','end');editor.insert('1.0',text);editor.focus_set();editor.event_generate('<KeyRelease>')
        self.status.set('Imported '+Path(name).name+' · Preview')

    def make_single(self):
        page=tk.Frame(self.area,bg=BG);self.pages['Text Analysis']=page
        page.columnconfigure(0,weight=6);page.columnconfigure(1,weight=5);page.rowconfigure(0,weight=1)
        left=self.card(page);left.grid(row=0,column=0,sticky='nsew',padx=(0,16));right=self.card(page);right.grid(row=0,column=1,sticky='nsew')
        h=tk.Frame(left,bg=WHITE);h.pack(fill='x',padx=20,pady=(18,0));self.label(h,'Input').pack(side='left')
        self.single=self.editor(left)
        self.button(h,'Import Document',lambda:self.import_txt(self.single)).pack(side='right')
        bottom=tk.Frame(left,bg=WHITE);bottom.pack(fill='x',padx=20,pady=(0,18))
        self.button(bottom,'Analyze',self.preview_action,True).pack(side='right')
        self.label(right,'Results').pack(anchor='w',padx=20,pady=(21,14))
        tabs=ttk.Notebook(right);tabs.pack(fill='both',expand=True,padx=16,pady=(0,16))
        for name in ['Overview','Passages','Suggestions']:
            f=tk.Frame(tabs,bg=WHITE);tabs.add(f,text=name)

    def make_pair(self):
        page=tk.Frame(self.area,bg=BG);self.pages['Compare Drafts']=page
        inputs=tk.Frame(page,bg=BG);inputs.pack(fill='both',expand=True)
        inputs.columnconfigure(0,weight=1);inputs.columnconfigure(1,weight=1);inputs.rowconfigure(0,weight=1)
        for i,title in enumerate(['Original','Revised']):
            card=self.card(inputs);card.grid(row=0,column=i,sticky='nsew',padx=(0,8) if i==0 else (8,0))
            self.label(card,title).pack(anchor='w',padx=20,pady=(20,0));self.editor(card,''+title)
        result=self.card(page);result.pack(fill='x',pady=(16,0))
        self.label(result,'Comparison').pack(anchor='w',padx=20,pady=(15,8))
        bar=tk.Frame(page,bg=BG);bar.pack(fill='x',pady=(14,0));self.button(bar,'Compare',self.preview_action,True).pack(side='right')

    def make_batch(self):
        page=tk.Frame(self.area,bg=BG);self.pages['Batch Analysis']=page
        card=self.card(page);card.pack(fill='both',expand=True)
        h=tk.Frame(card,bg=WHITE);h.pack(fill='x',padx=20,pady=16)
        self.filename=tk.StringVar(value='');tk.Label(h,textvariable=self.filename,bg=WHITE,fg=INK).pack(side='left')
        self.button(h,'Import CSV',self.import_csv).pack(side='right')
        area=tk.Frame(card,bg=WHITE);area.pack(fill='both',expand=True,padx=20,pady=(0,20))
        self.tree=ttk.Treeview(area,columns=('id','text','label','score','status'),show='headings',selectmode='browse')
        for key,title,width in [('id','ID',80),('text','Text',300),('label','Prediction',90),('score','Score',70),('status','Status',90)]:
            self.tree.heading(key,text=title);self.tree.column(key,width=width,minwidth=60,stretch=key=='text')
        self.tree.pack(fill='both',expand=True)
        bar=tk.Frame(page,bg=BG);bar.pack(fill='x',pady=(16,0))
        self.batch_count=tk.StringVar(value='');tk.Label(bar,textvariable=self.batch_count,bg=BG,fg=INK).pack(side='left')
        self.button(bar,'Analyze',self.preview_action,True).pack(side='right')
        export=self.button(bar,'Export',self.preview_action);export.configure(state='disabled',disabledforeground='#aab3ae');export.pack(side='right',padx=10)

    def import_csv(self):
        name=filedialog.askopenfilename(filetypes=[('CSV files','*.csv')])
        if not name:return
        try:
            with open(name,encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f)
                if not {'id','text'}<=set(reader.fieldnames or []):raise ValueError('CSV requires id and text columns.')
                rows=list(reader)
        except (UnicodeError,OSError,ValueError) as exc:messagebox.showerror('Import failed',str(exc));return
        for item in self.tree.get_children():self.tree.delete(item)
        for row in rows:self.tree.insert('','end',values=(row['id'],row['text'][:90].replace('\n',' '),'','','Pending'))
        self.filename.set(Path(name).name);self.batch_count.set(f'{len(rows)} documents')

    def show(self,name):
        for key,page in self.pages.items():
            page.pack_forget();self.nav[key].configure(bg=PALE if name==key else WHITE,fg=GREEN if name==key else INK)
        self.pages[name].pack(fill='both',expand=True);self.heading.configure(text=name)

if __name__=='__main__':
    import sys
    app=Preview()
    if '--smoke' in sys.argv:
        app.withdraw();app.update()
        for name in app.pages:app.show(name);app.update()
        assert len(app.pages)==3 and len(app.model['values'])==4
        print('Three-page UI smoke check passed. Preview performs no model inference.')
        app.destroy()
    else:app.mainloop()
