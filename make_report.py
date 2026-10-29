from pathlib import Path
import sys,re,subprocess
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from q2_results import read_stats,fmt
import numpy as np
import pypandoc

text=(root/'Assignment2_Report.md').read_text(encoding='utf-8')
labels={'MSE1':r'$\mathrm{MSE}_1$','MSE2':r'$\mathrm{MSE}_2$','MSE3':r'$\mathrm{MSE}_3$',
        'MSE4':r'$\mathrm{MSE}_4$','MSE5':r'$\mathrm{MSE}_5$','E5':r'$E^{(5)}_{1\times}$',
        'CM':r'$C_M$','EM':r'$E_M$','CR':r'$C_R$','ER':r'$E_R$',
        'E2_1':r'$E^{(2)}_{1\times}$','E2_2':r'$E^{(2)}_{2\times}$','E2_3':r'$E^{(2)}_{3\times}$',
        'E5_2':r'$E^{(5)}_{2\times}$','parameters':'Params','train':'Train (s/epoch)','inference':'Infer (s)'}
specs={2:(['A','Blank','Copy-last'],['MSE1','MSE2','MSE3','MSE4','MSE5','E5','parameters']),
       3:(['A','Blank','Copy-last'],['E5','CM','EM','CR','ER']),
       4:(['A','Copy-last'],['E2_1','E2_2','E2_3','E5','E5_2']),
       5:(['A','B','C','D'],['E5','CM','EM','CR','ER','E2_1','E2_2','E2_3','parameters','train','inference']),
       6:(['sym_B','sym_D'],['E5','CM','EM','CR','ER','E2_1','E2_2','E2_3','parameters','train','inference'])}
for n,(names,cols) in specs.items():
    original=(root/'results'/f'table{n}.md').read_text(encoding='utf-8')
    caption=original.splitlines()[0].replace(f'**Table {n}.**',f'\\textbf{{Table {n}.}}')
    lines=[r'\clearpage' if n in (5,6) else '',r'\begin{landscape}' if n in (5,6) else '',
           caption,r'\begin{center}',r'\small',r'\setlength{\tabcolsep}{5pt}',
           r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{l'+'c'*len(cols)+'}',r'\toprule',
           'Model & '+' & '.join(labels[c] for c in cols)+r' \\',r'\midrule']
    for name in names:
        ss=[read_stats(name,i) for i in ((0,) if name in ('Blank','Copy-last') else (0,1))]
        cells=[]
        for c in cols:
            vals=[s[c] for s in ss]
            fs=lambda x:str(int(x)) if c=='parameters' else fmt(x,n==6 and c in ('CM','CR'))
            if len(vals)==1: cell=fs(vals[0])
            else: cell=r'\makecell[c]{'+fs(np.mean(vals))+r'\\('+fs(vals[0])+r',\\'+fs(vals[1])+')}'
            cells.append(cell)
        lines.append(name.replace('_',r'\_')+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{center}',
        'Each learned cell shows mean (seed 0, seed 1), with line breaks for readability. Errors are pixel MSE on all 1,000 test videos. Train is the mean of all 30 epoch wall times, including compilation; inference is the synchronized 1,000-video total at batch size 32 after device transfer. Symmetrised timings include all eight passes and transformations.',
        r'\end{landscape}' if n in (5,6) else '',r'\clearpage' if n in (5,6) else '']
    assert original in text
    text=text.replace(original,'\n'.join(lines))
text=text.replace('–','-').replace('—','-').replace('‑','-')
# Every image keeps its explicit assignment figure number; disable Pandoc implicit captions.
for fig in ('figure3','figure6a'):
    number='3' if fig=='figure3' else '6(a)'
    blocks=[]
    for tag,label in (('m','mirror'),('r','rotation')):
        blocks += [r'\clearpage',r'\begin{center}',
            rf'\includegraphics[width=\linewidth]{{figures/{fig}_{tag}1.png}}',
            rf'\includegraphics[width=\linewidth]{{figures/{fig}_{tag}2.png}}',r'\end{center}']
        if tag=='m': blocks.append(f'Figure {number}, {label} panels. The rotation panels continue on the next page.')
    replacement='\n'.join(blocks)+'\n'
    text=re.sub(r'!\[.*\]\(figures/'+fig+r'\.png\)',lambda m:replacement,text)
pypandoc.convert_text(text,'latex',format='markdown-implicit_figures+tex_math_dollars',
    outputfile=str(root/'Assignment2_Report.tex'),extra_args=['--standalone','-V','mainfont:Times New Roman'])
for _ in range(2):
    r=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','Assignment2_Report.tex'],cwd=root,
                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    (root/'results'/'latex_build.log').write_text(r.stdout,encoding='utf-8')
    if r.returncode:raise RuntimeError(r.stdout[-6000:])
print(root/'Assignment2_Report.pdf')
