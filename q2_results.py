"""Required tables, figures and GIFs; all examples use seed 0 and test videos."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from q2_experiments import (ROOT, DEVICE, M, R, window, video_tensor, load_model,
                            load_data, evaluate)

def fmt(x, scientific=False):
    x=float(x)
    if scientific or (0<abs(x)<1e-3): return f'{x:.2e}'
    return f'{x:.3g}'

def stats(v):
    return dict(MSE1=v['mse1'][:,0].mean(),MSE2=v['mse1'][:,1].mean(),
                MSE3=v['mse1'][:,2].mean(),MSE4=v['mse1'][:,3].mean(),MSE5=v['mse1'][:,4].mean(),
                E5=v['mse1'].mean(), CM=v['cm'].mean(), EM=v['em'].mean(),
                CR=v['cr'].mean(), ER=v['er'].mean(),
                E2_1=v['mse1'][:,:2].mean(),E2_2=v['mse2'][:,:2].mean(),
                E2_3=v['mse3'][:,:2].mean(),E5_2=v['mse2'].mean(),
                inference=float(v['inference_seconds']))

def read_stats(name,seed=0):
    v=dict(np.load(ROOT/'results'/f'{name}_{seed}.npz'))
    s=stats(v)
    if name in ('Blank','Copy-last'): s.update(parameters=0,train=0.)
    else:
        base=name.removeprefix('sym_')
        j=json.loads((ROOT/'results'/f'training_{base}_{seed}.json').read_text())
        s.update(parameters=480784,train=np.mean(j['times']))
    return s

def table(names,cols,number):
    lines=[f'**Table {number}.** '+{
        2:'Normal-speed baseline accuracy and parameters.',
        3:'Mirror/rotation consistency and transformed correctness at 1x.',
        4:'Speed tests; E(2) uses only the first two forecast horizons.',
        5:'Models A–D: accuracy, consistency, parameters and timing.',
        6:'Symmetrised models. Final model: symmetrised D.'}[number],
        '', '| Model | '+' | '.join(cols)+' |', '|---|'+'---:|'*len(cols)]
    for name in names:
        ss=[read_stats(name,i) for i in ((0,) if name in ('Blank','Copy-last') else (0,1))]
        cells=[]
        for c in cols:
            vals=[s[c] for s in ss]
            sci=number==6 and c in ('CM','CR')
            if c=='parameters': cell=str(vals[0]) if len(vals)==1 else f'{vals[0]} ({vals[0]}, {vals[1]})'
            elif len(vals)==1: cell=fmt(vals[0],sci)
            else: cell=f'{fmt(np.mean(vals),sci)} ({fmt(vals[0],sci)}, {fmt(vals[1],sci)})'
            cells.append(cell)
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','Learned entries: mean (seed 0, seed 1). Errors are pixel MSE on 1,000 test videos. Timing columns are seconds; inference excludes host-to-device transfer and includes XPU synchronization.','']
    text='\n'.join(lines)
    (ROOT/'results'/f'table{number}.md').write_text(text,encoding='utf-8')
    return text

def tables(final=False):
    table(['A','Blank','Copy-last'],['MSE1','MSE2','MSE3','MSE4','MSE5','E5','parameters'],2)
    table(['A','Blank','Copy-last'],['E5','CM','EM','CR','ER'],3)
    table(['A','Copy-last'],['E2_1','E2_2','E2_3','E5','E5_2'],4)
    if final:
        cols=['E5','CM','EM','CR','ER','E2_1','E2_2','E2_3','parameters','train','inference']
        table(['A','B','C','D'],cols,5)
        table(['sym_B','sym_D'],cols,6)

@torch.no_grad()
def predict(model,x): return model(x.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()

def frame_grid(rows,labels,path,title,columns=None):
    n=len(rows[0]); fig,ax=plt.subplots(len(rows),n,figsize=(n*1.08,len(rows)*1.2),squeeze=False)
    for r,row in enumerate(rows):
        for c,frame in enumerate(row):
            ax[r,c].imshow(np.asarray(frame).squeeze(),cmap='gray',vmin=0,vmax=1)
            ax[r,c].set_xticks([]);ax[r,c].set_yticks([])
            for spine in ax[r,c].spines.values(): spine.set_visible(False)
            if c==0: ax[r,c].set_ylabel(labels[r],fontsize=9)
            if r==0: ax[r,c].set_title(columns[c] if columns else str(c+1),fontsize=9)
    fig.suptitle(title,fontsize=11);fig.subplots_adjust(left=.1,right=.995,bottom=.01,top=.88,wspace=.03,hspace=.06)
    fig.savefig(path,dpi=150);plt.close(fig)

def gif_panel(future_rows,labels,path,inputs):
    imgs=[]; scale=2; left=135; w=left+6*64*scale; h=24+4*64*scale
    for t in range(5):
        canvas=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(canvas)
        draw.text((left,5),f'Inputs 1-5 (fixed)                         Forecast horizon {t+1}',fill='black')
        for r,row in enumerate(future_rows):
            for c,frame in enumerate([*inputs[r],row[t]]):
                a=np.clip(np.asarray(frame).squeeze()*255,0,255).astype('uint8')
                im=Image.fromarray(a).resize((128,128),Image.Resampling.NEAREST).convert('RGB')
                canvas.paste(im,(left+c*128,24+r*128))
            draw.text((5,24+r*128+50),labels[r],fill='black')
        imgs.append(canvas)
    imgs[0].save(path,save_all=True,append_images=imgs[1:],duration=450,loop=0)

def components(frame):
    n,labels,stat,centers=cv2.connectedComponentsWithStats((frame.numpy().squeeze()>.2).astype('uint8'))
    return sorted([int(a) for a in stat[1:,cv2.CC_STAT_AREA] if a>=30],reverse=True)

def baseline_figures(ds):
    model=load_model('A',0)
    # A single connected foreground is only a candidate; visual QA confirms actual overlap.
    overlap=None; clear=None
    for idx in range(9000,10000):
        v=video_tensor(ds,idx)
        if overlap is None and any(len(components(v[t]))==1 and components(v[t])[0]>170 for t in range(4,10)):
            overlap=idx
        if clear is None and all(len(components(v[t]))==2 for t in range(19)):
            clear=idx
        if overlap is not None and clear is not None: break
    assert clear is not None and overlap is not None
    ids=[overlap]+[i for i in (9000,9001,9002,9003) if i!=overlap][:2]
    rows=[];labels=[]
    for idx in ids:
        x,y,_=window(video_tensor(ds,idx));p=predict(model,x)
        rows.extend([torch.cat((x,p)),torch.cat((x,y))]);labels.extend([f'{idx} prediction',f'{idx} truth'])
    frame_grid(rows,labels,ROOT/'figures'/'figure2.png','Figure 2. Model A, seed 0; inputs 1–5, future 6–10')
    v=np.load(ROOT/'results'/'A_0.npz')
    selected={tag:(9000+np.argsort(v['c'+tag])[-2:][::-1]).tolist() for tag in ('m','r')}
    selected.update(figure2=ids,overlap=overlap,speed=clear)
    (ROOT/'results'/'selected_videos.json').write_text(json.dumps(selected,indent=2))
    symmetry_panels(ds,model,selected,'figure3')
    fig,ax=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    for a,tag in zip(ax,('m','r')):
        values=v['c'+tag];a.hist(values,bins=35,color='#357b96')
        for j,idx in enumerate(selected[tag]):
            value=values[idx-9000];a.axvline(value,c=('red','orange')[j],ls='--',label=f'video {idx}: {value:.3g}')
        a.set(xlabel=f'C_{tag.upper()} (per-video MSE)',ylabel='Test videos',title=('Mirror' if tag=='m' else 'Rotation'));a.legend(fontsize=8)
    fig.suptitle('Figure 4. Model A seed-0 consistency gaps over 1,000 test videos');fig.savefig(ROOT/'figures'/'figure4.png',dpi=170);plt.close(fig)
    speed_panel(ds,model,clear,'figure5')
    tables(False)
    return selected

def symmetry_panels(ds,model,selected,prefix):
    panels=[]
    for tag,fn in (('m',M),('r',R)):
        for rank,idx in enumerate(selected[tag],1):
            x,y,_=window(video_tensor(ds,idx));p=predict(model,x);pt=predict(model,fn(x))
            rows=[torch.cat((x,p)),torch.cat((fn(x),fn(y))),torch.cat((fn(x),pt)),torch.cat((fn(x),fn(p)))]
            labels=['X / f(X)','T(X) / T(Y)','T(X) / f(T(X))','T(X) / T(f(X))']
            path=ROOT/'figures'/f'{prefix}_{tag}{rank}.png'
            frame_grid(rows,labels,path,f'{prefix.replace("figure","Figure ")} — {tag.upper()}, test video {idx}, seed 0')
            gif_panel([p,fn(y),pt,fn(p)],['X / f(X)','T(X) / T(Y)','T(X) / f(T(X))','T(X) / T(f(X))'],ROOT/'gifs'/f'{prefix}_{tag}{rank}.gif',
                      [x,fn(x),fn(x),fn(x)])
            panels.append(path)
    # Consolidation keeps figure numbering unambiguous in the report.
    ims=[Image.open(p).convert('RGB') for p in panels]
    canvas=Image.new('RGB',(max(i.width for i in ims),sum(i.height for i in ims)),'white')
    y=0
    for im in ims:canvas.paste(im,(0,y));y+=im.height
    canvas.save(ROOT/'figures'/f'{prefix}.png')

def speed_panel(ds,model,idx,prefix):
    rows=[];labels=[]
    for k in (1,2,3):
        x,y,_=window(video_tensor(ds,idx),1,k);p=predict(model,x)
        rows.extend([torch.cat((x,p[:2])),torch.cat((x,y[:2]))]);labels.extend([f'{k}x prediction',f'{k}x truth'])
    frame_grid(rows,labels,ROOT/'figures'/f'{prefix}.png',f'{prefix.replace("figure","Figure ")}. Test video {idx}; first two forecast horizons',
               [f'Input {i}' for i in range(1,6)]+['h=1','h=2'])

def final_figures(ds):
    selected=json.loads((ROOT/'results'/'selected_videos.json').read_text())
    model=load_model('D',0,True)
    symmetry_panels(ds,model,selected,'figure6a')
    speed_panel(ds,model,selected['speed'],'figure6b')
    tables(True)

if __name__=='__main__':
    import sys
    torch.set_num_threads(4); ds=load_data()
    if sys.argv[1]=='baseline':
        for name in ('Blank','Copy-last','A'):
            for seed in ((0,) if name in ('Blank','Copy-last') else (0,1)):evaluate(ds,name,seed)
        baseline_figures(ds)
    elif sys.argv[1]=='final':final_figures(ds)
