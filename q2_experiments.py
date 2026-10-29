"""Q2: exact assignment protocol; Intel XPU required for learned models."""
from pathlib import Path
import json, time, random, argparse, os, shutil, subprocess
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MovingMNIST

ROOT = Path(__file__).resolve().parent
for directory in ('data', 'checkpoints', 'results', 'figures', 'gifs'):
    (ROOT / directory).mkdir(exist_ok=True)
DEVICE = torch.device('xpu')

def M(x): return x.flip(-1)
def R(x, a=1): return torch.rot90(x, a, (-2, -1))
def transform(x, g): return M(R(x, g % 4)) if g >= 4 else R(x, g)
def inverse(x, g): return transform(x, g) if g >= 4 else R(x, -g)

def window(video, t=1, k=1):
    """t uses the handout's one-based frame numbering."""
    ix = torch.arange(t-1, t-1+5*k, k)
    iy = torch.arange(t-1+5*k, min(t-1+10*k, 20), k)
    assert len(ix) == 5 and len(iy) >= 2
    y = torch.zeros((5, *video.shape[1:]), dtype=video.dtype)
    mask = torch.zeros(5, dtype=torch.bool)
    y[:len(iy)] = video[iy]; mask[:len(iy)] = True
    return video[ix], y, mask

def load_data():
    ds = MovingMNIST(root=str(ROOT/'data'), split=None, download=True)
    assert len(ds) == 10000
    return ds

def video_tensor(ds, i):
    v = ds[i]
    if isinstance(v, tuple): v = v[0]
    assert tuple(v.shape) == (20, 1, 64, 64), v.shape
    return v.float().div(255)

class Cell(nn.Module):
    def __init__(self, cin, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(cin+64, 256, 3, padding=dilation,
                              dilation=dilation, bias=True)
    def forward(self, u, h, c):
        i, f, o, g = self.conv(torch.cat((u, h), 1)).chunk(4, 1)
        c = torch.sigmoid(f)*c + torch.sigmoid(i)*torch.tanh(g)
        return torch.sigmoid(o)*torch.tanh(c), c

class Forecaster(nn.Module):
    def __init__(self, larger_reach=False):
        super().__init__()
        self.cell1 = Cell(16, 1)
        self.cell2 = Cell(64, 3 if larger_reach else 1)
        self.head = nn.Conv2d(64, 16, 1, bias=True)
    @torch.autocast(device_type='xpu', dtype=torch.bfloat16)
    def forward(self, x):
        z = torch.zeros((x.shape[0], 64, 16, 16),device=x.device,dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
        h1, c1, h2, c2 = z, z, z, z
        pred, outputs = None, []
        for step in range(9):
            u = x[:, step] if step < 5 else pred
            h1, c1 = self.cell1(F.pixel_unshuffle(u, 4).contiguous(memory_format=torch.channels_last), h1, c1)
            h2, c2 = self.cell2(h1, h2, c2)
            pred = torch.sigmoid(F.pixel_shuffle(self.head(h2), 4))
            if step >= 4: outputs.append(pred)
        return torch.stack(outputs, 1).float()

class Symmetrised(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, x):
        return torch.stack([inverse(self.model(transform(x,g)),g)
                            for g in range(8)]).mean(0)

def masked_mse(pred, y, mask):
    per_h = (pred-y).square().mean((2,3,4))
    return ((per_h*mask).sum(1)/mask.sum(1)).mean()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)

def compiler_environment():
    """Use the already installed MSVC toolchain for Triton on Windows."""
    if os.name!='nt' or shutil.which('cl'): return
    vswhere=Path(os.environ.get('ProgramFiles(x86)','C:/Program Files (x86)'))/'Microsoft Visual Studio/Installer/vswhere.exe'
    installation=subprocess.check_output([str(vswhere),'-latest','-products','*','-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],text=True).strip()
    script=Path(installation)/'Common7/Tools/VsDevCmd.bat'
    output=subprocess.check_output(f'call "{script}" -arch=x64 >nul && set',shell=True,text=True)
    for line in output.splitlines():
        if '=' in line and not line.startswith('='):
            key,value=line.split('=',1);os.environ[key]=value

def train(ds, name, seed):
    assert torch.xpu.is_available(), 'Intel XPU unavailable: CPU fallback is disabled.'
    final = ROOT/'checkpoints'/f'{name}_seed{seed}.pt'
    if final.exists(): return
    seed_all(seed)
    model = Forecaster(name == 'D').to(device=DEVICE,memory_format=torch.channels_last)
    assert sum(p.numel() for p in model.parameters()) == 480784
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    resume = ROOT/'checkpoints'/f'{name}_seed{seed}_resume.pt'
    losses, times, start = [], [], 0
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        for v in opt.state.values():
            for key,value in v.items():
                if torch.is_tensor(value) and key != 'step': v[key] = value.to(DEVICE)
        start, losses, times = state['epoch'], state['losses'], state['times']
        torch.set_rng_state(state['rng']); random.setstate(state['python_rng'])
        torch.xpu.set_rng_state_all(state['xpu_rng'])
    print(f'TRAIN {name} seed={seed} start={start} device={torch.xpu.get_device_name(0)}', flush=True)
    compiler_environment()
    run_model = torch.compile(model)
    # Loading uint8 videos does not change the prescribed per-epoch window sampling.
    raw = ds.data
    for epoch in range(start,30):
        torch.xpu.synchronize(); tick = time.perf_counter()
        order = torch.randperm(8000).tolist()
        total = torch.zeros((),device=DEVICE)
        model.train()
        for offset in range(0,8000,32):
            batch=[]
            for idx in order[offset:offset+32]:
                k = random.choice((1,2,3)) if name in ('C','D') else 1
                max_t = {1:14,2:8,3:2}[k] if name in ('C','D') else 11
                t = random.randint(1,max_t)
                x,y,mask = window(video_tensor(ds,idx),t,k)
                if name == 'B':
                    g=random.randrange(8); x,y=transform(x,g),transform(y,g)
                batch.append((x,y,mask))
            x,y,mask=[torch.stack(v).to(DEVICE) for v in zip(*batch)]
            opt.zero_grad(set_to_none=True)
            pred=run_model(x); loss=masked_mse(pred,y,mask)
            loss.backward(); opt.step(); total += loss.detach()
            if offset == 0 and epoch == start:
                torch.xpu.synchronize()
                print(f'First batch {time.perf_counter()-tick:.2f}s; loss={loss.item():.6f}',flush=True)
        torch.xpu.synchronize(); seconds=time.perf_counter()-tick
        losses.append(total.item()/250); times.append(seconds)
        state={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,
               'losses':losses,'times':times,'rng':torch.get_rng_state(),
               'python_rng':random.getstate(),'xpu_rng':torch.xpu.get_rng_state_all(),
               'model_name':name,'seed':seed,'hardware':torch.xpu.get_device_name(0),
               'torch':torch.__version__,'precision':'float32 inputs/weights/loss; XPU bfloat16 autocast; channels-last convolutions'}
        torch.save(state,resume)
        (ROOT/'results'/f'training_{name}_{seed}.json').write_text(json.dumps(
            {key:state[key] for key in ('epoch','losses','times','hardware','torch','precision')},indent=2))
        print(f'{name}/{seed} epoch {epoch+1}/30 MSE={losses[-1]:.7f} time={seconds:.2f}s',flush=True)
    temporary=final.with_suffix('.tmp')
    torch.save({key:state[key] for key in ('model','epoch','losses','times','model_name','seed','hardware','torch','precision')},temporary)
    os.replace(temporary,final)

def load_model(name,seed,wrap=False):
    state=torch.load(ROOT/'checkpoints'/f'{name}_seed{seed}.pt',map_location='cpu',weights_only=False)
    assert state['epoch']==30
    model=Forecaster(name=='D').to(device=DEVICE,memory_format=torch.channels_last); model.load_state_dict(state['model']); model.eval()
    return Symmetrised(model).eval() if wrap else model

@torch.no_grad()
def evaluate(ds,name,seed=0,wrap=False):
    label = ('sym_' if wrap else '')+name
    path=ROOT/'results'/f'{label}_{seed}.npz'
    if path.exists(): return dict(np.load(path))
    if name in ('Blank','Copy-last'):
        model = lambda x: torch.zeros_like(x) if name=='Blank' else x[:,-1:].expand(-1,5,-1,-1,-1)
    else: model=load_model(name,seed,wrap)
    values={key:[] for key in ('mse1','mse2','mse3','cm','em','cr','er')}
    elapsed=0.
    for offset in range(9000,10000,32):
        videos=[video_tensor(ds,i) for i in range(offset,min(offset+32,10000))]
        for k in (1,2,3):
            batch=[window(v,1,k) for v in videos]
            x,y,mask=[torch.stack(v).to(DEVICE) for v in zip(*batch)]
            torch.xpu.synchronize(); tick=time.perf_counter()
            pred=model(x)
            torch.xpu.synchronize()
            if k==1: elapsed+=time.perf_counter()-tick
            per_h=(pred-y).square().mean((2,3,4)); per_h[:,~mask[0]]=torch.nan
            values[f'mse{k}'].append(per_h.cpu().numpy())
            if k==1:
                for tag,fn in (('m',M),('r',R)):
                    p=model(fn(x))
                    values['c'+tag].append((p-fn(pred)).square().mean((1,2,3,4)).cpu().numpy())
                    values['e'+tag].append((p-fn(y)).square().mean((1,2,3,4)).cpu().numpy())
        print(f'EVAL {label}/{seed}: {min(offset+32,10000)-9000}/1000',flush=True) if (offset-9000)%320==0 else None
    values={k:np.concatenate(v) for k,v in values.items()}
    values['inference_seconds']=np.array(elapsed)
    np.savez_compressed(path,**values)
    return values

def checks():
    torch.manual_seed(0); x=torch.rand(20,1,64,64)
    p=torch.zeros_like(x); p[7,0,10,20]=1
    rows=[]
    for tag,left,right in [('a',M(M(x)),x),('b',R(x,4),x),('c',M(R(M(x))),R(x,-1))]:
        assert torch.equal(left,right); rows.append([tag,0,bool(torch.equal(left,right))])
    for tag,fn,expected in [('d',M,[7,0,10,43]),('e',R,[7,0,43,10])]:
        actual=fn(p).nonzero()[0].tolist(); assert actual==expected
        rows.append([tag,[actual[0],actual[2],actual[3]]])
    numbered=torch.arange(1,21).view(20,1,1,1).expand(20,1,64,64)
    for k in (1,2,3):
        xx,yy,mask=window(numbered,1,k)
        rows.append([f'{k}x',xx[:,0,0,0].tolist(),yy[mask,0,0,0].tolist()])
    assert sum(p.numel() for p in Forecaster().parameters())==480784
    (ROOT/'results'/'table1.json').write_text(json.dumps(rows,indent=2))
    print(rows,flush=True)

def flow_figure(ds):
    import cv2
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    angles=[]; speeds=[]
    for i in range(50):
        v=(video_tensor(ds,i).squeeze(1).numpy()*255).astype(np.uint8)
        for s in range(19):
            f=cv2.calcOpticalFlowFarneback(v[s],v[s+1],None,.5,3,15,3,5,1.2,0)
            dx,dy=f[...,0],f[...,1]; speed=np.hypot(dx,dy)
            keep=(v[s]>51)&(v[s+1]>51)&(speed>.5)
            angles.append(np.degrees(np.arctan2(-dy[keep],dx[keep]))%360)
            speeds.append(speed[keep])
    a=np.concatenate(angles); s=np.concatenate(speeds)
    fig,ax=plt.subplots(1,2,figsize=(10,3.3),layout='constrained')
    ax[0].hist(a,bins=np.arange(0,361,20),weights=np.ones(len(a))/len(a),edgecolor='white')
    ax[0].axhline(1/18,c='red',ls='--'); ax[0].set(xlabel='Direction (degrees)',ylabel='Fraction of kept pixels',title='(a) Direction',xlim=(0,360))
    ax[1].hist(s,bins=35,weights=np.ones(len(s))/len(s),edgecolor='white')
    ax[1].axvline(np.median(s),c='red',ls='--'); ax[1].set(xlabel='Speed (pixels/frame)',ylabel='Fraction of kept pixels',title='(b) Speed')
    fig.suptitle(f'Figure 1. First 50 training videos; {len(s):,} kept pixels; median {np.median(s):.3g} pixels/frame')
    fig.savefig(ROOT/'figures'/'figure1.png',dpi=170); plt.close(fig)
    stats={'pixels':len(s),'median':float(np.median(s)),'direction_fractions':np.histogram(a,np.arange(0,361,20))[0].astype(float).tolist()}
    stats['direction_fractions']=(np.array(stats['direction_fractions'])/len(s)).tolist()
    (ROOT/'results'/'flow.json').write_text(json.dumps(stats,indent=2))
    print('FLOW',stats,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['setup','A','BD','evaluate']); args=parser.parse_args()
    torch.set_num_threads(4)
    if args.stage=='setup':
        checks(); ds=load_data(); flow_figure(ds)
    elif args.stage in ('A','BD'):
        ds=load_data()
        for name in (['A'] if args.stage=='A' else ['B','C','D']):
            for seed in (0,1): train(ds,name,seed)
    else:
        ds=load_data()
        for name in ('Blank','Copy-last','A','B','C','D'):
            for seed in ((0,) if name in ('Blank','Copy-last') else (0,1)): evaluate(ds,name,seed)
        for name in ('B','D'):
            for seed in (0,1): evaluate(ds,name,seed,True)
