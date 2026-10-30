from pathlib import Path
import sys, json, re, textwrap, base64
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from q2_results import read_stats,fmt
import nbformat as nbf

def md(c):
    cell=nbf.v4.new_markdown_cell(c.strip())
    attachments={}
    def attach(match):
        name=Path(match.group(2)).name
        attachments[name]={'image/png':base64.b64encode((root/match.group(2)).read_bytes()).decode('ascii')}
        return f'![{match.group(1)}](attachment:{name})'
    cell.source=re.sub(r'!\[(.*)\]\((figures/[^)]+)\)',attach,cell.source)
    if attachments:cell.attachments=attachments
    return cell
def code(c):return nbf.v4.new_code_cell(textwrap.dedent(c).strip())
def markdown_cells(text):
    return [md(part) for part in re.split(r'(?m)(?=^## |^### Q2\.)',text) if part.strip()]

def q2_text():
    flow=json.loads((root/'results'/'flow.json').read_text())
    median=flow['median']; selected=json.loads((root/'results'/'selected_videos.json').read_text())
    notes=json.loads((root/'results'/'visual_notes.json').read_text())
    s={name:read_stats(name,0) for name in ('Blank','Copy-last')}
    for name in ('A','B','C','D','sym_B','sym_D'):
        a,b=read_stats(name,0),read_stats(name,1)
        s[name]={key:(a[key]+b[key])/2 for key in a}
    A,B,C,D,F,SB=[s[n] for n in ('A','B','C','D','sym_D','sym_B')]
    p5=(root/'Q2_5_preregistered.md').read_text(encoding='utf-8').split('\n\n',1)[1].strip()
    p11=(root/'Q2_11_preregistered.md').read_text(encoding='utf-8').split('\n\n',1)[1].strip()
    def tab(n):return (root/'results'/f'table{n}.md').read_text(encoding='utf-8')
    def changed(old,new):return f'{fmt(old)} to {fmt(new)}'
    def direction(old,new):return 'improved' if new<old else 'worsened'
    def fig(name,caption):return f'![{caption}](figures/{name}.png)\n\n{caption}\n'
    table1=r'''**Table 1.** Transformation and window checks (the same functions used in all experiments).

| Check | Expected | Yours |
|---|---|---|
| (a) $\max\lvert M(M(x))-x\rvert$ | 0; equal True | 0; `torch.equal=True` |
| (b) $\max\lvert R^4(x)-x\rvert$ | 0; equal True | 0; `torch.equal=True` |
| (c) $\max\lvert MRM(x)-R^{-1}(x)\rvert$ | 0; equal True | 0; `torch.equal=True` |
| (d) bright pixel of $M(p)$, zero-based (frame,row,column) | (7,10,43) | (7,10,43) |
| (e) bright pixel of $R(p)$, zero-based (frame,row,column) | (7,43,10) | (7,43,10) |
| (f) 1x input / valid target frames | 1–5 / 6–10 | 1–5 / 6–10 |
| (g) 2x input / valid target frames | 1,3,5,7,9 / 11,13,15,17,19 | 1,3,5,7,9 / 11,13,15,17,19 |
| (h) 3x input / valid target frames | 1,4,7,10,13 / 16,19 | 1,4,7,10,13 / 16,19 |
'''
    setup=rf'''
# Question 2 — Motion or Memory?

## Setup and Table 1

Load `torchvision.datasets.MovingMNIST(split=None, download=True)`. Split by video index: 0–7999 training, 8000–8999 validation, 9000–9999 test. Inputs are float32 intensities divided by 255. Every scored model result uses all 1,000 test videos at $t=1$. The validation set is reserved and was not needed for design selection: D was fixed before its training. No optional D+B model is trained.

{table1}

{fig('figure1',f'Figure 1. Farneback flow from the first 50 training videos, all 950 consecutive frame pairs; {flow["pixels"]:,} pixels retained. Median speed {fmt(median)} pixels/frame. Both histograms show fractions of retained pixels; direction reference 1/18 and speed median are marked.')}

### Q2.1

Figure 1(a) covers all direction bins, although it is not perfectly uniform. Mirroring maps $\theta$ to $180^\circ-\theta$ modulo $360^\circ$; anticlockwise rotation adds $90^\circ$. Such rearrangements broadly preserve the occupied directions, so mirrored and rotated trajectories should already be familiar as motion types, even though the small sample is not exactly invariant. The {fmt(median)}-pixel/frame median in Figure 1(b) corresponds approximately to {fmt(2*median)} at 2x and {fmt(3*median)} at 3x, away from bounce-induced displacement cancellation. This is an optical-flow proxy, not digit-centre ground truth. Overlap can create erroneous flow and the intersection-intensity mask preferentially retains shared foreground, potentially biasing both histograms and the median; repeated pixels within a trajectory are strongly correlated. The plots support diverse directions, but cannot rule out quantitative bias or prove exact symmetry of the complete training distribution.

### Q2.2

Mirror and rotation change digit appearance. With an isotropic direction distribution they leave motion statistics unchanged, although each individual velocity vector is transformed; in this finite sample the direction histogram is only approximately symmetric, so its frequencies can change somewhat, while speed magnitudes are preserved. They therefore chiefly probe dependence on unfamiliar appearances, with finite directional imbalance as a caveat. Speed-up preserves each sampled digit's spatial appearance while changing temporal displacement, observed bounce/overlap timing and motion statistics. Together these are controlled interventions on appearance versus temporal motion, rather than a guarantee that the data distribution is invariant.

## Baseline and training protocol

Fold each frame with `pixel_unshuffle(4)`: $1\times64\times64\to16\times16\times16$. Both recurrent layers have 64 hidden channels, 3-by-3 gates, padding 1 and biases. The second takes the first layer's new hidden state. With no peepholes,

$$[i,f,o,g]=\operatorname{{Conv}}_{{3\times3}}([u_\tau,h_{{\tau-1}}]),\quad c_\tau=\sigma(f)\odot c_{{\tau-1}}+\sigma(i)\odot\tanh(g),\quad h_\tau=\sigma(o)\odot\tanh(c_\tau).$$

The output is $\widehat x_{{\tau+1}}=\sigma(\operatorname{{PixelShuffle}}_4(\operatorname{{Conv}}_{{1\times1}}(h^{{(2)}}_\tau)))$. All states start at zero per video. Steps 1–5 consume the five observed frames; steps 6–9 consume the previous prediction with its gradient attached. Only outputs at steps 5–9 are scored. No normalization, dropout, residual or extra layers are added to A–C. Gate parameters are $256(16+64)9+256=184576$ and $256(64+64)9+256=295168$; the head has $16\cdot64+16=1040$, giving **480,784** parameters.

Every learned model uses seeds 0 and 1, Adam with learning rate $10^{{-3}}$, batch size 32, 30 epochs, and final-epoch weights. Each epoch shuffles 8,000 videos and samples one window per video. A/B draw $t\in[1,11]$ at 1x; C/D draw $k$ uniformly from $\{{1,2,3\}}$, then $t$ uniformly from $[1,14]$, $[1,8]$ or $[1,2]$. B draws one of eight spatial symmetries uniformly, identically for input and target. C/D have no symmetry augmentation. All models generate five frames even when only two targets exist. The loss is

$$\mathcal L=\frac1B\sum_{{b=1}}^B\frac{{\sum_{{h=1}}^5m_{{bh}}\left(\frac1{{4096}}\sum_p(\widehat Y_{{bhp}}-Y_{{bhp}})^2\right)}}{{\sum_{{h=1}}^5m_{{bh}}}}.$$

Hardware: Intel Arc 140T GPU (23 GB reported shared device memory), PyTorch 2.14.1+xpu and torchvision 0.29.1+xpu. Inputs/weights/loss are float32, with XPU bfloat16 autocast and channels-last convolutions. Training uses `torch.compile`; A seed 0's first ten completed epochs used eager execution, and subsequent epochs use compilation. Saved timings retain actual completed-epoch wall time, including compilation when encountered; the excluded preliminary float32 run contributes no weights or timing. Other learned runs use compilation throughout. Evaluation uses the same model computations in eager mode, `eval()` and `no_grad()`. Each inference batch is already on XPU when its synchronized timer starts. Dataset download is excluded. No CPU training/inference fallback and no NPU conversion is used.

{tab(2)}

{fig('figure2',f'Figure 2. Model A seed 0 on test videos {selected["figure2"]}; input frames 1–5 are repeated for context, with predicted versus true frames 6–10. Video {selected["overlap"]} contains overlapping digits in the displayed truth (confirmed visually).')}

### Q2.3

One folded-cell displacement corresponds to four original pixels; patch width adds nothing under the stated convention. Within one update, input and previous layer-1 hidden-state paths can traverse the layer-1 3-by-3 gate and then the layer-2 3-by-3 gate. Their maximal offset is two folded cells along either axis, or **8 original pixels per axis**. The previous layer-2 state traverses only its own 3-by-3 gate (4 pixels per axis), and cell-state carry is pointwise. The head and shuffle do not expand reach. Thus the longest path has square support with $L_\infty$ radius 8; its greatest centre-to-centre Euclidean displacement is $8\sqrt2\approx11.3$ pixels at a corner. A diagonal maximum does not give an 11.3-pixel radius in every direction. Earlier recurrent steps enlarge total history support, but cannot be counted as additional layers in this one-step quantity.

### Q2.4

Table 2 gives model A $E^{{(5)}}={fmt(A['E5'])}$ versus Blank {fmt(s['Blank']['E5'])} and Copy-last {fmt(s['Copy-last']['E5'])}; it is {'better than both references' if A['E5']<min(s['Blank']['E5'],s['Copy-last']['E5']) else 'not better than both references'}. A's MSE increases at every forecast horizon, from {fmt(A['MSE1'])} at $h=1$ to {fmt(A['MSE5'])} at $h=5$. {notes['q24']}

### Q2.5 — preserved pre-experiment prediction

{p5}

## Testing the baseline

$$C_T=\operatorname{{MSE}}(f(TX),T f(X)),\qquad E_T=\operatorname{{MSE}}(f(TX),TY).$$

Both average pixels, five horizons and all 1,000 test videos at 1x.

{tab(3)}

{fig('figure3',f'Figure 3. Seed-0 model A: the two largest per-video mirror gaps (videos {selected["m"]}) and rotation gaps (videos {selected["r"]}). Each panel has the four specified rows and ten frame columns. GIFs: `gifs/figure3_m1.gif`, `gifs/figure3_m2.gif`, `gifs/figure3_r1.gif`, `gifs/figure3_r2.gif`; each keeps the same four rows and advances through five forecast horizons. Video indices refer to the original dataset.')}

{fig('figure4','Figure 4. Per-video mirror/rotation gap distributions for model A seed 0 over the complete test set. The Figure 3 videos are marked with their indices and gap values.')}

### Q2.6

Blank and Copy-last commute exactly with pixel permutations: $0=T0$ and copying the last transformed input equals transforming the copied last input. Their consistency gaps are zero despite substantial transformed error in Table 3. $C_T$ measures agreement between two prediction routes, not agreement with the real future. Reporting $E_T$ alongside it prevents a perfectly symmetric but useless predictor from appearing successful.

### Q2.7

{notes['q27']}

### Q2.8

For model A at 1x, Table 3 gives mirror $E_M={fmt(A['EM'])}$ and rotation $E_R={fmt(A['ER'])}$; rotation is {'harder' if A['ER']>A['EM'] else 'not harder'} by transformed forecast MSE. The corresponding gaps are $C_M={fmt(A['CM'])}$ and $C_R={fmt(A['CR'])}$; a larger consistency gap is not by itself greater forecast error. Figure 1 shows broad directional support and Q2.2 identifies a stronger appearance change under rotation, so appearance dependence is a plausible explanation for a rotation penalty. These results alone do not prove exclusive appearance causation: finite direction imbalance, spatial folding, padding and anisotropic learned filters can also contribute.

{tab(4)}

{fig('figure5',f'Figure 5. Model A seed 0 on the same non-overlapping test video {selected["speed"]} at 1x, 2x and 3x. Five inputs and only the first two predicted/true forecast horizons are displayed for fair comparison. The displayed true frames were checked for separated digits.')}

### Q2.9

Table 4 gives A's $E^{{(2)}}$ at 1x/2x/3x as {fmt(A['E2_1'])}, {fmt(A['E2_2'])}, {fmt(A['E2_3'])}; Copy-last gives {fmt(s['Copy-last']['E2_1'])}, {fmt(s['Copy-last']['E2_2'])}, {fmt(s['Copy-last']['E2_3'])}. {notes['q29']} The reach estimate is a spatial path bound rather than a sharp maximum trackable speed: vector direction, broad digit support, hidden velocity estimates from earlier frames, wall bounces and autoregressive error accumulation all matter. The flow median cannot predict the fastest individual digit, and 8 pixels per axis must not be confused with an isotropic Euclidean bound.

### Q2.10

A alone cannot separate unfamiliar fast motion from limited reach: increasing speed changes both displacement and training-distribution familiarity. Train C on the stated speed mixture with A's unchanged architecture, then compare it with D trained on the identical mixture using greater reach. Evaluate on the same test videos only after fixing designs; compare validation candidates if design selection is needed. C versus A tests data exposure, while D versus C tests the architectural intervention under equal data and training budget. Improvement supports these explanations but is not a proof of a hard physical speed cutoff, because dilation also changes inductive bias and optimization.

## What can data fix?

### Q2.11 — preserved prediction committed before B–D training

{p11}

{tab(5)}

### Q2.12

D retains layer 1's dilation 1 and changes only layer 2's 3-by-3 gate convolution to dilation 3 with padding 3. Hidden widths, biases, patch size, head, zero states and nine-step feedback rollout remain unchanged; it contains no attention. A longest within-step path crosses radii 1 and 3 on the folded grid, giving **16 original pixels per axis** ($L_\infty$ radius 16; Euclidean corner displacement $16\sqrt2\approx22.6$). The previous layer-2 state alone reaches 12 pixels per axis. Combined offsets $\{{-1,0,1\}}+\{{-3,0,3\}}=\{{-4,-3,\ldots,4\}}$ cover intermediate folded cells as well. D has exactly 480,784 parameters, below the 721,176 limit, since dilation changes spacing rather than weight count. No test performance was used to select D.

### Q2.13

The transformation failures have mixed evidence for an **appearance** explanation: A to B changes only symmetry augmentation, and Table 5 changes $E_M$ from {changed(A['EM'],B['EM'])}, $E_R$ from {changed(A['ER'],B['ER'])}, $C_M$ from {changed(A['CM'],B['CM'])}, and $C_R$ from {changed(A['CR'],B['CR'])}. B reduces both gaps and improves rotation correctness, supporting an appearance-exposure contribution to the rotation failure. Mirror correctness slightly worsens despite its smaller gap, so that accuracy failure remains unresolved; finite optimization and architectural anisotropy are alternatives. Lower gaps alone do not establish a successful appearance repair.

The speed failure has an unfamiliar-motion component {'supported' if C['E2_3']<A['E2_3'] else 'not resolved'} by C versus A at 3x: $E^{{(2)}}$ changes from {changed(A['E2_3'],C['E2_3'])}. At unchanged reach this {'shows that broader motion exposure helps' if C['E2_3']<A['E2_3'] else 'does not establish a successful data-only repair under this budget'}. The reach hypothesis is {'supported, though not uniquely proven,' if D['E2_3']<C['E2_3'] else 'not supported as a demonstrated repair'} by D versus C at 3x: {changed(C['E2_3'],D['E2_3'])}; their 2x errors are {fmt(C['E2_2'])} and {fmt(D['E2_2'])}. More data cannot add absent edges to a one-step computational graph. It can, however, teach earlier states to infer motion more effectively, so poor C and improved D would support an architectural bottleneck, not an absolute impossibility of forecasting fast motion with C. Any failure not improved by these comparisons remains unresolved rather than being assigned a convenient causal label.

### Q2.14

At original 1x, A/B/C/D have $E^{{(5)}}$ {fmt(A['E5'])}/{fmt(B['E5'])}/{fmt(C['E5'])}/{fmt(D['E5'])} (Table 5). Relative to A, B {direction(A['E5'],B['E5'])}, C {direction(A['E5'],C['E5'])}, and D {direction(A['E5'],D['E5'])}. C's speed-training tradeoff is worthwhile for the stated robustness objective: it substantially lowers both 2x and 3x error, at a measurable original-speed cost. D's additional architectural change is not justified by accuracy here: C has lower original, 2x and 3x errors, while D's smaller consistency gaps can partly reflect its diffuse predictions. B buys better rotation generalization and consistency with a small original-speed cost, but does not improve every transformed-error column. Two seeds do not establish statistical certainty about small differences. The final wrapped D is retained because the handout specifies it, not because D won every comparison.

## The final model

Use **symmetrised D**, as required, rather than adding the optional D+B training runs. This combines D's prescribed larger reach with guaranteed square-symmetry consistency; it does not assume D is the best model on every accuracy column.

$$\overline f(X)=\frac18\sum_{{g\in G}}g^{{-1}}f(gX).$$

For $a\in G$, set $h=ga$. Closure makes this a permutation of the eight terms and $g^{{-1}}=a h^{{-1}}$, so

$$\overline f(aX)=\frac18\sum_{{h\in G}}a h^{{-1}}f(hX)=a\overline f(X).$$

The last equality uses linearity of pixel permutations. Consequently $C_M=C_R=0$ in exact arithmetic for any underlying $f$, regardless of its correctness. The implemented group is $\{{R^a,MR^a\}}$ with rotation first, then mirror; $MR^a$ is its own inverse. Inputs are transformed before prediction and outputs inversely transformed before float32 averaging. Table 6 reports the remaining floating-point gaps. It is the full eight-element average, not just a mirror/rotation pair.

{tab(6)}

{fig('figure6a',f'Figure 6(a). Final model (symmetrised D, seed 0) on exactly the four transformation/video cases from Figure 3. GIFs: `gifs/figure6a_m1.gif`, `gifs/figure6a_m2.gif`, `gifs/figure6a_r1.gif`, `gifs/figure6a_r2.gif`. Rows 3 and 4 agree up to numerical error; row 2 remains necessary to judge correctness.')}

{fig('figure6b',f'Figure 6(b). Final model on exactly test video {selected["speed"]} from Figure 5, with the same speeds, input frames and first two forecast horizons.')}

### Q2.15

B learns approximate symmetry: its Table 5 gaps are $C_M={fmt(B['CM'])}$, $C_R={fmt(B['CR'])}$. The wrapper enforces exact equivariance mathematically; Table 6 gives symmetrised B gaps {fmt(SB['CM'],True)}/{fmt(SB['CR'],True)} and symmetrised D gaps {fmt(F['CM'],True)}/{fmt(F['CR'],True)}. This guarantees consistency, not accuracy. For B, wrapping changes original $E^{{(5)}}$ from {changed(B['E5'],SB['E5'])}, mirror error from {changed(B['EM'],SB['EM'])}, and rotation error from {changed(B['ER'],SB['ER'])}. For D, these changes are {changed(D['E5'],F['E5'])}, {changed(D['EM'],F['EM'])}, and {changed(D['ER'],F['ER'])}.

Timed inference over 1,000 original clips is {fmt(B['inference'])} seconds for B versus {fmt(SB['inference'])} for symmetrised B, and {fmt(D['inference'])} for D versus {fmt(F['inference'])} for symmetrised D. The eight passes plus transformations/averaging explain the increased cost; measured ratios need not equal exactly eight. The first timed call is included, so cold-start overhead also affects these totals. Averaging incompatible trajectories or phase-shifted digit locations can blur edges and hurt accuracy when the original orientation was already predicted better than the inverse-aligned alternatives. Here wrapping D {'improved' if F['E5']<D['E5'] else 'worsened'} original-speed MSE by only {fmt(abs(F['E5']-D['E5']))}, an essentially negligible change hidden by Table 6's three-significant-figure rounding. Its diffuse predictions were already present in the underlying D. The wrapper requires no additional training or trainable parameters.

### Q2.16

{notes['q216']} The exact row-3/row-4 agreement is fixed by group averaging; any gain in fast-motion accuracy is attributable to the learned underlying D and its training only through the controlled comparisons in Table 5, not to an increase in reach caused by the wrapper.

### Q2.17 — final verdict (under 150 words)

Exact symmetry is established; robust motion learning is not. At 3x the final model's $E^{{(2)}}$ is {fmt(F['E2_3'])}, versus Blank {fmt(s['Blank']['E2_3'])} and Copy-last {fmt(s['Copy-last']['E2_3'])}; at 1x its $E^{{(5)}}$ is {fmt(F['E5'])}. Those gains show useful predictive information, but Figure 6's blurred, weakly moving foreground does not demonstrate correct object tracking. C's improvement over A supports learning from broader motion exposure; D's larger reach brings no measured accuracy gain over C. A skeptic needs both seeds, per-horizon errors and the same-video true-versus-predicted trajectories alongside consistency. The wrapper guarantees agreement between transformed routes, not correct physics. One unpassed test is transfer to unfamiliar shapes or more than two objects at held-out speeds. These experiments therefore support partial predictive learning, not a general motion simulator.
'''
    return setup.replace('\\\\','\\')

def build_notebooks(q2):
    q1=(root/'Q1_River_Forecasting.md').read_text(encoding='utf-8')
    n=nbf.v4.new_notebook(cells=markdown_cells(q1))
    n.metadata.kernelspec={'display_name':'Python (Intel Arc XPU - AI651)','language':'python','name':'ai651-assignment2-xpu'}
    nbf.write(n,root/'Q1_River_Proposal.ipynb')
    # Keep only required experiment stages and answers in the executed Q2 notebook.
    cells=[md('# Question 2 — Motion or Memory?\n\nEight prescribed runs; seed-0 figures; all performance numbers use the complete test split. Cached completed runs are loaded on re-execution. The accompanying `q2_experiments.py` and `q2_results.py` contain the implementation used below.')]
    cells.append(code('''
        from pathlib import Path
        import torch
        from q2_experiments import *
        from q2_results import baseline_figures, final_figures, tables
        torch.set_num_threads(4)
        assert torch.xpu.is_available(), 'Select the Intel XPU kernel.'
        print(torch.__version__, torch.xpu.get_device_name(0))
        ds = load_data()
        checks()
        if not (ROOT/'figures'/'figure1.png').exists(): flow_figure(ds)
    '''))
    # Split prose at the same experiment boundaries used in the run.
    p0,qrest=q2.split('## Baseline and training protocol',1)
    cells.extend(markdown_cells(p0.replace('# Question 2 — Motion or Memory?','',1)))
    protocol,qrest=qrest.split('**Table 2.',1)
    cells.append(md('## Baseline and training protocol'+protocol))
    cells.append(code('''
        # Implementation: q2_experiments.Cell, Forecaster, masked_mse and train.
        assert sum(p.numel() for p in Forecaster().parameters()) == 480784
        for seed in (0,1): train(ds,'A',seed)
    '''))
    cells.append(code('''
        # Cached outputs were originally computed only after the Q2.5 commit.
        assert (ROOT/'Q2_5_preregistered.md').exists()
        for name in ('Blank','Copy-last','A'):
            for seed in ((0,) if name in ('Blank','Copy-last') else (0,1)):
                evaluate(ds,name,seed)
        if not (ROOT/'figures'/'figure5.png').exists(): baseline_figures(ds)
        tables(False)
    '''))
    baseline,rest=('**Table 2.'+qrest).split('## What can data fix?',1)
    cells.extend(markdown_cells(baseline))
    before_bd,after_bd=rest.split('**Table 5.',1)
    cells.extend(markdown_cells('## What can data fix?'+before_bd))
    cells.append(code('''
        # Q2.11 was committed before the first B-D training run.
        assert (ROOT/'Q2_11_preregistered.md').exists()
        for name in ('B','C','D'):
            for seed in (0,1): train(ds,name,seed)
        for name in ('B','C','D'):
            for seed in (0,1): evaluate(ds,name,seed)
    '''))
    middle,final=('**Table 5.'+after_bd).split('## The final model',1)
    cells.extend(markdown_cells(middle))
    cells.extend(markdown_cells('## The final model'+final.split('**Table 6.',1)[0]))
    cells.append(code('''
        for name in ('B','D'):
            for seed in (0,1): evaluate(ds,name,seed,wrap=True)
        if not (ROOT/'figures'/'figure6a.png').exists(): final_figures(ds)
        tables(True)
    '''))
    cells.extend(markdown_cells('**Table 6.'+final.split('**Table 6.',1)[1]))
    n=nbf.v4.new_notebook(cells=cells)
    n.metadata.kernelspec={'display_name':'Python (Intel Arc XPU - AI651)','language':'python','name':'ai651-assignment2-xpu'}
    nbf.write(n,root/'Q2_Motion_or_Memory.ipynb')

if __name__=='__main__':
    q2=q2_text();(root/'Q2_Results_and_Answers.md').write_text(q2,encoding='utf-8')
    build_notebooks(q2)
    q1=(root/'Q1_River_Forecasting.md').read_text(encoding='utf-8')
    report='''---
title: "AI651: Deep Learning for Space, Time and Graphs"
subtitle: "Assignment 2 — River Forecasting; Motion or Memory?"
date: "29 October 2026"
geometry: margin=20mm
papersize: a4
fontsize: 10pt
colorlinks: true
header-includes:
  - \\usepackage{pdflscape}
  - \\usepackage{makecell}
  - \\usepackage{graphicx}
  - \\usepackage{float}
---

**Name:** Muhammad Sami Ul Basit

**Roll No:** 25280025

**GitHub:** [github.com/msbasit/dl4stg-pa2](https://github.com/msbasit/dl4stg-pa2)

**Execution:** All eight learned runs use final epoch 30, with seeds 0 and 1 on the Intel Arc 140T GPU. The separate Q2 notebook, checkpoints, required figures and GIFs accompany this report.

\\newpage

'''+q1+'\n\n\\newpage\n\n'+q2
    (root/'Assignment2_Report.md').write_text(report,encoding='utf-8')
