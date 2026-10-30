from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
root=Path(__file__).resolve().parent
fig,ax=plt.subplots(figsize=(8.2,9.5)); ax.set(xlim=(0,8.2),ylim=(0,9.5)); ax.axis('off')
ink='#173f53'; blue='#edf4f8'; green='#e5f2eb'; gold='#fff3da'
def box(x,y,w,h,title,body,color=blue,fs=10.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.035',facecolor=color,edgecolor=ink,linewidth=1.15))
    ax.text(x+.13,y+h-.16,title,ha='left',va='top',fontsize=11.4,fontweight='bold',color=ink)
    for index,line in enumerate(body.split('\n')):
        ax.text(x+.13,y+h-.42-index*.20,line,ha='left',va='top',fontsize=fs,color='#172c36')
def arrow(a,b,dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=ink,linewidth=1.2,linestyle='--' if dashed else '-'))
ax.text(.1,9.30,'Q1 | One-hour river-level forecast',fontsize=16,fontweight='bold',color=ink,va='top')
ax.text(.1,8.97,'Issued every 10 minutes using information received by issuance time.',fontsize=10.5,color=ink,va='top')
box(.1,7.65,2.5,1.05,'Water levels','S1, S2, S3, S4 and T1\nValues: B × H × 5 × 1')
box(2.85,7.65,2.5,1.05,'Actual releases','Q + past plan error\nValues: B × H × 1 × 1')
box(5.6,7.65,2.5,1.05,'Rainfall','Gauges R1 and R2\nValues: B × H × 2 × 1')
for x in (1.35,4.1,6.85): arrow((x,7.65),(x,7.31))
box(.1,5.78,8,1.49,'Align, project and combine sources',
    'Align to the same 10-minute grid; scale using training data only.\n'
    'Carry each value with its availability mask and observation age.\n'
    'Project levels / releases to d features; stack N source nodes.\n'
    'Project rain; weight gauges for each node; add rain and time / node embeddings.\n'
    'Combined history: B × H × N × d',fs=10.3)
arrow((2.7,5.78),(2.7,5.39))
box(.1,3.53,5.22,1.82,'Encode the historical river state',
    'Reshape time × node into HN = 864 tokens.\n'
    'Two directed, causal attention blocks.\n'
    'Each token uses its own / upstream nodes\n'
    'at the same or earlier historical times.\n'
    'Learn distance- and state-dependent lag weights.\n'
    'Encoded history: B × 864 × d',fs=10.5)
box(5.75,3.53,2.35,1.82,'Physical context',
    'River connectivity\nand river distances\nNode / gauge locations\nand source types\n'
    'Guides attention and\nrain-gauge weighting.',fs=10)
arrow((5.75,4.4),(5.34,4.4),True); arrow((6.9,5.35),(6.9,5.76),True)
arrow((2.7,3.53),(2.7,3.32))
box(.1,1.85,5.22,1.43,'Decode four station forecasts',
    'One query each for S1, S2, S3 and S4.\n'
    'Cross-attend to upstream history + known plan.\n'
    'Contexts: B × 4 × d; predict a level increment.\n'
    'Add last observed level; calibrate intervals.',green,fs=10.5)
box(5.75,1.85,2.35,1.43,'Known future plan',
    'K release intervals\nAmount / mask / age\nB × K × 1 × 3\nSqueeze + project:\nB × K × d',gold,fs=10)
arrow((5.75,2.45),(5.34,2.45)); arrow((2.7,1.85),(2.7,1.52))
box(.1,.73,8,.75,'Output: S1, S2, S3 and S4 at issuance + 60 minutes',
    'Point levels, lower bounds and upper bounds: each B × 4',green,fs=11)
ax.text(.1,.51,'AXIS KEY   B = issuance examples   H = 144 historical grid points   N = 6 source nodes',fontsize=10,color=ink,va='top')
ax.text(.1,.26,'d = 32 learned features   K = 6 future 10-minute intervals   × separates tensor axes',fontsize=10,color=ink,va='top')
fig.savefig(root/'figures'/'q1_architecture.png',dpi=240,bbox_inches='tight',pad_inches=.07)
fig.savefig(root/'figures'/'q1_architecture.svg',bbox_inches='tight',pad_inches=.07)
plt.close(fig)
