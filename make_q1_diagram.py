from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
root=Path(__file__).resolve().parent
fig,ax=plt.subplots(figsize=(12,8));ax.set(xlim=(0,12),ylim=(0,8));ax.axis('off')
def box(x,y,w,h,text,color='#e9f1f6'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07',facecolor=color,edgecolor='#24566b',linewidth=1.3))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color='#24566b',linewidth=1.3))
box(.2,6.6,3.4,1,'Levels S1-S4, T1\nB x 144 x 5 x 1\n+ masks / observation ages')
box(4.1,6.6,3.4,1,'Actual reservoir releases\nB x 144 x 1 x 1\n+ masks / ages / discrepancies')
box(8,6.6,3.6,1,'Rain R1, R2\nB x 144 x 2 x 1\n+ masks / observation ages')
box(.5,4.85,6.7,1.05,'As-of alignment; training-only scaling\nType-specific projections to width 32, then stack\n5 level nodes + 1 reservoir: B x 144 x 6 x 32')
box(8,4.85,3.6,1.05,'Project rain to width 32\nGauge-to-node weighted fusion\nB x 144 x 6 x 32')
arrow((1.9,6.6),(2.1,5.9));arrow((5.8,6.6),(5.6,5.9));arrow((9.8,6.6),(9.8,5.9));arrow((8,5.38),(7.25,5.38))
box(.5,3.05,6.7,1.05,'Add projected rain and static/time embeddings\nFlatten (144, 6) to 864 historical tokens\n2 directed, causal space-time attention blocks\nB x 864 x 32')
arrow((3.8,4.85),(3.8,4.15))
box(8,3.05,3.6,1.05,'Connectivity / river distance\n6 x 6; node metadata 6 x 4\nGauge locations 2 x 2\nGauge-node distances 6 x 2')
arrow((8,3.57),(7.25,3.57));arrow((9.8,4.1),(9.8,4.8))
box(8,1.25,3.6,1.05,'Announced future release plan\n6 intervals; B x 6 x 1 x 3\nAmount / mask / age\nSqueeze + project: B x 6 x 32','#fcf0d9')
box(.5,1.25,6.7,1.05,'4 station queries attend to history + known plan\nB x 4 x 32; residual level forecast head\nPoint forecasts + calibrated intervals: B x 4\nS1, S2, S3, S4 at issuance + 60 minutes','#dcefe6')
arrow((3.8,3.05),(3.8,2.35));arrow((8,1.78),(7.25,1.78))
ax.text(6,.45,'Only data available at issuance enter the model. No Q1 training or benchmark is claimed.',ha='center',fontsize=10,color='#24566b')
fig.savefig(root/'figures'/'q1_architecture.png',dpi=180,bbox_inches='tight');plt.close(fig)
