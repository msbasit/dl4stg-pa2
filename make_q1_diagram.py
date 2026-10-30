from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
root=Path(__file__).resolve().parent
fig,ax=plt.subplots(figsize=(9,11.6));fig.subplots_adjust(0,0,1,1)
ax.set(xlim=(0,9),ylim=(0,11.6));ax.axis('off')
ink='#163a52';blue='#eaf3f7';green='#e4f1eb';gold='#fff3d7'
def box(x,y,w,h,title,lines,color=blue,fs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025',facecolor=color,edgecolor=ink,linewidth=1.3))
    ax.text(x+.1,y+h-.13,title,fontsize=11.8,fontweight='bold',color=ink,va='top')
    for k,line in enumerate(lines):ax.text(x+.1,y+h-.40-k*.205,line,fontsize=fs,color='#1d2b36',va='top')
def arrow(a,b,static=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,linewidth=1.35,color=ink,linestyle='--' if static else '-'))
ax.text(.12,11.45,'Q1 | One-hour river-level forecast',fontsize=18,fontweight='bold',color=ink,va='top')
ax.text(.12,11.10,'Issue every 10 minutes at t₀; use only information received by t₀.',fontsize=11.5,color=ink,va='top')
xs=[.12,2.37,4.62,6.87]
box(xs[0],9.55,2.02,1.20,'Water levels',['S1-S4 and T1','value / mask / age','B × H × 5 × 3'])
box(xs[1],9.55,2.02,1.20,'Actual releases',['Reservoir release Q','value / mask / age','B × H × 1 × 3'])
box(xs[2],9.55,2.02,1.20,'Past schedules',['Archived release plans','Amount, announcement','and effective times'])
box(xs[3],9.55,2.02,1.20,'Rainfall',['R1 and R2','value / mask / age','B × H × 2 × 3'])
for x in xs:arrow((x+1.01,9.55),(x+1.01,9.29))
box(.12,8.18,8.77,1.07,'Align to the issuance grid and scale',[
    'H = 144 grid points: t₀ − 23 h 50 min through t₀; exclude arrivals after t₀.',
    'Rain: interval totals. Training-only scalers; retain observation masks and ages.',
    'Release features: Q, mask, age, error e = Q − Qplan, error-availability mask (5).'])
for x in (1.15,3.31,5.46):arrow((x,8.18),(x,7.94))
box(.12,6.79,2.01,1.11,'Project levels',['Linear 3 → d','B × H × 5 × 3','→ B × H × 5 × d'])
box(2.33,6.79,2.01,1.11,'Project releases',['Linear 5 → d','B × H × 1 × 5','→ B × H × 1 × d'])
box(4.54,6.79,2.01,1.11,'Project rain',['Linear 3 → d','Weight usable gauges','→ B × H × 6 × d'])
box(6.83,3.77,2.06,4.13,'Physical context',[
    'Upstream mask: 6 × 6','River distances: 6 × 6','Gauge-node distances:','6 × 2; node / gauge','locations; source types.','','Guides rain weights,','node embeddings,','attention / decoder','masks and lag biases.','','Rain weights also use','availability / quality;','normalize over usable','gauges only.'],color='#f2f4f5',fs=10.8)
arrow((6.83,7.25),(6.58,7.25),True)
for x in (1.15,3.31,5.46):arrow((x,6.79),(x,6.54))
box(.12,5.50,6.43,1.00,'Combine sources: X has shape B × H × N × d',[
    'Stack 5 level nodes + reservoir: B × H × 6 × d.',
    'Add weighted rain + time and node / source-type embeddings.'])
arrow((6.83,6.0),(6.58,6.0),True);arrow((3.34,5.50),(3.34,5.25))
box(.12,3.77,6.43,1.44,'Encode the historical river state',[
    'Reshape (H, N) → HN = 144 × 6 = 864 tokens.',
    'Two directed, causal attention blocks: each token attends',
    'to its own / upstream nodes at the same or earlier times.',
    'Distance, lag and state bias learn lag preferences.',
    'Encoded history Z: B × 864 × d.'])
arrow((6.83,4.45),(6.58,4.45),True);arrow((3.34,3.77),(3.34,3.49))
box(.12,2.03,6.43,1.42,'Decode four point forecasts',[
    'Queries for S1-S4: B × 4 × d.',
    'Cross-attend to own / upstream history + known plan.',
    'Keys / values: B × (864 + K) × d; context: B × 4 × d.',
    'Point-increment head: B × 4; scale to level units and',
    'add each station’s last available level (age in query).'])
box(6.83,2.03,2.06,1.42,'Known future plan',[
    'K = 6; received by t₀.',
    'Amount / mask / age:',
    'B × K × 1 × 3.',
    'Squeeze + project:',
    'B × K × d + time.'],gold,fs=10.2)
arrow((7.75,3.77),(6.25,3.47),True);arrow((6.83,2.65),(6.58,2.65));arrow((3.34,2.03),(3.34,1.77))
box(.12,.85,8.77,.88,'Output: levels at S1-S4, all at t₀ + 60 minutes',[
    'Point forecasts: B × 4. Separate split-conformal calibration gives lower / upper bounds.',
    'Use held-out calibration residuals; each bound is B × 4 (packed together: B × 4 × 3).'],green,fs=11)
ax.text(.12,.58,'B = issuance examples   H = 144 historical grid points   N = 6 source nodes',fontsize=11,color=ink,va='top')
ax.text(.12,.35,'d = 32 learned features   K = 6 future intervals   × separates tensor axes',fontsize=11,color=ink,va='top')
ax.text(.12,.12,'Solid arrows: data flow     Dashed arrows: static context     Q1 is a proposed design.',fontsize=10.5,color=ink,va='top')
fig.savefig(root/'figures'/'q1_architecture.png',dpi=240,bbox_inches='tight',pad_inches=.06)
fig.savefig(root/'figures'/'q1_architecture.svg',bbox_inches='tight',pad_inches=.06)
plt.close(fig)
