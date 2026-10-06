import os
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT='/home/user/Random/ebt-explainer'
OUT=os.environ.get('EBT_CHECK_DIR', '/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/ebt/digitize_check')
d=json.load(open(ROOT+'/data/paper_heatmaps.json'))
hm=d['heatmaps']
fig,axs=plt.subplots(len(hm),2,figsize=(12,4.2*len(hm)))
for k,h in enumerate(hm):
    axs[k,0].imshow(Image.open(ROOT+'/'+h['image'])); axs[k,0].axis('off'); axs[k,0].set_title('original '+h['figure'],fontsize=9)
    E=np.array(h['energy'])
    axs[k,1].imshow(E,cmap='viridis_r',vmin=0,vmax=1,aspect='auto')
    axs[k,1].set_yticks(range(len(h['rows']))); axs[k,1].set_yticklabels(h['rows'],fontsize=7)
    for i in range(E.shape[0]):
        for j in range(E.shape[1]): axs[k,1].text(j,i,f'{E[i,j]:.2f}',ha='center',va='center',fontsize=5,color='w' if E[i,j]>0.5 else 'k')
    axs[k,1].set_title('digitized '+h['id'],fontsize=9)
fig.tight_layout(); fig.savefig(OUT+'/heatmaps_check.png',dpi=80)
