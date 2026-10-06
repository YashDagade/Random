import os
import json
crops=json.load(open('/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/ebt/digitize_check/crops.json'))
KIND={'fig01':'diagram','fig02':'diagram','fig03':'diagram','fig04':'plot','fig05':'plot','fig06':'plot','fig07':'plot','fig08':'heatmap','fig09':'plot',
      'fig10':'image grid','fig11':'heatmap','fig12':'plot','figB1':'plot','figB2':'heatmap','figB3':'plot','figC1':'plot','figE1':'diagram','figH1':'diagram'}
ALT={
 'fig01':'Four block diagrams: AR Transformer, RNN, Diffusion Transformer (outputs noise), EBT (takes context and candidate, outputs energy).',
 'fig02':'EBT thinking for text (next-token distribution) and video (next frame): predictions refined over steps 0-4 by gradient descent on energy.',
 'fig03':'3D energy landscape for context "The dog caught the ___"; a prediction descends from high to low energy over steps 0-4.',
 'fig04':'Text pretraining scaling vs data, batch size, depth: EBT vs Transformer++.',
 'fig05':'Text pretraining scaling vs parameters, FLOPs, embedding width: EBT vs Transformer++.',
 'fig06':'(a) OOD perplexity increase vs forward passes; (b) BoN-5 verification gain vs tokens trained.',
 'fig07':'% perplexity improvement from thinking vs OOD magnitude shift (5 datasets) with a linear trend.',
 'fig08':'Heatmaps of normalized energy per token across 12 thinking iterations for two sentences.',
 'fig09':'Video (SSV2) scaling vs embedding dimension and non-embedding parameters: EBT vs Transformer++.',
 'fig10':'OOD image denoising: EBT after 1-3 steps vs DiT after 100-300 steps, with ground truth.',
 'fig11':'Heatmap of normalized energy per video frame across thinking iterations, with frame thumbnails.',
 'fig12':'PSNR vs number of forward passes (log x): EBT at 1-3 passes vs DiT at 100-300 passes.',
 'figB1':'(a) BoN-10 vs BoN-2 gain vs tokens; (b) Fig 6b trend extrapolated to 15T tokens.',
 'figB2':'Energy heatmaps: an in-distribution sentence vs a random OOD token sequence.',
 'figB3':'Larger-scale (FineWeb) data scaling: EBT vs Transformer++, full range and zoomed.',
 'figC1':'EBT System 1 vs System 2 hyperparameters: perplexity vs non-embedding parameters.',
 'figE1':'Diffusion (supervision at every denoising step) vs EBM (supervision only at the end of optimization).',
 'figH1':'Feed-forward model (x to y-hat) vs energy-based model (x and y-hat to energy).',
}
DIG={'fig04a':'scaling:fig4a','fig04b':'scaling:fig4b','fig04c':'scaling:fig4c','fig05a':'scaling:fig5a','fig05b':'scaling:fig5b','fig05c':'scaling:fig5c',
     'fig06a':'scaling:fig6a','fig06b':'scaling:fig6b','fig07':'scaling:fig7','fig09a':'scaling:fig9a','fig09b':'scaling:fig9b','fig12':'scaling:fig12',
     'figB1a':'scaling:figB1a','figB1b':'scaling:figB1b','figB3a':'scaling:figB3a','figB3b':'scaling:figB3b','figC1':'scaling:figC1',
     'fig08a':'paper_heatmaps:fig8a','fig08b':'paper_heatmaps:fig8b','fig11':'paper_heatmaps:fig11','figB2':'paper_heatmaps:figB2_seq1,figB2_seq2'}
figs=[]
for c in crops:
    parent=c.get('parent') or c['id']
    rec=dict(id=c['id'], file=c['file'], caption=c['caption'], page=c['page'], panel=c.get('panel'),
             parent=c.get('parent'), kind=KIND[parent], width=c['size_px'][0], height=c['size_px'][1])
    if c.get('panels'): rec['panels']=c['panels']
    if c['id'] in ALT: rec['alt']=ALT[c['id']]
    if c['id'] in DIG: rec['digitized']=DIG[c['id']]
    if c['id'] in ('fig08a','fig08b'): rec['panel_note']='The paper prints no subcaption for these panels; the caption refers to them as (a) and (b). Panel text written by the explainer.'
    figs.append(rec)
out={'_meta':{'source':'cropped from the paper PDF (Gladstone et al., "Energy-Based Transformers are Scalable Learners and Thinkers", arXiv 2507.02092v1) with pymupdf at 200 dpi',
              'note':'Crops are tight around the figure content (vector paths, embedded images and figure text found inside the figure area). The main "Figure N:" caption is never inside an image. Whole-figure crops keep the printed (a)/(b) subcaption lines; panel crops (e.g. fig04a) contain only the panel. White background (paper figures). Captions are the exact caption text from the PDF, with line-break hyphenation removed. Paper colours: orange = Transformer++ / DiT, blue = EBT.',
              'count':len(figs)},
     'figures':figs}
json.dump(out, open('/home/user/Random/ebt-explainer/data/paper_figs.json','w'), ensure_ascii=False, indent=1)
print(len(figs), 'figures written')
