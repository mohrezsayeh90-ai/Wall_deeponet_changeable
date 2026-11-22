"""Recover cached plot bytes and estimate probe endpoints from raster marker positions.

No missing training record or checkpoint is reconstructed. Image estimates are
calibrated using the already archived exact prefix of the same plotted series.
"""
from pathlib import Path
import base64,html,json,re,hashlib
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/molab_soap_cosine100k/retained'
text=html.unescape((OUT/'notebook.html').read_text())
encoded=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',text)
assert len(encoded)==1
png=base64.b64decode(encoded[0]);plot=OUT/'retained_plot_0.png';plot.write_bytes(png)
im=np.asarray(Image.open(plot).convert('RGB'))
assert im.shape[:2]==(1623,2023),im.shape
rows=[json.loads(x) for x in (ROOT/'artifacts/molab/soap-cosine100k-v1/continuation/probe.jsonl').read_text().splitlines()]
# Axis/marker coordinates refer to this preserved original PNG. No curve is redrawn.
x0,xlast=316,1923
result={}
for field,color in [('control',(31,119,180)),('audit',(255,127,14))]:
 mask=np.all(im==np.array(color),axis=-1)
 def marker_y(offset):
  x=round(x0+(xlast-x0)*offset/100000)
  ys=np.where(mask[582:997,x-2:x+3])[0]+582
  return float(np.median(ys)) if len(ys)>0 else None
 pairs=[(row,marker_y(row['offset'])) for row in rows]
 pairs=[(row,y) for row,y in pairs if y is not None]
 yy=np.array([y for row,y in pairs]);lv=np.log([row[field] for row,y in pairs]);a,b=np.polyfit(lv,yy,1)
 residual=yy-(a*lv+b)
 # Refit after excluding marker overlaps with errors exceeding3 raster pixels.
 keep=np.abs(residual)<3
 assert keep.sum()>=15,(field,keep.sum())
 a,b=np.polyfit(lv[keep],yy[keep],1)
 yend=marker_y(100000);estimate=float(np.exp((yend-b)/a));start=rows[0][field]
 result[field]={'source':'raster estimate, not recovered raw log','exact_start_loss':start,'estimated_final_loss':estimate,'estimated_relative_reduction_percent':100*(1-estimate/start),'endpoint_y_pixel':yend,'calibration_points':int(keep.sum()),'calibration_rms_pixels':float(np.sqrt(np.mean((yy[keep]-(a*lv[keep]+b))**2))),'illustrative_plus_minus3_pixel_range':sorted([float(np.exp((yend+d-b)/a)) for d in [-3,3]])}
train=[json.loads(x) for x in (ROOT/'artifacts/molab/soap-cosine100k-v1/continuation/training.jsonl').read_text().splitlines()]
orange=np.all(im==np.array((255,127,14)),axis=-1)
def line_y(offset):
 x=round(x0+(xlast-x0)*offset/100000);ys=np.where(orange[63:477,x-1:x+2])[0]+63
 return float(np.median(ys)) if len(ys) else None
samples=[]
for i in range(99,len(train),100):
 y=line_y(train[i]['step']-322400)
 if y is not None:samples.append((float(np.mean([t['loss'] for t in train[i-99:i+1]])),y))
lv=np.log([m for m,y in samples]);yy=np.array([y for m,y in samples]);a,b=np.polyfit(lv,yy,1);keep=np.abs(yy-(a*lv+b))<2
assert keep.sum()>=15
a,b=np.polyfit(lv[keep],yy[keep],1);final_mean=float(np.exp((line_y(100000)-b)/a));initial_mean=float(np.mean([t['loss'] for t in train[:100]]))
train_estimate={'source':'Raster estimate of final plotted100-record mean; exact archived initial100-record mean','initial_mean':initial_mean,'estimated_final_mean':final_mean,'estimated_relative_reduction_percent':100*(1-final_mean/initial_mean),'calibration_points':int(keep.sum())}
report={'scope':'Cached final figure only; no final weights or raw post347400 logs available','image_sha256':hashlib.sha256(png).hexdigest(),'image_dimensions':list(im.shape[:2]),'reported_completed_step_in_cached_html':422400,'probe_estimates':result,'training_mean_estimate':train_estimate}
(OUT/'image_estimates.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
