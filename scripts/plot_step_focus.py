"""Make the whole downstream step and recirculation prominent using cached fields."""
from pathlib import Path
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/rounded_backward_step_82216'


def main():
    with np.load(OUT/'fields.npz',allow_pickle=False) as d:
        x,y,b,t=d['x'],d['y'],d['bottom'],d['top']
        fields=[d['model'],d['fem']]
    plt.rcParams.update({'font.size':11,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(13,10),layout='constrained')
    gs=fig.add_gridspec(3,1,height_ratios=[.8,1.5,1.5])
    overview=fig.add_subplot(gs[0])
    overview.fill_between(x,b,t,color='#dbeafe')
    overview.fill_between(x,0,b,color='#cbd5e1')
    overview.plot(x,b,'k',lw=1.5);overview.plot(x,t,'k',lw=1.5)
    limits=(1.65,3.4,-.012,.45)
    overview.add_patch(Rectangle((limits[0],limits[2]),limits[1]-limits[0],limits[3]-limits[2],fill=False,edgecolor='#dc2626',lw=2))
    overview.set(xlim=(0,5),ylim=(-.03,1.03),aspect='equal',xlabel='x',ylabel='y',title='Full channel: red rectangle locates the enlarged step below')
    for row,field,label in zip([1,2],fields,['Stored PINN / DeepONet','Independent FEM']):
        ax=fig.add_subplot(gs[row])
        u=np.ma.masked_invalid(field[:,:,0]);v=np.ma.masked_invalid(field[:,:,1])
        pc=ax.pcolormesh(x,y,u,cmap='coolwarm',norm=TwoSlopeNorm(vmin=-.035,vcenter=0,vmax=.3),shading='auto',rasterized=True)
        ax.streamplot(x,y,u,v,color='#172554',density=(4,3),linewidth=.8,arrowsize=1.)
        ax.contour(x,y,u,levels=[-.001],colors=['#dc2626'],linewidths=1.5)
        ax.plot(x,b,color='black',lw=2.2)
        ax.fill_between(x,-.02,b,color='#cbd5e1',zorder=4)
        ax.set(xlim=limits[:2],ylim=limits[2:],aspect='equal',xlabel='x',ylabel='y',title=label+' | whole downstream step and reverse-flow region')
        ax.annotate('',xy=(1.77,float(b.max())),xytext=(1.77,0),arrowprops={'arrowstyle':'<->','color':'#334155'},zorder=6)
        ax.text(1.80,.17,f'h = {b.max():.3f}',color='#334155',zorder=6)
        fig.colorbar(pc,ax=ax,label='Axial velocity u; blue indicates reverse flow',shrink=.85,extend='max')
    fig.suptitle('Enlarged view of the rounded backward-facing step | Re = 100 | Adam step 82,216\nSame geometry and solution; equal physical x/y scale in every panel',fontsize=14)
    fig.savefig(OUT/'step_enlarged.png',dpi=220)
    fig.savefig(OUT/'step_enlarged.pdf');plt.close(fig)
    readme=OUT/'README.md'
    text=readme.read_text()
    if '[Enlarged whole-step view]' not in text:
        text=text.replace('![Flow and streamlines](streamlines.png)', '[Enlarged whole-step view](step_enlarged.png) · [Enlarged view PDF](step_enlarged.pdf)\n\n![Whole step, enlarged](step_enlarged.png)\n\nThe enlarged view includes the full descending floor and downstream reverse-flow region. The red rectangle locates the crop in the full channel. Both axes retain equal physical scale; geometry, weights and numerical results are unchanged. Reproduce this view with `python -m scripts.plot_step_focus`.\n\n![Flow and streamlines](streamlines.png)')
        readme.write_text(text)
    with zipfile.ZipFile(OUT/'rounded_step_82216.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.iterdir()):
            if path.suffix!='.zip':z.write(path,arcname=OUT.name+'/'+path.name)
    with zipfile.ZipFile(OUT/'rounded_step_82216.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():assert z.read(name)==(OUT/Path(name).name).read_bytes()

if __name__=='__main__':main()
