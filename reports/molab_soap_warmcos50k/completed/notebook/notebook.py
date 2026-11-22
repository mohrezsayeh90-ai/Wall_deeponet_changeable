import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import json
    from pathlib import Path

    return Path, json, mo


@app.cell(hide_code=True)
def wall_training_refresh(mo):
    training_refresh = mo.ui.refresh(options=[10, 30, 60], default_interval=10, label="تازه‌سازی خودکار؛ برای تازه‌سازی فوری دکمهٔ چرخش را بزنید")
    return (training_refresh,)


@app.cell(hide_code=True, expand_output=True)
def wall_soap_dashboard(Path, json, mo, training_refresh):
    training_refresh.value
    import matplotlib.pyplot as _plt
    import numpy as _np
    _old=Path('/home/marimo/wall_soap_sweep50k_v1');_new=Path('/home/marimo/wall_soap_warmcos50k_v1')
    def _read(_p):
     try:return json.loads(_p.read_text())
     except (FileNotFoundError,json.JSONDecodeError):return {}
    def _rows(_p):
     _out=[]
     for _l in (_p.read_text().splitlines() if _p.exists() else []):
      try:_out.append(json.loads(_l))
      except json.JSONDecodeError:pass
     return _out
    _s=_read(_new/'status.json');_fail=_read(_new/'failure.json');_cohort=_read(_new/'fem_cohort.json');_hard=set(_cohort.get('hard_cases',[]));_nf=[_read(p) for p in sorted((_new/'fem').glob('fem_*.json'))];_nf=[x for x in _nf if x]
    _header=mo.md(f'''## ران جدید ۵۰هزارگامی: افزایش نرم 1e-4 → 5e-4 سپس کسینوسی تا 8e-5
    **وضعیت:** {_s.get('phase','آماده‌سازی')} · **گام جدید:** {_s.get('offset',0):,} از ۵۰٬۰۰۰ · **گام مدل:** {_s.get('step',392400):,} · **نرخ:** {_s.get('learning_rate',8e-5 if _s.get('phase')=='completed' else 1e-4):.8g}

    ادامهٔ وزن نهایی ۳۹۲٬۴۰۰؛ منحنی جدید روی محور تجمعی از ۳۴۲٬۴۰۰؛ batch=32. نمودارهای قبلی حفظ شده‌اند و منحنی جدید روی آن‌ها اضافه شده است.

    **FEM فقط ارزیابی نمایشی است:** پاسخ‌ها در فرایند جداگانه خوانده می‌شوند؛ هیچ جواب FEM وارد ورودی، loss، گرادیان، انتخاب نرخ یا انتخاب وزنِ آموزش نمی‌شود. ۲۰ کانال داخل محدودهٔ اصلی، شامل شکل‌های چندبرآمدگی و نامنظم؛ ۱۲ مورد با انحنای بالاتر جداگانه نمایش داده می‌شوند. Re=30.
    ''')
    _names=['constant_1e4','constant_2e5','constant_5e5','constant_2e4','cosine_1e4_2e5','constant_1e4_extend25k'];_series=[(_n,_old/_n,False) for _n in _names]+[('Previous50k cosine',Path('/home/marimo/wall_soap_highlr50k_v1'),False),('NEW:warmup+cosine',_new,True)]
    _fig,_axs=_plt.subplots(3,1,figsize=(11,9),layout='constrained')
    for _name,_r,_isnew in _series:
     _t=_rows(_r/'continuation/training.jsonl');_v=_rows(_r/'continuation/probe.jsonl');_style={'color':'black','lw':2.4} if _isnew else {'alpha':.65,'lw':1.1}
     if _t:
      _y=[x['loss'] for x in _t];_x=[x['step']-342400 for x in _t]
      _axs[0].plot(_x,[sum(_y[max(0,i-99):i+1])/min(i+1,100) for i in range(len(_y))],label=_name,**_style)
      _axs[2].plot(_x,[x['learning_rate'] for x in _t],label=_name,**_style)
     if _v:
      _axs[1].plot([x['step']-342400 for x in _v],[x['control'] for x in _v],'o-',label=_name,**_style)
      if _isnew:_axs[1].plot([x['step']-342400 for x in _v],[x['audit'] for x in _v],'s--',color='black',label='NEW audit')
    for _a in _axs:
     _a.grid(alpha=.2);_a.set(xlabel='Updates since common checkpoint342400',yscale='log')
     if _a.lines:_a.legend(fontsize=7,ncol=2)
    _axs[0].set(ylabel='Training mean loss');_axs[1].set(ylabel='Fixed physics probe loss');_axs[2].set(ylabel='Actual applied LR')
    _training_plot=mo.as_html(_fig);_plt.close(_fig)
    _f,_aa=_plt.subplots(3,2,figsize=(12,12),layout='constrained')
    for _name,_r,_isnew in _series:
     if _isnew:_vals=[dict(v,offset=v['step']-342400) for v in _nf]
     elif _name=='Previous50k cosine':_vals=[dict(_read(p),offset=int(p.stem.split('_')[-1])-342400) for p in sorted((_r/'fem').glob('fem_*.json'))]
     else:
      _files=sorted((_r/'continuation').glob('fem_*.json'));_vals=[_read(p) for p in _files]
      for _v,_p in zip(_vals,_files):_v['offset']=int(_p.stem.split('_')[-1])-342400
     for _ax,_key in zip(_aa[0],['velocity_l2','pressure_l2']):
      _xx=[];_yy=[]
      for _v in _vals:
       _arr=[x[_key] for x in _v.get('cases',[]) if str(x['case']) in _hard and x.get(_key) is not None]
       if _arr:_xx.append(_v['offset']);_yy.append(100*float(_np.median(_arr)))
      if _xx:_ax.plot(_xx,_yy,'o-',label=_name,**({'color':'black','lw':2.4} if _isnew else {'alpha':.65,'lw':1.1}))
    for _ax,_key in zip(_aa[1],['u_l2','v_l2']):
     for _group in ['all20','hard12']:
      if _nf:_ax.plot([v['step']-342400 for v in _nf],[100*v[_group][_key+'_median'] for v in _nf],'o-',label='NEW '+_group)
    for _ax,_key in zip(_aa[2],['velocity_l2','pressure_l2']):
     if _nf:
      for _case in _nf[0]['cases']:
       _id=_case['case'];_vv=[(v['step']-342400,next(x[_key] for x in v['cases'] if x['case']==_id)) for v in _nf];_vv=[(x,y) for x,y in _vv if y is not None]
       if _vv:_ax.plot([x for x,y in _vv],[100*y for x,y in _vv],label='C'+_id+('*' if _id in _hard else ''),lw=1.5 if _id in _hard else .8,alpha=.9 if _id in _hard else .45)
    for _ax,_title in zip(_aa.flat,['Higher-curvature12: velocity median, same FEM','Higher-curvature12: pressure median, same FEM','NEW u median; straight-zero v excluded separately','NEW v median; undefined zero-reference ratios omitted','NEW per-case velocity; * higher-curvature group','NEW per-case pressure; * higher-curvature group']):
     _ax.set(title=_title,xlabel='Updates since342400',ylabel='Relative L2 error (%)',yscale='log');_ax.grid(alpha=.2)
     if _ax.lines:_ax.legend(fontsize=6,ncol=3)
    _fem_plot=mo.as_html(_f);_plt.close(_f)
    _last=_nf[-1]['step'] if _nf else 'در انتظار'
    _dl=mo.download(data=lambda:(_new/'continuation/checkpoint.npz').read_bytes(),filename='warmcos50k_last_saved.npz',label='دانلود آخرین وزن ذخیره‌شدهٔ ران جدید') if (_new/'continuation/checkpoint.npz').exists() else mo.md('وزن اولیه در حال آماده‌سازی است.')
    mo.vstack([training_refresh,_header,mo.md(str(_fail)) if _fail else mo.md(''),_training_plot,mo.md(f'### ارزیابی مستقل FEM — آخرین وزن بررسی‌شده: {_last}\nعلامت * فقط معیار انحنای هندسه است؛ دشواری جریان را به‌تنهایی تضمین نمی‌کند. تمام ۲۰ کیس، از جمله شکل‌های نامنظم و پلهٔ نرم، در نمودارهای تک‌کیس دیده می‌شوند.'),_fem_plot,_dl])
    return


@app.cell(hide_code=True)
def wall_adam_dashboard(Path, json, mo):
    import matplotlib.pyplot as _plt
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo as _ZI
    _r=Path('/home/marimo/wall_rad_ab_15k_v1')
    def _read(_p):
     try:return json.loads(_p.read_text())
     except (FileNotFoundError,json.JSONDecodeError):return {}
    def _rows(_p):
     _out=[]
     for _l in (_p.read_text().splitlines() if _p.exists() else []):
      try:_out.append(json.loads(_l))
      except json.JSONDecodeError:pass
     return _out
    _state=_read(_r/'status.json');_sup=_read(_r/'supervisor-status.json');_fail=_read(_r/'failure.json');_done=0;_table=[]
    _fig,_axs=_plt.subplots(2,1,figsize=(10,5),layout='constrained')
    for _name,_label in [('baseline','ادامهٔ عادی'),('rad','نقاط تطبیقی')]:
     _t=_rows(_r/_name/'training.jsonl');_v=_rows(_r/_name/'probe.jsonl');_off=(_t[-1]['step']-287400) if _t else 0;_done+=_off
     _table.append(f'| {_label} | {_off:,} / ۱۵٬۰۰۰ |')
     if _t:
      _y=[x['loss'] for x in _t];_axs[0].plot([x['step']-287400 for x in _t],[sum(_y[max(0,i-99):i+1])/min(i+1,100) for i in range(len(_y))],label=_name)
     if _v:
      _axs[1].plot([x['offset'] for x in _v],[x['audit'] for x in _v],'o-',label=_name+' audit')
      _axs[1].plot([x['offset'] for x in _v],[x['control'] for x in _v],'--',alpha=.6,label=_name+' control')
    for _ax in _axs:
     _ax.set(yscale='log',xlabel='Updates from identical step 287,400',ylabel='Loss');_ax.grid(alpha=.2)
     if _ax.lines:_ax.legend()
    _axs[0].set_title('Training means: objectives differ; use fixed probes and FEM to compare')
    _axs[1].set_title('Identical fixed development probes')
    _phase='خطا؛ نیاز به بررسی' if _fail else ('پایان دو آزمایش' if _state.get('phase')=='completed' else ('آموزش '+_state.get('branch','') if _state else 'بررسی فنی و ارزیابی FEM اولیه'))
    _updated=_dt.now(_ZI('Asia/Tehran')).strftime('%H:%M:%S')
    _current=_state.get('step',287400)
    _cards=mo.Html(f'<section dir="rtl" style="font-family:Tahoma;background:#f0fdfa;padding:22px;border-radius:15px"><h2>آزمایش دو روش ادامهٔ آموزش</h2><h3>{_phase} · گام {_current:,}</h3><p>آخرین تازه‌سازی: {_updated} — تهران</p><p>شروع هر شاخه: ۲۸۷٬۴۰۰ · نرخ ثابت: 2e-5 · هدف هر شاخه: ۳۰۲٬۴۰۰</p><progress value="{_done}" max="30000" style="width:100%;height:24px"></progress><p>{_done:,} از ۳۰٬۰۰۰ گام مجموع دو آزمایش</p><p>عادی ابتدا اجرا می‌شود؛ تطبیقی سپس از همان وزن اولیه. ارزیابی با همان ۲۰ هندسه و ۸ نمونهٔ FEM انجام می‌شود.</p><p>وزن‌ها هر ۱۰۰ گام ذخیره می‌شوند؛ نسخه‌های دوره‌ای با یک فرایند مستقل به گیت‌هاب منتقل می‌شوند.</p></section>')
    _plot=mo.as_html(_fig);_plt.close(_fig)
    _bundle=_r/'results.zip';_dl=mo.download(data=_bundle.read_bytes(),filename='rad_ab_15k_results.zip',label='دانلود نتایج دو آزمایش') if _bundle.exists() else mo.md('خروجی نهایی پس از پایان دو شاخه قابل دانلود خواهد بود.')
    mo.vstack([_cards,mo.md('| روش | گام جدید |\n|---|---:|\n'+'\n'.join(_table)),_plot,_dl])
    return


@app.cell(hide_code=True)
def wall_training_download(mo):
    mo.md("""
    نمودار و وضعیت زنده در پنل بالا قرار دارد. دکمهٔ تازه‌سازی فقط نمایش را به‌روز می‌کند.
    """)
    return


if __name__ == "__main__":
    app.run()
