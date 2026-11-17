training_refresh.value
import matplotlib.pyplot as _plt
_root = Path('/home/marimo/wall_lr_trials')
_manifest = json.loads((_root/'manifest.json').read_text())
_suite_file = _root/'suite-status.json'
_suite = json.loads(_suite_file.read_text()) if _suite_file.exists() else {'phase':'starting'}
_base = _manifest['base_step']
_active = _suite.get('active_case', _manifest['cases'][0]['id'])
_case = next(c for c in _manifest['cases'] if c['id']==_active)
_trials = {}
for _spec in _manifest['cases']:
    _f = _root/'cases'/_spec['id']/'training.jsonl'
    _rr = []
    for _line in (_f.read_text().splitlines() if _f.exists() else []):
        try:
            _rr.append(json.loads(_line))
        except json.JSONDecodeError:
            pass
    _trials[_spec['id']] = _rr
_current = _trials[_active]
_last = _current[-1] if _current else {}
_offset = _last.get('step',_base)-_base
_done = len(_suite.get('completed',[]))
_percent = min(100,100*(_done*15000+(0 if _active in _suite.get('completed',[]) else _offset))/60000)
_phase = _suite.get('phase','starting')
_label = {'running':'در حال اجرای آزمون','starting':'در حال آماده‌سازی','completed':'هر چهار آزمون تمام شدند','failed':'آزمون متوقف شده؛ نیاز به بررسی'}.get(_phase,_phase)
_loss = f"{_last['loss']:.6f}" if _last else 'در حال آماده‌سازی'
_cards = mo.Html(f'''<section dir="rtl" style="font-family:Tahoma,Arial;background:#f0fdfa;border:1px solid #99f6e4;border-radius:16px;padding:22px;color:#134e4a">
<h1>مقایسهٔ زندهٔ نرخ‌های یادگیری Adam</h1><h3>{_label} — {_case['label']}</h3>
<div style="display:flex;gap:22px;flex-wrap:wrap;font-size:20px"><div>پیشرفت این حالت: <b>{_offset:,} / ۱۵٬۰۰۰</b></div><div>آخرین Loss: <b>{_loss}</b></div><div>آزمون‌های کامل: <b>{_done}/4</b></div></div>
<progress max="100" value="{_percent}" style="width:100%;height:24px;margin-top:18px"></progress>
<div>پیشرفت کل: {_percent:.1f}٪ — هر چهار حالت مستقل از گام {_base:,}، با Adam و RNG یکسان شروع می‌شوند.</div>
<div style="margin-top:10px">نمودار اول ادامهٔ همان تاریخچه است؛ شاخه‌ها در نقطهٔ شروع مشترک روی هم مقایسه می‌شوند. نمودار دوم ارزیابی ثابت است.</div></section>''')
_fig,_axes = _plt.subplots(2,1,figsize=(11,8),layout='constrained')
_old_path = Path('/home/marimo/wall_adam_1h/run/training.jsonl')
_old = [json.loads(l) for l in _old_path.read_text().splitlines() if l.strip()]
_old = [r for r in _old if r['step']<=_base]
if _old:
    _y = [r['loss'] for r in _old]
    _axes[0].plot([r['step'] for r in _old],_y,color='#94a3b8',alpha=.25,lw=.5)
    _axes[0].plot([r['step'] for r in _old],[sum(_y[max(0,i-99):i+1])/min(i+1,100) for i in range(len(_y))],color='#64748b',label='Previous molab history',lw=1.5)
for _spec,_color in zip(_manifest['cases'],['#0f766e','#2563eb','#dc2626','#9333ea']):
    _records = _trials[_spec['id']]
    if _records:
        _x = [r['step'] for r in _records];_y = [r['loss'] for r in _records]
        _axes[0].plot(_x,_y,color=_color,alpha=.14,lw=.5)
        _axes[0].plot(_x,[sum(_y[max(0,i-99):i+1])/min(i+1,100) for i in range(len(_y))],color=_color,label=_spec['label'],lw=1.8)
    _pp = _root/'cases'/_spec['id']/'probe.jsonl'
    _probe = []
    for _line in (_pp.read_text().splitlines() if _pp.exists() else []):
        try:
            _probe.append(json.loads(_line))
        except json.JSONDecodeError:
            pass
    if _probe:_axes[1].plot([r['offset'] for r in _probe],[r['loss'] for r in _probe],'o-',color=_color,label=_spec['label'])
_axes[0].axvline(_base,color='black',ls=':',label=f'Common checkpoint {_base:,}')
_axes[0].set(xlabel='Global Adam step (independent branches aligned)',ylabel='Training loss (log)',yscale='log',title='Previous history + LR trials; bold: mean over up to 100 logged records')
_axes[1].set(xlabel='Updates from common checkpoint',ylabel='Fixed development-probe loss (log)',yscale='log',title='Same fixed geometries, Re and points; checked at 0 / 5k / 10k / 15k',xlim=(0,15000))
for _ax in _axes:
    _ax.grid(alpha=.2)
    if _ax.lines:_ax.legend(fontsize=8)
_plot = mo.as_html(_fig)
_plt.close(_fig)
_rows = []
for _spec in _manifest['cases']:
    _ff = _root/'cases'/_spec['id']/'fem_pilot.json'
    if _ff.exists():
        _fm = json.loads(_ff.read_text())
        _rows.append(f"| {_spec['label']} | {_fm['median_velocity_l2']:.2%} | {_fm['median_pressure_l2']:.2%} |")
_table = mo.md('| حالت | میانهٔ خطای سرعت FEM | میانهٔ خطای فشار FEM |\n|---|---:|---:|\n'+'\n'.join(_rows)) if _rows else mo.md('مقایسهٔ هشت نمونهٔ FEM پس از پایان هر حالت اضافه می‌شود. loss آموزش به‌تنهایی معیار انتخاب برنده نیست.')
_bundle = _root/'lr_trials_results.zip'
_download = mo.download(data=_bundle.read_bytes(),filename='lr_trials_results.zip',label='دانلود مقایسه و وزن‌های چهار حالت') if (_root/'comparison_ready.json').exists() else mo.md('بستهٔ دانلود پس از پایان چهار آزمون آماده می‌شود.')
mo.vstack([_cards,_plot,_table,_download])
