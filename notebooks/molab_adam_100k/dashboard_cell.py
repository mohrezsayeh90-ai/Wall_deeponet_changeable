training_refresh.value
import matplotlib.pyplot as _plt
_root = Path('/home/marimo/wall_adam_100k')
_manifest = json.loads((_root/'manifest.json').read_text())
_out = _root/'cases/cosine_100k'
_state_file = _root/'supervisor-status.json'
_state = json.loads(_state_file.read_text()) if _state_file.exists() else {'phase':'starting'}
def _read_records(_p):
    _rr=[]
    for _l in (_p.read_text().splitlines() if _p.exists() else []):
        try:_rr.append(json.loads(_l))
        except json.JSONDecodeError:pass
    return _rr
_records = _read_records(_out/'training.jsonl')
_last = _records[-1] if _records else {}
_base = _manifest['base_step']
_step = _last.get('step',_base)
_offset = _step-_base
_phase = _state.get('phase','starting')
_best_path = _out/'best.json'
_best = json.loads(_best_path.read_text()) if _best_path.exists() else {}
_loss = f"{_last['loss']:.6f}" if _last else 'در حال آماده‌سازی'
_cards = mo.Html(f'''<section dir="rtl" style="font-family:Tahoma,Arial;background:#eff6ff;border:1px solid #93c5fd;border-radius:16px;padding:22px;color:#1e3a8a">
<h1>ادامهٔ ۱۰۰هزارگامی — کنترلر کسینوسی</h1><h3>وضعیت: {_phase}</h3>
<div style="display:flex;gap:28px;font-size:22px;flex-wrap:wrap"><div>گام فعلی: <b>{_step:,}</b></div><div>گام جدید: <b>{_offset:,} / ۱۰۰٬۰۰۰</b></div><div>Loss: <b>{_loss}</b></div></div>
<progress max="100000" value="{_offset}" style="width:100%;height:24px;margin-top:20px"></progress>
<div>شروع از وزن کسینوسی گام {_base:,} — هدف ۲۸۷٬۴۰۰ — نرخ یادگیری: 1e-4 → 2e-5</div>
<div>بهترین ارزیابی ثابت: {_best.get('probe_loss','در حال ارزیابی')} در گام {_best.get('step',_base):,}</div>
<div style="margin-top:10px">وزن آخر، بهترین وزنِ ارزیابی ثابت و نسخه‌های دوره‌ای جدا ذخیره می‌شوند. تعمیم برنامه به ۱۰۰هزار گام یک آزمایش جدید است.</div></section>''')
_fig,_axes = _plt.subplots(2,1,figsize=(11,8),layout='constrained')
_old = _read_records(Path('/home/marimo/wall_adam_1h/run/training.jsonl'))
_old = [r for r in _old if r['step']<=172400]
_series=[('Previous molab',_old,'#94a3b8')]
_trials = Path('/home/marimo/wall_lr_trials')
_tm = json.loads((_trials/'manifest.json').read_text())
for _c,_color in zip(_tm['cases'],['#0f766e','#2563eb','#dc2626','#9333ea']):
    _series.append((_c['label'],_read_records(_trials/'cases'/_c['id']/'training.jsonl'),_color))
_series.append(('Selected cosine +100k',_records,'#111827'))
for _label,_rr,_color in _series:
    if _rr:
        _y=[r['loss'] for r in _rr];_x=[r['step'] for r in _rr]
        _axes[0].plot(_x,[sum(_y[max(0,i-99):i+1])/min(i+1,100) for i in range(len(_y))],color=_color,label=_label,lw=2 if _label.startswith('Selected') else 1)
_probe = _read_records(_out/'probe.jsonl')
if _probe:_axes[1].plot([r['step'] for r in _probe],[r['loss'] for r in _probe],'o-',color='#111827',label='Fixed dev probe')
_axes[0].axvline(_base,ls=':',color='black');_axes[0].set(title='Loss history and selected continuation (rolling mean)',xlabel='Global Adam step',ylabel='Training loss',yscale='log')
_axes[1].set(title='Fixed evaluation every 5,000 updates; best checkpoint retained',xlabel='Global Adam step',ylabel='Fixed probe loss',yscale='log')
for _ax in _axes:
    _ax.grid(alpha=.2)
    if _ax.lines:_ax.legend(fontsize=8)
_plot=mo.as_html(_fig);_plt.close(_fig)
_bundle=_root/'adam_100k_results.zip'
_download=mo.download(data=_bundle.read_bytes(),filename='adam_100k_results.zip',label='دانلود نتایج نهایی ران ۱۰۰هزارگامی') if _bundle.exists() else mo.md('دانلود وزن نهایی و بهترین وزن پس از پایان فعال می‌شود.')
mo.vstack([_cards,_plot,_download])
