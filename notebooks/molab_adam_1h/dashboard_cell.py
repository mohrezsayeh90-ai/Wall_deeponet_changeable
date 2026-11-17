training_refresh.value
import matplotlib.pyplot as _plt
_root = Path('/home/marimo/wall_adam_1h')
try:
    _supervisor = json.loads((_root/'supervisor-status.json').read_text())
    _sp = _root/'run/status.json'
    _status = json.loads(_sp.read_text()) if _sp.exists() else {}
    _lp = _root/'run/training.jsonl'
    _records = []
    for _line in (_lp.read_text().splitlines() if _lp.exists() else []):
        try:
            _records.append(json.loads(_line))
        except json.JSONDecodeError:
            pass
    _last = _records[-1] if _records else {}
    _step = _last.get('step', _status.get('step', 132216))
    _elapsed = _last.get('elapsed_seconds', 0)
    _phase = _supervisor.get('phase', 'starting')
    _percent = 100 if _phase == 'completed' else min(100, 100*_elapsed/3600)
    _label = {'running':'در حال آموزش','completed':'آموزش تمام شد','failed':'آموزش متوقف شده؛ نیاز به بررسی','starting':'در حال آماده‌سازی'}.get(_phase, _phase)
    _loss_text = f"{_last['loss']:.6f}" if _records else 'در حال کامپایل'
    _cards = mo.Html(f'''<section dir="rtl" style="font-family:Tahoma,Arial,sans-serif;background:#f0fdfa;border:1px solid #99f6e4;border-radius:16px;padding:24px;margin:12px 0;color:#134e4a">
<h1 style="margin:0 0 12px">داشبورد زندهٔ آموزش</h1>
<div style="font-size:20px;font-weight:bold;margin-bottom:16px">● {_label}</div>
<div style="display:grid;grid-template-columns:repeat(3,minmax(100px,1fr));gap:12px">
<div style="background:white;padding:16px;border-radius:10px">گام فعلی<br><strong style="font-size:28px">{_step:,}</strong></div>
<div style="background:white;padding:16px;border-radius:10px">گام‌های جدید<br><strong style="font-size:28px">{_step-132216:,}</strong></div>
<div style="background:white;padding:16px;border-radius:10px">آخرین Loss<br><strong style="font-size:28px">{_loss_text}</strong></div>
</div>
<div style="margin-top:20px;font-size:17px">پیشرفت زمانی: {_percent:.1f}٪ — {_elapsed/60:.1f} دقیقه از ۶۰ دقیقه</div>
<progress max="100" value="{_percent}" style="width:100%;height:24px;accent-color:#0f766e;margin:10px 0"></progress>
<div>زمان باقی‌مانده: {max(0,3600-_elapsed)/60:.1f} دقیقه · آخرین گام ذخیره‌شده: {_status.get('step',132216):,}</div>
<div style="margin-top:12px;font-size:13px">RTX PRO 6000 Blackwell · Adam · شروع از گام ۱۳۲٬۲۱۶ · تازه‌سازی خودکار</div>
</section>''')
    _views = [_cards]
    if _records:
        _fig, _ax = _plt.subplots(figsize=(11,3.8), layout='constrained')
        _xs = [r['step'] for r in _records]
        _ys = [r['loss'] for r in _records]
        _ax.plot(_xs,_ys,color='#0891b2',alpha=.45,lw=.8,label='Logged loss')
        if len(_records)>=20:
            _means=[sum(_ys[max(0,i-99):i+1])/min(i+1,100) for i in range(len(_ys))]
            _ax.plot(_xs,_means,color='#7c3aed',lw=2,label='Mean over up to 100 logged records')
        _ax.set(xlabel='Global Adam step',ylabel='Loss (log scale)',title='Live training loss',yscale='log')
        _ax.grid(alpha=.2);_ax.legend();_views.append(mo.as_html(_fig));_plt.close(_fig)
    _views.append(mo.md('وضعیت و نمودار به‌صورت خودکار تازه می‌شوند. پس از اتمام، دکمهٔ دانلود نتایج و وزن‌ها در پایین فعال می‌شود.'))
    _dashboard_view = mo.vstack(_views)
except (FileNotFoundError, json.JSONDecodeError):
    _dashboard_view = mo.md('در حال خواندن وضعیت آموزش؛ چند ثانیه دیگر دوباره تازه می‌شود.')
_dashboard_view
