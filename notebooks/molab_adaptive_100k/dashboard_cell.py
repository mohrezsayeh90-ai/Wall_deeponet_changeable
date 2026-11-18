training_refresh.value
import matplotlib.pyplot as _plt
_root = Path('/home/marimo/wall_adaptive_100k')
def _json_read(_p,_default):
    try:return json.loads(_p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return _default
def _records(_p):
    _r=[]
    for _line in (_p.read_text().splitlines() if _p.exists() else []):
        try:_r.append(json.loads(_line))
        except json.JSONDecodeError:pass
    return _r
_state = _json_read(_root/'status.json',{})
_data = _records(_root/'run/training.jsonl')
_validation = _records(_root/'run/validation.jsonl')
_decisions = _records(_root/'run/controller.jsonl')
_last = _data[-1] if _data else {}
_step = _state.get('accepted_step',287400)
_lr = _state.get('current_lr',2e-5)
_phase = _state.get('phase','starting')
_label = {'running':'آموزش','comparing_rates':'مقایسهٔ نرخ‌های یادگیری؛ پیشرفت اصلی موقتاً ثابت می‌ماند','completed':'پایان آموزش','failed':'خطا؛ نیاز به بررسی','starting':'آماده‌سازی'}.get(_phase,_phase)
_loss = f"{_last['loss']:.6f}" if _last else 'در حال آماده‌سازی'
_cards=mo.Html(f'''<section dir="rtl" style="font-family:Tahoma,Arial;background:#f0fdfa;border:1px solid #99f6e4;padding:22px;border-radius:16px;color:#134e4a">
<h1>ادامهٔ ۱۰۰هزارگامی با نرخ یادگیری تطبیقی</h1><h3>{_label}</h3>
<div style="display:flex;gap:24px;font-size:21px;flex-wrap:wrap"><div>گام: <b>{_step:,}</b></div><div>نرخ فعلی: <b>{_lr:.2e}</b></div><div>Loss: <b>{_loss}</b></div></div>
<progress value="{_step-287400}" max="100000" style="width:100%;height:24px;margin-top:18px"></progress>
<div>{_step-287400:,} از ۱۰۰٬۰۰۰ گام پذیرفته‌شده — هدف: ۳۸۷٬۴۰۰</div>
<div>هر ۱۰هزار گام: آزمایش ۵۰۰گامی نرخ فعلی، نصف و دوبرابر آن، از وضعیت Adam و RNG مشترک.</div>
<div>مجموعهٔ کنترل و مجموعهٔ پایش مستقل، هرکدام ۳۲ هندسهٔ متفاوت دارند. آزمایش‌های ردشده جزو گام‌های اصلی نیستند.</div>
<div style="font-size:13px;margin-top:10px">زمان سپری‌شده: {_state.get('elapsed_seconds',0)/60:.1f} دقیقه · محاسبات انجام‌شده: {_state.get('physical_updates',0):,} به‌روزرسانی</div></section>''')
_fig,_axes = _plt.subplots(3,1,figsize=(11,10),layout='constrained')
if _data:
    _x=[r['step'] for r in _data];_y=[r['loss'] for r in _data]
    _axes[0].plot(_x,_y,alpha=.2,lw=.5,color='#0891b2')
    _axes[0].plot(_x,[sum(_y[max(0,i-99):i+1])/min(i+1,100) for i in range(len(_y))],color='#7c3aed',label='Mean over up to 100 logged records')
    _axes[2].step(_x,[r['lr'] for r in _data],where='post',color='#0f766e',label='Accepted learning rate')
if _validation:
    _axes[1].plot([r['step'] for r in _validation],[r['control_loss'] for r in _validation],'o-',label='Control: selects LR',color='#2563eb')
    _axes[1].plot([r['step'] for r in _validation],[r['audit_loss'] for r in _validation],'o-',label='Disjoint audit: not used by LR controller',color='#dc2626')
_axes[0].set(title='Accepted training path',ylabel='Training loss (log)')
_axes[1].set(title='Fixed development objectives on disjoint geometry cohorts',ylabel='Probe loss (log)')
_axes[2].set(title='Adaptive learning-rate history',ylabel='Learning rate (log)')
for _ax in _axes:
    _ax.set(yscale='log',xlabel='Global accepted Adam step');_ax.grid(alpha=.2)
    if _ax.lines:_ax.legend(fontsize=8)
_plot=mo.as_html(_fig);_plt.close(_fig)
_rows=['| گام تصمیم | نرخ قبلی | نرخ انتخاب‌شده |','|---|---:|---:|']
for _d in _decisions:_rows.append(f"| {_d['anchor_step']:,} | {_d['previous_lr']:.2e} | {_d['chosen_lr']:.2e} |")
_table=mo.md('\n'.join(_rows)) if _decisions else mo.md('نخستین تصمیم کنترلر پس از ۱۰هزار گام ثبت می‌شود؛ نرخ شروع همان `2e-5` است.')
_bundle=_root/'adaptive_results.zip'
_download=mo.download(data=_bundle.read_bytes(),filename='adaptive_results.zip',label='دانلود وزن‌ها و خروجی‌های نهایی') if _bundle.exists() else mo.md('وزن آخر، بهترین وزنِ کنترل و بهترین وزنِ پایش مستقل، جداگانه ذخیره می‌شوند.')
mo.vstack([_cards,_plot,_table,_download])
