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
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo as _ZI
    _r=Path('/home/marimo/wall_soap_cosine100k_v1')
    def _read(_p):
     try:return json.loads(_p.read_text())
     except (FileNotFoundError,json.JSONDecodeError):return {}
    def _rows(_p):
     _out=[]
     for _l in (_p.read_text().splitlines() if _p.exists() else []):
      try:_out.append(json.loads(_l))
      except json.JSONDecodeError:pass
     return _out
    _s=_read(_r/'status.json');_fail=_read(_r/'failure.json');_pre=_read(_r/'preflight.json');_logs=_rows(_r/'continuation/training.jsonl');_probes=_rows(_r/'continuation/probe.jsonl')
    _phase='خطا؛ بررسی لازم است' if _fail else {'running':'آموزش در حال اجرا','preflight':'بررسی بازیابی کامل','completed':'پایان آموزش','stopped':'متوقف شده'}.get(_s.get('phase'),'در انتظار آماده‌سازی')
    _step=_s.get('step',322400);_offset=_s.get('offset',0);_lr=_s.get('learning_rate',1e-4);_now=_dt.now(_ZI('Asia/Tehran')).strftime('%H:%M:%S')
    _header=mo.Html(f'<section dir="rtl" style="font-family:Tahoma;background:#eef2ff;padding:20px;border-radius:14px"><h2>ادامهٔ ۱۰۰هزارگامی SOAP با کاهش پیوستهٔ نرخ</h2><h3>{_phase} · گام {_step:,}</h3><p>آخرین تازه‌سازی: {_now} تهران · نرخ فعلی {_lr:.8g}</p><progress value="{_offset}" max="100000" style="width:100%;height:22px"></progress><p>{_offset:,} از ۱۰۰٬۰۰۰ گام جدید · هدف ۴۲۲٬۴۰۰</p><p>شروع از وزن کامل ۳۲۲٬۴۰۰ با حفظ وضعیت SOAP؛ batch=32 و هندسه‌ها و نقاط تازه در هر گام.</p><p>نرخ کسینوسی پیوسته از 1e-4 تا 5e-6؛ بازیابی دقیق: {"موفق" if _pre.get("passed") else "در حال بررسی"}</p></section>')
    _fig,_axs=_plt.subplots(3,1,figsize=(10,8),layout='constrained')
    if _logs:
     _y=[x['loss'] for x in _logs];_x=[x['step']-322400 for x in _logs]
     _axs[0].plot(_x,_y,alpha=.25,lw=.5,label='Logged loss')
     _axs[0].plot(_x,[sum(_y[max(0,i-99):i+1])/min(i+1,100) for i in range(len(_y))],label='100-record mean')
     _axs[2].plot(_x,[x['learning_rate'] for x in _logs],label='Actual applied LR')
    if _probes:
     for _key in ['control','audit']:_axs[1].plot([x['offset'] for x in _probes],[x[_key] for x in _probes],'o-',label=_key)
    for _a in _axs:
     _a.grid(alpha=.2);_a.set(xlabel='New parameter updates since322400')
     if _a.lines:_a.legend(fontsize=8)
    _axs[0].set(yscale='log',ylabel='Physics loss',title='SOAP cosine continuation')
    _axs[1].set(yscale='log',ylabel='Same fixed probe loss')
    _axs[2].set(ylabel='Learning rate')
    _plot=mo.as_html(_fig);_plt.close(_fig)
    _guard=_read(_r/'regression_guard.json');_note=mo.md(f'آخرین ارزیابی ۲۸ نمونهٔ FEM: گام **{_guard.get("step",322400):,}**. وزن کامل هر ۵۰۰۰ گام ذخیره می‌شود. کاهش loss به‌تنهایی به معنی کاهش خطای همهٔ میدان‌ها نیست.')
    _error=mo.md(str(_fail)) if _fail else mo.md('')
    mo.vstack([training_refresh,_header,_error,_plot,_note])
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
