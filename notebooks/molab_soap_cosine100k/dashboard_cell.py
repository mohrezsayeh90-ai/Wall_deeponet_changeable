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
