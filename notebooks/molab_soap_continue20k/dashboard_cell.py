training_refresh.value
import matplotlib.pyplot as _plt
from datetime import datetime as _dt
from zoneinfo import ZoneInfo as _ZI
_r=Path('/home/marimo/wall_soap_continue20k_v1');_prev=Path('/home/marimo/wall_soap_trials_v1');_old=Path('/home/marimo/wall_rad_ab_15k_v1');_middle=Path('/home/marimo/wall_soap_continue10k_v1')
def _read(_p):
 try:return json.loads(_p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def _rows(_p):
 _out=[]
 for _l in (_p.read_text().splitlines() if _p.exists() else []):
  try:_out.append(json.loads(_l))
  except json.JSONDecodeError:pass
 return _out
_s=_read(_r/'status.json');_pre=_read(_r/'preflight.json');_fail=_read(_r/'failure.json');_off=_s.get('offset',0);_step=_s.get('step',302400)
_phase='خطا؛ بررسی لازم است' if _fail else {'completed':'پایان ۲۰هزار گام','stopped':'توقف طبق معیار ارزیابی','running':'آموزش در حال اجرا'}.get(_s.get('phase'),'بررسی بازیابی دقیق SOAP')
_now=_dt.now(_ZI('Asia/Tehran')).strftime('%H:%M:%S')
_header=mo.Html(f'<section dir="rtl" style="font-family:Tahoma;background:#eef2ff;padding:20px;border-radius:14px"><h2>ادامهٔ ۲۰هزارگامی SOAP</h2><h3>{_phase} · گام {_step:,}</h3><p>آخرین تازه‌سازی: {_now} تهران · نرخ 2e-5</p><progress value="{_off}" max="20000" style="width:100%;height:22px"></progress><p>{_off:,} از ۲۰٬۰۰۰ گام جدید — مجموع SOAP: {_off+15000:,} از ۳۵٬۰۰۰</p><p>ادامه از وزن ۲۹۲٬۴۰۰ با حفظ کامل وضعیت SOAP و نمونه‌برداری جدید در هر گام؛ هدف ۳۰۲٬۴۰۰.</p><p>ارزیابی ۲۸ نمونهٔ FEM هر ۲۵۰۰ گام؛ توقف در صورت پسرفت پایدار. بازیابی دقیق: {"موفق" if _pre.get("passed") else "در حال بررسی"}</p></section>')
_fig,_axs=_plt.subplots(2,1,figsize=(10,5),layout='constrained')
for _name,_t,_v in [('Adam15k (shorter reference)',_rows(_old/'baseline/training.jsonl'),_rows(_old/'baseline/probe.jsonl')),('SOAP',_rows(_prev/'soap_2e5/training.jsonl')+_rows(_middle/'continuation/training.jsonl')+_rows(_r/'continuation/training.jsonl'),_rows(_prev/'soap_2e5/probe.jsonl')+_rows(_middle/'continuation/probe.jsonl')+_rows(_r/'continuation/probe.jsonl'))]:
 if _t:
  _y=[x['loss'] for x in _t];_axs[0].plot([x['step']-287400 for x in _t],[sum(_y[max(0,i-99):i+1])/min(i+1,100) for i in range(len(_y))],label=_name)
 if _v:_axs[1].plot([x['step']-287400 for x in _v],[x['audit'] for x in _v],'o-',label=_name+' audit')
for _a in _axs:
 _a.axvline(15000,color='gray',ls=':',alpha=.6);_a.set(yscale='log',xlabel='Total parameter updates since base287400',ylabel='Loss');_a.grid(alpha=.2)
 if _a.lines:_a.legend(fontsize=8)
_axs0=_axs[0];_axs0.set_title('Combined SOAP history: 15k completed + 20k continuation')
_axs[1].set_title('Identical fixed audit; compare physical FEM errors separately')
_plot=mo.as_html(_fig);_plt.close(_fig)
_mark=_r/'regression_guard.json';_guard=_read(_mark)
_guard_text=mo.md(f'آخرین بررسی FEM: گام **{_guard.get("step",302400):,}** · تعداد بررسی‌های پیاپی با پسرفت بیش از آستانه: **{_guard.get("bad_streak",0)}**')
_zip=_r/'results.zip';_dl=mo.download(data=lambda:_zip.read_bytes(),filename='soap_continue20k_results.zip',label='دانلود خروجی کامل') if _zip.exists() else mo.md('وزن‌های دوره‌ای به گیت‌هاب منتقل می‌شوند؛ خروجی نهایی پس از پایان آماده می‌شود.')
mo.vstack([training_refresh,_header,_guard_text,_plot,_dl])
