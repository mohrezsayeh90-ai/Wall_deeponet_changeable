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
_comparison_note=mo.md('**آبی: ران تاریخی Adam با لاگ تا گام ۳۷۸٬۶۰۰؛ نارنجی: SOAP فعلی. بازهٔ مشترک از گام ۲۸۷٬۴۰۰ نمایش داده می‌شود. نرخ Adam در آن ران تطبیقی بود، ولی نرخ SOAP ثابت 2e-5 است؛ این مقایسهٔ دو روش اجراشده است.**')
_phase='خطا؛ بررسی لازم است' if _fail else {'completed':'پایان ۲۰هزار گام','stopped':'توقف طبق معیار ارزیابی','running':'آموزش در حال اجرا'}.get(_s.get('phase'),'بررسی بازیابی دقیق SOAP')
_now=_dt.now(_ZI('Asia/Tehran')).strftime('%H:%M:%S')
_header=mo.Html(f'<section dir="rtl" style="font-family:Tahoma;background:#eef2ff;padding:20px;border-radius:14px"><h2>ادامهٔ ۲۰هزارگامی SOAP</h2><h3>{_phase} · گام {_step:,}</h3><p>آخرین تازه‌سازی: {_now} تهران · نرخ 2e-5</p><progress value="{_off}" max="20000" style="width:100%;height:22px"></progress><p>{_off:,} از ۲۰٬۰۰۰ گام جدید — مجموع SOAP: {_off+15000:,} از ۳۵٬۰۰۰</p><p>ادامه از وزن ۳۰۲٬۴۰۰ با حفظ کامل وضعیت SOAP و نمونه‌برداری جدید در هر گام؛ هدف ۳۲۲٬۴۰۰.</p><p>ارزیابی ۲۸ نمونهٔ FEM هر ۲۵۰۰ گام؛ توقف در صورت پسرفت پایدار. بازیابی دقیق: {"موفق" if _pre.get("passed") else "در حال بررسی"}</p></section>')
_soap_t=_rows(_prev/'soap_2e5/training.jsonl')+_rows(_middle/'continuation/training.jsonl')+_rows(_r/'continuation/training.jsonl')
_soap_v=_rows(_prev/'soap_2e5/probe.jsonl')+_rows(_middle/'continuation/probe.jsonl')+_rows(_r/'continuation/probe.jsonl')
_end=max([x['step'] for x in _soap_t],default=302400)
_ht=[dict(step=x[0],loss=x[1],learning_rate=x[2]) for x in _read(_r/'historical_adam_training.json').get('rows',[]) if x[0]<=_end]
_hv=[dict(step=x['step'],audit=x['audit_loss'],control=x['control_loss']) for x in _read(_r/'historical_adam_probe.json').get('validation',[]) if x['step']<=_end]
_fig,_axs=_plt.subplots(3,1,figsize=(10,7),layout='constrained')
for _name,_t,_v,_color in [('Historical Adam (adaptive LR)',_ht,_hv,'#2563eb'),('SOAP (fixed LR)',_soap_t,_soap_v,'#f97316')]:
 if _t:
  _y=[x['loss'] for x in _t];_axs[0].plot([x['step']-287400 for x in _t],[sum(_y[max(0,i-99):i+1])/min(i+1,100) for i in range(len(_y))],label=_name,color=_color)
  _axs[2].step([x['step']-287400 for x in _t],[x.get('learning_rate',2e-5) for x in _t],where='post',label=_name,color=_color)
 if _v:_axs[1].plot([x['step']-287400 for x in _v],[x['audit'] for x in _v],'o-',label=_name+' audit',color=_color)
for _a in _axs:
 _a.set(yscale='log',xlabel='Updates since common checkpoint287400');_a.grid(alpha=.2)
 if _a.lines:_a.legend(fontsize=8)
_axs[0].set(title='Actual historical Adam vs SOAP: same starting checkpoint',ylabel='Training loss mean')
_axs[1].set(title='Same fixed audit cohort; markers are actual recorded evaluations',ylabel='Audit loss')
_axs[2].set(title='Learning-rate policies differ',ylabel='Learning rate')
_plot=mo.as_html(_fig);_plt.close(_fig)
_mark=_r/'regression_guard.json';_guard=_read(_mark)
_guard_text=mo.md(f'آخرین بررسی FEM: گام **{_guard.get("step",302400):,}** · تعداد بررسی‌های پیاپی با پسرفت بیش از آستانه: **{_guard.get("bad_streak",0)}**')
_zip=_r/'results.zip';_dl=mo.download(data=lambda:_zip.read_bytes(),filename='soap_continue20k_results.zip',label='دانلود خروجی کامل') if _zip.exists() else mo.md('وزن‌های دوره‌ای به گیت‌هاب منتقل می‌شوند؛ خروجی نهایی پس از پایان آماده می‌شود.')
_diagroot=Path('/home/marimo/wall_branch_only_probe_v1')
_ds=_read(_diagroot/'status.json');_du=_ds.get('optimizer_updates',0)
_drows=_rows(_diagroot/'branch_only/probe.jsonl');_da=_drows[-1].get('audit') if _drows else None
_dtext=mo.md('**بررسی توقف کاهش خطا کامل شد:** ادامهٔ اصلی تا گام ۳۲۲٬۴۰۰ پایان یافت. loss ارزیابی ثابت از ۰٫۰۱۲۸۱۵ به ۰٫۰۱۲۴۸۴ رسید؛ فقط ۲٫۵۸٪ کاهش در ۲۰هزار گام. میانهٔ سرعت ۲۰ کیس: ۱٫۳۶۱٪ → ۱٫۳۴۰٪؛ فشار: ۱٫۱۳۶٪ → ۱٫۱۷۱٪. دو آزمون batch128 و آزمون branch-only کاهش بزرگ و یکدستی ندادند و جایگزین مدل نشدند. همهٔ حالت‌ها و وزن‌ها جداگانه نگه داشته شدند. **آموزش جدیدی بر پایهٔ این نتایج شروع نشده است.**')
mo.vstack([training_refresh,_header,_dtext,_guard_text,_comparison_note,_plot,_dl])
