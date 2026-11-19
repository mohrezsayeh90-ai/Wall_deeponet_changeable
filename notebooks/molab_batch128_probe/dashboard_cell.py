training_refresh.value
import matplotlib.pyplot as _plt
from datetime import datetime as _dt
from zoneinfo import ZoneInfo as _ZI
_r=Path('/home/marimo/wall_batch128_probe_v1');_old=Path('/home/marimo/wall_soap_continue20k_v1')
def _read(_p):
 try:return json.loads(_p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def _rows(_p):
 _out=[]
 for _l in (_p.read_text().splitlines() if _p.exists() else []):
  try:_out.append(json.loads(_l))
  except json.JSONDecodeError:pass
 return _out
_s=_read(_r/'status.json');_pre=_read(_r/'preflight.json');_failure=_read(_r/'failure.json');_comparison=_read(_r/'comparison.json');_count=_s.get('optimizer_updates',0)
_phase='خطا؛ نیاز به بررسی' if _failure else ('پایان مقایسهٔ کوتاه' if _comparison else ('آزمایش بچ ۱۲۸ در حال اجرا' if _s else 'بررسی یکسان‌بودن گرادیان‌ها'))
_now=_dt.now(_ZI('Asia/Tehran')).strftime('%H:%M:%S')
_header=mo.Html(f'<section dir="rtl" style="font-family:Tahoma;background:#fff7ed;padding:20px;border-radius:14px"><h2>بررسی توقف کاهش لاس: بچ ۳۲ در برابر ۱۲۸</h2><h3>{_phase}</h3><p>تازه‌سازی: {_now} تهران</p><progress value="{_count}" max="500" style="width:100%;height:22px"></progress><p>{_count:,} از ۵۰۰ به‌روزرسانی با بچ ۱۲۸؛ معادل {_count*4:,} از ۲۰۰۰ بچ ۳۲ از نظر تعداد هندسه‌های دیده‌شده.</p><p>ران قبلی موقتاً با حفظ وضعیت متوقف است. مرجع بچ ۳۲ قبلاً اجرا شده؛ وزن شروع هر دو گام ۳۱۳٬۴۰۰ است.</p><p>فقط اندازهٔ بچ تغییر کرده؛ نرخ 2e-5، معادلات و بازنمونه‌برداری همان است. موفقیت با ارزیابی ثابت و FEM سنجیده می‌شود.</p></section>')
_fig,_axs=_plt.subplots(2,1,figsize=(10,5),layout='constrained')
_ref=[x for x in _rows(_old/'continuation/probe.jsonl') if 313400<=x['step']<=315400];_new=_rows(_r/'batch128/probe.jsonl');_base=_read(_r/'base_metrics.json')
if _ref:_axs[0].plot([x['step']-313400 for x in _ref],[x['audit'] for x in _ref],'o-',label='Batch32: cached2000 updates')
if _base:
 _xs=[0]+[x['equivalent_batch32_updates'] for x in _new];_ys=[_base['probe']['audit']]+[x['audit'] for x in _new];_axs[0].plot(_xs,_ys,'o-',label='Batch128:500 optimizer updates')
for _name,_t,_mult in [('Batch32', [x for x in _rows(_old/'continuation/training.jsonl') if 313400<x['step']<=315400],1),('Batch128',_rows(_r/'batch128/training.jsonl'),4)]:
 if _t:
  _y=[x['loss'] for x in _t];_w=100 if _mult==1 else 50
  _xx=[x['step']-313400 if _mult==1 else x['equivalent_batch32_updates'] for x in _t];_axs[1].plot(_xx,[sum(_y[max(0,i-_w+1):i+1])/min(i+1,_w) for i in range(len(_y))],label=_name)
for _a in _axs:
 _a.set(xlabel='Equivalent batch32 draws since shared checkpoint',ylabel='Loss');_a.grid(alpha=.2)
 if _a.lines:_a.legend(fontsize=8)
_axs[0].set_title('Fixed audit: actual improvement, not only smoother training loss');_axs[1].set_title('Same physics objective; matched geometry-draw budget')
_plot=mo.as_html(_fig);_plt.close(_fig)
_result=mo.md('مقایسهٔ نهایی FEM و ذخیرهٔ وزن‌ها پس از پایان آماده می‌شود.')
if _comparison:
 _bs=_comparison['batch32_reference']['fem']['saved20'];_bg=_comparison['batch128']['fem']['saved20'];_result=mo.md(f'| روش | میانهٔ سرعت۲۰کیس | میانهٔ فشار۲۰کیس |\n|---|---:|---:|\n| بچ۳۲ | {_bs["velocity_l2_median"]:.3%} | {_bs["pressure_l2_median"]:.3%} |\n| بچ۱۲۸ | {_bg["velocity_l2_median"]:.3%} | {_bg["pressure_l2_median"]:.3%} |')
mo.vstack([training_refresh,_header,_plot,_result])
