training_refresh.value
import matplotlib.pyplot as _plt
from datetime import datetime as _dt
from zoneinfo import ZoneInfo as _ZI
_r=Path('/home/marimo/wall_soap_trials_v1');_old=Path('/home/marimo/wall_rad_ab_15k_v1')
def _read(_p):
 try:return json.loads(_p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def _rows(_p):
 _out=[]
 for _l in (_p.read_text().splitlines() if _p.exists() else []):
  try:_out.append(json.loads(_l))
  except json.JSONDecodeError:pass
 return _out
_state=_read(_r/'status.json');_fail=_read(_r/'failure.json');_pre=_read(_r/'preflight.json');_total=0;_lines=[]
_fig,_ax=_plt.subplots(2,1,figsize=(10,5),layout='constrained')
for _name,_folder,_label in [('adam',_old/'baseline','Adam reference'),('soap_2e5',_r/'soap_2e5','SOAP 2e-5'),('soap_1e4',_r/'soap_1e4','SOAP 1e-4')]:
 _t=[x for x in _rows(_folder/'training.jsonl') if x['step']<=292400];_v=[x for x in _rows(_folder/'probe.jsonl') if x['step']<=292400]
 if _name!='adam':
  _n=_t[-1]['step']-287400 if _t else 0;_total+=_n;_lines.append(f'| {_label} | {_n:,} / ۵٬۰۰۰ |')
 if _t:
  _y=[x['loss'] for x in _t];_ax[0].plot([x['step']-287400 for x in _t],[sum(_y[max(0,i-99):i+1])/min(i+1,100) for i in range(len(_y))],label=_label)
 if _v:_ax[1].plot([x['offset'] for x in _v],[x['audit'] for x in _v],'o-',label=_label+' audit')
for _a in _ax:
 _a.set(yscale='log',xlabel='Parameter updates from identical base',ylabel='Loss');_a.grid(alpha=.2)
 if _a.lines:_a.legend(fontsize=8)
_ax[0].set_title('Same baseline sampling: mean of up to 100 logged records');_ax[1].set_title('Identical fixed audit cohort; compare FEM and elapsed time too')
_phase='خطا؛ بررسی لازم است' if _fail else ('پایان آزمایش‌ها' if _state.get('phase')=='completed' else ('آموزش '+_state.get('branch','') if _state else 'بررسی گرادیان، پایداری و بازیابی وضعیت SOAP'))
_now=_dt.now(_ZI('Asia/Tehran')).strftime('%H:%M:%S')
_header=mo.Html(f'<section dir="rtl" style="font-family:Tahoma;background:#eef2ff;padding:20px;border-radius:14px"><h2>ادامهٔ آموزش با SOAP</h2><h3>{_phase}</h3><p>تازه‌سازی: {_now} تهران · شروع: ۲۸۷٬۴۰۰ · هدف هر شاخه: ۲۹۲٬۴۰۰</p><progress value="{_total}" max="10000" style="width:100%;height:20px"></progress><p>{_total:,} از ۱۰٬۰۰۰ گام دو شاخه · بررسی اولیه: {"موفق" if _pre.get("passed") else "در حال اجرا"}</p><p>دو نرخ 2e-5 و 1e-4؛ مقایسه با ۵هزار گام Adam ذخیره‌شده. هر شاخه وضعیت SOAP مستقل دارد.</p></section>')
_plot=mo.as_html(_fig);_plt.close(_fig)
_zip=_r/'results.zip';_download=mo.download(data=lambda: _zip.read_bytes(),filename='soap_trials_results.zip',label='دانلود خروجی کامل SOAP') if _zip.exists() else mo.md('خروجی نهایی پس از پایان دو شاخه آماده می‌شود؛ وزن‌های دوره‌ای جداگانه ذخیره می‌شوند.')
mo.vstack([training_refresh,_header,mo.md('| شاخه | پیشرفت |\n|---|---:|\n'+'\n'.join(_lines)),_plot,_download])
