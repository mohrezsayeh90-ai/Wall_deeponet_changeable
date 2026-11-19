training_refresh.value
import matplotlib.pyplot as _plt
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
_fig,_axs=_plt.subplots(2,1,figsize=(11,8),layout='constrained')
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
_cards=mo.Html(f'<section dir="rtl" style="font-family:Tahoma;background:#f0fdfa;padding:22px;border-radius:15px"><h2>آزمایش دو روش ادامهٔ آموزش</h2><h3>{_phase}</h3><p>شروع هر شاخه: ۲۸۷٬۴۰۰ · نرخ ثابت: 2e-5 · هدف هر شاخه: ۳۰۲٬۴۰۰</p><progress value="{_done}" max="30000" style="width:100%;height:24px"></progress><p>{_done:,} از ۳۰٬۰۰۰ گام مجموع دو آزمایش</p><p>عادی ابتدا اجرا می‌شود؛ تطبیقی سپس از همان وزن اولیه. ارزیابی با همان ۲۰ هندسه و ۸ نمونهٔ FEM انجام می‌شود.</p><p>وزن‌ها هر ۱۰۰ گام ذخیره می‌شوند؛ نسخه‌های دوره‌ای با یک فرایند مستقل به گیت‌هاب منتقل می‌شوند.</p></section>')
_plot=mo.as_html(_fig);_plt.close(_fig)
_bundle=_r/'results.zip';_dl=mo.download(data=_bundle.read_bytes(),filename='rad_ab_15k_results.zip',label='دانلود نتایج دو آزمایش') if _bundle.exists() else mo.md('خروجی نهایی پس از پایان دو شاخه قابل دانلود خواهد بود.')
mo.vstack([_cards,mo.md('| روش | گام جدید |\n|---|---:|\n'+'\n'.join(_table)),_plot,_dl])
