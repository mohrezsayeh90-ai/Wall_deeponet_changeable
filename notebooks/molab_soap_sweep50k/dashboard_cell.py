training_refresh.value
import matplotlib.pyplot as _plt
_r=Path('/home/marimo/wall_soap_sweep50k_v1')
def _read(_p):
 try:return json.loads(_p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def _rows(_p):
 return [json.loads(x) for x in _p.read_text().splitlines()] if _p.exists() else []
_names=['constant_1e4','constant_2e5','constant_5e5','constant_2e4','cosine_1e4_2e5','constant_1e4_extend25k']
_s=_read(_r/'status.json');_active=_s.get('active_case','constant_1e4');_a=_read(_r/_active/'status.json');_fail=_read(_r/'failure.json');_budget=_s.get('completed_updates',0)+(_a.get('offset',0) if _s.get('phase')=='running' else 0)
_budget=max(_budget,_s.get('consumed_updates_lower_bound',0))
_header=mo.md(f'''## آزمون ۵۰هزارگامی نرخ SOAP
**وضعیت:** {_s.get('phase','آماده‌سازی')} · **حالت فعال:** {_active} · **گام حالت:** {_a.get('step',342400):,}

**بودجه مصرف‌شده:** {_budget:,} از ۵۰٬۰۰۰ · **نرخ فعلی:** {_a.get('learning_rate',1e-4):.7g}

پنج مقایسهٔ ۵هزارگامی از وزن یکسان ۳۴۲٬۴۰۰؛ سپس ۲۵هزار گام ادامهٔ حالت ثابت 1e-4. مجموع آن حالت ۳۰هزار گام است. batch=32 و نقاط و هندسه‌ها در هر گام تازه‌اند.
''')
_fig,_axs=_plt.subplots(3,1,figsize=(10,9),layout='constrained')
for _name in _names:
 _d=_r/_name/'continuation';_t=_rows(_d/'training.jsonl');_v=_rows(_d/'probe.jsonl');_shift=5000 if _name.endswith('extend25k') else 0;_cfg=_read(_r/_name/'run_config.json');_anchor=_cfg.get('anchor',342400)
 if _t:
  _y=[x['loss'] for x in _t];_x=[x['step']-_anchor+_shift for x in _t]
  _axs[0].plot(_x,[sum(_y[max(0,i-99):i+1])/min(i+1,100) for i in range(len(_y))],label=_name)
  _axs[2].plot(_x,[x['learning_rate'] for x in _t],label=_name)
 if _v:_axs[1].plot([x['offset']+_shift for x in _v],[x['control'] for x in _v],'o-',label=_name)
for _ax in _axs:
 _ax.grid(alpha=.2);_ax.set(xlabel='Parameter updates from common342400 anchor')
 if _ax.lines:_ax.legend(fontsize=7)
_axs[0].set(yscale='log',title='100-record training mean; compare common first5000 updates',ylabel='Loss')
_axs[1].set(yscale='log',title='Same fixed control cohort; FEM assessed separately',ylabel='Control loss')
_axs[2].set(ylabel='Learning rate')
_plot=mo.as_html(_fig);_plt.close(_fig)
_downloads=[]
for _name in _names:
 _p=_r/_name/'continuation/checkpoint.npz'
 if _p.exists():_downloads.append(mo.download(data=lambda _path=_p:_path.read_bytes(),filename=_name+'_last_saved.npz',label='دانلود آخرین وزن ذخیره‌شده: '+_name))
mo.vstack([training_refresh,_header,mo.md(str(_fail)) if _fail else mo.md(''),_plot,mo.md('مرحلهٔ ۳۰هزارگامی بودجهٔ بیشتری دارد؛ برتری اپتیمایزر را از مقایسهٔ نابرابر نتیجه نمی‌گیریم. وزن‌های کامل هر ۲۵۰۰ گام و ارزیابی FEM ذخیره می‌شوند. برای داشتن نسخهٔ شخصی، پیش از بستن ماشین Molab وزن‌ها را از دکمه‌های زیر دانلود کنید.'),*_downloads])
