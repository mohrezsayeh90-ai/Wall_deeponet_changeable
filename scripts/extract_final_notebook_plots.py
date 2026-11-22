"""Extract original PNG bytes from the immutable final notebook export."""
from pathlib import Path
import re,html,base64,json,hashlib,zipfile
from PIL import Image
R=Path(__file__).resolve().parents[1];O=R/'reports/molab_soap_warmcos50k/completed';N=O/'notebook';seen=set();out=[]
for encoded in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',html.unescape((N/'notebook.html').read_text())):
 data=base64.b64decode(encoded);sha=hashlib.sha256(data).hexdigest()
 if sha in seen:continue
 seen.add(sha);p=O/f'figure_{len(out)+1:02d}.png';p.write_bytes(data)
 with Image.open(p) as im:im.verify()
 out.append(dict(file=p.name,bytes=len(data),sha256=sha))
assert out
(O/'plot_verification.json').write_text(json.dumps({'original_png_bytes_extracted':True,'plots':out},indent=2)+'\n')
with zipfile.ZipFile(O/'reports_and_notebook.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(O.rglob('*')):
  if p.is_file() and p.suffix!='.zip' and not p.name.endswith('.download'):z.write(p,p.relative_to(O))
with zipfile.ZipFile(O/'reports_and_notebook.zip') as z:assert z.testzip() is None
print(json.dumps(out))
