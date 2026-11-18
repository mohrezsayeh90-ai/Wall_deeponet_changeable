"""Build the dynamic-LR core without modifying the frozen original trainer."""
from pathlib import Path
import argparse,hashlib,json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);r=ap.parse_args().root
    package=r/'frozen/output/project/geometry_operator';source=(package/'train.py').read_text()
    signature='def step(params,m,v,count,data):';schedule='lr=jnp.maximum(2e-5,1e-3*.95**(count/2000))'
    assert source.count(signature)==1 and source.count(schedule)==1
    modified=source.replace(signature,'def step(params,m,v,count,data,learning_rate):').replace(schedule,'lr=learning_rate')
    (package/'adaptive_core.py').write_text(modified)
    (r/'core-source-hashes.json').write_text(json.dumps({'original':hashlib.sha256(source.encode()).hexdigest(),'dynamic':hashlib.sha256(modified.encode()).hexdigest(),'changes':'Only explicit LR input replaces old schedule; Adam/gradient/physics math unchanged'},indent=2))
if __name__=='__main__':main()
