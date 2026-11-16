"""Continue the successful Adam checkpoint for another 50,000 updates."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    template=(ROOT/'notebooks/kaggle_continue_50k/run.py').read_text()
    marker='# Reuse the original frozen trainer and optimizer, not fresh training.'
    prefix,body=template.split(marker,1)
    body=body.replace('82216','__NEXT_TARGET__').replace('32216','82216').replace('__NEXT_TARGET__','132216')
    body=body.replace('a7e9e35b91d6e3c7f3a37404b940289427bfe78f9b38e39661c10382babd5235','84276c11feecd5bf230c79d1077250018b356ae118491678a4ed69e46ab83140')
    body=body.replace("source_kernel='mohrezsayeh/geometry-pi-deeponet-main'","source_kernel='mohrezsayeh/geometry-pi-deeponet-continue-50k'")
    body=body.replace('runs/continued_50k','runs/adam_next50k')
    script=prefix+marker+body;compile(script,'run.py','exec')
    assert "assert meta['step']==82216" in script and "'--steps','132216'" in script
    assert 'train_lbfgs' not in script
    output=ROOT/'notebooks/kaggle_adam_next50k';output.mkdir(exist_ok=True)
    (output/'run.py').write_text(script)
    metadata=dict(id='mohrezsayeh/geometry-pi-deeponet-adam-next50k',title='Geometry PI DeepONet Adam Next50k',code_file='run.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[],competition_sources=[],kernel_sources=['mohrezsayeh/geometry-pi-deeponet-continue-50k'])
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(path=str(output),script_sha256=hashlib.sha256(script.encode()).hexdigest(),start_step=82216,target_step=132216,additional_steps=50000)))


if __name__=='__main__':main()
