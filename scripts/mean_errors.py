"""Area-weighted absolute MAE and relative L1 for the five saved flow examples."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/random_channel_chain_5'


def calculate(predicted,reference,weights,vector=False):
    difference=predicted-reference
    magnitude=np.linalg.norm(difference,axis=1) if vector else np.abs(difference)
    reference_magnitude=np.linalg.norm(reference,axis=1) if vector else np.abs(reference)
    numerator=float(weights@magnitude)
    denominator=float(weights@reference_magnitude)
    return dict(mae=numerator/float(weights.sum()),relative_l1_percent=100*numerator/max(denominator,1e-16),reference_mean_magnitude=denominator/float(weights.sum()),denominator_floor_active=denominator<1e-16)


def main():
    original=json.loads((BASE/'summary.json').read_text());results=[]
    for case in original['cases']:
        path=BASE/case['directory']/'fields.npz'
        with np.load(path,allow_pickle=False) as data:
            pred=data['predicted'];ref=data['fem'];w=data['weights']
            assert pred.shape==ref.shape and np.isfinite(pred).all() and np.isfinite(ref).all() and np.all(w>0)
            assert str(data['checkpoint_sha256'])==original['checkpoint_sha256']
            values={'velocity_vector':calculate(pred[:,:2],ref[:,:2],w,True),
                    'u':calculate(pred[:,0],ref[:,0],w),'v':calculate(pred[:,1],ref[:,1],w),
                    'pressure':calculate(pred[:,2],ref[:,2],w),
                    'speed':calculate(np.linalg.norm(pred[:,:2],axis=1),np.linalg.norm(ref[:,:2],axis=1),w)}
            # Independently check MAE = reference mean magnitude * relative L1.
            for value in values.values():
                if not value['denominator_floor_active']:
                    np.testing.assert_allclose(value['mae'],value['reference_mean_magnitude']*value['relative_l1_percent']/100,rtol=1e-13)
            results.append(dict(index=case['index'],title=case['title'],directory=case['directory'],reynolds=case['reynolds'],fields_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),evaluated_area=float(w.sum()),metrics=values))
    assert len(results)==5
    output=BASE/'mean_errors';output.mkdir(exist_ok=True)
    report=dict(checkpoint_sha256=original['checkpoint_sha256'],checkpoint_step=original['checkpoint_step'],reynolds=30,case_count=5,weighting='Physical-area quadrature on the original common FEM domain; same samples as the L2 reports.',relative_definition='100 * sum(w * norm(prediction-reference)) / max(sum(w * norm(reference)), 1e-16)',absolute_definition='sum(w * norm(prediction-reference)) / sum(w)',norm='Euclidean for velocity vector; absolute value for scalar u, v, speed and pressure.',all_mae_relative_identity_checks_passed=True,cases=results)
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['case','title','quantity','mae_absolute','relative_l1_percent','reference_mean_magnitude','denominator_floor_active'])
        for case in results:
            for name,m in case['metrics'].items():writer.writerow([case['index'],case['title'],name,m['mae'],m['relative_l1_percent'],m['reference_mean_magnitude'],m['denominator_floor_active']])
    lines=['# Mean absolute and relative errors for the five examples','', '[All quantities CSV](metrics.csv) · [Full precision JSON and source hashes](metrics.json) · [Original figures and L2 comparison](../README.md)','', '**Absolute MAE** = sum(w × magnitude(model − FEM)) / sum(w).','', '**Relative L1 (%)** = 100 × sum(w × magnitude(model − FEM)) / max(sum(w × magnitude(FEM)), 1e−16).','', 'For velocity, magnitude is the Euclidean norm of the two-component vector (u,v). For pressure or another scalar, it is absolute value. This uses the same positive physical-area weights, common-domain quadrature points and original step-32,216 checkpoint as the five-case L2 report, all at Re=30. No new model inference, FEM solve or training is needed. Original field arrays and figures remain unchanged.','', 'MAE is an absolute error in the corresponding nondimensional field units, **not a percentage**. Relative L1 is a ratio of weighted integrals, not the mean of pointwise percentage errors (MAPE). Pressure is compared without a fitted constant shift. Near-zero reference v can make its relative error large even when absolute v error is small; the reference mean magnitude and denominator-floor flag are included in JSON/CSV. These ratios are not guaranteed to be smaller than relative L2.','', '## Velocity vector and pressure','', '| Case | Velocity MAE | Velocity relative L1 | Pressure MAE | Pressure relative L1 |','| --- | ---: | ---: | ---: | ---: |']
    for case in results:
        v=case['metrics']['velocity_vector'];p=case['metrics']['pressure'];lines.append(f"| {case['index']:02d} {case['title']} | {v['mae']:.6f} | {v['relative_l1_percent']:.3f}% | {p['mae']:.6f} | {p['relative_l1_percent']:.3f}% |")
    lines+=['','## Components and speed magnitude','', '| Case | Quantity | Absolute MAE | Relative L1 |','| --- | --- | ---: | ---: |']
    for case in results:
        for name in ['u','v','speed']:
            m=case['metrics'][name];lines.append(f"| {case['index']:02d} | {name} | {m['mae']:.6f} | {m['relative_l1_percent']:.3f}% |")
    lines+=['','Reproduce from repository root: `python -m scripts.mean_errors`. All 25 quantity-level MAE/relative identities are verified during generation.','']
    (output/'README.md').write_text('\n'.join(lines))
    p=BASE/'README.md';text=p.read_text();link='[Mean absolute MAE and relative L1 errors](mean_errors/README.md)'
    if link not in text:text=text.replace('# Five varied random channel inputs and flow predictions','# Five varied random channel inputs and flow predictions\n\n'+link)
    p.write_text(text)
    (BASE/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(BASE).as_posix()+'\n' for p in sorted(BASE.rglob('*')) if p.is_file() and p!=BASE/'SHA256SUMS'))
    for case in results:print(json.dumps(dict(case=case['index'],velocity=case['metrics']['velocity_vector'],pressure=case['metrics']['pressure'])))


if __name__=='__main__':main()
