import json
from pathlib import Path
import numpy as np
import pytest
from geometry_operator.geometry import generate,reconstruct_samples,sample_numpy,validate
from geometry_operator.sampling import generate_pools,load_pool
from geometry_operator.evaluate import weighted_relative


def test_immutable_splits_and_tamper_detection(tmp_path):
    m=generate_pools(tmp_path,(8,8,8,8),seed=9)
    assert generate_pools(tmp_path,(8,8,8,8),seed=9)==m
    ids=[c['id'] for s in m['splits'].values() for c in s['cases']]
    assert len(set(ids))==32
    with pytest.raises(ValueError):generate_pools(tmp_path,(16,8,8,8),seed=9)
    with open(tmp_path/'dev.npz','ab') as f:f.write(b'tampered')
    with pytest.raises(ValueError):load_pool(tmp_path,'dev')


def test_sensor_only_reconstruct_and_reject():
    flat=np.zeros(257);top=np.ones(257)
    c,d=reconstruct_samples(flat,top);assert abs(c).max()<1e-12
    with pytest.raises(ValueError):reconstruct_samples(top,flat)
    with pytest.raises(ValueError):reconstruct_samples(flat[:10],top[:10])
    with pytest.raises(ValueError):validate(np.ones((2,6,80)))


def test_physical_weighted_metric():
    truth=np.array([1.,1.]);pred=np.array([1.,2.]);weights=np.array([9.,1.])
    assert weighted_relative(pred,truth,weights)==pytest.approx(np.sqrt(.1))


def test_reference_cache_cannot_mix_geometries(tmp_path):
    from geometry_operator.reference import build_case
    path=tmp_path/'reference.npz'
    np.savez(path,metadata=np.array(json.dumps(dict(geometry_id='wrong',reynolds=30.))))
    with pytest.raises(ValueError,match='geometry/Re mismatch'):
        build_case(np.zeros((2,6,80)),30.,path)
