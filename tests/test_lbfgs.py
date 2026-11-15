import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from jax.flatten_util import ravel_pytree
from geometry_operator.model import Config,initialize
from geometry_operator.geometry import generate,sensors
from geometry_operator.train import make_step
from geometry_operator.train_lbfgs import make_objective,fixed_banks


def test_fixed_objective_gradient_and_lbfgs_decrease():
    c=Config(sensors=65,width=4,depth=1,rank=2)
    params=initialize(c);mean=jnp.zeros(130);std=jnp.ones(130)
    coeff=generate(np.random.default_rng(81),5)
    data=dict(coeff=jnp.asarray(coeff)[None],sensors=jnp.asarray(sensors(coeff,65))[None],re=jnp.array([30.]),z=jnp.array([[[.4,.3],[.6,.7]]]),outlet=jnp.array([[.3,.7]]))
    objective=make_objective(c,mean,std);flat,unravel=ravel_pytree(params)
    value,gradient=objective(params,data)
    _,one=make_step(c,mean,std)
    expected,_=one(params,{k:v[0] for k,v in data.items()});np.testing.assert_allclose(value,expected,rtol=1e-12)
    direction=np.random.default_rng(4).normal(size=flat.shape);direction/=np.linalg.norm(direction);eps=1e-5
    fd=(float(objective(unravel(flat+eps*direction),data)[0])-float(objective(unravel(flat-eps*direction),data)[0]))/(2*eps)
    analytic=np.asarray(ravel_pytree(gradient)[0])@direction;np.testing.assert_allclose(fd,analytic,rtol=1e-4,atol=1e-8)
    def fg(x):
        v,g=objective(unravel(jnp.asarray(x)),data);return float(v),np.asarray(ravel_pytree(g)[0],dtype=float)
    result=minimize(fg,np.asarray(flat),jac=True,method='L-BFGS-B',options={'maxiter':3,'maxls':20})
    assert np.isfinite(result.fun) and result.fun<float(value)
    a=objective(params,data);b=objective(params,data)
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(ravel_pytree(a[1])[0],ravel_pytree(b[1])[0])


def test_disjoint_reproducible_banks():
    pool=dict(coeff=np.zeros((10,2,6,80)),sensors=np.zeros((10,514)))
    a,indices,_=fixed_banks(pool,4,3,2,2,42);b,other,_=fixed_banks(pool,4,3,2,2,42)
    assert not set(indices[:4])&set(indices[4:]);np.testing.assert_array_equal(indices,other)
    for first,second in zip(a,b):
        for k in first:np.testing.assert_array_equal(first[k],second[k])
