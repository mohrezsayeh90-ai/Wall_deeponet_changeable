import json
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from geometry_operator.geometry import generate,validate,forward,inverse,walls,local,sensors,sample_numpy
from geometry_operator.model import Config,initialize,context,fields
from geometry_operator.operators import residual,traction,mapped_derivative,mapped_laplacian
from geometry_operator.checkpoint import save,load


@pytest.fixture(scope='module')
def curved():return jnp.asarray(generate(np.random.default_rng(81),5))
@pytest.fixture(scope='module')
def model():
    c=Config(sensors=65,width=8,rank=4,depth=1)
    p=initialize(c)
    # Exercise non-negligible random corrections, not only a null initializer.
    p=jax.tree_util.tree_map(lambda x:x*3,p)
    return c,p


def test_flat_map_and_roundtrip(curved):
    flat=jnp.zeros((2,6,80));z=jnp.array([.37,.42])
    np.testing.assert_allclose(forward(flat,z),[1.85,.42])
    np.testing.assert_allclose(inverse(curved,forward(curved,z)),z,atol=1e-13)
    jac=jax.jacfwd(lambda q:forward(curved,q))(z)
    gap=1+local(curved,5*z[0])[1]
    np.testing.assert_allclose(jnp.linalg.det(jac),5*gap,rtol=1e-12)


@pytest.mark.parametrize('nonpolynomial',[False,True])
def test_mapped_chain_rule(curved,nonpolynomial):
    physical=(lambda p:jnp.sin(p[0]) * jnp.exp(.3*p[1])) if nonpolynomial else (lambda p:p[0]**3*p[1]**2+.7*p[1]**4)
    ref=lambda z:physical(forward(curved,z));z=jnp.array([.43,.36]);xy=forward(curved,z)
    grad=jax.grad(physical)(xy);hess=jax.hessian(physical)(xy)
    for a in (0,1):np.testing.assert_allclose(mapped_derivative(ref,curved,z,a),grad[a],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(mapped_laplacian(ref,curved,z),jnp.trace(hess),rtol=1e-10,atol=1e-10)
    xy_mixed=mapped_derivative(lambda q:mapped_derivative(ref,curved,q,0),curved,z,1)
    yx_mixed=mapped_derivative(lambda q:mapped_derivative(ref,curved,q,1),curved,z,0)
    np.testing.assert_allclose(xy_mixed,yx_mixed,atol=1e-10)


def test_wall_inlet_flux_divergence(curved,model):
    c,p=model;b=context(p,jnp.asarray(sensors(curved,c.sensors)),55.,jnp.zeros(2*c.sensors),jnp.ones(2*c.sensors))
    fn=jax.jit(lambda xy:fields(p,b,curved,55.,xy,c))
    for x in [.1,1.1,2.7,4.8]:
        for eta in [0.,1.]:np.testing.assert_allclose(fn(forward(curved,jnp.array([x/5,eta])))[:2],0.,atol=1e-10)
    for y in [.1,.4,.8]:np.testing.assert_allclose(fn(jnp.array([0.,y]))[:2],[6*y*(1-y),0],atol=1e-10)
    e,w=np.polynomial.legendre.leggauss(40);eta=(e+1)/2
    for x in [1.,2.5,4.]:
        gap=float(1+local(curved,x)[1]);q=jax.vmap(lambda e:forward(curved,jnp.array([x/5,e])))(jnp.asarray(eta))
        u=jax.vmap(fn)(q)[:,0];np.testing.assert_allclose(np.dot(w,np.asarray(u))*gap/2,1.,atol=1e-9)
    g=jax.jacfwd(lambda q:fields(p,b,curved,55.,q,c))(forward(curved,jnp.array([.4,.3])))
    assert abs(float(g[0,0]+g[1,1]))<1e-10


def test_poiseuille_exact(model):
    c,p=model;p=jax.tree_util.tree_map(jnp.zeros_like,p);flat=jnp.zeros((2,6,80));b=jnp.zeros((2,c.rank))
    np.testing.assert_allclose(fields(p,b,flat,40.,jnp.array([2.,.4]),c),[1.44,0.,.9],atol=1e-12)
    np.testing.assert_allclose(residual(p,b,flat,40.,jnp.array([2.,.4]),c),0,atol=1e-12)
    np.testing.assert_allclose(traction(p,b,flat,40.,.4,c),0,atol=1e-12)


def test_families_and_spline_smoothness():
    for i in range(8):
        coeff=generate(np.random.default_rng(i),i);validate(coeff)
        # Exact one-sided derivatives at every polynomial join, through order 4.
        from scipy.interpolate import PPoly
        for component in coeff:
            for order in range(5):
                c=component.copy()
                for _ in range(order):c=c[:-1]*np.arange(len(c)-1,0,-1)[:,None]
                right=np.polyval(c[:,:-1],.0625)
                np.testing.assert_allclose(right,c[-1,1:],atol=2e-7,rtol=1e-7)


def test_checkpoint_roundtrip(tmp_path,model):
    c,p=model;z=jax.tree_util.tree_map(jnp.zeros_like,p);rng=np.random.default_rng(8)
    save(tmp_path/'x.npz',p,c,7,rng.bit_generator.state,np.zeros(130),np.ones(130),z,z)
    q,cc,meta,norm,state=load(tmp_path/'x.npz',True)
    assert cc==c and meta['step']==7
    for a,b in zip(jax.tree_util.tree_leaves(p),jax.tree_util.tree_leaves(q)):np.testing.assert_array_equal(a,b)


def test_reference_flat():
    from geometry_operator.reference import solve_reference,mesh_for
    coeff=np.zeros((2,6,80));fn,meta=solve_reference(coeff,30.,nx=12,ny=6)
    xy=np.array([[1.,.2],[2.5,.5],[4.,.7]])
    truth=np.column_stack([6*xy[:,1]*(1-xy[:,1]),np.zeros(3),12*(5-xy[:,0])/30])
    np.testing.assert_allclose(fn(xy),truth,atol=2e-6)
    mesh=mesh_for(coeff,12,6)
    assert all(len(mesh.boundaries[k])>0 for k in ['inlet','outlet','top','bottom'])
    with pytest.raises(ValueError):fn(np.array([[2.,2.]]))


def test_gradient_and_deterministic_update(curved,model):
    from geometry_operator.train import make_step
    c,p=model;z=jax.tree_util.tree_map(jnp.zeros_like,p)
    data=dict(coeff=curved[None],sensors=jnp.asarray(sensors(curved,65))[None],re=jnp.array([30.]),
        z=jnp.array([[[.35,.4],[.7,.7]]]),outlet=jnp.array([[.3,.8]]))
    step,one=make_step(c,jnp.zeros(130),jnp.ones(130))
    updated,m,v,value,terms,gn=step(p,z,z,1,data)
    assert np.isfinite(value) and float(gn)>0
    # Same Adam/RNG-independent batch after serialization gives same next step.
    a=step(updated,m,v,2,data);b=step(updated,m,v,2,data)
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):np.testing.assert_array_equal(x,y)
