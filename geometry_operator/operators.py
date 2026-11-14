"""Production differentiation in physical coordinates; independent mapped path."""
import jax
import jax.numpy as jnp
from .geometry import local, walls, forward
from .model import fields


def residual(params,branch,coeff,re,xy,config):
    fn=lambda q:fields(params,branch,coeff,re,q,config)
    u=fn(xy);g=jax.jacfwd(fn)(xy);h=jax.jacfwd(jax.jacfwd(fn))(xy)
    return jnp.array([u[0]*g[0,0]+u[1]*g[0,1]+g[2,0]-(h[0,0,0]+h[0,1,1])/re,
        u[0]*g[1,0]+u[1]*g[1,1]+g[2,1]-(h[1,0,0]+h[1,1,1])/re,g[0,0]+g[1,1]])


def traction(params,branch,coeff,re,eta,config):
    xy=jnp.array([5.,eta])
    fn=lambda q:fields(params,branch,coeff,re,q,config)
    u=fn(xy);g=jax.jacfwd(fn)(xy)
    return jnp.array([g[0,0]/re-u[2],g[1,0]/re])


def mapped_derivative(fn,coeff,z,axis):
    grad=jax.grad(fn)(z);x=5*z[0]
    cp,gp=local(coeff,x,1);_,dg=local(coeff,x);gap=1+dg
    a=cp-gp/2+z[1]*gp
    return grad[0]/5-a/gap*grad[1] if axis==0 else grad[1]/gap


def mapped_laplacian(fn,coeff,z):
    return sum(mapped_derivative(lambda q:mapped_derivative(fn,coeff,q,axis),coeff,z,axis) for axis in (0,1))
