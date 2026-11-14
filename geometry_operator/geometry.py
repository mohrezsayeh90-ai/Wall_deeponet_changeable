"""C4 quintic spline geometry, canonical sensors, physical map.

Two deviations (centerline minus .5, gap minus 1), stored as local
descending-power polynomials. Basis support is restricted to [.5,4.5].
"""
from dataclasses import dataclass
import hashlib
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline, PPoly, make_interp_spline

LENGTH = 5.
BREAKS = np.linspace(0., LENGTH, 81)
FAMILIES = ('mild', 'centerline', 'symmetric_gap', 'one_wall',
            'multiple_gap', 'independent_walls', 'oscillatory', 'mixed_spline')


def local(coeff, x, order=0):
    breaks=jnp.asarray(BREAKS,dtype=coeff.dtype)
    i = jnp.clip(jnp.searchsorted(breaks, x, side='right')-1, 0, 79)
    c = coeff[:, :, i]
    for _ in range(order):
        c = c[:, :-1] * jnp.arange(c.shape[1]-1, 0, -1)
    value = jnp.zeros(2, dtype=coeff.dtype)
    z = x-breaks[i]
    for k in range(c.shape[1]):
        value = value*z+c[:, k]
    return value


def walls(coeff, x):
    c, g = local(coeff, x)
    return .5+c-(1+g)/2, .5+c+(1+g)/2


def forward(coeff, z):
    x = LENGTH*z[0]
    b, t = walls(coeff, x)
    return jnp.array([x, b+(t-b)*z[1]])


def inverse(coeff, xy):
    x, y = xy
    b, t = walls(coeff, x)
    return jnp.array([x/LENGTH, (y-b)/(t-b)])


def sample_numpy(coeff, x, order=0):
    return np.stack([PPoly(c, BREAKS)(x, nu=order) for c in np.asarray(coeff)])


def sensors(coeff, count=257):
    return sample_numpy(coeff, np.linspace(0, LENGTH, count)).reshape(-1)


def descriptors(coeff):
    x = np.linspace(0, LENGTH, 4001)
    c, dg = sample_numpy(coeff, x)
    cp, gp = sample_numpy(coeff, x, 1)
    cpp, gpp = sample_numpy(coeff, x, 2)
    return dict(min_gap=float(np.min(1+dg)), max_gap=float(np.max(1+dg)),
                center_deviation=float(np.max(abs(c))),
                max_slope=float(np.max(abs(np.r_[cp-gp/2, cp+gp/2]))),
                max_curvature=float(np.max(abs(np.r_[cpp-gpp/2, cpp+gpp/2]))))


def validate(coeff, stress=False):
    coeff = np.asarray(coeff)
    if coeff.shape != (2,6,80) or not np.isfinite(coeff).all():
        raise ValueError('Geometry must be finite coefficients of shape (2,6,80)')
    d = descriptors(coeff)
    limits = (.42, 1.7, .45, 1.1) if stress else (.60, 1.50, .35, .7)
    if not (d['min_gap'] >= limits[0]-1e-8 and d['max_gap'] <= limits[1]+1e-8
            and d['center_deviation'] <= limits[2]+1e-8 and d['max_slope'] <= limits[3]+1e-8):
        raise ValueError(f'Geometry outside declared envelope: {d}')
    xx = np.r_[np.linspace(0,.5,31),np.linspace(4.5,5,31)]
    if np.max(abs(sample_numpy(coeff,xx))) > 1e-8:
        raise ValueError('First and last 10% must be flat')
    return d


def _canonical(values):
    # Simple interior knots, degree five: C4 at every join. B-spline basis
    # functions intersecting the flat strips have exactly zero coefficients.
    knots = np.r_[np.zeros(6), BREAKS[1:-1], np.full(6,LENGTH)]
    n = len(knots)-6
    active = (knots[:n] >= .5) & (knots[6:6+n] <= 4.5)
    out=[]
    for v in values:
        cv = np.where(active,v,0.)
        pp = PPoly.from_spline(BSpline(knots,cv,5))
        keep=np.diff(pp.x)>0
        out.append(pp.c[:,keep])
    return np.stack(out)


def generate(rng, family, stress=False):
    knots=np.r_[np.zeros(6),BREAKS[1:-1],np.full(6,LENGTH)]
    x=np.array([np.mean(knots[i+1:i+6]) for i in range(len(knots)-6)])
    def feature():
        center=rng.uniform(1.35,3.65); width=rng.uniform(.4,1.1)
        z=(x-center)/width
        return np.exp(1-1/np.maximum(1-z*z,1e-12))*(abs(z)<1)
    c=np.zeros_like(x); g=c.copy()
    sign=lambda:rng.choice([-1.,1.])
    if family==0: c=.035*rng.normal()*feature();g=.08*rng.normal()*feature()
    elif family==1: c=.25*(feature()-rng.uniform(.3,1)*feature())*sign();g=.07*rng.normal()*feature()
    elif family==2: g=rng.uniform(.15,.65)*sign()*feature()
    elif family==3:
        g=rng.uniform(.15,.6)*sign()*feature();c=sign()*g/2
    elif family==4:
        for _ in range(rng.integers(2,5)):g+=rng.uniform(.1,.4)*sign()*feature()
    elif family==5:
        b=sum(.2*rng.normal()*feature() for _ in range(3))
        t=sum(.2*rng.normal()*feature() for _ in range(3));c=(b+t)/2;g=t-b
    elif family==6:
        for k in range(1,5):
            c+=rng.normal()*.04/k*np.sin(2*np.pi*k*x/4+rng.uniform(0,6.28))
            g+=rng.normal()*.07/k*np.sin(2*np.pi*k*x/4+rng.uniform(0,6.28))
    elif family==7:
        # Smooth random coefficient family, no interpolation of wall samples.
        from scipy.ndimage import gaussian_filter1d
        c=gaussian_filter1d(rng.normal(size=len(x)),rng.uniform(3,7))*.3
        g=gaussian_filter1d(rng.normal(size=len(x)),rng.uniform(3,7))*.5
    else:raise ValueError('Unknown family')
    coeff=_canonical([c,g]);d=descriptors(coeff)
    # Radial amplitude draw reaches both mild and envelope-edge geometries.
    bounds=(.56,.7,.45,1.1) if stress else (.4,.5,.35,.7)
    xx=np.linspace(0,5,1001);cc,gg=sample_numpy(coeff,xx)
    ratios=[bounds[0]/max(-gg.min(),1e-9),bounds[1]/max(gg.max(),1e-9),
            bounds[2]/max(abs(cc).max(),1e-9),bounds[3]/max(d['max_slope'],1e-9)]
    scale=min(ratios)*rng.uniform(.65, .995) if family!=0 else min(1.,min(ratios))
    coeff*=scale
    validate(coeff,stress)
    return coeff


def geometry_id(coeff):
    return hashlib.sha256(np.asarray(coeff,dtype='<f8').tobytes()).hexdigest()


def reconstruct_samples(bottom, top):
    """Deterministic quintic interpolation then canonical B-spline projection.

    Sensor-only mode uses the reconstructed geometry for ALL model operations.
    Input arrays must use the canonical uniform 0..5 grid. Caller must inspect
    reconstruction diagnostics against known-wall truth when available.
    """
    bottom,top=np.asarray(bottom),np.asarray(top)
    if bottom.ndim!=1 or top.shape!=bottom.shape or len(bottom)<65:
        raise ValueError('At least 65 paired canonical wall samples required')
    x=np.linspace(0,5,len(bottom))
    if not np.isfinite(np.r_[bottom,top]).all() or np.any(top<=bottom):
        raise ValueError('Crossed/nonfinite walls')
    knots=np.r_[np.zeros(6),BREAKS[1:-1],np.full(6,5.)]
    gx=np.array([np.mean(knots[i+1:i+6]) for i in range(len(knots)-6)])
    out=_canonical([make_interp_spline(x,(bottom+top)/2-.5,k=5)(gx),
                    make_interp_spline(x,top-bottom-1,k=5)(gx)])
    validate(out)
    cc,gg=sample_numpy(out,x)
    err=max(np.max(abs(cc-(bottom+top)/2+.5)),np.max(abs(gg-(top-bottom-1))))
    if err>.025:raise ValueError(f'Under-resolved/reconstruction error {err:.4g} exceeds .025')
    return out,dict(max_sensor_reconstruction_error=float(err),interpolant='quintic-to-canonical-C4')
