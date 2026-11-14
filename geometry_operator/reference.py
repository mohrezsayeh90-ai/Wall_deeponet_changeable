"""Independent Taylor-Hood P2/P1 FEM, mapped triangles and tagged facets.

Laplacian diffusion, advective convection, natural zero-traction outlet.
Pressure is anchored by this weak outlet; no gauge projection is applied.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import bmat,csr_matrix
from skfem import MeshTri,Basis,ElementTriP2,ElementTriP1,ElementVector,BilinearForm,asm,condense,solve
from skfem.helpers import ddot,dot,div
from .geometry import sample_numpy
from .sampling import load_pool,sha

@BilinearForm
def diffusion(u,v,w):return ddot(u.grad,v.grad)
@BilinearForm
def divergence(u,q,w):return -q*div(u)
@BilinearForm
def convection(u,v,w):return dot(np.einsum('j...,ij...->i...',w.wind,u.grad),v)


def mesh_for(coeff,nx,ny):
    mesh=MeshTri.init_tensor(np.linspace(0,5,nx+1),np.linspace(0,1,ny+1))
    facets={name:mesh.facets_satisfying(fn,boundaries_only=True) for name,fn in
        dict(bottom=lambda q:np.isclose(q[1],0),top=lambda q:np.isclose(q[1],1),
             inlet=lambda q:np.isclose(q[0],0),outlet=lambda q:np.isclose(q[0],5)).items()}
    p=mesh.p.copy();c,g=sample_numpy(coeff,p[0]);p[1]=c-g/2+(1+g)*p[1]
    mapped=MeshTri(p,mesh.t).with_boundaries(facets)
    e1=p[:,mesh.t[1]]-p[:,mesh.t[0]];e2=p[:,mesh.t[2]]-p[:,mesh.t[0]]
    det=e1[0]*e2[1]-e1[1]*e2[0]
    if np.min(abs(det))<1e-12:raise RuntimeError('Degenerate mapped FEM elements')
    # Reference tensor mesh can contain both element orientations. The map
    # must preserve each triangle's original orientation.
    a=mesh.p[:,mesh.t[1]]-mesh.p[:,mesh.t[0]];b=mesh.p[:,mesh.t[2]]-mesh.p[:,mesh.t[0]]
    if np.any(det*(a[0]*b[1]-a[1]*b[0])<=0):raise RuntimeError('Inverted FEM element')
    return mapped


def solve_reference(coeff,re,nx=80,ny=24,tolerance=2e-8):
    begin=time.monotonic();mesh=mesh_for(coeff,nx,ny)
    ub=Basis(mesh,ElementVector(ElementTriP2()),intorder=5);pb=Basis(mesh,ElementTriP1(),intorder=5)
    a,b=asm(diffusion,ub),asm(divergence,ub,pb)
    walls=ub.get_dofs(['bottom','top']);entrance=ub.get_dofs('inlet')
    fixed=np.union1d(walls.all(),entrance.all());bc=np.zeros(ub.N+pb.N)
    ids=entrance.all(['u^1']);y=ub.doflocs[1,ids];bc[ids]=6*y*(1-y)
    zero=csr_matrix((pb.N,pb.N));free=np.setdiff1d(np.arange(len(bc)),fixed)
    mat=lambda reynolds,c:bmat([[a/reynolds+c,b.T],[b,zero]],format='csr')
    solution=solve(*condense(mat(re,0*a),np.zeros_like(bc),x=bc,D=fixed))
    total=0
    for reynolds in np.linspace(min(10,re),re,max(1,int(np.ceil((re-10)/15))+1)):
        for i in range(160):
            c=asm(convection,ub,wind=ub.interpolate(solution[:ub.N]));matrix=mat(reynolds,c)
            updated=solve(*condense(matrix,np.zeros_like(bc),x=bc,D=fixed))
            # Pressure must converge too: for flat Poiseuille velocity can be
            # exact while continuation still changes pressure by a large factor.
            change=np.linalg.norm(updated-solution)/max(np.linalg.norm(updated),1e-12)
            solution=.7*updated+.3*solution;solution[fixed]=bc[fixed];total+=1
            if change<tolerance:break
        else:raise RuntimeError(f'FEM nonlinear iteration failed at Re={reynolds}')
    c=asm(convection,ub,wind=ub.interpolate(solution[:ub.N]));res=float(np.linalg.norm((mat(re,c)@solution)[free]))
    if res>1e-6 or not np.isfinite(solution).all():raise RuntimeError(f'FEM residual {res}')
    components=ub.split(solution[:ub.N])
    interpolators=[basis.interpolator(values) for values,basis in components]
    pressure_interpolator=pb.interpolator(solution[ub.N:])
    grid=np.linspace(0,5,nx+1);center,dgap=sample_numpy(coeff,grid)
    bottom=center-dgap/2;top=bottom+1+dgap
    def contains(xy):
        xx,yy=np.asarray(xy).T
        return (xx>=0)&(xx<=5)&(yy>=np.interp(xx,grid,bottom)-1e-12)&(yy<=np.interp(xx,grid,top)+1e-12)
    def evaluate(xy):
        xy=np.asarray(xy);finder=mesh.element_finder();finder(*xy.T)  # raises outside; never extrapolate
        return np.column_stack([fn(xy.T) for fn in interpolators]+[pressure_interpolator(xy.T)])
    evaluate.contains=contains
    return evaluate,dict(nx=nx,ny=ny,iterations=total,algebraic_residual=res,seconds=time.monotonic()-begin,
                         velocity_dofs=int(ub.N),pressure_dofs=int(pb.N),outlet='du_dx/Re-p=0,dv_dx/Re=0')


def quadrature(coeff,nx=48,ny=24):
    from numpy.polynomial.legendre import leggauss
    xx,wx=leggauss(nx);eta,wy=leggauss(ny);x=(xx+1)*2.5;eta=(eta+1)/2
    c,g=sample_numpy(coeff,x);gap=1+g;b=c-g/2
    xy=np.stack([np.broadcast_to(x[:,None],(nx,ny)),b[:,None]+gap[:,None]*eta],-1).reshape(-1,2)
    weights=(wx[:,None]*2.5*wy[None,:]/2*gap[:,None]).ravel()
    return xy,weights


def pressure_sections(evaluate,coeff):
    from numpy.polynomial.legendre import leggauss
    e,w=leggauss(32);e=(e+1)/2
    p=[]
    for x in [.25,4.75]:
        c,g=sample_numpy(coeff,np.array([x]));gap=1+g[0];b=c[0]-g[0]/2
        p.append(float(np.dot(w,evaluate(np.column_stack([np.full(len(e),x),b+gap*e]))[:,2])/2))
    return p[0]-p[1]


def build_case(coeff,re,path,nx=80,ny=24,mesh_rtol=.01):
    path=Path(path)
    if path.exists():return json.loads(str(np.load(path,allow_pickle=False)['metadata']))
    xy,w=quadrature(coeff);coarse,mc=solve_reference(coeff,re,nx,ny);fine,mf=solve_reference(coeff,re,nx*2,ny*2)
    # Chordal meshes approximate the exact walls. Explicitly quantify the common
    # valid interior; never extrapolate or silently discard the boundary strip.
    inside=coarse.contains(xy)&fine.contains(xy)
    excluded=float(np.sum(w[~inside])/np.sum(w));xy,w=xy[inside],w[inside]
    if excluded>.005:
        if nx<320:return build_case(coeff,re,path,nx*2,ny*2,mesh_rtol)
        raise RuntimeError(f'Common-domain excluded physical area {excluded} exceeds .5%')
    fc,ff=coarse(xy),fine(xy)
    rel=lambda a,b:float(np.sqrt(np.sum(w[:,None]*(a-b)**2)/max(np.sum(w[:,None]*b*b),1e-20)))
    du,dp=rel(fc[:,:2],ff[:,:2]),rel(fc[:,2:],ff[:,2:]);drop=pressure_sections(fine,coeff);dc=pressure_sections(coarse,coeff)
    uncertainty=dict(velocity=du,pressure=dp,pressure_drop=abs(drop-dc)/max(abs(drop),1e-8))
    if max(uncertainty.values())>mesh_rtol:
        if nx<160:return build_case(coeff,re,path,nx*2,ny*2,mesh_rtol)
        raise RuntimeError(f'Mesh convergence unresolved: {uncertainty}')
    metadata=dict(reynolds=float(re),coarse=mc,fine=mf,mesh_uncertainty=uncertainty,pressure_drop=drop,
                  pressure_sections=[.25,4.75],pressure_denominator_floor=1e-8,wall_stress_verified=False,
                  common_domain_excluded_area_fraction=excluded,common_domain_rule='both chordal meshes; exclude <=.5% physical quadrature area')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,xy=xy,weights=w,fields=ff,metadata=np.array(json.dumps(metadata)))
    return metadata


def main():
    p=argparse.ArgumentParser();p.add_argument('--pool',default='data/geometries');p.add_argument('--split',choices=['dev','test','stress'],default='dev')
    p.add_argument('--output',default='data/references');p.add_argument('--limit',type=int);p.add_argument('--nx',type=int,default=80);p.add_argument('--ny',type=int,default=24)
    p.add_argument('--unlock-test',action='store_true');a=p.parse_args()
    if a.split=='test' and not a.unlock_test:p.error('Locked test: explicit --unlock-test after development selection required')
    pool,m=load_pool(a.pool,a.split);out=Path(a.output)/a.split;out.mkdir(parents=True,exist_ok=True)
    statuses=[]
    for i in range(min(len(pool['coeff']),a.limit or len(pool['coeff']))):
        try:
            info=build_case(pool['coeff'][i],pool['reynolds'][i],out/f'{i:04d}.npz',a.nx,a.ny)
            status=dict(index=i,status='passed',**info)
        except Exception as e:status=dict(index=i,status='failed',error=str(e))
        statuses.append(status);print(json.dumps(status),flush=True)
        (out/'status.json').write_text(json.dumps(dict(split=a.split,pool_sha256=m['splits'][a.split]['sha256'],cases=statuses),indent=2))
if __name__=='__main__':main()
