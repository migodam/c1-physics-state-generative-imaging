"""Candidate-computable toy and vector electric dipole forward states."""
from __future__ import annotations
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from .accounting import PhysicsLedger

def toy_forward(x,kappa):
    x=np.asarray(x); kappa=np.asarray(kappa)
    return np.stack([x[...,0]**2,x[...,1]**2,kappa*(x[...,0]+0.5*x[...,1])],axis=-1)

def toy_jacobian(x,kappa):
    return np.array([[2*x[0],0],[0,2*x[1]],[kappa,0.5*kappa]])

class DipoleProblem:
    """Normalized vector DDA, one k; not continuum-Maxwell convergence validation."""
    solver_id='c1-vector-dipole13-rr-v1'
    def __init__(self,ledger=None):
        self.ledger=ledger or PhysicsLedger()
        self.k=4.; self.spacing=.08; self.volume=self.spacing**3; self.eps_b=2.
        xyz=[]
        for i in range(-2,3):
            for j in range(-2,3):
                if i*i+j*j<=4:xyz.append([i*self.spacing,j*self.spacing,0.])
        self.xyz=np.array(xyz);self.n=len(xyz)
        self.core=np.linalg.norm(self.xyz,axis=1)<=self.spacing*1.01
        self.basis=np.stack([self.core,~self.core],axis=1).astype(float)
        G=np.zeros((3*self.n,3*self.n),complex)
        for i in range(self.n):
            for j in range(self.n):
                if i==j:continue
                rvec=self.xyz[i]-self.xyz[j];r=np.linalg.norm(rvec);n=rvec/r
                kr=self.k*r;nn=np.outer(n,n)
                G[3*i:3*i+3,3*j:3*j+3]=np.exp(1j*kr)/(4*np.pi*r**3)*(
                    kr**2*(np.eye(3)-nn)+(1-1j*kr)*(3*nn-np.eye(3)))
        self.G=G
        self.source_directions=np.array([[1.,0,0],[0,1.,0]])
        self.polarizations=np.array([[0.,1.,0],[1.,0,0]])
        self.inc=np.stack([(np.exp(1j*self.k*(self.xyz@d))[:,None]*e).reshape(-1)
                           for d,e in zip(self.source_directions,self.polarizations)],axis=1)
        recv=np.array([[0.,0.,1.],[1/np.sqrt(2),0.,1/np.sqrt(2)]])
        self.S=[]
        for s in range(2):
            for n in recv:
                e=self.polarizations[s]-(n@self.polarizations[s])*n
                e=e/np.linalg.norm(e)
                v=self.k**2/(4*np.pi)*np.exp(-1j*self.k*(self.xyz@n))[:,None]*e
                self.S.append((s,v.reshape(-1)))
        self.scale=None

    def polarizability(self,x,loss=.02):
        eps=self.eps_b+self.basis@np.asarray(x)+1j*loss
        cm=3*self.volume*(eps-self.eps_b)/(eps+2*self.eps_b)
        den=1-1j*self.k**3*cm/(6*np.pi)
        alpha=cm/den
        da=9*self.volume*self.eps_b/(eps+2*self.eps_b)**2/den**2
        return eps,alpha,da[:,None]*self.basis

    def forward(self,x,derivative=False,kind='forward',loss=.02):
        with self.ledger.reserve(kind,derivative_rhs=4 if derivative else 0):
            eps,a,da=self.polarizability(x,loss)
            ar=np.repeat(a,3)
            L=np.eye(3*self.n)-ar[:,None]*self.G
            fac=lu_factor(L)
            p=lu_solve(fac,ar[:,None]*self.inc)
            E=self.inc+self.G@p
            amplitude=np.array([v@p[:,s] for s,v in self.S])
            scale=np.ones(4) if self.scale is None else self.scale
            f=np.abs(amplitude)**2/scale
            J=None
            if derivative:
                dp=lu_solve(fac,np.concatenate([np.repeat(da[:,q],3)[:,None]*E for q in range(2)],axis=1))
                dam=np.array([[v@dp[:,2*q+s] for q in range(2)] for s,v in self.S])
                J=2*np.real(np.conj(amplitude)[:,None]*dam)/scale[:,None]
            return dict(f=f,J=J,eps=eps,p=p,E=E,L=L,amplitude=amplitude,
                        lu=fac,material_coordinates=np.asarray(x).copy(),
                        constitutive_residual=float(np.linalg.norm(p-ar[:,None]*E)/(np.linalg.norm(p)+1e-30)),
                        source_count=2,passive=bool(np.min(eps.imag)>=0),factorization_count=1)

    def adjoint_gradient(self,state,x,y,sigma):
        if not np.array_equal(state['material_coordinates'],x):raise ValueError('cached state belongs to different candidate')
        with self.ledger.reserve('cached_candidate_adjoint',material_frequency=0,forward_rhs=0,adjoint_rhs=2):
            b=np.zeros_like(state['p'])
            cot=2*state['amplitude']*(state['f']-y)/(sigma**2*self.scale)
            for value,(source,v) in zip(cot,self.S):b[:,source]+=np.conj(v)*value
            lam=lu_solve(state['lu'],b,trans=2)
            _,_,da=self.polarizability(x)
            return np.array([np.real(np.sum(np.conj(lam)*np.repeat(da[:,q],3)[:,None]*state['E'])) for q in range(2)])

    def jvp(self,state,x,h):
        if not np.array_equal(state['material_coordinates'],x):raise ValueError('cached state belongs to different candidate')
        with self.ledger.reserve('cached_candidate_jvp',material_frequency=0,forward_rhs=0,derivative_rhs=2):
            _,_,da=self.polarizability(x)
            dp=lu_solve(state['lu'],np.repeat(da@np.asarray(h),3)[:,None]*state['E'])
            dam=np.array([v@dp[:,s] for s,v in self.S])
            return 2*np.real(np.conj(state['amplitude'])*dam)/self.scale

    def set_scale(self):
        if self.scale is None:
            s=self.forward([1.,1.],kind='normalization_setup')
            self.scale=np.maximum(np.abs(s['amplitude'])**2,1e-20)
        return self.scale

    def context(self,y,sigma,initial=(.65,-.45)):
        """API deliberately has no truth material, fields, or posterior reference."""
        x=np.array(initial,float); states=[]
        for step in range(2):
            s=self.forward(x,True,kind='candidate_pilot')
            r=(s['f']-y)/sigma;J=s['J']/sigma;g=J.T@r;H=J.T@J
            patches=[]
            for mask in [self.core,~self.core]:
                for key in ['E','p']:
                    v=s[key].reshape(self.n,3,2)[mask].mean(axis=0)
                    # Co-polarized electric component for each source, with physical scaling.
                    w=np.array([v[:,q]@self.polarizations[q] for q in range(2)])
                    if key=='p':w=w/self.volume
                    patches.extend(w.real);patches.extend(w.imag)
            c=np.r_[x,s['f'],r,g,J.ravel(),np.diag(H),patches]
            states.append(c)
            if step==0:
                delta=np.linalg.solve(H+np.eye(2),g)
                delta*=min(1,.5/(np.linalg.norm(delta)+1e-30))
                x=np.clip(x-delta,-1.45,1.45)
        return np.stack(states)

def toy_context(y,kappa,sigma,initial=(.65,-.45)):
    x=np.array(initial,float); states=[]
    for step in range(2):
        f=toy_forward(x,kappa);J=toy_jacobian(x,kappa)/sigma;r=(f-y)/sigma;g=J.T@r;H=J.T@J
        states.append(np.r_[x,f,r,g,J.ravel(),np.diag(H)])
        if step==0:
            delta=np.linalg.solve(H+np.eye(2),g)
            delta*=min(1,.5/(np.linalg.norm(delta)+1e-30))
            x=np.clip(x-delta,-1.45,1.45)
    return np.stack(states)
