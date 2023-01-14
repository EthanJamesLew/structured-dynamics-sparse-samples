import numpy as np
import itertools
import typing as typ

import scipy.linalg

import scipy.linalg


def mldivide(A, b):
    """helper to match the matlab A \ b
    
    NOTE: lstsq is much slower than this approach (not as safe though)
    """
    return np.linalg.solve(A.T@A, A.T@b)


def douglas_rachford(A, b, sigma, tau, mu, MaxIt, tol):
    """Douglas–Rachford splitting algorithm for sparse optimization"""

    def ProxF(x, u):
        x1 = np.copy(x)
        for idx in range(x.shape[0]):
            x1[idx][0] = max(np.abs(x1[idx][0]) - tau, 0)
        x1 *= np.sign(x)
        u1 = b + (u-b) * min(sigma/np.linalg.norm(u-b,ord=2),1)
        return x1, u1
    
    def rProxF(x,u):
        x1, u1 = ProxF(x, u)
        x1 = 2*x1-x
        u1 = 2*u1-u
        return x1, u1
    
    def ProxG(x,u):
        y = x + A.T@u
        ls = mldivide(pAlower, y)
        x1 = mldivide(pAlower.T, ls)
        u1 = A@x1
        return x1, u1
    
    def rProxG(x,u):
        x1, u1 = ProxG(x,u)
        x1 = 2*x1-x
        u1 = 2*u1-u
        return x1, u1
    
    T = 1
    N = A.shape[1]
    M = len(b)
    x = np.zeros((N, 1))
    x1 = np.zeros((N, 1))
    u1 = np.zeros((M, 1))
    error = tol+1
    
    pAlower = np.linalg.cholesky(np.eye(N) + A.T@A);
    mu1 = 1-mu
    
    while T<=MaxIt and error >=tol:
        pass
        # first step
        p,q = rProxF(x1,u1)
        
        p,q = rProxG(p,q)
        #print(p, q)
        x1 = mu1*x1 + mu*p
        u1 = mu1*u1 + mu*q

        # second step
        xnew,_ = ProxF(x1,u1)

        # update
        error = np.linalg.norm(x-xnew, ord=2)
        print(f"[Douglas-Rachford Opt] Iter: {T}, Tol: {tol:2.4E}, Err: {error:2.4E}")
        
        x = xnew
        T = T+1
    return x


def legendre_dictionary(u: np.ndarray, p: int, r: int) -> typ.Tuple[
    np.ndarray, np.ndarray, np.ndarray, 
    np.ndarray, np.ndarray, np.ndarray, 
    np.ndarray, np.ndarray, np.ndarray
    ]:
    """
    Legendre Dictionary Matrix

    TODO: build p = 2 case
    TODO: build the key names L

    returns: Dmon, Dleg, Ind1, Ind20, Ind11, Ind300, Ind210, Ind120, Ind11
    """
    m, n = u.shape

    if n >= 2*r + 1:
        umon = np.hstack([np.ones((m,1)), u[:, :r+1], u[:, n-r:n]])
        uleg = np.hstack([
            np.ones((m, 1)), 
            u[:,:r+1]*np.sqrt(3.0), 
            u[:,n-r:n]*np.sqrt(3)
        ])
        #l1 = l1
    else:
        umon = np.hstack([np.ones((m, 1)), u])
        uleg = np.hstack([np.ones((m, 1)), u*np.sqrt(3.0)])
        
    n = min(2*r+1, n)
        
    C = []
    for r in itertools.product(*[np.arange(1, n+2) for _ in range(p)]):
        C.append(r)
    C = np.array(C)[:, ::-1].T
    # sort columns in ascending order
    C = np.sort(C, axis=0)
    # remove duplicate rows
    C = np.unique(C, axis=1).T
    
    nD = C.shape[0]
    Dmon = np.ones((m,nD))
    Dleg = np.ones((m,nD))
    
    #print(umon.shape, Dmon.shape, nD)
    for ii in range(0, nD):
        for jj in range(p):
            Dmon[:,ii]
            umon[:,C[ii,jj]-1]
            Dmon[:,ii] = Dmon[:,ii] * umon[:,C[ii,jj]-1]
            Dleg[:,ii] = Dleg[:,ii] * uleg[:,C[ii,jj]-1]
    
    if p == 2:
        raise ValueError
    elif p == 3:
        # P2: (3*ui^2-1)*sqrt(5)/2
        N = int(n**2/2.0+ 3*n/2.0 + 1) 
        Ind20 = []
        for ii in range(2,n+2):
            ind = np.where((C[:N, 1:3] == (ii, ii)).all(axis=1))[0]
            Dleg[:,ind] = (Dleg[:,ind]-1.0)*np.sqrt(5.0)/2.0;
            Ind20 = [*Ind20, *ind]
        Ind1 = np.arange(1, n+1)
        Ind11 = np.arange(0, N)
        Ind11 = np.delete(Ind11, [0, *Ind1, *list(Ind20)], axis=0)
        
        #   P3: sqrt(15)*(3*ui^2-1)*uj for i~=j
        Ind210 = []
        for ii in range(2,n+2):
            ind = np.where((C[N:, :2] == (ii, ii)).all(axis=1))[0]
            for jj in range(len(ind)):
                Dleg[:, ind[jj]+N] = ( 3*u[:,ii-2]**2 - 1 ) * uleg[:,C[ind[jj]+N,2]-1] * np.sqrt(5.0)/2.0
            Ind210 = [*Ind210, *ind]
        Ind210 = np.sort(Ind210)
        Ind210 = np.unique(Ind210) + N
        
        Ind120 = []
        for ii in range(2, n+2):
            #ind = find(ismember(C(N+1:end,2:3),[1 1]*ii,'rows'));
            ind = np.where((C[N:, 1:3] == (ii, ii)).all(axis=1))[0]
            for jj in range(len(ind)):
                Dleg[:,ind[jj]+N] = ( 3*u[:,ii-2]**2 - 1 ) * uleg[:,C[ind[jj]+N,0] - 1] * np.sqrt(5.0)/2.0
            Ind120 = [*Ind120, *ind];
        Ind120 = np.sort(Ind120)
        Ind120 = np.unique(Ind120) + N 
        
         #   P3: (5*ui^3-3*ui)*sqrt(7)/2
        Ind300 = []
        for ii in range(2,n+2):
            #ind = find(ismember(C,[1 1 1]*ii,'rows'));
            ind = np.where((C == (ii, ii, ii)).all(axis=1))[0]
            #Dleg(:,ind) = ( 5*u(:,ii-1).^3 - 3*u(:,ii-1) ) * sqrt(7)/2;
            for idx in ind:
                Dleg[:, idx] = ( 5.0*u[:,ii-2]**3.0 - 3.0*u[:,ii-2] ) * np.sqrt(7.0)/2.0
            Ind300 = [*Ind300, *ind]
        #Ind210(find(ismember(Ind210,Ind300))) = [];
        #Ind120(find(ismember(Ind120,Ind300))) = [];
        Ind210 = np.delete(Ind210, np.where(np.isin(Ind210, Ind300))[0], axis=0)
        Ind120 = np.delete(Ind120, np.where(np.isin(Ind120, Ind300))[0], axis=0)

        # P3: sqrt(27)*ui*uj*uk
        Ind111 = np.arange(0, nD) 
        Ind111 = np.delete(Ind111, [1, *Ind1, *Ind20, *Ind11, *Ind300, *Ind210, *Ind120], axis=0)
        
        return Dmon, Dleg, Ind1, Ind20, Ind11, Ind300, Ind210, Ind120, Ind11
    else:
        raise ValueError
