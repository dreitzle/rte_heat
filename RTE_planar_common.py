# coding: utf-8

"""
RTE_planar_common.py: Planar PN eigenvalues, eigenvectors
and particular solution moments
"""

# Planar RTE

import numpy as np
from scipy.linalg.lapack import dsterf
from scipy.linalg.lapack import dgtsv
from scipy.special import eval_legendre

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2021, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"

class calc_RTE:

    def __init__(self,musp,mua,g,NN,mu0=1.0):

        self.musp = musp
        self.mua = mua
        self.g = g
        self.NN = NN

        self.N = 2*NN+1
        
        if hasattr(g, '__len__'):
            
            self.mus = musp / (1.0-g[1])
    
            self.gm = np.zeros(self.N+2)
            nel = min(g.size,self.N+2)
            self.gm[:nel] = g[:nel]

        else:
            
            self.mus = musp / (1.0-g)

            self.gm = np.empty(self.N+2)
            self.gm[0] = 1.0
            for i in range(1,self.N+2):
                self.gm[i] = self.gm[i-1]*g

        self.alpha = self.gm[self.N+1]
        self.mut = mua + (1.0-self.alpha)*self.mus

        self.sigma = mua + self.mus*(1.0-self.gm[:-1])

        l2 = np.square(np.arange(self.N) + 1)
        self.beta = np.sqrt(l2/((4*l2-1)*self.sigma[:-1]*self.sigma[1:]))

        self.ew = self.calc_ew(self.beta)

        self.ev = self.calc_ev(self.beta,self.ew)

        if mu0 is not None:
            self.mu0 = mu0
            self.eps = self.calc_eps(mu0)
        else:
            self.mu0 = None
            self.eps = None

    def calc_ew(self,beta):

        if self.NN == 0:
            return np.atleast_1d(beta[0])

        d = np.empty(self.NN+1)
        d[0] = beta[0]**2
        d[1:] = beta[1::2]**2 + beta[2::2]**2
        e = beta[0:-1:2]*beta[1:-1:2]

        dsterf(d,e,overwrite_d=1,overwrite_e=1)

        return np.sqrt(d)

    def calc_ev(self,beta,ew):

        N = self.N

        ev = np.empty([ew.size,N+1])
        ev[:,N] = 1.0e250*beta[-1]/ew
        ev[:,N-1] = 1.0e250

        for i in reversed(range(0,N-1)):
            nx = (ew*ev[:,i+1] - beta[i+1]*ev[:,i+2])/beta[i]
            gr = 1.0e250/nx
            idx = gr < 1.0
            scale = np.where(idx, gr , 1.0)
            ev[:,i+1:N+1] = ev[:,i+1:N+1] * scale[:, np.newaxis]
            ev[:,i] = np.where(idx, 1.0e250 , nx)

        return ev/np.sqrt(self.sigma)

    def calc_eps(self,mu0):

        N = self.N

        self.mu0 = mu0
        sigma = self.sigma.copy()

        l = np.arange(N)
        avec = -self.mut/mu0*(l+1)/np.sqrt((2*l+1)*(2*l+3))
        eps = self.mus/mu0*(self.gm[:-1]-self.alpha) \
            *np.sqrt((2*np.arange(N+1)+1)/(4*np.pi)) \
            *eval_legendre(np.arange(N+1),mu0)

        dgtsv(avec, sigma, avec, eps, overwrite_dl = 0, overwrite_d = 1, overwrite_du = 0, overwrite_b = 1)

        self.eps = eps

        return eps

    def get_res(self):
        return self.mut,self.ew,self.ev,self.eps
