# coding: utf-8

"""frensnel.py: Compute coefficients for RTE Fresnel boundary conditions"""

import os.path
import numpy as np
from math import factorial as fac
from scipy.special import lpmv
from scipy.special import eval_legendre
from scipy.special import gammaln
import scipy.integrate

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2021, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"

def _Rfres(n,mu):
    """Compute Fresnel reflection coefficient without checks

    :param n: Refractive index quotient (inner/outer)
    :param mu: Cosine of incitent angle to the normal
    :returns: Fresnel reflection coefficient for transition inner->outer

    """
    mu0 = np.sqrt(1.0-n*n*(1.0-mu*mu))
    f1 = (mu-n*mu0)/(mu+n*mu0)
    f2 = (mu0-n*mu)/(mu0+n*mu)
    return 0.5*f1*f1+0.5*f2*f2

def Rfres(n,mu=1.0):
    """Compute Fresnel reflection coefficient with check for total reflection

    :param n: Refractive index quotient (inner/outer)
    :param mu: Cosine of incitent angle to the normal
    :returns: Fresnel reflection coefficient for transition inner->outer

    """
    # critical angle
    muc = np.sqrt(n*n-1)/n if n > 1.0 else 0.0

    if mu <= muc: return 1.0
    return _Rfres(n,mu)

def calc_R_marshak(n,ms,ls,l):
    """Compute Fresnel coefficient for 3D Marshak boundary conditions
    If there is total reflection, the integral is split at the
    critical angle.

    :param n: Refractive index quotient (inner/outer)
    :param ms: Index m prime
    :param ls: Index l prime
    :param l: Index l
    :returns: Fresnel coefficient

    """

    # critical angle
    muc = np.sqrt(n*n-1)/n if n > 1.0 else 0.0

    vf = 0.5*np.sqrt((2*l+1)*(2*ls+1)*fac(l-ms)*fac(ls-ms)/(fac(l+ms)*fac(ls+ms)))

    def Rint_even(mu):
        """Integrand for even orders

        :param mu: Cosine of incitent angle to the normal
        :returns: Integrand at mu

        """
        return (1.0-_Rfres(n,mu))*lpmv(ms,l,mu)*lpmv(ms,ls,mu)

    def Rint_odd(mu):
        """Integrand for odd orders

        :param mu: Cosine of incitent angle to the normal
        :returns: Integrand at mu

        """
        return (1.0+_Rfres(n,mu))*lpmv(ms,l,mu)*lpmv(ms,ls,mu)

    def Rint_odd_TR(mu):
        """Integrand for odd orders and total reflection

        :param mu: Cosine of incitent angle to the normal
        :returns: Integrand at mu

        """
        return 2.0*lpmv(ms,l,mu)*lpmv(ms,ls,mu)

    odd = (l+ms)&1

    R2func = Rint_odd if odd else Rint_even

    R1 = 0.0
    R2 = 0.0

    # This part vanishes without total reflection or for even orders with
    # total reflection
    if muc > 0.0 and odd:
        R1 = scipy.integrate.quadrature(Rint_odd_TR,0.0,muc,maxiter=1000)[0]

    R2 = scipy.integrate.quadrature(R2func,muc,1.0,maxiter=1000)[0]

    return vf*(R1+R2)

def calc_R_marshak_planar(n,ls,l):
    """Compute Fresnel coefficient for planar symmetric Marshak boundary
    conditions. If there is total reflection, the integral is split at the
    critical angle.

    :param n: Refractive index quotient (inner/outer)
    :param ls: Index l prime
    :param l: Index l
    :returns: Fresnel coefficient

    """

    # critical angle
    muc = np.sqrt(n*n-1)/n if n > 1.0 else 0.0

    vf = 0.5*np.sqrt((2*l+1)*(2*ls+1))

    def Rint_even(mu):
        """Integrand for even orders

        :param mu: Cosine of incitent angle to the normal
        :returns: Integrand at mu

        """
        return (1.0-_Rfres(n,mu))*eval_legendre(l,mu)*eval_legendre(ls,mu)

    def Rint_odd(mu):
        """Integrand for odd orders

        :param mu: Cosine of incitent angle to the normal
        :returns: Integrand at mu

        """
        return (1.0+_Rfres(n,mu))*eval_legendre(l,mu)*eval_legendre(ls,mu)

    def Rint_odd_TR(mu):
        """Integrand for odd orders and total reflection

        :param mu: Cosine of incitent angle to the normal
        :returns: Integrand at mu

        """
        return 2.0*eval_legendre(l,mu)*eval_legendre(ls,mu)

    odd = l&1

    R2func = Rint_odd if odd else Rint_even

    R1 = 0.0
    R2 = 0.0

    if muc > 0.0 and odd:
        R1 = scipy.integrate.quadrature(Rint_odd_TR,0.0,muc,maxiter=1000)[0]

    R2 = scipy.integrate.quadrature(R2func,muc,1.0,maxiter=1000)[0]

    return vf*(R1+R2)

def calc_R_marshak_planar_1_analytic(ls,l):
    """Compute Fresnel coefficient for matched planar symmetric Marshak
    boundary (n=1). In this case, the involved integrals can be solved
    analytically. Therefore, no numerical integration is required.

    :param ls: Index l prime
    :param l: Index l
    :returns: Fresnel coefficient (n=1)

    """

    if (ls == l):
        return 0.5

    vf = np.sqrt((2*l+1)*(2*ls+1))/np.pi

    sin = {0 : 0.0, 1: 1.0, 2: 0.0, 3: -1.0}
    cos = {0 : 1.0, 1: 0.0, 2: -1.0, 3: 0.0}

    lA = gammaln(0.5*(ls+1.0)) - gammaln(0.5*(l+1.0)) + gammaln(0.5*(l+2.0)) - gammaln(0.5*(ls+2.0))
    A = np.exp(lA)

    t1 = A*sin[l&3]*cos[ls&3]
    t2 = sin[ls&3]*cos[l&3]/A

    return vf*(t1-t2)/((l-ls)*(l+ls+1))

def calc_Rphi(n):
    """Compute R_phi coefficient from diffusion theory PC
    boundary condition

    :param n: Refractive index quotient (inner/outer)
    :returns: R_phi

    """

    muc = np.sqrt(n*n-1)/n if n > 1.0 else 0.0

    def Rint(mu):

        return 2.0*mu*_Rfres(n,mu)

    R1 = muc*muc
    R2 = scipy.integrate.quad(Rint,muc,1.0)[0]

    return R1+R2

def calc_Rj(n):
    """Compute R_j coefficient from diffusion theory PC
    boundary condition

    :param n: Refractive index quotient (inner/outer)
    :returns: R_j

    """

    muc = np.sqrt(n*n-1)/n if n > 1.0 else 0.0

    def Rint(mu):

        return 3.0*mu*mu*_Rfres(n,mu)

    R1 = muc*muc*muc
    R2 = scipy.integrate.quad(Rint,muc,1.0)[0]

    return R1+R2

def calc_R_P3(n):
    """Compute and return all Marshak boundary coefficients for
    P3 solution.

    :param n: Refractive index quotient (inner/outer)
    :returns: Matrix of Marshak coefficients up to P3

    """

    R = np.zeros([9,4])

    count = 0

    for m in range(3):
        for ls in range(m+1,4,2):
            count2 = 0
            start = ((2*4-m+1)*m) // 2
            for l in range(m,4):
                R[start+count2,count] = calc_R_marshak(n,m,ls,l)
                count2 = count2 + 1
            count = count + 1

    return R

def calc_R_P1(n):
    """Compute and return all Marshak boundary coefficients for
    P1 solution.

    :param n: Refractive index quotient (inner/outer)
    :returns: Matrix of Marshak coefficients up to P1

    """

    R = np.empty([3,1])
    R[0,0] = calc_R_marshak(n,0,1,0)
    R[1,0] = calc_R_marshak(n,0,1,1)
    R[2,0] = 0.0

    return R

def save_R_marshak_planar(n,filename,NNmax):
    """Compute all Marshak coefficients for P(NN+1)
    solution and save them to file

    :param n: Refractive index quotient (inner/outer)
    :param filename: Name of output file
    :param NNmax: maximum half order

    """

    from progress.bar import IncrementalBar
    from datetime import timedelta
    
    if(os.path.isfile(filename)):
        print('File exists: n={} N={}'.format(n,NNmax))
        return

    Nmax = 2*NNmax+1

    R = np.zeros([Nmax+1,NNmax+1])

    class RBar(IncrementalBar):
        """ """
        suffix='%(percent).1f%% - ETA: %(remaining_time)s'
        @property
        def remaining_time(self):
            """ """
            return str(timedelta(seconds=self.eta))

    bar = RBar('Calculating: n={} N={}'.format(n,NNmax), max=NNmax+1)

    for i in range(NNmax+1):
        for j in np.arange(i,NNmax+1):
            R[2*i,j] = calc_R_marshak_planar(n,2*j+1,2*i)
        for j in np.arange(2*i+1,Nmax+1):
            R[j,i] = calc_R_marshak_planar(n,2*i+1,j)
        R[2*i+1,i+1:] = R[2*i+3::2,i]
        bar.next()

    np.savez(filename,NNmax=NNmax,Rf=R)
    bar.finish()

def save_R_marshak_planar_1_analytic(filename,NNmax):
    """Compute coefficients for matched planar symmetric Marshak
    boundary (n=1) and save them to file. Since no integrals
    must be solved numerically, this is much faster than the
    generic version.

    :param filename: Name of output file
    :param NNmax: maximum half order

    """

    from progress.bar import IncrementalBar
    from datetime import timedelta
    
    if(os.path.isfile(filename)):
        print('File exists: n=1.0 N={}'.format(NNmax))
        return

    Nmax = 2*NNmax+1

    R = np.empty([Nmax+1,NNmax+1])

    class RBar(IncrementalBar):
        """ """
        suffix='%(percent).1f%% - ETA: %(remaining_time)s'
        @property
        def remaining_time(self):
            """ """
            return str(timedelta(seconds=self.eta))

    bar = RBar('Calculating: n=1.0 N={}'.format(NNmax), max=NNmax+1)

    for i in range(NNmax+1):
        for j in np.arange(i,NNmax+1):
            R[2*i,j] = calc_R_marshak_planar_1_analytic(2*j+1,2*i)
        for j in np.arange(2*i+1,Nmax+1):
            R[j,i] = calc_R_marshak_planar_1_analytic(2*i+1,j)
        R[2*i+1,i+1:] = R[2*i+3::2,i]
        bar.next()

    np.savez(filename,NNmax=NNmax,Rf=R)
    bar.finish()

def load_R_planar(filename,NN):
    """Load Marshak coefficients from file and return
    the part for maxium half order NN as matrix

    :param filename: Name of input file
    :param NN: maximum half order

    """

    N = 2*NN+1

    Rfile = np.load(filename)
    NNmax = Rfile['NNmax']

    if NN > NNmax:
        raise ValueError("Requested order exceeds maximum order in file.")

    R = Rfile['Rf']

    return R[:N+1,:NN+1]
