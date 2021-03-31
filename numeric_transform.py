#!/usr/bin/env python
# coding: utf-8

"""numeric_transform.py: Numeric inverse Laplace and Fourier transforms."""

import unittest
import numpy as np
from scipy.special import jn_zeros
from scipy.special import j0
from scipy.special import j1
from scipy.special import k0
from functools import partial

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2021, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"


def ILT_hyperbolic(func, t, N, alpha=0.0):
    """
    Hyperbolic contour integratration ILT

    References:
     J. A. C. Weideman, L. Trefethen, Parabolic and hyperbolic contours
       for computing the bromwich integral, Mathematics of Computation 76
       (2007) 1341–1356. doi:10.1090/S0025-5718-07-01945-X
     A. Liemert, A. Kienle, Application of the laplace transform in time-
       domain optical spectroscopy and imaging, Journal of Biomedical Optics
       20 (2015) 110502. doi:10.1117/1.JBO.20.11.110502

    :param func: Function in Laplace space.
    :param t: Transform points (usually time values).
    :param N: Number of points in Laplace space. Usually 10-20 are sufficient.
    :param alpha: Convergence abscissa.
    :returns: Transform result at t.
    """
    tc = 4.492075287*N/1e4

    phi = 1.172104229
    h = 1.081792140/N
    mu = 4.492075287*N / np.where(t < tc, tc, t)

    # integration contour
    p = np.transpose(((np.arange(N) + 0.5)*h)[np.newaxis,np.newaxis])
    s = alpha + mu + 1j*mu*np.sinh(p+1j*phi)
    ds = 1j*mu*np.cosh(p+1j*phi)

    # compute integral
    F = np.transpose(func(s)[np.newaxis,np.newaxis])

    vec = np.imag(F*np.exp(s*t)*ds)

    return h/np.pi*np.sum(vec, axis=0)


def ILT_brancik(func, t, N, P, alpha=0.0):
    """
    Accelerated complex fourier series ILT

    References:
    L. Brancik, Numerical inverse laplace transforms for electrical
      engineering simulation, in: K. Perutka (Ed.), MATLAB for Engineers,
      InTech, London, 2011, Ch. 3, pp. 51 – 74.

    :param func: Function in Laplace space.
    :param t: Transform points (usually time values).
    :param N: Number of points in Laplace space. Usually 30-40
        are sufficient.
    :param P: Extra points for convergence acceleration. Usually
        4-6 are sufficient.
    :param alpha: Convergence abscissa.
    :returns: Transform result at t.
    """
    prec = 16.0

    tau = 2.0*np.amax(t)

    if(tau < 1e-12):
        tau = 1e-12

    domega = 2.0*np.pi / tau
    r = (alpha + prec/tau)

    # Evaluation points
    omega = 1j*domega*np.arange(N+2*P+1)
    sk = r + omega
    pts = func(sk)

    # exponentials
    mexp = np.exp(np.outer(t, omega[:N]))

    # compute finite fourier series
    res1 = np.dot(mexp, pts[:N])

    # compute e and q arrays
    e = np.zeros(2*P+1, dtype=complex)
    q = pts[N+1:] / pts[N:-1]

    for i in range(1, P):
        e[i:2*P-i+1] += (q[i:2*P-i+1] - q[i-1:2*P-i])
        q[i:2*P-i] *= (e[i+1:2*P-i+1] / e[i:2*P-i])

    e[P] += (q[P] - q[P-1])

    # compute correction sum
    z = mexp[:, 1]
    qdvf = np.exp(omega[N]*np.array(t))

    Am1 = np.zeros_like(z)
    Bm1 = np.ones_like(z)
    A = Bm1*pts[N]
    B = Bm1

    for i in range(1, P+1):
        tmp1 = A - q[i-1]*z*Am1
        tmp2 = tmp1 - e[i]*z*A

        A = tmp2
        Am1 = tmp1

        tmp1 = B - q[i-1]*z*Bm1
        tmp2 = tmp1 - e[i]*z*B

        B = tmp2
        Bm1 = tmp1

    res2 = qdvf*A/B

    # calculate result

    res = 2.0*(np.real(res1) + np.real(res2)) - np.real(pts[0])

    return res*np.exp(np.multiply(r, t)) / tau

def IFT_pts(Nq,Np):

    (phi_int, dphi) = np.linspace(0.0, 2.0*np.pi, Np,
                        endpoint=False, retstep=True)

    r_limit = 2.8

    (r, dr) = np.linspace(-1.5*r_limit, r_limit, Nq, retstep=True)

    q_int = np.exp(2.0*np.sinh(r))

    qdq_int = 2.0*q_int*q_int*np.cosh(r)

    return q_int,phi_int,r,dr,dphi,qdq_int

def IFT(func, profile, x, y, Nq, Np):
    """
    Direct integration DE substitution IFT

    References:
    L. N. Trefethen, J. A. C. Weideman, The exponentially con-
      vergent trapezoidal rule, SIAM Review 56 (2014) 385 – 458.
      doi:10.1137/130932132
    H. Takahasi, M. Mori, Double exponential formulas for numerical
      integration, Publications of the Research Institute for
      Mathematical Sciences 9 (1974) 721 – 741.
    W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery,
      Numerical Recipes 3rd Edition: The Art of Scientific Computing,
      3rd Edition, Cambridge University Press, New York, 2007.

    :param func: Function in Laplace space.
    :param profile: Beam profile
    :param x: Transform at (x,y)
    :param y: Transform at (x,y)
    :param Nq: Number of radial points
    :param Np: Number of angular points
    :returns: Transform result at (x,y).
    """
    def diff_wrapper(q, phi):
        d = np.exp(1j*q*(x*np.cos(phi)+y*np.sin(phi)))
        return func(q, phi)*d*profile(q, phi)

    q_int,phi_int,r,dr,dphi,qdq_int = IFT_pts(Nq,Np)

    F = diff_wrapper(q_int[:, None], phi_int[None, :])

    F = np.sum(F, axis=-1)
    F = np.sum(F*qdq_int, axis=-1)

    return F*dr*dphi/(4.0*np.pi*np.pi)


def IHT(func, profile, dist, Nq):

    def psi(t):
        p1 = np.exp(t)
        m1 = np.exp(-t)
        e1 = np.exp(-0.5*np.pi*p1)
        e2 = np.exp(-0.5*np.pi*m1)
        t1 = (e2-e1)/(e2+e1)
        return t*t1
    def dpsi(t):
        p1 = np.exp(t)
        m1 = np.exp(-t)
        e1 = np.exp(-0.5*np.pi*p1)
        e2 = np.exp(-0.5*np.pi*m1)
        t1 = (e2-e1)/(e2+e1)
        t2 = np.pi*t*(p1+m1)*e1*e2/(e1+e2)**2
        return t1+t2

    def diff_wrapper(q):
        return func((q/dist),0.0)*q/(dist*dist)*profile(q/dist, 0.0)

    xi = jn_zeros(0, Nq)/np.pi

    #~ w = y0(np.pi*xi)/j1(np.pi*xi)
    t = j1(np.pi*xi)
    w = 2.0/(np.pi*np.pi*xi*t*t)

    h = np.pi/Nq

    q = np.pi/h*psi(h*xi)

    d = dpsi(h*xi)

    f = w*diff_wrapper(q)*j0(q)*d

    return 0.5*np.sum(f)

def IHT2(func, profile, dist, Nq):

    def diff_wrapper(q):
        d = j0(q*dist)
        return func(q,0.0)*d*profile(q, 0.0)

    r_limit = 2.6
    (r, dr) = np.linspace(-1.5*r_limit, r_limit, Nq, retstep=True)

    q_int = np.exp(0.5*np.pi*np.sinh(r))
    qdq_int = 0.5*np.pi*q_int*q_int*np.cosh(r)
    #~ q_int = np.exp(r-np.exp(-r))
    #~ qdq_int = q_int*q_int*(1.0+np.exp(-r))

    F = diff_wrapper(q_int)
    F = np.sum(F*qdq_int)

    return F*dr/(2.0*np.pi)

def ILT_IFT_hyperbolic(func, profile, x, y, t, Ns, Nq, Np, alpha=0.0):
    """
    Direct integration IFT + hyperbolic ILT

    :param func: Function in Laplace space.
    :param profile: Beam profile
    :param x: Transform at (x,y)
    :param y: Transform at (x,y)
    :param t: Transform at t
    :param Ns: Nomber of points in Laplace space
    :param Nq: Number of radial points
    :param Np: Number of angular points
    :param alpha: Laplace convergence abszissa
    :returns: Transform result at (x,y) and t.
    """

    tc = 4.492075287*Ns/1e4

    phi = 1.172104229
    h = 1.081792140/Ns
    mu = 4.492075287*Ns/np.where(t < tc, tc, t)

    p = np.transpose(((np.arange(Ns) + 0.5)*h))

    s_int = alpha + mu + 1j*mu*np.sinh(p+1j*phi)
    ds_int = 1j*mu*np.cosh(p+1j*phi)

    q_int,phi_int,r,dr,dphi,qdq_int = IFT_pts(Nq,Np)

    d = np.exp(1j*q_int[:, None]*(x*np.cos(phi_int[None, :])+y*np.sin(phi_int[None, :])))
    prof = profile(q_int[:, None], phi_int[None, :])

    res = 0.0

    for i in range(Ns):
        F = func(s_int[i], q_int[:, None], phi_int[None, :])*d*prof
        F = np.sum(F, axis=-1)
        F = np.sum(F*qdq_int, axis=-1)
        res += np.imag(F*np.exp(s_int[i]*t)*ds_int[i])

    return res*dr*dphi*h/(4.0*np.pi*np.pi*np.pi)


def ILT_FFT_hyperbolic(funcxy, profilexy, n, d, t, Ns, alpha=0.0):
    """
    2D FFT + hyperbolic ILT

    :param funcxy: Function in Laplace space (Carthesian coordinates)
    :param profilexy: Beam profile (Carthesian coordinates)
    :param n: FFT length
    :param d: FFT sample spacing
    :param t: Transform at t
    :param Ns: Number of points in Laplace space
    :param alpha: Laplace convergence abszissa
    :returns: Transform result at FFT points and t.
    """
    def diff_wrapper(s, qx, qy):
        return funcxy(s, qx, qy)*profilexy(qx, qy)

    freq = 2.0*np.pi*np.fft.fftfreq(n, d)

    phi = 1.172104229
    h = 1.081792140/Ns
    mu = 4.492075287*Ns/t

    p = np.transpose(((np.arange(Ns) + 0.5)*h))

    s_int = alpha + mu + 1j*mu*np.sinh(p+1j*phi)
    ds_int = 1j*mu*np.cosh(p+1j*phi)

    res = np.zeros((n, n))

    for i in range(Ns):
        F = diff_wrapper(s_int[i], freq[:, None], freq[None, :])
        res += np.imag(np.fft.ifft2(F)*np.exp(s_int[i]*t)*ds_int[i])

    return h/np.pi*np.fft.fftshift(res)/(d*d)


def ILT_IFT_brancik(func, profile, x, y, t, Ns, Nsp, Nq, Np, Q=1.0,
                    add_laplace=None, Ql=1.0, alpha=0.0):
    """
    Direct integration IFT + complex fourier series ILT

    :param func: Function in Laplace space (Carthesian coordinates)
    :param profile: Beam profile (Carthesian coordinates)
    :param x: Transform at (x,y)
    :param y: Transform at (x,y)
    :param t: Transform at t
    :param Ns: Number of points in Laplace space
    :param Nsp: Extra Laplace space points for convergence acceleration.
    :param Nq: Number of radial points
    :param Np: Number of angular points
    :param Q: Source strength
    :param add_laplace: Additional source independent of x and y
    :param Ql: Additional source strength
    :param alpha: Laplace convergence abszissa
    :returns: Transform result (x,y) and t.
    """
    prec = 16.0

    tau = 2.0*np.amax(t)

    if(tau < 1e-12):
        tau = 1e-12

    domega = 2.0*np.pi / tau
    r = (alpha + prec/tau)

    # Evaluation points
    omega = 1j*domega*np.arange(Ns+2*Nsp+1)
    sk = r + omega

    # Invert 2D fourier transform to get points in s-space
    pts = np.empty_like(sk)
    for idx, s in np.ndenumerate(sk):
        fn = partial(func, s)
        pts[idx] = Q*IFT(fn, profile, x, y, Nq, Np)

    # Add additional function
    if add_laplace is not None:
        pts += Ql*add_laplace(sk)

    # exponentials
    mexp = np.exp(np.outer(t, omega[:Ns]))

    # compute finite fourier series
    res1 = np.dot(mexp, pts[:Ns])

    # compute e and q arrays
    e = np.zeros(2*Nsp+1, dtype=complex)
    q = pts[Ns+1:] / pts[Ns:-1]

    for i in range(1, Nsp):
        e[i:2*Nsp-i+1] += (q[i:2*Nsp-i+1] - q[i-1:2*Nsp-i])
        q[i:2*Nsp-i] *= (e[i+1:2*Nsp-i+1] / e[i:2*Nsp-i])

    e[Nsp] += (q[Nsp] - q[Nsp-1])

    # compute correction sum
    z = mexp[:, 1]
    qdvf = np.exp(omega[Ns]*np.array(t))

    Am1 = np.zeros_like(z)
    Bm1 = np.ones_like(z)
    A = Bm1*pts[Ns]
    B = Bm1

    for i in range(1, Nsp+1):
        tmp1 = A - q[i-1]*z*Am1
        tmp2 = tmp1 - e[i]*z*A

        A = tmp2
        Am1 = tmp1

        tmp1 = B - q[i-1]*z*Bm1
        tmp2 = tmp1 - e[i]*z*B

        B = tmp2
        Bm1 = tmp1

    res2 = qdvf*A/B

    # result

    res = 2.0*(np.real(res1) + np.real(res2)) - np.real(pts[0])

    return res*np.exp(np.multiply(r, t)) / tau


def ILT_IHT_brancik(func, profile, dist, t, Ns, Nsp, Nq, Q=1.0,
                    add_laplace=None, Ql=1.0, alpha=0.0):
    """
    Direct integration IFT + complex fourier series ILT

    :param func: Function in Laplace space (Carthesian coordinates)
    :param profile: Beam profile (Carthesian coordinates)
    :param x: Transform at (x,y)
    :param y: Transform at (x,y)
    :param t: Transform at t
    :param Ns: Number of points in Laplace space
    :param Nsp: Extra Laplace space points for convergence acceleration.
    :param Nq: Number of radial points
    :param Np: Number of angular points
    :param Q: Source strength
    :param add_laplace: Additional source independent of x and y
    :param Ql: Additional source strength
    :param alpha: Laplace convergence abszissa
    :returns: Transform result (x,y) and t.
    """
    prec = 16.0

    tau = 2.0*np.amax(t)

    if(tau < 1e-12):
        tau = 1e-12

    domega = 2.0*np.pi / tau
    r = (alpha + prec/tau)

    # Evaluation points
    omega = 1j*domega*np.arange(Ns+2*Nsp+1)
    sk = r + omega

    # Invert 2D fourier transform to get points in s-space
    pts = np.empty_like(sk)
    for idx, s in np.ndenumerate(sk):
        fn = partial(func, s)
        pts[idx] = Q*IHT(fn, profile, dist, Nq)

    # Add additional function
    if add_laplace is not None:
        pts += Ql*add_laplace(sk)

    # exponentials
    mexp = np.exp(np.outer(t, omega[:Ns]))

    # compute finite fourier series
    res1 = np.dot(mexp, pts[:Ns])

    # compute e and q arrays
    e = np.zeros(2*Nsp+1, dtype=complex)
    q = pts[Ns+1:] / pts[Ns:-1]

    for i in range(1, Nsp):
        e[i:2*Nsp-i+1] += (q[i:2*Nsp-i+1] - q[i-1:2*Nsp-i])
        q[i:2*Nsp-i] *= (e[i+1:2*Nsp-i+1] / e[i:2*Nsp-i])

    e[Nsp] += (q[Nsp] - q[Nsp-1])

    # compute correction sum
    z = mexp[:, 1]
    qdvf = np.exp(omega[Ns]*np.array(t))

    Am1 = np.zeros_like(z)
    Bm1 = np.ones_like(z)
    A = Bm1*pts[Ns]
    B = Bm1

    for i in range(1, Nsp+1):
        tmp1 = A - q[i-1]*z*Am1
        tmp2 = tmp1 - e[i]*z*A

        A = tmp2
        Am1 = tmp1

        tmp1 = B - q[i-1]*z*Bm1
        tmp2 = tmp1 - e[i]*z*B

        B = tmp2
        Bm1 = tmp1

    res2 = qdvf*A/B

    # result

    res = 2.0*(np.real(res1) + np.real(res2)) - np.real(pts[0])

    return res*np.exp(np.multiply(r, t)) / tau


# Tests

class test_ILT_hyperbolic(unittest.TestCase):

    def test_ILT_hyperbolic_single_point_zero(self):

        alpha = 0.8

        def F_exp(s):
            return 1.0/(s+alpha)

        try:
            ILT_hyperbolic(F_exp, 0.0, 25)
        except ZeroDivisionError:
            # should not happen
            self.fail("Division by zero.")

    def test_ILT_hyperbolic_single_point_exp(self):

        alpha = 0.8
        t1 = 0.001
        t2 = 1.2

        def F_exp(s):
            return 1.0/(s+alpha)

        f1_ana = np.exp(-alpha*t1)
        f1_num = ILT_hyperbolic(F_exp, t1, 15, -alpha)

        f2_ana = np.exp(-alpha*t2)
        f2_num = ILT_hyperbolic(F_exp, t2, 15, -alpha)

        # accuracy is low only for very small t
        self.assertAlmostEqual(f1_num, f1_ana, delta=5e-4)
        self.assertAlmostEqual(f2_num, f2_ana, delta=1e-10)

    def test_ILT_hyperbolic_vector_cosh(self):

        alpha = 0.8
        Np = 11
        t = np.linspace(0.01, 4, num=Np)

        def F_cosh(s):
            return s/(s*s - alpha*alpha)

        f_ana = np.cosh(alpha*t)
        f_num = ILT_hyperbolic(F_cosh, t, 25, alpha)

        for x in range(Np):
            self.assertAlmostEqual(f_num[x], f_ana[x], delta=1e-10)


class test_ILT_brancik(unittest.TestCase):

    def test_ILT_brancik_single_point_zero(self):

        alpha = 0.8

        def F_exp(s):
            return 1.0/(s+alpha)

        try:
            f_num = ILT_brancik(F_exp, 0.0, 100, 5, -alpha)
        except ZeroDivisionError:
            # should not happen
            self.fail("Division by zero.")

        # Converges to 0.5 instead of 1.0 (jump)
        # Also, convergence is slow here
        self.assertAlmostEqual(f_num, 0.5, delta=1e-2)

    def test_ILT_brancik_single_point_exp(self):

        alpha = 0.8
        t1 = 0.001
        t2 = 1.2

        def F_exp(s):
            return 1.0/(s+alpha)

        f1_ana = np.exp(-alpha*t1)
        f1_num = ILT_brancik(F_exp, t1, 50, 5, -alpha)

        f2_ana = np.exp(-alpha*t2)
        f2_num = ILT_brancik(F_exp, t2, 50, 5, -alpha)

        self.assertAlmostEqual(f1_num, f1_ana, delta=1e-5)
        self.assertAlmostEqual(f2_num, f2_ana, delta=1e-7)

    def test_ILT_brancik_vector_cosh(self):
        alpha = 0.8
        Np = 11
        t = np.linspace(1, 4, num=Np)

        def F_cosh(s):
            return s/(s*s - alpha*alpha)

        f_ana = np.cosh(alpha*t)
        f_num = ILT_brancik(F_cosh, t, 50, 5, alpha)

        for x in range(Np):
            self.assertAlmostEqual(f_num[x], f_ana[x], delta=1e-5)


class test_IFT(unittest.TestCase):

    def test_IFT_single_point_gauss(self):

        sigmax = 2.3
        sigmay = 1.1

        def f_gauss(x, y):
            vf = 1.0/(2.0*np.pi*sigmax*sigmay)
            e = np.exp(-0.5*(x*x/(sigmax*sigmax)+y*y/(sigmay*sigmay)))
            return vf*e

        def F_gauss(q, phiq):
            qx = q*np.cos(phiq)
            qy = q*np.sin(phiq)
            return np.exp(-0.5*(sigmax*sigmax*qx*qx+sigmay*sigmay*qy*qy))

        def prof_id(q, phi):
            return 1.0

        f_ana = f_gauss(1.3, 0.5)
        f_num = IFT(F_gauss, prof_id, 1.3, 0.5, 90, 60)

        self.assertAlmostEqual(f_num, f_ana, delta=1e-10)

class test_IHT(unittest.TestCase):

    def test_IHT_single_point_gauss(self):

        import matplotlib.pyplot as plt

        def f(r):
            return k0(r)

        def F(q, phiq):
            return 2.0*np.pi/(1+q**2)

        def prof_id(q, phi):
            rw = 0.0
            return np.exp(-0.25*q*q*rw*rw)

        r = np.linspace(0.001,10.0,500)

        res1 = np.zeros_like(r)
        res2 = np.zeros_like(r)

        for i in range(500):
            res1[i] = IHT(F, prof_id, r[i], 300)
            res2[i] = IHT2(F, prof_id, r[i], 300)

        res_ana = f(r)
        #~ plt.plot(r,res1)
        #~ plt.plot(r,res2)
        plt.plot(r,np.abs(res1-res_ana)/res_ana)
        plt.plot(r,np.abs(res2-res_ana)/res_ana)
        plt.plot(r,res_ana)
        plt.yscale('log')
        #~ plt.ylim(1e-7,1.0)
        plt.show()

class test_ILT_IFT(unittest.TestCase):

    sigmax = 2.3
    sigmay = 1.1
    alpha = 0.2

    @classmethod
    def f(cls, x, y, t):
        p1 = 1.0/(2.0*np.pi*cls.sigmax*cls.sigmay) \
                * np.exp(-0.5*(x*x/(cls.sigmax*cls.sigmax)
                         + y*y/(cls.sigmay*cls.sigmay)))
        p2 = 1.0 - np.exp(-cls.alpha*t)
        return p1*p2

    @classmethod
    def F(cls, s, q, phiq):
        qx = q*np.cos(phiq)
        qy = q*np.sin(phiq)
        p1 = np.exp(-0.5*(cls.sigmax*cls.sigmax*qx*qx
                    + cls.sigmay*cls.sigmay*qy*qy))
        p2 = cls.alpha/(s*(s+cls.alpha))
        return p1*p2

    @classmethod
    def prof_id(cls, q, phiq):
        return 1.0

    def test_ILT_IFT_single_point_hyperbolic(self):
        f_ana = test_ILT_IFT.f(1.1, 0.5, 1.3)
        f_num = ILT_IFT_hyperbolic(test_ILT_IFT.F, test_ILT_IFT.prof_id,
                                   1.1, 0.5, 1.3, 30, 80, 40)
        self.assertAlmostEqual(f_num, f_ana, delta=1e-8)

    def test_ILT_IFT_single_point_brancik(self):
        f_ana = test_ILT_IFT.f(1.1, 0.5, 1.3)
        f_num = ILT_IFT_brancik(test_ILT_IFT.F, test_ILT_IFT.prof_id,
                                1.1, 0.5, 1.3, 30, 4, 80, 40)
        self.assertAlmostEqual(f_num, f_ana, delta=1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
