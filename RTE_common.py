#!/usr/bin/env python3
# coding: utf-8

"""RTE_common.py: 3D P1 and P3 solution"""

import numpy as np

from fresnel import calc_R_P3
from fresnel import calc_R_P1

from scipy.special import sph_harm as Ylm
from scipy.linalg.lapack import zgesv

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2021, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"

class RTE_P3:

    def __init__(self,musp,mua,g,mu0=1.0):

        self.mua = mua
        self.mus = musp / (1.0-g)
        self.gm = np.array([1.0,g,g**2,g**3,g**4])
        self.alpha = self.gm[4]
        self.mut = mua + (1.0-self.alpha)*self.mus
        self.mut2 = self.mut*self.mut

        self.s0 = mua
        self.s1 = mua + self.mus*(1.0-self.gm[1])
        self.s2 = mua + self.mus*(1.0-self.gm[2])
        self.s3 = mua + self.mus*(1.0-self.gm[3])

        self.s23 = self.s2*self.s3
        self.s123 = self.s1*self.s23

        a = 3.0/2.0*self.s0*self.s1 + 28.0/18.0*self.s0*self.s3 + 35.0/18.0*self.s23
        b = 35.0/3.0*self.s0*self.s123
        c = np.sqrt(a*a-b)

        self.D = np.empty(4)

        self.D[0] = a+c
        self.D[1] = a-c
        self.D[2] = 35.0*self.s123 / (8.0*self.s1+7.0*self.s3)
        self.D[3] = 7.0*self.s23

        self.mu0 = mu0

    def get_res(self,q,phiq=np.array([0.0],ndmin=2),squeeze=False):

        q.shape += (1,) * (2 - q.ndim)

        s0,s1,s2,s3 = self.s0,self.s1,self.s2,self.s3
        s23,s123 = self.s23,self.s123
        gm,D = self.gm,self.D
        mus,mut,mut2,alpha = self.mus,self.mut,self.mut2,self.alpha
        mu0 = self.mu0

        iq = 1j*q
        q2 = q*q

        # Eigenvalues
        W = np.empty((4,)+q.shape)
        W[0] = np.sqrt(q2+D[0])
        W[1] = np.sqrt(q2+D[1])
        W[2] = np.sqrt(q2+D[2])
        W[3] = np.sqrt(q2+D[3])

        # Eigenvector components (last component is not needed)
        V = np.zeros(((4,9)+q.shape),dtype=np.complex)

        V[0,0] = np.sqrt(7.0)/42.0*(27.0*s1+28.0*s3)*D[0] - 2.5*np.sqrt(7.0)*s123
        V[0,1] = 1.0/np.sqrt(84.0)*(9.0*D[0]-35.0*s23)*W[0]
        V[0,2] = -np.sqrt(35.0)/6.0*s3*(2.0*D[0]+3.0*q2)
        V[0,3] = -0.5*(2.0*D[0]+5.0*q2)*W[0]
        V[0,4] = 1j*q/np.sqrt(168.0)*(9.0*D[0]-35.0*s23)
        V[0,5] = -1j*q*np.sqrt(35.0/6.0)*s3*W[0]
        V[0,6] = -1j*q*np.sqrt(3.0)/4.0*(4.0*D[0]+5.0*q2)
        V[0,7] = 0.5*np.sqrt(35.0/6.0)*s3*q2
        V[0,8] = 1.5*np.sqrt(5.0/6.0)*q2*W[0]
        #V[0,9] = 1j*np.sqrt(5.0)/4.0*q*q2

        V[1,0] = np.sqrt(7.0)/42.0*(27.0*s1+28.0*s3)*D[1] - 2.5*np.sqrt(7.0)*s123
        V[1,1] = 1.0/np.sqrt(84.0)*(9.0*D[1]-35.0*s23)*W[1]
        V[1,2] = -np.sqrt(35.0)/6.0*s3*(2.0*D[1]+3.0*q2)
        V[1,3] = -0.5*(2.0*D[1]+5.0*q2)*W[1]
        V[1,4] = 1j*q/np.sqrt(168.0)*(9.0*D[1]-35.0*s23)
        V[1,5] = -1j*q*np.sqrt(35.0/6.0)*s3*W[1]
        V[1,6] = -1j*q*np.sqrt(3.0)/4.0*(4.0*D[1]+5.0*q2)
        V[1,7] = V[0,7]
        V[1,8] = 1.5*np.sqrt(5.0/6.0)*q2*W[1]
        #V[1,9] = 1j*np.sqrt(5.0)/4.0*q*q2

        #V[2,0] = 0.0
        V[2,1] = 1j*q/np.sqrt(28.0)*(35.0*s23-8.0*D[2])
        V[2,2] = 1j*q*np.sqrt(105.0)/2.0*s3*W[2]
        V[2,3] = 1j*q*np.sqrt(3.0)/2.0*(4.0*D[2]+5.0*q2)
        V[2,4] = np.sqrt(14.0)/28.0*(8.0*D[2]-35.0*s23)*W[2]
        V[2,5] = -np.sqrt(35.0/8.0)*s3*(2.0*q2+D[2])
        V[2,6] = -0.25*W[2]*(4.0*D[2]+15.0*q2)
        V[2,7] = -1j*q*W[2]*np.sqrt(35.0/8.0)*s3
        V[2,8] = -0.5j*np.sqrt(2.5)*q*(2.0*D[2]+3.0*q2)
        #V[2,9] = np.sqrt(15.0)/4.0*q2*W[2]

        #V[3,0] = 0.0
        #V[3,1] = 0.0
        V[3,2] = np.sqrt(10.5)*q2*s3
        V[3,3] = np.sqrt(7.5)*W[3]*q2
        #V[3,4] = 0.0
        V[3,5] = 1j*q*W[3]*np.sqrt(7.0)*s3
        V[3,6] = 0.5j*np.sqrt(2.5)*(2.0*D[3]+3.0*q2)*q
        V[3,7] = -0.5*(2.0*D[3]+q2)*np.sqrt(7.0)*s3
        V[3,8] = -0.5*W[3]*(2.0*D[3]+3.0*q2)
        #V[3,9] = -1j*q*np.sqrt(6.0)/4.0*(2.0*D[3]+q2)
        
        # Partial analytic LU decomposition for particular solution system

        muc = mut/mu0 + iq*np.sqrt(1.0-mu0*mu0)/mu0*np.cos(phiq)
        muc2 = muc*muc

        h3m1q = 3.0*muc2 + q2
        h1m3q = muc2 + 3.0*q2
        h3m4q = 3.0*muc2 + 4.0*q2

        h11 = -muc/np.sqrt(5.0)
        h12 = iq/np.sqrt(30.0)
        h13 = -iq/np.sqrt(5.0)
        h14 = -iq/np.sqrt(6.0)
        h17 = s1

        h21 = -iq*np.sqrt(6.0/7.0)/muc
        h22 = -h3m1q/(np.sqrt(35.0)*muc)
        h23 = np.sqrt(6.0/35.0)/muc*q2
        h24 = q2/(np.sqrt(7.0)*muc)
        h27 = iq*s1*np.sqrt(6.0/7.0)/muc
        h28 = s3

        h31 = iq*np.sqrt(5.0/7.0)/muc
        h32 = -q2*np.sqrt(5.0/6.0)/h3m1q
        h33 = -muc*h3m4q/(np.sqrt(7.0)*h3m1q)
        h34 = -np.sqrt(15.0/14.0)*muc*q2/h3m1q
        h37 = -np.sqrt(45.0/7.0)*muc*iq*s1/h3m1q
        h38 = np.sqrt(5.0/6.0)*q2*s3/h3m1q
        h39 = s3

        h41 = np.sqrt(2.0)*iq/muc
        h42 = np.sqrt(7.0/3.0)*(2.0*muc2-q2)/h3m1q
        h43 = np.sqrt(70.0)*q2/h3m4q
        h44 = -np.sqrt(3.0)*muc*h1m3q/h3m4q
        h47 = -np.sqrt(50.0)*iq*muc*s1/h3m4q
        h48 = -np.sqrt(7.0/3.0)*s3*(2.0*muc2+q2)/h3m4q
        h49 = -np.sqrt(70.0)*q2*s3/h3m4q
        h40 = s1

        h51 = np.sqrt(8.0/7.0)
        h52 = 5.0*iq*muc/(np.sqrt(3.0)*h3m1q)
        h53 = np.sqrt(10.0)*iq*(q2-3.0*muc2)/(2.0*muc*h3m4q)
        h54 = iq*(3.0*q2-4.0*muc2)/(np.sqrt(28.0)*muc*h1m3q)
        h55 = s3
        h57 = s1*(3.0*q2-4.0*muc2)/(np.sqrt(14.0)*h1m3q)
        h58 = -np.sqrt(3.0/4.0)*iq*s3*(2.0*muc2+q2)/(muc*h1m3q)
        h59 = np.sqrt(5.0/2.0)*iq*muc*s3/h1m3q
        h50 = iq*s1*(4.0*muc2-3.0*q2)/(np.sqrt(28.0)*muc*h1m3q)

        h63 = np.sqrt(6.0)*iq*h3m1q/(2.0*muc*h3m4q)
        h64 = -iq*q2*np.sqrt(15.0/7.0)/(2.0*muc*h1m3q)
        h66 = s3
        h67 = -np.sqrt(15.0/14.0)*q2*s1/h1m3q
        h68 = -iq*q2*np.sqrt(5.0/4.0)*s3/(muc*h1m3q)
        h69 = -iq*np.sqrt(3.0/2.0)*s3*(muc2+2.0*q2)/(muc*h1m3q)
        h60 = np.sqrt(15.0/7.0)*iq*q2*s1/(2.0*muc*h1m3q)

        h71 = -np.sqrt(5.0)*s2/muc
        h72 = -np.sqrt(35.0/6.0)*iq*s2/h3m1q
        h73 = np.sqrt(63.0)*iq*s2/h3m4q
        h74 = np.sqrt(5.0/2.0)*iq*s2/h1m3q
        h75 = -np.sqrt(8.0/35.0)*muc/s3

        h82 = -np.sqrt(35.0)*muc*s2/h3m1q
        h83 = -np.sqrt(42.0)*q2*s2/(muc*h3m4q)
        h84 = -np.sqrt(5.0/3.0)*q2*s2/(muc*h1m3q)
        h85 = -np.sqrt(12.0/35.0)*iq/s3

        h93 = -np.sqrt(7.0)*s2*h3m1q/(muc*h3m4q)
        h94 = np.sqrt(5.0/2.0)*q2*s2/(muc*h1m3q)
        h95 = iq/(np.sqrt(70.0)*s3)
        h96 = -np.sqrt(3.0/14.0)*iq/s3

        h04 = -s0*h3m4q/(np.sqrt(3.0)*muc*h1m3q)

        h77 = -muc/np.sqrt(5.0) - (h71*h17 + h72*h27 + h73*h37 + h74*h47 + h75*h57)
        h78 = iq*np.sqrt(3.0/70.0) - (h72*h28 + h73*h38 + h74*h48 + h75*h58)
        h79 = -iq*np.sqrt(1.0/7.0) - (h73*h39 + h74*h49 + h75*h59)
        h70 = -iq*np.sqrt(1.0/10.0) - (h74*h40 + h75*h50)

        h87 = (np.sqrt(2.0/15.0)*iq - (h82*h27 + h83*h37 + h84*h47 + h85*h57))/h77
        h97 = (-np.sqrt(1.0/5.0)*iq - (h93*h37 + h94*h47 + h95*h57 + h96*h67))/h77
        h07 = (-np.sqrt(2.0/3.0)*iq - h04*h47)/h77

        h88 = -3.0*muc/np.sqrt(35.0) - (h82*h28 + h83*h38 + h84*h48 + h85*h58 + h87*h78)
        h89 = -(h83*h39 + h84*h49 + h85*h59 + h87*h79)
        h80 = -2.0*muc/np.sqrt(15.0) - (h84*h40 + h85*h50 + h87*h70)

        h98 = -(h93*h38 + h94*h48 + h95*h58 + h96*h68 + h97*h78)/h88
        h08 = -(h04*h48 + h07*h78)/h88

        h99 = -muc/np.sqrt(7.0) - (h93*h39 + h94*h49 + h95*h59 + h96*h69 + h97*h79 + h98*h89)
        h90 = -(h94*h40 + h95*h50 + h96*h60 + h97*h70 + h98*h80)

        h09 = -(h04*h49 + h07*h79 + h08*h89)/h99

        h00 = -muc/np.sqrt(3.0) - (h04*h40 + h07*h70 + h08*h80 + h09*h90)

        # particular solution components
        H = np.empty(((10,)+np.broadcast(q,phiq).shape),dtype=np.complex)

        phi = np.arccos(mu0)

        H[0] = mus/mu0*(gm[1]-alpha)*Ylm(1,1,phiq,phi).real
        H[1] = mus/mu0*(gm[3]-alpha)*Ylm(0,3,phiq,phi).real
        H[2] = mus/mu0*(gm[3]-alpha)*Ylm(2,3,phiq,phi).real
        H[3] = mus/mu0*(gm[1]-alpha)*Ylm(0,1,phiq,phi).real
        H[4] = mus/mu0*(gm[3]-alpha)*Ylm(1,3,phiq,phi).real
        H[5] = mus/mu0*(gm[3]-alpha)*Ylm(3,3,phiq,phi).real
        H[6] = mus/mu0*(gm[2]-alpha)*Ylm(1,2,phiq,phi).real
        H[7] = mus/mu0*(gm[2]-alpha)*Ylm(0,2,phiq,phi).real
        H[8] = mus/mu0*(gm[2]-alpha)*Ylm(2,2,phiq,phi).real
        H[9] = mus/mu0*(gm[0]-alpha)*Ylm(0,0,phiq,phi).real

        #H[0] = H[0]
        H[1] = H[1]-h21*H[0]
        H[2] = H[2]-h31*H[0]-h32*H[1]
        H[3] = H[3]-h41*H[0]-h42*H[1]-h43*H[2]
        H[4] = H[4]-h51*H[0]-h52*H[1]-h53*H[2]-h54*H[3]
        H[5] = H[5]-h63*H[2]-h64*H[3]
        H[6] = H[6]-h71*H[0]-h72*H[1]-h73*H[2]-h74*H[3]-h75*H[4]
        H[7] = H[7]-h82*H[1]-h83*H[2]-h84*H[3]-h85*H[4]-h87*H[6]
        H[8] = H[8]-h93*H[2]-h94*H[3]-h95*H[4]-h96*H[5]-h97*H[6]-h98*H[7]
        H[9] = H[9]-h04*H[3]-h07*H[6]-h08*H[7]-h09*H[8]

        H[9] = H[9]/h00
        H[8] = (H[8] - h90*H[9])/h99
        H[7] = (H[7] - h89*H[8] - h80*H[9])/h88
        H[6] = (H[6] - h78*H[7] - h79*H[8] - h70*H[9])/h77
        H[5] = (H[5] - h67*H[6] - h68*H[7] - h69*H[8] - h60*H[9])/h66
        H[4] = (H[4] - h57*H[6] - h58*H[7] - h59*H[8] - h50*H[9])/h55
        H[3] = (H[3] - h47*H[6] - h48*H[7] - h49*H[8] - h40*H[9])/h44
        H[2] = (H[2] - h34*H[3] - h37*H[6] - h38*H[7] - h39*H[8])/h33
        H[1] = (H[1] - h23*H[2] - h24*H[3] - h27*H[6] - h28*H[7])/h22
        H[0] = (H[0] - h12*H[1] - h13*H[2] - h14*H[3] - h17*H[6])/h11

        if squeeze:
            return np.squeeze(muc), np.squeeze(1.0/W), np.squeeze(V), np.squeeze(H[[3,9,1,7,6,0,4,2,8]])
        else:
            return muc, 1.0/W, V, H[[3,9,1,7,6,0,4,2,8]]

#-------

class RTE_P1:

    def __init__(self,musp,mua,g,mu0=1.0):

        self.mua = mua
        self.mus = musp / (1.0-g)
        self.gm = np.array([1.0,g,g**2])
        self.alpha = self.gm[2]
        self.mut = mua + (1.0-self.alpha)*self.mus
        self.mut2 = self.mut*self.mut

        self.s0 = mua
        self.s1 = mua + self.mus*(1.0-self.gm[1])

        self.D = np.sqrt(3.0*self.s0*self.s1)

        self.mu0 = mu0

    def get_res(self,q,phiq=np.array([0.0],ndmin=2),squeeze=False):

        q.shape += (1,) * (2 - q.ndim)

        s0,s1 = self.s0,self.s1
        gm,D = self.gm,self.D
        mus,mut,mut2,alpha = self.mus,self.mut,self.mut2,self.alpha
        mu0 = self.mu0

        g = gm[1]

        iq = 1j*q
        q2 = q*q

        # Eigenvalue
        W = np.empty((1,)+q.shape)
        W[0] = np.sqrt(q2+D*D)

        # Eigenvector components
        V = np.zeros(((1,3)+q.shape),dtype=np.complex)

        V[0,0] = -np.sqrt(3.0)*s1
        V[0,1] = -W
        V[0,2] = -np.sqrt(0.5)*iq

        muc = mut/mu0 + iq*np.sqrt(1.0-mu0*mu0)/mu0*np.cos(phiq)
        muc2 = muc*muc

        # Analytic LU decomposition for particular solution system
        h11 = s1
        #h12 = 0
        h13 = -iq/np.sqrt(6.0)
        #h21 = 0
        h22 = s1
        h23 = -muc/np.sqrt(3.0)
        h31 = -iq*np.sqrt(2.0/3.0)/s1
        h32 = -muc/(np.sqrt(3.0)*s1)
        h33 = s0+(q2-muc2)/(3.0*s1)

        # particular solution components
        H = np.empty(((3,)+np.broadcast(q,phiq).shape),dtype=np.complex)

        phi = np.arccos(mu0)

        H[0] = mus/mu0*(gm[1]-alpha)*Ylm(1,1,phiq,phi).real
        H[1] = mus/mu0*(gm[1]-alpha)*Ylm(0,1,phiq,phi).real
        H[2] = mus/mu0*(gm[0]-alpha)*Ylm(0,0,phiq,phi).real

        #H[0] = H[0]
        #H[1] = H[1]
        H[2] = H[2]-h31*H[0]-h32*H[1]

        H[2] = (H[2])/h33
        H[1] = (H[1] - h23*H[2])/h22
        H[0] = (H[0] - h13*H[2])/h11

        if squeeze:
            return np.squeeze(muc), np.squeeze(1.0/W), np.squeeze(V)[None,...], np.squeeze(H[[2,1,0]])
        else:
            return muc, 1.0/W, V, H[[2,1,0]]

# P3 boundary system

def get_calc_c_marshak_3L_P3(n):

    R = calc_R_P3(n)
    T = calc_R_P3(1.0)

    def calc_c_marshak_3L_P3(
        L1,mut1,ew1,ev1,eps1,
        L2,mut2,ew2,ev2,eps2,
        L3,mut3,ew3,ev3,eps3
        ):

        NN_P3 = 1
        N_P3 = 3

        d = (NN_P3+1)*(2*NN_P3+3) - 1

        m1l = -2*(np.arange(d)&1)+1

        Ts = -np.transpose(np.transpose(T)*m1l)

        ex1 = np.exp(-L1/ew1)
        ex2 = np.exp(-L2/ew2)
        ex3 = np.exp(-L3/ew3)

        e = (N_P3+1)*(N_P3+1) // 2

        mbc = np.zeros([3*e,3*e],dtype=np.complex)

        b = e // 2

        # Upper boundary
        mbc[0:b,0:b] = np.transpose(ev1.dot(R))
        mbc[0:b,b:2*b] = np.transpose((ev1*m1l).dot(R))*ex1
        # Boundary Layer 1 -> Layer 2
        mbc[b:2*b,0:b] = np.transpose(ev1.dot(T))*ex1
        mbc[b:2*b,b:2*b] = np.transpose((ev1*m1l).dot(T))
        mbc[b:2*b,2*b:3*b] = -np.transpose(ev2.dot(T))
        mbc[b:2*b,3*b:4*b] = -np.transpose((ev2*m1l).dot(T))*ex2
        mbc[2*b:3*b,0:b] = np.transpose(ev1.dot(Ts))*ex1
        mbc[2*b:3*b,b:2*b] = np.transpose((ev1*m1l).dot(Ts))
        mbc[2*b:3*b,2*b:3*b] = -np.transpose(ev2.dot(Ts))
        mbc[2*b:3*b,3*b:4*b] = -np.transpose((ev2*m1l).dot(Ts))*ex2
        # Boundary Layer 2 -> Layer 3
        mbc[3*b:4*b,2*b:3*b] = np.transpose(ev2.dot(T))*ex2
        mbc[3*b:4*b,3*b:4*b] = np.transpose((ev2*m1l).dot(T))
        mbc[3*b:4*b,4*b:5*b] = -np.transpose(ev3.dot(T))
        mbc[3*b:4*b,5*b:6*b] = -np.transpose((ev3*m1l).dot(T))*ex3
        mbc[4*b:5*b,2*b:3*b] = np.transpose(ev2.dot(Ts))*ex2
        mbc[4*b:5*b,3*b:4*b] = np.transpose((ev2*m1l).dot(Ts))
        mbc[4*b:5*b,4*b:5*b] = -np.transpose(ev3.dot(Ts))
        mbc[4*b:5*b,5*b:6*b] = -np.transpose((ev3*m1l).dot(Ts))*ex3
        # Lower bondary
        mbc[5*b:6*b,4*b:5*b] = -np.transpose((ev3*m1l).dot(R))*ex3
        mbc[5*b:6*b,5*b:6*b] = -np.transpose(ev3.dot(R))

        rbc = np.transpose(np.concatenate([
            -eps1.dot(R),
            (eps2-eps1*np.exp(-mut1*L1)).dot(T),
            (eps2-eps1*np.exp(-mut1*L1)).dot(Ts),
            (eps3-eps2*np.exp(-mut2*L2)).dot(T),
            (eps3-eps2*np.exp(-mut2*L2)).dot(Ts),
            (eps3*m1l).dot(R)*np.exp(-mut3*L3)
            ],axis=-1))

        zgesv(mbc,rbc,overwrite_a = 1, overwrite_b = 1)

        return np.transpose(rbc)

    return calc_c_marshak_3L_P3

# P1 boundary system

def get_calc_c_marshak_3L_P1(n):

    R = calc_R_P1(n)
    T = calc_R_P1(1.0)

    def calc_c_marshak_3L_P1(
        L1,mut1,ew1,ev1,eps1,
        L2,mut2,ew2,ev2,eps2,
        L3,mut3,ew3,ev3,eps3
        ):

        #NN_P1 = 0
        #N_P1 = 1

        m1l = np.array([1,-1,1])

        Ts = -np.transpose(np.transpose(T)*m1l)

        ex1 = np.exp(-L1/ew1)
        ex2 = np.exp(-L2/ew2)
        ex3 = np.exp(-L3/ew3)

        mbc = np.zeros([6,6],dtype=np.complex)

        mbc[0,0] = ev1[0,0]*R[0,0]+ev1[0,1]*R[1,0]
        mbc[0,1] = (ev1[0,0]*R[0,0]-ev1[0,1]*R[1,0])*ex1

        mbc[1,0] = (ev1[0,0]*T[0,0]+ev1[0,1]*T[1,0])*ex1
        mbc[1,1] = ev1[0,0]*T[0,0]-ev1[0,1]*T[1,0]
        mbc[1,2] = -ev2[0,0]*T[0,0]-ev2[0,1]*T[1,0]
        mbc[1,3] = (-ev2[0,0]*T[0,0]+ev2[0,1]*T[1,0])*ex2
        mbc[2,0] = (ev1[0,0]*Ts[0,0]+ev1[0,1]*Ts[1,0])*ex1
        mbc[2,1] = ev1[0,0]*Ts[0,0]-ev1[0,1]*Ts[1,0]
        mbc[2,2] = -ev2[0,0]*Ts[0,0]-ev2[0,1]*Ts[1,0]
        mbc[2,3] = (-ev2[0,0]*Ts[0,0]+ev2[0,1]*Ts[1,0])*ex2

        mbc[3,2] = (ev2[0,0]*T[0,0]+ev2[0,1]*T[1,0])*ex2
        mbc[3,3] = ev2[0,0]*T[0,0]-ev2[0,1]*T[1,0]
        mbc[3,4] = -ev3[0,0]*T[0,0]-ev3[0,1]*T[1,0]
        mbc[3,5] = (-ev3[0,0]*T[0,0]+ev3[0,1]*T[1,0])*ex3
        mbc[4,2] = (ev2[0,0]*Ts[0,0]+ev2[0,1]*Ts[1,0])*ex2
        mbc[4,3] = ev2[0,0]*Ts[0,0]-ev2[0,1]*Ts[1,0]
        mbc[4,4] = -ev3[0,0]*Ts[0,0]-ev3[0,1]*Ts[1,0]
        mbc[4,5] = (-ev3[0,0]*Ts[0,0]+ev3[0,1]*Ts[1,0])*ex3

        mbc[5,4] = (-ev3[0,0]*R[0,0]+ev3[0,1]*R[1,0])*ex3
        mbc[5,5] = -ev3[0,0]*R[0,0]-ev3[0,1]*R[1,0]

        rbc = np.transpose(np.concatenate([
            -eps1.dot(R),
            (eps2-eps1*np.exp(-mut1*L1)).dot(T),
            (eps2-eps1*np.exp(-mut1*L1)).dot(Ts),
            (eps3-eps2*np.exp(-mut2*L2)).dot(T),
            (eps3-eps2*np.exp(-mut2*L2)).dot(Ts),
            (eps3*m1l).dot(R)*np.exp(-mut3*L3)
            ],axis=-1))

        zgesv(mbc,rbc,overwrite_a = 1, overwrite_b = 1)

        return np.transpose(rbc)

    return calc_c_marshak_3L_P1
