#!/usr/bin/env python3
# coding: utf-8

"""
RTE_heat_3L_thick.py: solution implementation for optically thick slabs.
It is valid, if the lower RTE source strength is negligible compared to
the upper one. Then, it saves about half of the computation time.
"""

import numpy as np
from typing import NamedTuple

from heat3L_det import *

from fresnel import Rfres

from RTE_planar_common import calc_RTE
from RTE_planar_slab_3L import get_calc_c_marshak_3L
from RTE_common import RTE_P3
from RTE_common import RTE_P1
from RTE_common import get_calc_c_marshak_3L_P3
from RTE_common import get_calc_c_marshak_3L_P1

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2021, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"

# default source function

def _src_switch(s):
    return 1.0/s

# layers

class Layer(NamedTuple):
    rho:    np.double # in kg/m^3
    cp:     np.double # in J/kg/K
    k:      np.double # in W/m/K
    musp:   np.double # in 1/mm
    mua:    np.double # in 1/mm
    g:      np.double # unitless
    L:      np.double # in mm

# solution implementation

class rte_heat_3L:

    def __init__(self, layer1, layer2, layer3, h1=0, h2=0, ux=0, uy=0,
                 f_src=_src_switch, mu1=1.0, NN=100,
                 n=1.5, Rn_file_n='Rn__1-5.npz', Rn_file_1='Rn__1-0_analytic.npz'):

        # Heat conduction parameters (rescale length units to mm)

        self.rho1 = layer1.rho / 1e9
        self.cp1 = layer1.cp
        self.k1 = layer1.k / 1e3
        self.u1x = ux
        self.u1y = uy

        self.rho2 = layer2.rho / 1e9
        self.cp2 = layer2.cp
        self.k2 = layer2.k / 1e3
        self.u2x = ux
        self.u2y = uy

        self.rho3 = layer3.rho / 1e9
        self.cp3 = layer3.cp
        self.k3 = layer3.k / 1e3
        self.u3x = ux
        self.u3y = uy

        self.l1 = layer1.L
        self.l2 = layer2.L
        self.l3 = layer3.L

        self.h1 = h1 / self.k1 /1e6
        self.h2 = h2 / self.k3 /1e6

        self.lg = self.l1+self.l2+self.l3

        if f_src is not None:
            self.f_src = f_src
        else:
            self.f_src = lambda s: 0.0

        # RTE general

        self.mua1 = layer1.mua
        self.mua2 = layer2.mua
        self.mua3 = layer3.mua

        self.n = n

        self.mu0 = np.sqrt(1.0 - (1.0/(n*n)*(1.0-mu1*mu1)))

        self.R0 = Rfres(self.n,self.mu0)
        self.Rd = Rfres(1.0/self.n,mu1)

        # RTE PN

        self.NN = NN

        print("Planar PN Order : P%d(P3)"%(2*NN+1))

        Qo,Qu,R,T,Abs1o,Abs2o,Abs3o,Abs1u,Abs2u,Abs3u = self.calc_PN(
            Rn_file_n,Rn_file_1,NN,
            self.l1,layer1.musp,layer1.mua,layer1.g,
            self.l2,layer2.musp,layer2.mua,layer2.g,
            self.l3,layer3.musp,layer3.mua,layer3.g
            )

        Abs1 = Abs1o+Abs1u
        Abs2 = Abs2o+Abs2u
        Abs3 = Abs3o+Abs3u

        # RTE P3

        self.RTE1_P3 = RTE_P3(layer1.musp,layer1.mua,layer1.g,self.mu0)
        self.RTE2_P3 = RTE_P3(layer2.musp,layer2.mua,layer2.g,self.mu0)
        self.RTE3_P3 = RTE_P3(layer3.musp,layer3.mua,layer3.g,self.mu0)

        Qo_p3,Qu_p3,R_p3,T_p3,Abs1o_p3,Abs2o_p3,Abs3o_p3,Abs1u_p3,Abs2u_p3,Abs3u_p3 = self.calc_P3(
            n,
            self.l1,layer1.musp,layer1.mua,layer1.g,
            self.l2,layer2.musp,layer2.mua,layer2.g,
            self.l3,layer3.musp,layer3.mua,layer3.g
            )

        Abs1_p3 = Abs1o_p3+Abs1u_p3
        Abs2_p3 = Abs2o_p3+Abs2u_p3
        Abs3_p3 = Abs3o_p3+Abs3u_p3

        # RTE P1

        self.RTE1_P1 = RTE_P1(layer1.musp,layer1.mua,layer1.g,self.mu0)
        self.RTE2_P1 = RTE_P1(layer2.musp,layer2.mua,layer2.g,self.mu0)
        self.RTE3_P1 = RTE_P1(layer3.musp,layer3.mua,layer3.g,self.mu0)

        Qo_p1,Qu_p1,R_p1,T_p1,Abs1o_p1,Abs2o_p1,Abs3o_p1,Abs1u_p1,Abs2u_p1,Abs3u_p1 = self.calc_P1(
            n,
            self.l1,layer1.musp,layer1.mua,layer1.g,
            self.l2,layer2.musp,layer2.mua,layer2.g,
            self.l3,layer3.musp,layer3.mua,layer3.g
            )

        Abs1_p1 = Abs1o_p1+Abs1u_p1
        Abs2_p1 = Abs2o_p1+Abs2u_p1
        Abs3_p1 = Abs3o_p1+Abs3u_p1

        # Output

        print("Reflectance: {0:.8}({1:.8}/{2:.8})".format(R,R_p3,R_p1))
        print("Transmittance: {0:.8}({1:.8}/{2:.8})".format(T,T_p3,T_p1))

        print("Qo: {0:.8}({1:.8}/{2:.8})".format(Qo,Qo_p3,Qo_p1))
        print("Qu: {0:.8}({1:.8}/{2:.8})".format(Qu,Qu_p3,Qu_p1))

        print("Layer 1 Absorbance: {0:.8}({1:.8}/{2:.8})".format(Abs1,Abs1_p3,Abs1_p1))
        print("Layer 2 Absorbance: {0:.8}({1:.8}/{2:.8})".format(Abs2,Abs2_p3,Abs2_p1))
        print("Layer 3 Absorbance: {0:.8}({1:.8}/{2:.8})".format(Abs3,Abs3_p3,Abs3_p1))

        print("Total Absorbance: {0:.8}({1:.8}/{2:.8})".format(Abs1+Abs2+Abs3,Abs1_p3+Abs2_p3+Abs3_p3,Abs1_p1+Abs2_p1+Abs3_p1))

        print("P3 Absorbance Correction upper: {0:.8}% / {1:.8}% / {2:.8}%".format(
            (Abs1o/Abs1o_p3-1.0)*100,
            (Abs2o/Abs2o_p3-1.0)*100,
            (Abs3o/Abs3o_p3-1.0)*100 ))
        print("P3 Absorbance Correction lower: {0:.8}% / {1:.8}% / {2:.8}%".format(
            (Abs1u/Abs1u_p3-1.0)*100,
            (Abs2u/Abs2u_p3-1.0)*100,
            (Abs3u/Abs3u_p3-1.0)*100 ))
        print("P1 Absorbance Correction upper: {0:.8}% / {1:.8}% / {2:.8}%".format(
            (Abs1o/Abs1o_p1-1.0)*100,
            (Abs2o/Abs2o_p1-1.0)*100,
            (Abs3o/Abs3o_p1-1.0)*100 ))
        print("P1 Absorbance Correction lower: {0:.8}% / {1:.8}% / {2:.8}%".format(
            (Abs1u/Abs1u_p1-1.0)*100,
            (Abs2u/Abs2u_p1-1.0)*100,
            (Abs3u/Abs3u_p1-1.0)*100 ))


        self.Q1o_p3 = self.mua1*Qo_p3*Abs1o/Abs1o_p3
        self.Q2o_p3 = self.mua2*Qo_p3*Abs2o/Abs2o_p3
        self.Q3o_p3 = self.mua3*Qo_p3*Abs3o/Abs3o_p3

        self.Q1u_p3 = self.mua1*Qu_p3*Abs1u/Abs1u_p3
        self.Q2u_p3 = self.mua2*Qu_p3*Abs2u/Abs2u_p3
        self.Q3u_p3 = self.mua3*Qu_p3*Abs3u/Abs3u_p3

        self.Q1o_p1 = self.mua1*Qo_p1*Abs1o/Abs1o_p1
        self.Q2o_p1 = self.mua2*Qo_p1*Abs2o/Abs2o_p1
        self.Q3o_p1 = self.mua3*Qo_p1*Abs3o/Abs3o_p1

        self.Q1u_p1 = self.mua1*Qu_p1*Abs1u/Abs1u_p1
        self.Q2u_p1 = self.mua2*Qu_p1*Abs2u/Abs2u_p1
        self.Q3u_p1 = self.mua3*Qu_p1*Abs3u/Abs3u_p1

    # PN solution

    def calc_PN(
        self,Rn_file_n,Rn_file_1,NN,
        L1, musp1, mua1, g1,
        L2, musp2, mua2, g2,
        L3, musp3, mua3, g3,
        ):

        mu0 = self.mu0
        R0 = self.R0

        PN1 = calc_RTE(musp1,mua1,g1,NN,mu0)
        PN2 = calc_RTE(musp2,mua2,g2,NN,mu0)
        PN3 = calc_RTE(musp3,mua3,g3,NN,mu0)

        mut1,ew1,ev1,eps1 = PN1.get_res()
        mut2,ew2,ev2,eps2 = PN2.get_res()
        mut3,ew3,ev3,eps3 = PN3.get_res()

        calc_c_marshak_3L = get_calc_c_marshak_3L(Rn_file_n,Rn_file_1,NN)

        e = np.exp(-(mut1*L1+mut2*L2+mut3*L3)/mu0)
        Re = R0*e
        s = 1.0/(1.0-Re*Re)

        Ro = self.Rd + (1.0-self.Rd)*(1.0-R0)*R0*e*e*s
        Qo = (1.0-self.Rd)*s
        Qu = Re*Qo
        Ru = (1.0-self.Rd)*(1.0-R0)*e*s

        c = calc_c_marshak_3L(
            self.l1,mut1,ew1,ev1,eps1,
            self.l2,mut2,ew2,ev2,eps2*np.exp(-mut1/self.mu0*self.l1),
            self.l3,mut3,ew3,ev3,eps3*np.exp(-mut1/self.mu0*self.l1-mut2/self.mu0*self.l2),
            mu0
            )

        c_rev = calc_c_marshak_3L(
            self.l3,mut3,ew3,ev3,eps3,
            self.l2,mut2,ew2,ev2,eps2*np.exp(-mut3/self.mu0*self.l3),
            self.l1,mut1,ew1,ev1,eps1*np.exp(-mut3/self.mu0*self.l3-mut2/self.mu0*self.l2),
            mu0
            )

        b = NN + 1

        F1 = 0.0
        F2 = 0.0
        G1 = 0.0
        G2 = 0.0

        for i in range(NN+1):
            i2 = i + b
            F1 += (c[i]-c[i2]*np.exp(-L1/ew1[i]))*ev1[i,1]
            F2 += (c[i+4*b]*np.exp(-L3/ew3[i])-c[i2+4*b])*ev3[i,1]
            G1 += (c_rev[i]-c_rev[i2]*np.exp(-L3/ew3[i]))*ev3[i,1]
            G2 += (c_rev[i+4*b]*np.exp(-L1/ew1[i])-c_rev[i2+4*b])*ev1[i,1]

        # Reflectance from upper and lower source
        R1 = -Qo*(np.sqrt(4.0/3.0*np.pi)*(F1+eps1[1]))
        R2 = Qu*(np.sqrt(4.0/3.0*np.pi)*(G2+eps1[1]*np.exp(-mut1/mu0*L1-mut2/mu0*L2-mut3/mu0*L3)))

        # Transmittance from lower and upper source
        T1 = Qo*(np.sqrt(4.0/3.0*np.pi)*(F2+eps3[1]*np.exp(-mut1/mu0*L1-mut2/mu0*L2-mut3/mu0*L3)))
        T2 = -Qu*(np.sqrt(4.0/3.0*np.pi)*(G1+eps3[1]))

        # Absorbance, upper source
        A1 = Qo*mua1*np.sqrt(4.0*np.pi)*(np.sum(ev1[:,0]*ew1*(1.0-np.exp(-L1/ew1))*(c[0:b]+c[b:2*b]))+eps1[0]*mu0/mut1*(1.0-np.exp(-mut1/mu0*L1)))
        A2 = Qo*mua2*np.sqrt(4.0*np.pi)*(np.sum(ev2[:,0]*ew2*(1.0-np.exp(-L2/ew2))*(c[2*b:3*b]+c[3*b:4*b]))+eps2[0]*np.exp(-mut1/mu0*L1)*mu0/mut2*(1.0-np.exp(-mut2/mu0*L2)))
        A3 = Qo*mua3*np.sqrt(4.0*np.pi)*(np.sum(ev3[:,0]*ew3*(1.0-np.exp(-L3/ew3))*(c[4*b:5*b]+c[5*b:6*b]))+eps3[0]*np.exp(-mut1/mu0*L1-mut2/mu0*L2)*mu0/mut3*(1.0-np.exp(-mut3/mu0*L3)))

        # Absorbance contribution from direct upper part
        A1 += Qo*mua1/mut1*(1.0-np.exp(-mut1/mu0*L1))
        A2 += Qo*mua2/mut2*(1.0-np.exp(-mut2/mu0*L2))*np.exp(-mut1/mu0*L1)
        A3 += Qo*mua3/mut3*(1.0-np.exp(-mut3/mu0*L3))*np.exp(-mut1/mu0*L1-mut2/mu0*L2)

        # Absorbance, lower source
        B1 = Qu*mua3*np.sqrt(4.0*np.pi)*(np.sum(ev3[:,0]*ew3*(1.0-np.exp(-L3/ew3))*(c_rev[0:b]+c_rev[b:2*b]))+eps3[0]*mu0/mut3*(1.0-np.exp(-mut3/mu0*L3)))
        B2 = Qu*mua2*np.sqrt(4.0*np.pi)*(np.sum(ev2[:,0]*ew2*(1.0-np.exp(-L2/ew2))*(c_rev[2*b:3*b]+c_rev[3*b:4*b]))+eps2[0]*np.exp(-mut3/mu0*L3)*mu0/mut2*(1.0-np.exp(-mut2/mu0*L2)))
        B3 = Qu*mua1*np.sqrt(4.0*np.pi)*(np.sum(ev1[:,0]*ew1*(1.0-np.exp(-L1/ew1))*(c_rev[4*b:5*b]+c_rev[5*b:6*b]))+eps1[0]*np.exp(-mut3/mu0*L3-mut2/mu0*L2)*mu0/mut1*(1.0-np.exp(-mut1/mu0*L1)))

        # Absorbance contribution from direct lower part
        B1 += Qu*mua3/mut3*(1.0-np.exp(-mut3/mu0*L3))
        B2 += Qu*mua2/mut2*(1.0-np.exp(-mut2/mu0*L2))*np.exp(-mut3/mu0*L3)
        B3 += Qu*mua1/mut1*(1.0-np.exp(-mut1/mu0*L1))*np.exp(-mut3/mu0*L3-mut2/mu0*L2)

        Rtot = Ro+R1+R2
        Ttot = Ru+T1+T2

        return Qo,Qu,Rtot,Ttot,A1,A2,A3,B3,B2,B1

    # P3 solution

    def calc_P3(
        self,n,
        L1, musp1, mua1, g1,
        L2, musp2, mua2, g2,
        L3, musp3, mua3, g3,
        ):

        NN = 1

        mu0 = self.mu0
        R0 = self.R0

        mut1,ew1,ev1,eps1 = self.RTE1_P3.get_res(np.array([0.0]),squeeze=True)
        mut2,ew2,ev2,eps2 = self.RTE2_P3.get_res(np.array([0.0]),squeeze=True)
        mut3,ew3,ev3,eps3 = self.RTE3_P3.get_res(np.array([0.0]),squeeze=True)

        calc_c_marshak_3L_P3 = get_calc_c_marshak_3L_P3(n)

        e = np.exp(-mut1*L1-mut2*L2-mut3*L3)
        Re = R0*e
        s = 1.0/(1.0-Re*Re)

        Ro = self.Rd + (1.0-self.Rd)*(1.0-R0)*R0*e*e*s
        Qo = (1.0-self.Rd)*s
        Qu = Re*Qo
        Ru = (1.0-self.Rd)*(1.0-R0)*e*s

        c = calc_c_marshak_3L_P3(
            self.l1,mut1,ew1,ev1,eps1,
            self.l2,mut2,ew2,ev2,eps2*np.exp(-mut1*self.l1),
            self.l3,mut3,ew3,ev3,eps3*np.exp(-mut1*self.l1-mut2*self.l2)
            )

        c_rev = calc_c_marshak_3L_P3(
            self.l3,mut3,ew3,ev3,eps3,
            self.l2,mut2,ew2,ev2,eps2*np.exp(-mut3*self.l3),
            self.l1,mut1,ew1,ev1,eps1*np.exp(-mut3*self.l3-mut2*self.l2)
            )

        b = 4

        F1 = 0.0
        F2 = 0.0
        G1 = 0.0
        G2 = 0.0

        for i in range(b):
            i2 = i + b
            F1 += (c[i]-c[i2]*np.exp(-L1/ew1[i]))*ev1[i,1]
            F2 += (c[i+4*b]*np.exp(-L3/ew3[i])-c[i2+4*b])*ev3[i,1]
            G1 += (c_rev[i]-c_rev[i2]*np.exp(-L3/ew3[i]))*ev3[i,1]
            G2 += (c_rev[i+4*b]*np.exp(-L1/ew1[i])-c_rev[i2+4*b])*ev1[i,1]

        # Reflectance from upper and lower source
        R1 = -Qo*(np.sqrt(4.0/3.0*np.pi)*(F1+eps1[1]))
        R2 = Qu*(np.sqrt(4.0/3.0*np.pi)*(G2+eps1[1]*np.exp(-mut1*L1-mut2*L2-mut3*L3)))

        # Transmittance from lower and upper source
        T1 = Qo*(np.sqrt(4.0/3.0*np.pi)*(F2+eps3[1]*np.exp(-mut1*L1-mut2*L2-mut3*L3)))
        T2 = -Qu*(np.sqrt(4.0/3.0*np.pi)*(G1+eps3[1]))

        # Absorbance, upper source
        A1 = Qo*mua1*np.sqrt(4.0*np.pi)*(np.sum(ev1[:,0]*ew1*(1.0-np.exp(-L1/ew1))*(c[0:b]+c[b:2*b]))+eps1[0]/mut1*(1.0-np.exp(-mut1*L1)))
        A2 = Qo*mua2*np.sqrt(4.0*np.pi)*(np.sum(ev2[:,0]*ew2*(1.0-np.exp(-L2/ew2))*(c[2*b:3*b]+c[3*b:4*b]))+eps2[0]*np.exp(-mut1*L1)/mut2*(1.0-np.exp(-mut2*L2)))
        A3 = Qo*mua3*np.sqrt(4.0*np.pi)*(np.sum(ev3[:,0]*ew3*(1.0-np.exp(-L3/ew3))*(c[4*b:5*b]+c[5*b:6*b]))+eps3[0]*np.exp(-mut1*L1-mut2*L2)/mut3*(1.0-np.exp(-mut3*L3)))

        # Absorbance contribution from direct upper part
        A1 += Qo*mua1/mut1/mu0*(1.0-np.exp(-mut1*L1))
        A2 += Qo*mua2/mut2/mu0*(1.0-np.exp(-mut2*L2))*np.exp(-mut1*L1)
        A3 += Qo*mua3/mut3/mu0*(1.0-np.exp(-mut3*L3))*np.exp(-mut1*L1-mut2*L2)

        # Absorbance, lower source
        B1 = Qu*mua3*np.sqrt(4.0*np.pi)*(np.sum(ev3[:,0]*ew3*(1.0-np.exp(-L3/ew3))*(c_rev[0:b]+c_rev[b:2*b]))+eps3[0]/mut3*(1.0-np.exp(-mut3*L3)))
        B2 = Qu*mua2*np.sqrt(4.0*np.pi)*(np.sum(ev2[:,0]*ew2*(1.0-np.exp(-L2/ew2))*(c_rev[2*b:3*b]+c_rev[3*b:4*b]))+eps2[0]*np.exp(-mut3*L3)/mut2*(1.0-np.exp(-mut2*L2)))
        B3 = Qu*mua1*np.sqrt(4.0*np.pi)*(np.sum(ev1[:,0]*ew1*(1.0-np.exp(-L1/ew1))*(c_rev[4*b:5*b]+c_rev[5*b:6*b]))+eps1[0]*np.exp(-mut3*L3-mut2*L2)/mut1*(1.0-np.exp(-mut1*L1)))

        # Absorbance contribution from direct lower part
        B1 += Qu*mua3/mut3/mu0*(1.0-np.exp(-mut3*L3))
        B2 += Qu*mua2/mut2/mu0*(1.0-np.exp(-mut2*L2))*np.exp(-mut3*L3)
        B3 += Qu*mua1/mut1/mu0*(1.0-np.exp(-mut1*L1))*np.exp(-mut3*L3-mut2*L2)

        Rtot = Ro+R1+R2
        Ttot = Ru+T1+T2

        return Qo,Qu,np.abs(Rtot),np.abs(Ttot),np.abs(A1),np.abs(A2),np.abs(A3),np.abs(B3),np.abs(B2),np.abs(B1)

    # P1 solution

    def calc_P1(
        self,n,
        L1, musp1, mua1, g1,
        L2, musp2, mua2, g2,
        L3, musp3, mua3, g3,
        ):

        NN = 0

        mu0 = self.mu0
        R0 = self.R0

        mut1,ew1,ev1,eps1 = self.RTE1_P1.get_res(np.array([0.0]),squeeze=True)
        mut2,ew2,ev2,eps2 = self.RTE2_P1.get_res(np.array([0.0]),squeeze=True)
        mut3,ew3,ev3,eps3 = self.RTE3_P1.get_res(np.array([0.0]),squeeze=True)

        calc_c_marshak_3L_P1 = get_calc_c_marshak_3L_P1(n)

        e = np.exp(-mut1*L1-mut2*L2-mut3*L3)
        Re = R0*e
        s = 1.0/(1.0-Re*Re)

        Ro = self.Rd + (1.0-self.Rd)*(1.0-R0)*R0*e*e*s
        Qo = (1.0-self.Rd)*s
        Qu = Re*Qo
        Ru = (1.0-self.Rd)*(1.0-R0)*e*s

        c = calc_c_marshak_3L_P1(
            self.l1,mut1,ew1,ev1,eps1,
            self.l2,mut2,ew2,ev2,eps2*np.exp(-mut1*self.l1),
            self.l3,mut3,ew3,ev3,eps3*np.exp(-mut1*self.l1-mut2*self.l2)
            )

        c_rev = calc_c_marshak_3L_P1(
            self.l3,mut3,ew3,ev3,eps3,
            self.l2,mut2,ew2,ev2,eps2*np.exp(-mut3*self.l3),
            self.l1,mut1,ew1,ev1,eps1*np.exp(-mut3*self.l3-mut2*self.l2)
            )

        b = 1
        i = 0
        i2 = 1

        F1 = (c[i]-c[i2]*np.exp(-L1/ew1))*ev1[0,1]
        F2 = (c[i+4*b]*np.exp(-L3/ew3)-c[i2+4*b])*ev3[0,1]
        G1 = (c_rev[i]-c_rev[i2]*np.exp(-L3/ew3))*ev3[0,1]
        G2 = (c_rev[i+4*b]*np.exp(-L1/ew1)-c_rev[i2+4*b])*ev1[0,1]

        # Reflectance from upper and lower source
        R1 = -Qo*(np.sqrt(4.0/3.0*np.pi)*(F1+eps1[1]))
        R2 = Qu*(np.sqrt(4.0/3.0*np.pi)*(G2+eps1[1]*np.exp(-mut1*L1-mut2*L2-mut3*L3)))

        # Transmittance from lower and upper source
        T1 = Qo*(np.sqrt(4.0/3.0*np.pi)*(F2+eps3[1]*np.exp(-mut1*L1-mut2*L2-mut3*L3)))
        T2 = -Qu*(np.sqrt(4.0/3.0*np.pi)*(G1+eps3[1]))

        # Absorbance, upper source
        A1 = Qo*mua1*np.sqrt(4.0*np.pi)*((ev1[0,0]*ew1*(1.0-np.exp(-L1/ew1))*(c[0]+c[1]))+eps1[0]/mut1*(1.0-np.exp(-mut1*L1)))
        A2 = Qo*mua2*np.sqrt(4.0*np.pi)*((ev2[0,0]*ew2*(1.0-np.exp(-L2/ew2))*(c[2]+c[3]))+eps2[0]*np.exp(-mut1*L1)/mut2*(1.0-np.exp(-mut2*L2)))
        A3 = Qo*mua3*np.sqrt(4.0*np.pi)*((ev3[0,0]*ew3*(1.0-np.exp(-L3/ew3))*(c[4]+c[5]))+eps3[0]*np.exp(-mut1*L1-mut2*L2)/mut3*(1.0-np.exp(-mut3*L3)))

        # Absorbance contribution from direct upper part
        A1 += Qo*mua1/mut1/mu0*(1.0-np.exp(-mut1*L1))
        A2 += Qo*mua2/mut2/mu0*(1.0-np.exp(-mut2*L2))*np.exp(-mut1*L1)
        A3 += Qo*mua3/mut3/mu0*(1.0-np.exp(-mut3*L3))*np.exp(-mut1*L1-mut2*L2)

        # Absorbance, lower source
        B1 = Qu*mua3*np.sqrt(4.0*np.pi)*((ev3[0,0]*ew3*(1.0-np.exp(-L3/ew3))*(c_rev[0]+c_rev[1]))+eps3[0]/mut3*(1.0-np.exp(-mut3*L3)))
        B2 = Qu*mua2*np.sqrt(4.0*np.pi)*((ev2[0,0]*ew2*(1.0-np.exp(-L2/ew2))*(c_rev[2]+c_rev[3]))+eps2[0]*np.exp(-mut3*L3)/mut2*(1.0-np.exp(-mut2*L2)))
        B3 = Qu*mua1*np.sqrt(4.0*np.pi)*((ev1[0,0]*ew1*(1.0-np.exp(-L1/ew1))*(c_rev[4]+c_rev[5]))+eps1[0]*np.exp(-mut3*L3-mut2*L2)/mut1*(1.0-np.exp(-mut1*L1)))

        # Absorbance contribution from direct lower part
        B1 += Qu*mua3/mut3/mu0*(1.0-np.exp(-mut3*L3))
        B2 += Qu*mua2/mut2/mu0*(1.0-np.exp(-mut2*L2))*np.exp(-mut3*L3)
        B3 += Qu*mua1/mut1/mu0*(1.0-np.exp(-mut1*L1))*np.exp(-mut3*L3-mut2*L2)

        Rtot = Ro+R1+R2
        Ttot = Ru+T1+T2

        return Qo,Qu,np.abs(Rtot),np.abs(Ttot),np.abs(A1),np.abs(A2),np.abs(A3),np.abs(B3),np.abs(B2),np.abs(B1)

    def calc_laplace_pre_P3(self,q,phiq):
        """
        Precompute P3 solution coefficients. This speeds up the solution,
        since we use the steady state RTE that does not depend on time and
        therefore the Laplace variable s. One of calc_laplace_pre_P3() and
        calc_laplace_pre_P1() must be called before calc_laplace()
        """

        q = np.array(q)
        q = q.reshape(q.size,1)

        phiq = np.array(phiq)
        phiq = phiq.reshape(1,phiq.size)

        mut1,ew1,ev1,eps1 = self.RTE1_P3.get_res(q,phiq)
        mut2,ew2,ev2,eps2 = self.RTE2_P3.get_res(q,phiq)
        mut3,ew3,ev3,eps3 = self.RTE3_P3.get_res(q,phiq)

        calc_c_marshak_3L_P3 = get_calc_c_marshak_3L_P3(self.n)

        self.mut1 = mut1
        self.mut2 = mut2
        self.mut3 = mut3

        self.ew1 = np.moveaxis(ew1, 0, -1)
        self.ev1 = np.moveaxis(ev1, [0,1], [-2,-1])
        self.eps1 = np.moveaxis(eps1, 0, -1)

        self.ew2 = np.moveaxis(ew2, 0, -1)
        self.ev2 = np.moveaxis(ev2, [0,1], [-2,-1])
        self.eps2 = np.moveaxis(eps2, 0, -1)

        self.ew3 = np.moveaxis(ew3, 0, -1)
        self.ev3 = np.moveaxis(ev3, [0,1], [-2,-1])
        self.eps3 = np.moveaxis(eps3, 0, -1)

        c_p3 = np.empty([q.size,phiq.size,3*8],dtype=np.complex)

        for i in range(q.size):
            c_p3[i] = calc_c_marshak_3L_P3(
                self.l1,mut1[i][...,None],self.ew1[i,0],self.ev1[i,0],self.eps1[i],
                self.l2,mut2[i][...,None],self.ew2[i,0],self.ev2[i,0],self.eps2[i]*np.exp(-mut1[i]*self.l1)[...,None],
                self.l3,mut3[i][...,None],self.ew3[i,0],self.ev3[i,0],self.eps3[i]*np.exp(-mut1[i]*self.l1-mut2[i]*self.l2)[...,None]
                )

        self.ci_1 = c_p3[...,0:4]
        self.cr_1 = c_p3[...,4:8]

        self.ci_2 = c_p3[...,8:12]
        self.cr_2 = c_p3[...,12:16]

        self.ci_3 = c_p3[...,16:20]
        self.cr_3 = c_p3[...,20:24]

        self.Q1o = self.Q1o_p3
        self.Q2o = self.Q2o_p3
        self.Q3o = self.Q3o_p3

        self.Q1u = 0
        self.Q2u = 0
        self.Q3u = 0

    def calc_laplace_pre_P1(self,q,phiq):
        """
        Precompute P1 solution coefficients. This speeds up the solution,
        since we use the steady state RTE that does not depend on time and
        therefore the Laplace variable s. One of calc_laplace_pre_P3() and
        calc_laplace_pre_P1() must be called before calc_laplace()
        """

        q = np.array(q)
        q = q.reshape(q.size,1)

        phiq = np.array(phiq)
        phiq = phiq.reshape(1,phiq.size)

        mut1,ew1,ev1,eps1 = self.RTE1_P1.get_res(q,phiq)
        mut2,ew2,ev2,eps2 = self.RTE2_P1.get_res(q,phiq)
        mut3,ew3,ev3,eps3 = self.RTE3_P1.get_res(q,phiq)

        calc_c_marshak_3L_P1 = get_calc_c_marshak_3L_P1(self.n)

        self.mut1 = mut1
        self.mut2 = mut2
        self.mut3 = mut3

        self.ew1 = np.moveaxis(ew1, 0, -1)
        self.ev1 = np.moveaxis(ev1, [0,1], [-2,-1])
        self.eps1 = np.moveaxis(eps1, 0, -1)

        self.ew2 = np.moveaxis(ew2, 0, -1)
        self.ev2 = np.moveaxis(ev2, [0,1], [-2,-1])
        self.eps2 = np.moveaxis(eps2, 0, -1)

        self.ew3 = np.moveaxis(ew3, 0, -1)
        self.ev3 = np.moveaxis(ev3, [0,1], [-2,-1])
        self.eps3 = np.moveaxis(eps3, 0, -1)

        c_p1 = np.empty([q.size,phiq.size,3*2],dtype=np.complex)

        for i in range(q.size):
            c_p1[i] = calc_c_marshak_3L_P1(
                self.l1,mut1[i][...,None],self.ew1[i,0],self.ev1[i,0],self.eps1[i],
                self.l2,mut2[i][...,None],self.ew2[i,0],self.ev2[i,0],self.eps2[i]*np.exp(-mut1[i]*self.l1)[...,None],
                self.l3,mut3[i][...,None],self.ew3[i,0],self.ev3[i,0],self.eps3[i]*np.exp(-mut1[i]*self.l1-mut2[i]*self.l2)[...,None]
                )

        self.ci_1 = c_p1[...,0,None]
        self.cr_1 = c_p1[...,1,None]

        self.ci_2 = c_p1[...,2,None]
        self.cr_2 = c_p1[...,3,None]

        self.ci_3 = c_p1[...,4,None]
        self.cr_3 = c_p1[...,5,None]

        self.Q1o = self.Q1o_p1
        self.Q2o = self.Q2o_p1
        self.Q3o = self.Q3o_p1

        self.Q1u = 0
        self.Q2u = 0
        self.Q3u = 0

    def calc_laplace(self, s, q, phiq, z):
        """
        Compute solution. One of calc_laplace_pre_P3() and
        calc_laplace_pre_P1() must be called before calc_laplace()
        """

        def calc_f(V00,ci,cr,ew,eps,mut,gamma,L,d=1.0):

            src = self.f_src(s)

            t1 = ew*(1.0-np.exp(-L*(gamma[...,None]+1.0/ew)))/(1.0+gamma[...,None]*ew)
            t2 = ew*(np.exp(-gamma[...,None]*L)-np.exp(-L/ew))/(1.0-gamma[...,None]*ew)

            t3 = d*(1.0-np.exp(-L*(gamma+mut)))/(gamma+mut)

            t4 = np.sqrt(4.0*np.pi)*(np.sum(V00*(ci*t1+cr*t2),axis=-1)+eps[...,0]*t3)
            t5 = t3/self.mu0

            f1 = 0.5*src*(t4+t5)

            t6 = d*(np.exp(-L*mut)-np.exp(-L*gamma))/(gamma-mut)

            t7 = np.sqrt(4.0*np.pi)*(np.sum(V00*(ci*t2+cr*t1),axis=-1)+eps[...,0]*t6)
            t8 = t6/self.mu0

            f2 = 0.5*src*(t7+t8)

            return f1,f2

        def calc_part(V00,ci,cr,ew,eps,mut,gamma,k,L,z,d=1.0):

            src = self.f_src(s)

            t1 = -ew/(1.0-ew*ew*gamma[...,None]*gamma[...,None])
            t2 = 2.0*gamma[...,None]*ew*np.exp(-z/ew) + (1.0-gamma[...,None]*ew)*np.exp(-gamma[...,None]*(L-z)-L/ew) - (1.0+gamma[...,None]*ew)*np.exp(-gamma[...,None]*z)
            t3 = 2.0*gamma[...,None]*ew*np.exp(-(L-z)/ew) + (1.0-gamma[...,None]*ew)*np.exp(-gamma[...,None]*z-L/ew) - (1.0+gamma[...,None]*ew)*np.exp(-gamma[...,None]*(L-z))

            lb = 1.0/mut

            t4 = -lb/(1.0-lb*lb*gamma*gamma)
            t5 = 2.0*gamma*lb*np.exp(-z/lb) + (1.0-gamma*lb)*np.exp(-gamma*(L-z)-L/lb) - (1.0+gamma*lb)*np.exp(-gamma*z)

            t6 = t1*t2
            t7 = t1*t3
            t8 = d*t4*t5

            t9 = np.sqrt(4.0*np.pi)*(np.sum(V00*(ci*t6+cr*t7),axis=-1)+eps[...,0]*t8)
            t10 = t8/self.mu0

            particular = 0.5*src/(gamma*k)*(t9+t10)

            return particular

        if(z < 0.0 or z > self.lg):
            raise ValueError('z outside medium.')

        L1 = self.l1
        L2 = self.l2
        L3 = self.l3
        Lg = self.lg

        q = np.array(q)
        q = q.reshape(q.size,1)

        phiq = np.array(phiq)
        phiq = phiq.reshape(1,phiq.size)

        qx = q*np.cos(phiq)
        qy = q*np.sin(phiq)

        qx2 = qx*qx
        qy2 = qy*qy

        # Heat conduction parameters

        gamma1 = np.sqrt(self.rho1*self.cp1/self.k1*(s+1j*(qx*self.u1x+qy*self.u1y)) + qx2 + qy2)
        gamma2 = np.sqrt(self.rho2*self.cp2/self.k2*(s+1j*(qx*self.u2x+qy*self.u2y)) + qx2 + qy2)
        gamma3 = np.sqrt(self.rho3*self.cp3/self.k3*(s+1j*(qx*self.u3x+qy*self.u3y)) + qx2 + qy2)

        e1 = np.exp(-gamma1*self.l1)
        e2 = np.exp(-gamma2*self.l2)
        e3 = np.exp(-gamma3*self.l3)

        p1 = (1.0 + e1*e1)
        p2 = (1.0 + e2*e2)
        p3 = (1.0 + e3*e3)

        m1 = (1.0 - e1*e1)
        m2 = (1.0 - e2*e2)
        m3 = (1.0 - e3*e3)

        detM = calc_M_3L_delta(self.k1, self.k2, self.k3,
                                gamma1, gamma2, gamma3, e1, e2, e3,
                                p1, p2, p3, m1, m2, m3, self.h1, self.h2)

        # P1/P3 solution

        mut1 = self.mut1
        mut2 = self.mut2
        mut3 = self.mut3

        ew1 = self.ew1
        eps1 = self.eps1
        ew2 = self.ew2
        eps2 = self.eps2
        ew3 = self.ew3
        eps3 = self.eps3

        # Layer 1 source part

        V00_1 = self.ev1[...,0]

        f1_1,f2_1 = calc_f(V00_1,self.ci_1,self.cr_1,ew1,eps1,mut1,gamma1,L1)
        f1_1 = f1_1*(1.0-self.h1/gamma1)

        # Layer 2 source part

        V00_2 = self.ev2[...,0]

        d2 = np.exp(-mut1*L1)

        f1_2,f2_2 = calc_f(V00_2,self.ci_2,self.cr_2,ew2,eps2,mut2,gamma2,L2,d2)

        # Layer 3 source part

        V00_3 = self.ev3[...,0]

        d3 = np.exp(-mut1*L1-mut2*L2)

        f1_3,f2_3 = calc_f(V00_3,self.ci_3,self.cr_3,ew3,eps3,mut3,gamma3,L3,d3)
        f2_3 = f2_3*(1.0-self.h2/gamma3)

        if(z < self.l1):

            detA = calc_A_3L_delta_q1_z1(
                self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                e1, e2, e3, p1, p2, p3, m1, m2, m3,
                self.h1, self.h2, f1_1, f2_1)
            detB = calc_B_3L_delta_q1_z1(
                self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                e1, e2, e3, p1, p2, p3, m1, m2, m3,
                self.h1, self.h2, f1_1, f2_1)

            detAg = self.Q1o*detA/detM
            detBg = self.Q1o*detB/detM

            particular = calc_part(V00_1,self.ci_1,self.cr_1,ew1,eps1,mut1,gamma1,self.k1,L1,z)

            partg = self.Q1o*particular

            res1 = np.squeeze((detAg*np.exp(gamma1*(z-self.l1))
                    + detBg*np.exp(-gamma1*z)) + partg)

            detA = calc_A_3L_delta_q2_z1(
                self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                e1, e2, e3, p1, p2, p3, m1, m2, m3,
                self.h1, self.h2, f1_2, f2_2)
            detB = calc_B_3L_delta_q2_z1(
                self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                e1, e2, e3, p1, p2, p3, m1, m2, m3,
                self.h1, self.h2, f1_2, f2_2)

            detAg = self.Q2o*detA/detM
            detBg = self.Q2o*detB/detM

            res2 = np.squeeze(detAg*np.exp(gamma1*(z-self.l1))
                    + detBg*np.exp(-gamma1*z))

            detA = calc_A_3L_delta_q3_z1(
                self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                e1, e2, e3, p1, p2, p3, m1, m2, m3,
                self.h1, self.h2, f1_3, f2_3)
            detB = calc_B_3L_delta_q3_z1(
                self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                e1, e2, e3, p1, p2, p3, m1, m2, m3,
                self.h1, self.h2, f1_3, f2_3)

            detAg = self.Q3o*detA/detM
            detBg = self.Q3o*detB/detM

            res3 = np.squeeze(detAg*np.exp(gamma1*(z-self.l1))
                    + detBg*np.exp(-gamma1*z))

            return res1+res2+res3

        elif(z < self.l1+self.l2):

            detA = calc_A_3L_delta_q1_z2(
                self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                e1, e2, e3, p1, p2, p3, m1, m2, m3,
                self.h1, self.h2, f1_1, f2_1)
            detB = calc_B_3L_delta_q1_z2(
                self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                e1, e2, e3, p1, p2, p3, m1, m2, m3,
                self.h1, self.h2, f1_1, f2_1)

            detAg = self.Q1o*detA/detM
            detBg = self.Q1o*detB/detM

            res1 = np.squeeze(detAg*np.exp(gamma2*(z-self.l1-self.l2))
                    + detBg*np.exp(-gamma2*(z-self.l1)))

            detA = calc_A_3L_delta_q2_z2(
                self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                e1, e2, e3, p1, p2, p3, m1, m2, m3,
                self.h1, self.h2, f1_2, f2_2)
            detB = calc_B_3L_delta_q2_z2(
                self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                e1, e2, e3, p1, p2, p3, m1, m2, m3,
                self.h1, self.h2, f1_2, f2_2)

            detAg = self.Q2o*detA/detM
            detBg = self.Q2o*detB/detM

            particular = calc_part(V00_2,self.ci_2,self.cr_2,ew2,eps2,mut2,gamma2,self.k2,L2,z-L1,d2)

            partg = self.Q2o*particular

            res2 = np.squeeze((detAg*np.exp(gamma2*(z-self.l1-self.l2))
                    + detBg*np.exp(-gamma2*(z-self.l1))) + partg)

            detA = calc_A_3L_delta_q3_z2(
                self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                e1, e2, e3, p1, p2, p3, m1, m2, m3,
                self.h1, self.h2, f1_3, f2_3)
            detB = calc_B_3L_delta_q3_z2(
                self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                e1, e2, e3, p1, p2, p3, m1, m2, m3,
                self.h1, self.h2, f1_3, f2_3)

            detAg = self.Q3o*detA/detM
            detBg = self.Q3o*detB/detM

            res3 = np.squeeze(detAg*np.exp(gamma2*(z-self.l1-self.l2))
                    + detBg*np.exp(-gamma2*(z-self.l1)))

            return res1+res2+res3

        else:

            detA = calc_A_3L_delta_q1_z3(
                    self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                    e1, e2, e3, p1, p2, p3, m1, m2, m3,
                    self.h1, self.h2, f1_1, f2_1)
            detB = calc_B_3L_delta_q1_z3(
                    self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                    e1, e2, e3, p1, p2, p3, m1, m2, m3,
                    self.h1, self.h2, f1_1, f2_1)

            detAg = self.Q1o*detA/detM
            detBg = self.Q1o*detB/detM

            res1 = np.squeeze(detAg*np.exp(gamma3*(z-Lg))
                    + detBg*np.exp(-gamma3*(z-self.l1-self.l2)))

            detA = calc_A_3L_delta_q2_z3(
                    self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                    e1, e2, e3, p1, p2, p3, m1, m2, m3,
                    self.h1, self.h2, f1_2, f2_2)
            detB = calc_B_3L_delta_q2_z3(
                    self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                    e1, e2, e3, p1, p2, p3, m1, m2, m3,
                    self.h1, self.h2, f1_2, f2_2)

            detAg = self.Q2o*detA/detM
            detBg = self.Q2o*detB/detM

            res2 = np.squeeze(detAg*np.exp(gamma3*(z-Lg))
                    + detBg*np.exp(-gamma3*(z-self.l1-self.l2)))

            detA = calc_A_3L_delta_q3_z3(
                    self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                    e1, e2, e3, p1, p2, p3, m1, m2, m3,
                    self.h1, self.h2, f1_3, f2_3)
            detB = calc_B_3L_delta_q3_z3(
                    self.k1, self.k2, self.k3, gamma1, gamma2, gamma3,
                    e1, e2, e3, p1, p2, p3, m1, m2, m3,
                    self.h1, self.h2, f1_3, f2_3)

            detAg = self.Q3o*detA/detM
            detBg = self.Q3o*detB/detM

            particular = calc_part(V00_3,self.ci_3,self.cr_3,ew3,eps3,mut3,gamma3,self.k3,L3,z-L1-L2,d3)

            partg = self.Q3o*particular

            res3 = np.squeeze((detAg*np.exp(gamma3*(z-Lg))
                    + detBg*np.exp(-gamma3*(z-self.l1-self.l2))) + partg )

            return res1+res2+res3

    def calc_absorbance(self, q, phiq, z):
        """
        Compute P3 absorbance only.
        """

        if(z < 0.0 or z > self.lg):
            raise ValueError('z outside medium.')

        L1 = self.l1
        L2 = self.l2
        L3 = self.l3
        Lg = self.lg

        q = np.array(q)
        q = q.reshape(q.size,1)

        phiq = np.array(phiq)
        phiq = phiq.reshape(1,phiq.size)

        qx = q*np.cos(phiq)
        qy = q*np.sin(phiq)

        # P1/P3 solution

        mut1 = self.mut1
        mut2 = self.mut2
        mut3 = self.mut3

        ew1 = self.ew1
        eps1 = self.eps1
        ew2 = self.ew2
        eps2 = self.eps2
        ew3 = self.ew3
        eps3 = self.eps3

        # Layer 1 source part

        V00_1 = self.ev1[...,0]
        ci_1 = self.ci_1
        cr_1 = self.cr_1

        # Layer 2 source part

        V00_2 = self.ev2[...,0]
        ci_2 = self.ci_2
        cr_2 = self.cr_2

        d2 = np.exp(-mut1*L1)

        # Layer 3 source part

        V00_3 = self.ev3[...,0]
        ci_3 = self.ci_3
        cr_3 = self.cr_3

        d3 = np.exp(-mut1*L1-mut2*L2)

        if(z < self.l1):

            t1 = np.exp(-z/ew1)
            t2 = np.exp((z-L1)/ew1)
            t3 = np.exp(-mut1*z)

            res1 = np.sqrt(4.0*np.pi)*(np.sum(V00_1*(ci_1*t1+cr_1*t2),axis=-1)+eps1[...,0]*t3)
            res1 += t3/self.mu0

            res = self.Q1o*res1

            return np.squeeze(res)

        elif(z < self.l1+self.l2):

            zr = z - L1

            t1 = np.exp(-zr/ew2)
            t2 = np.exp((zr-L2)/ew2)
            t3 = np.exp(-mut2*zr)*d2

            res1 = np.sqrt(4.0*np.pi)*(np.sum(V00_2*(ci_2*t1+cr_2*t2),axis=-1)+eps2[...,0]*t3)
            res1 += t3/self.mu0

            res = self.Q2o*res1

            return np.squeeze(res)

        else:

            zr = z - L1 - L2

            t1 = np.exp(-zr/ew3)
            t2 = np.exp((zr-L3)/ew3)
            t3 = np.exp(-mut3*zr)*d3

            res1 = np.sqrt(4.0*np.pi)*(np.sum(V00_3*(ci_3*t1+cr_3*t2),axis=-1)+eps3[...,0]*t3)
            res1 += t3/self.mu0

            res = self.Q3o*res1

            return np.squeeze(res)
