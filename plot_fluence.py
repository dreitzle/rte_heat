#!/usr/bin/env python3
# coding: utf-8

"""plot_fluence.py: Generate depth-resolved total fluence (Figure 2)"""

import numpy as np

from fresnel import Rfres

from RTE_planar_common import calc_RTE
from RTE_planar_slab_3L import get_calc_c_marshak_3L

from RTE_common import RTE_P3
from RTE_common import RTE_P1
from RTE_common import get_calc_c_marshak_3L_P3
from RTE_common import get_calc_c_marshak_3L_P1

import matplotlib.pyplot as plt
# from tikzplotlib import save as tikz_save

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2021, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"

# Medium Parameters

musp1 = 2.0
mua1 = 0.02
g1 = 0.8

musp2 = 1.0
mua2 = 0.003
g2 = 0.8

musp3 = 0.5
mua3 = 0.04
g3 = 0.8

n = 1.4
coeff_file = 'Rn__1-4.npz'

L1 = 1.0
L2 = 2.0
L3 = 10.0

# Incidence angle
mu1 = np.cos(0/180*np.pi)
# mu1 = np.cos(60/180*np.pi)

# PN order
NN = 100
N = 2*NN+1

print("Planar PN Order : P%d"%N)

mu0 = np.sqrt(1.0 - (1.0/(n*n)*(1.0-mu1*mu1)))

# %% Calculate PN coefficients

calc_c = get_calc_c_marshak_3L(coeff_file,'Rn__1-0_analytic.npz',NN)

RTE1 = calc_RTE(musp1,mua1,g1,NN,mu0)
RTE2 = calc_RTE(musp2,mua2,g2,NN,mu0)
RTE3 = calc_RTE(musp3,mua3,g3,NN,mu0)

mut1,ew1,ev1,eps1 = RTE1.get_res()
mut2,ew2,ev2,eps2 = RTE2.get_res()
mut3,ew3,ev3,eps3 = RTE3.get_res()


c = calc_c(L1,mut1,ew1,ev1,eps1,
                L2,mut2,ew2,ev2,eps2*np.exp(-mut1/mu0*L1),
                L3,mut3,ew3,ev3,eps3*np.exp(-mut1/mu0*L1-mut2/mu0*L2),
                mu0)

c_rev = calc_c(L3,mut3,ew3,ev3,eps3,
                    L2,mut2,ew2,ev2,eps2*np.exp(-mut3/mu0*L3),
                    L1,mut1,ew1,ev1,eps1*np.exp(-mut3/mu0*L3-mut2/mu0*L2),
                    mu0)

# %% Calculate P3 coefficients

calc_c_p3 = get_calc_c_marshak_3L_P3(n)

RTE1_P3 = RTE_P3(musp1,mua1,g1,mu0)
RTE2_P3 = RTE_P3(musp2,mua2,g2,mu0)
RTE3_P3 = RTE_P3(musp3,mua3,g3,mu0)

mut1_p3,ew1_p3,ev1_p3,eps1_p3 = RTE1_P3.get_res(np.array(0.0),squeeze=True)
mut2_p3,ew2_p3,ev2_p3,eps2_p3 = RTE2_P3.get_res(np.array(0.0),squeeze=True)
mut3_p3,ew3_p3,ev3_p3,eps3_p3 = RTE3_P3.get_res(np.array(0.0),squeeze=True)

c_p3 = calc_c_p3(L1,mut1_p3,ew1_p3,ev1_p3,eps1_p3,
                L2,mut2_p3,ew2_p3,ev2_p3,eps2_p3*np.exp(-mut1_p3*L1),
                L3,mut3_p3,ew3_p3,ev3_p3,eps3_p3*np.exp(-mut1_p3*L1-mut2_p3*L2)
                )

c_rev_p3 = calc_c_p3(L3,mut3_p3,ew3_p3,ev3_p3,eps3_p3,
                    L2,mut2_p3,ew2_p3,ev2_p3,eps2_p3*np.exp(-mut3_p3*L3),
                    L1,mut1_p3,ew1_p3,ev1_p3,eps1_p3*np.exp(-mut3_p3*L3-mut2_p3*L2)
                    )

# %% Calculate P1 coefficients

calc_c_p1 = get_calc_c_marshak_3L_P1(n)

RTE1_P1 = RTE_P1(musp1,mua1,g1,mu0)
RTE2_P1 = RTE_P1(musp2,mua2,g2,mu0)
RTE3_P1 = RTE_P1(musp3,mua3,g3,mu0)

mut1_p1,ew1_p1,ev1_p1,eps1_p1 = RTE1_P1.get_res(np.array(0.0),squeeze=True)
mut2_p1,ew2_p1,ev2_p1,eps2_p1 = RTE2_P1.get_res(np.array(0.0),squeeze=True)
mut3_p1,ew3_p1,ev3_p1,eps3_p1 = RTE3_P1.get_res(np.array(0.0),squeeze=True)

c_p1 = calc_c_p1(L1,mut1_p1,ew1_p1,ev1_p1,eps1_p1,
                L2,mut2_p1,ew2_p1,ev2_p1,eps2_p1*np.exp(-mut1_p1*L1),
                L3,mut3_p1,ew3_p1,ev3_p1,eps3_p1*np.exp(-mut1_p1*L1-mut2_p1*L2)
                )

c_rev_p1 = calc_c_p1(L3,mut3_p1,ew3_p1,ev3_p1,eps3_p1,
                    L2,mut2_p1,ew2_p1,ev2_p1,eps2_p1*np.exp(-mut3_p1*L3),
                    L1,mut1_p1,ew1_p1,ev1_p1,eps1_p1*np.exp(-mut3_p1*L3-mut2_p1*L2)
                    )

# %% Calculate sources

R0 = Rfres(n,mu0)

e = np.exp(-(mut1*L1+mut2*L2+mut3*L3)/mu0)
e_p3 = np.exp(-(mut1_p3*L1+mut2_p3*L2+mut3_p3*L3))
e_p1 = np.exp(-(mut1_p1*L1+mut2_p1*L2+mut3_p1*L3))

Re = R0*e
Re_p3 = R0*e_p3
Re_p1 = R0*e_p1

Rd = Rfres(1.0/n,mu1)

s = 1.0/(1.0-Re*Re)
s_p3 = 1.0/(1.0-Re_p3*Re_p3)
s_p1 = 1.0/(1.0-Re_p1*Re_p1)

Ro = Rd + (1.0-Rd)*(1.0-R0)*R0*e*e*s
Qo = (1.0-Rd)*s
Qu = Re*Qo
Ru = (1.0-Rd)*(1.0-R0)*e*s

Ro_p3 = Rd + (1.0-Rd)*(1.0-R0)*R0*e_p3*e_p3*s_p3
Qo_p3 = (1.0-Rd)*s_p3
Qu_p3 = Re_p3*Qo_p3
Ru_p3 = (1.0-Rd)*(1.0-R0)*e_p3*s_p3

Ro_p1 = Rd + (1.0-Rd)*(1.0-R0)*R0*e_p1*e_p1*s_p1
Qo_p1 = (1.0-Rd)*s_p1
Qu_p1 = Re_p1*Qo_p1
Ru_p1 = (1.0-Rd)*(1.0-R0)*e_p1*s_p1

Ps1 = Qo
Ps2 = Qu

Ps1_p3 = np.abs(Qo_p3)
Ps2_p3 = np.abs(Qu_p3)

Ps1_p1 = np.abs(Qo_p1)
Ps2_p1 = np.abs(Qu_p1)

# %% P1 total

print("P1 upper source strength: %f " % Ps1_p1)
print("P1 lower source strength: %f " % Ps2_p1)

F1_p1 = 0.0
F2_p1 = 0.0
G1_p1 = 0.0
G2_p1 = 0.0

i = 0
i2 = 1
F1_p1 = (c_p1[i]-c_p1[i2]*np.exp(-L1/ew1_p1))*ev1_p1[0,1]
F2_p1 = (c_p1[i+4*1]*np.exp(-L3/ew3_p1)-c_p1[i2+4*1])*ev3_p1[0,1]
G1_p1 = (c_rev_p1[i]-c_rev_p1[i2]*np.exp(-L3/ew3_p1))*ev3_p1[0,1]
G2_p1 = (c_rev_p1[i+4*1]*np.exp(-L1/ew1_p1)-c_rev_p1[i2+4*1])*ev1_p1[0,1]

# Reflectance from upper and lower source
R1_p1 = -Ps1_p1*(np.sqrt(4.0/3.0*np.pi)*(F1_p1+eps1_p1[1]))
R2_p1 = Ps2_p1*(np.sqrt(4.0/3.0*np.pi)*(G2_p1+eps1_p1[1]*np.exp(-mut1_p1*L1-mut2_p1*L2-mut3_p1*L3)))
print("P1 Reflectance: %f " % np.abs(Ro_p1+R1_p1+R2_p1))

# Transmittance from lower and upper source
T1_p1 = Ps1_p1*(np.sqrt(4.0/3.0*np.pi)*(F2_p1+eps3_p1[1]*np.exp(-mut1_p1*L1-mut2_p1*L2-mut3_p1*L3)))
T2_p1 = -Ps2_p1*(np.sqrt(4.0/3.0*np.pi)*(G1_p1+eps3_p1[1]))
print("P1 Transmittance: %f " % np.abs(Ru_p1+T1_p1+T2_p1))

# Absorbance, upper source
A1_p1 = Ps1_p1*mua1*np.sqrt(4.0*np.pi)*((ev1_p1[0,0]*ew1_p1*(1.0-np.exp(-L1/ew1_p1))*(c_p1[0]+c_p1[1]))+eps1_p1[0]/mut1_p1*(1.0-np.exp(-mut1_p1*L1)))
A2_p1 = Ps1_p1*mua2*np.sqrt(4.0*np.pi)*((ev2_p1[0,0]*ew2_p1*(1.0-np.exp(-L2/ew2_p1))*(c_p1[2]+c_p1[3]))+eps2_p1[0]*np.exp(-mut1_p1*L1)/mut2_p1*(1.0-np.exp(-mut2_p1*L2)))
A3_p1 = Ps1_p1*mua3*np.sqrt(4.0*np.pi)*((ev3_p1[0,0]*ew3_p1*(1.0-np.exp(-L3/ew3_p1))*(c_p1[4]+c_p1[5]))+eps3_p1[0]*np.exp(-mut1_p1*L1-mut2_p1*L2)/mut3_p1*(1.0-np.exp(-mut3_p1*L3)))

# Absorbance contribution from direct upper part
A1_p1 += Ps1_p1*mua1/mu0/mut1_p1*(1.0-np.exp(-mut1_p1*L1))
A2_p1 += Ps1_p1*mua2/mu0/mut2_p1*(1.0-np.exp(-mut2_p1*L2))*np.exp(-mut1_p1*L1)
A3_p1 += Ps1_p1*mua3/mu0/mut3_p1*(1.0-np.exp(-mut3_p1*L3))*np.exp(-mut1_p1*L1-mut2_p1*L2)

# Absorbance, lower source
B1_p1 = Ps2_p1*mua3*np.sqrt(4.0*np.pi)*((ev3_p1[0,0]*ew3_p1*(1.0-np.exp(-L3/ew3_p1))*(c_rev_p1[0]+c_rev_p1[1]))+eps3_p1[0]/mut3_p1*(1.0-np.exp(-mut3_p1*L3)))
B2_p1 = Ps2_p1*mua2*np.sqrt(4.0*np.pi)*((ev2_p1[0,0]*ew2_p1*(1.0-np.exp(-L2/ew2_p1))*(c_rev_p1[2]+c_rev_p1[3]))+eps2_p1[0]*np.exp(-mut3_p1*L3)/mut2_p1*(1.0-np.exp(-mut2_p1*L2)))
B3_p1 = Ps2_p1*mua1*np.sqrt(4.0*np.pi)*((ev1_p1[0,0]*ew1_p1*(1.0-np.exp(-L1/ew1_p1))*(c_rev_p1[4]+c_rev_p1[5]))+eps1_p1[0]*np.exp(-mut3_p1*L3-mut2_p1*L2)/mut1_p1*(1.0-np.exp(-mut1_p1*L1)))

# Absorbance contribution from direct lower part
B1_p1 += Ps2_p1*mua3/mu0/mut3_p1*(1.0-np.exp(-mut3_p1*L3))
B2_p1 += Ps2_p1*mua2/mu0/mut2_p1*(1.0-np.exp(-mut2_p1*L2))*np.exp(-mut3_p1*L3)
B3_p1 += Ps2_p1*mua1/mu0/mut1_p1*(1.0-np.exp(-mut1_p1*L1))*np.exp(-mut3_p1*L3-mut2_p1*L2)

print("P1 Layer 1 Absorbance: %f" % np.abs(A1_p1+B3_p1))
print("P1 Layer 2 Absorbance: %f" % np.abs(A2_p1+B2_p1))
print("P1 Layer 3 Absorbance: %f" % np.abs(A3_p1+B1_p1))
print("P1 Total: %0.14f" % np.abs(Ro_p1+R1_p1+R2_p1+Ru_p1+T1_p1+T2_p1+A1_p1+A2_p1+A3_p1+B1_p1+B2_p1+B3_p1))

# %% P3 total

print("P3 upper source strength: %f " % Ps1_p3)
print("P3 lower source strength: %f " % Ps2_p3)

F1_p3 = 0.0
F2_p3 = 0.0
G1_p3 = 0.0
G2_p3 = 0.0

for i in range(4):
    i2 = i + 4
    F1_p3 += (c_p3[i]-c_p3[i2]*np.exp(-L1/ew1_p3[i]))*ev1_p3[i,1]
    F2_p3 += (c_p3[i+4*4]*np.exp(-L3/ew3_p3[i])-c_p3[i2+4*4])*ev3_p3[i,1]
    G1_p3 += (c_rev_p3[i]-c_rev_p3[i2]*np.exp(-L3/ew3_p3[i]))*ev3_p3[i,1]
    G2_p3 += (c_rev_p3[i+4*4]*np.exp(-L1/ew1_p3[i])-c_rev_p3[i2+4*4])*ev1_p3[i,1]

# Reflectance from upper and lower source
R1_p3 = -Ps1_p3*(np.sqrt(4.0/3.0*np.pi)*(F1_p3+eps1_p3[1]))
R2_p3 = Ps2_p3*(np.sqrt(4.0/3.0*np.pi)*(G2_p3+eps1_p3[1]*np.exp(-mut1_p3*L1-mut2_p3*L2-mut3_p3*L3)))
print("P3 Reflectance: %f " % np.abs(Ro_p3+R1_p3+R2_p3))

# Transmittance from lower and upper source
T1_p3 = Ps1_p3*(np.sqrt(4.0/3.0*np.pi)*(F2_p3+eps3_p3[1]*np.exp(-mut1_p3*L1-mut2_p3*L2-mut3_p3*L3)))
T2_p3 = -Ps2_p3*(np.sqrt(4.0/3.0*np.pi)*(G1_p3+eps3_p3[1]))
print("P3 Transmittance: %f " % np.abs(Ru_p3+T1_p3+T2_p3))

# Absorbance, upper source
A1_p3 = Ps1_p3*mua1*np.sqrt(4.0*np.pi)*(np.sum(ev1_p3[:,0]*ew1_p3*(1.0-np.exp(-L1/ew1_p3))*(c_p3[0:4]+c_p3[4:2*4]))+eps1_p3[0]/mut1_p3*(1.0-np.exp(-mut1_p3*L1)))
A2_p3 = Ps1_p3*mua2*np.sqrt(4.0*np.pi)*(np.sum(ev2_p3[:,0]*ew2_p3*(1.0-np.exp(-L2/ew2_p3))*(c_p3[2*4:3*4]+c_p3[3*4:4*4]))+eps2_p3[0]*np.exp(-mut1_p3*L1)/mut2_p3*(1.0-np.exp(-mut2_p3*L2)))
A3_p3 = Ps1_p3*mua3*np.sqrt(4.0*np.pi)*(np.sum(ev3_p3[:,0]*ew3_p3*(1.0-np.exp(-L3/ew3_p3))*(c_p3[4*4:5*4]+c_p3[5*4:6*4]))+eps3_p3[0]*np.exp(-mut1_p3*L1-mut2_p3*L2)/mut3_p3*(1.0-np.exp(-mut3_p3*L3)))

# Absorbance contribution from direct upper part
A1_p3 += Ps1_p3*mua1/mu0/mut1_p3*(1.0-np.exp(-mut1_p3*L1))
A2_p3 += Ps1_p3*mua2/mu0/mut2_p3*(1.0-np.exp(-mut2_p3*L2))*np.exp(-mut1_p3*L1)
A3_p3 += Ps1_p3*mua3/mu0/mut3_p3*(1.0-np.exp(-mut3_p3*L3))*np.exp(-mut1_p3*L1-mut2_p3*L2)

# Absorbance, lower source
B1_p3 = Ps2_p3*mua3*np.sqrt(4.0*np.pi)*(np.sum(ev3_p3[:,0]*ew3_p3*(1.0-np.exp(-L3/ew3_p3))*(c_rev_p3[0:4]+c_rev_p3[4:2*4]))+eps3_p3[0]/mut3_p3*(1.0-np.exp(-mut3_p3*L3)))
B2_p3 = Ps2_p3*mua2*np.sqrt(4.0*np.pi)*(np.sum(ev2_p3[:,0]*ew2_p3*(1.0-np.exp(-L2/ew2_p3))*(c_rev_p3[2*4:3*4]+c_rev_p3[3*4:4*4]))+eps2_p3[0]*np.exp(-mut3_p3*L3)/mut2_p3*(1.0-np.exp(-mut2_p3*L2)))
B3_p3 = Ps2_p3*mua1*np.sqrt(4.0*np.pi)*(np.sum(ev1_p3[:,0]*ew1_p3*(1.0-np.exp(-L1/ew1_p3))*(c_rev_p3[4*4:5*4]+c_rev_p3[5*4:6*4]))+eps1_p3[0]*np.exp(-mut3_p3*L3-mut2_p3*L2)/mut1_p3*(1.0-np.exp(-mut1_p3*L1)))

# Absorbance contribution from direct lower part
B1_p3 += Ps2_p3*mua3/mu0/mut3_p3*(1.0-np.exp(-mut3_p3*L3))
B2_p3 += Ps2_p3*mua2/mu0/mut2_p3*(1.0-np.exp(-mut2_p3*L2))*np.exp(-mut3_p3*L3)
B3_p3 += Ps2_p3*mua1/mu0/mut1_p3*(1.0-np.exp(-mut1_p3*L1))*np.exp(-mut3_p3*L3-mut2_p3*L2)

print("P3 Layer 1 Absorbance: %f" % np.abs(A1_p3+B3_p3))
print("P3 Layer 2 Absorbance: %f" % np.abs(A2_p3+B2_p3))
print("P3 Layer 3 Absorbance: %f" % np.abs(A3_p3+B1_p3))
print("P3 Total: %0.14f" % np.abs(Ro_p3+R1_p3+R2_p3+Ru_p3+T1_p3+T2_p3+A1_p3+A2_p3+A3_p3+B1_p3+B2_p3+B3_p3))

# %% PN total

print("PN upper source strength: %f " % Ps1)
print("PN lower source strength: %f " % Ps2)

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
R1 = -Ps1*(np.sqrt(4.0/3.0*np.pi)*(F1+eps1[1]))
R2 = Ps2*(np.sqrt(4.0/3.0*np.pi)*(G2+eps1[1]*np.exp(-mut1/mu0*L1-mut2/mu0*L2-mut3/mu0*L3)))

# Transmittance from lower and upper source
T1 = Ps1*(np.sqrt(4.0/3.0*np.pi)*(F2+eps3[1]*np.exp(-mut1/mu0*L1-mut2/mu0*L2-mut3/mu0*L3)))
T2 = -Ps2*(np.sqrt(4.0/3.0*np.pi)*(G1+eps3[1]))

print("PN Reflectance: %f" % (Ro+R1+R2))
print("PN Transmittance: %f" % (Ru+T1+T2))

# Absorbance, upper source
A1 = Ps1*mua1*np.sqrt(4.0*np.pi)*(np.sum(ev1[:,0]*ew1*(1.0-np.exp(-L1/ew1))*(c[0:b]+c[b:2*b]))+eps1[0]*mu0/mut1*(1.0-np.exp(-mut1/mu0*L1)))
A2 = Ps1*mua2*np.sqrt(4.0*np.pi)*(np.sum(ev2[:,0]*ew2*(1.0-np.exp(-L2/ew2))*(c[2*b:3*b]+c[3*b:4*b]))+eps2[0]*np.exp(-mut1/mu0*L1)*mu0/mut2*(1.0-np.exp(-mut2/mu0*L2)))
A3 = Ps1*mua3*np.sqrt(4.0*np.pi)*(np.sum(ev3[:,0]*ew3*(1.0-np.exp(-L3/ew3))*(c[4*b:5*b]+c[5*b:6*b]))+eps3[0]*np.exp(-mut1/mu0*L1-mut2/mu0*L2)*mu0/mut3*(1.0-np.exp(-mut3/mu0*L3)))

# Absorbance contribution from direct upper part
A1 += Ps1*mua1/mut1*(1.0-np.exp(-mut1/mu0*L1))
A2 += Ps1*mua2/mut2*(1.0-np.exp(-mut2/mu0*L2))*np.exp(-mut1/mu0*L1)
A3 += Ps1*mua3/mut3*(1.0-np.exp(-mut3/mu0*L3))*np.exp(-mut1/mu0*L1-mut2/mu0*L2)

# Absorbance, lower source
B1 = Ps2*mua3*np.sqrt(4.0*np.pi)*(np.sum(ev3[:,0]*ew3*(1.0-np.exp(-L3/ew3))*(c_rev[0:b]+c_rev[b:2*b]))+eps3[0]*mu0/mut3*(1.0-np.exp(-mut3/mu0*L3)))
B2 = Ps2*mua2*np.sqrt(4.0*np.pi)*(np.sum(ev2[:,0]*ew2*(1.0-np.exp(-L2/ew2))*(c_rev[2*b:3*b]+c_rev[3*b:4*b]))+eps2[0]*np.exp(-mut3/mu0*L3)*mu0/mut2*(1.0-np.exp(-mut2/mu0*L2)))
B3 = Ps2*mua1*np.sqrt(4.0*np.pi)*(np.sum(ev1[:,0]*ew1*(1.0-np.exp(-L1/ew1))*(c_rev[4*b:5*b]+c_rev[5*b:6*b]))+eps1[0]*np.exp(-mut3/mu0*L3-mut2/mu0*L2)*mu0/mut1*(1.0-np.exp(-mut1/mu0*L1)))

# Absorbance contribution from direct lower part
B1 += Ps2*mua3/mut3*(1.0-np.exp(-mut3/mu0*L3))
B2 += Ps2*mua2/mut2*(1.0-np.exp(-mut2/mu0*L2))*np.exp(-mut3/mu0*L3)
B3 += Ps2*mua1/mut1*(1.0-np.exp(-mut1/mu0*L1))*np.exp(-mut3/mu0*L3-mut2/mu0*L2)

print("PN Layer 1 Absorbance: %f" % (A1+B3))
print("PN Layer 2 Absorbance: %f" % (A2+B2))
print("PN Layer 3 Absorbance: %f" % (A3+B1))
print("PN Total: %0.14f" % (Ro+R1+R2+Ru+T1+T2+A1+A2+A3+B1+B2+B3))

z1 = np.linspace(0,L1,1000)
z2 = np.linspace(0,L2,1000)
z3 = np.linspace(0,L3,1000)

# %% P1 fluence

i = 0
i2 = 1

K1u_p1 = (c_p1[i]*np.exp(-z1/ew1_p1)+c_p1[i2]*np.exp((z1-L1)/ew1_p1))*ev1_p1[0,0]
K2u_p1 = (c_p1[2*1+i]*np.exp(-z2/ew2_p1)+c_p1[2*1+i2]*np.exp((z2-L2)/ew2_p1))*ev2_p1[0,0]
K3u_p1 = (c_p1[4*1+i]*np.exp(-z3/ew3_p1)+c_p1[4*1+i2]*np.exp((z3-L3)/ew3_p1))*ev3_p1[0,0]
K3l_p1 = (c_rev_p1[i]*np.exp(-(L3-z3)/ew3_p1)+c_rev_p1[i2]*np.exp(((L3-z3)-L3)/ew3_p1))*ev3_p1[0,0]
K2l_p1 = (c_rev_p1[2*1+i]*np.exp(-(L2-z2)/ew2_p1)+c_rev_p1[2*1+i2]*np.exp(((L2-z2)-L2)/ew2_p1))*ev2_p1[0,0]
K1l_p1 = (c_rev_p1[4*1+i]*np.exp(-(L1-z1)/ew1_p1)+c_rev_p1[4*1+i2]*np.exp(((L1-z1)-L1)/ew1_p1))*ev1_p1[0,0]

K1u_p1 = Ps1_p1*(np.sqrt(4.0*np.pi)*(K1u_p1+eps1_p1[0]*np.exp(-mut1_p1*z1)) + np.exp(-mut1_p1*z1)/mu0)
K2u_p1 = Ps1_p1*(np.sqrt(4.0*np.pi)*(K2u_p1+eps2_p1[0]*np.exp(-mut1_p1*L1-mut2_p1*z2)) + np.exp(-mut1_p1*L1-mut2_p1*z2)/mu0)
K3u_p1 = Ps1_p1*(np.sqrt(4.0*np.pi)*(K3u_p1+eps3_p1[0]*np.exp(-mut1_p1*L1-mut2_p1*L2-mut3_p1*z3)) + np.exp(-mut1_p1*L1-mut2_p1*L2-mut3_p1*z3)/mu0)
K3l_p1 = Ps2_p1*(np.sqrt(4.0*np.pi)*(K3l_p1+eps3_p1[0]*np.exp(-mut3_p1*(L3-z3))) + np.exp(-mut3_p1*(L3-z3))/mu0)
K2l_p1 = Ps2_p1*(np.sqrt(4.0*np.pi)*(K2l_p1+eps2_p1[0]*np.exp(-mut3_p1*L3-mut2_p1*(L2-z2))) + np.exp(-mut3_p1*L3-mut2_p1*(L2-z2))/mu0)
K1l_p1 = Ps2_p1*(np.sqrt(4.0*np.pi)*(K1l_p1+eps1_p1[0]*np.exp(-mut3_p1*L3-mut2_p1*L2-mut1_p1*(L1-z1))) + np.exp(-mut3_p1*L3-mut2_p1*L2-mut1_p1*(L1-z1))/mu0)


# %% P3 fluence

K1u_p3 = np.zeros_like(z1,dtype=np.complex)
K2u_p3 = np.zeros_like(z2,dtype=np.complex)
K3u_p3 = np.zeros_like(z3,dtype=np.complex)

K1l_p3 = np.zeros_like(z1,dtype=np.complex)
K2l_p3 = np.zeros_like(z2,dtype=np.complex)
K3l_p3 = np.zeros_like(z3,dtype=np.complex)

for i in range(4):
    i2 = i + 4
    K1u_p3 += (c_p3[i]*np.exp(-z1/ew1_p3[i])+c_p3[i2]*np.exp((z1-L1)/ew1_p3[i]))*ev1_p3[i,0]
    K2u_p3 += (c_p3[2*4+i]*np.exp(-z2/ew2_p3[i])+c_p3[2*4+i2]*np.exp((z2-L2)/ew2_p3[i]))*ev2_p3[i,0]
    K3u_p3 += (c_p3[4*4+i]*np.exp(-z3/ew3_p3[i])+c_p3[4*4+i2]*np.exp((z3-L3)/ew3_p3[i]))*ev3_p3[i,0]
    K3l_p3 += (c_rev_p3[i]*np.exp(-(L3-z3)/ew3_p3[i])+c_rev_p3[i2]*np.exp(((L3-z3)-L3)/ew3_p3[i]))*ev3_p3[i,0]
    K2l_p3 += (c_rev_p3[2*4+i]*np.exp(-(L2-z2)/ew2_p3[i])+c_rev_p3[2*4+i2]*np.exp(((L2-z2)-L2)/ew2_p3[i]))*ev2_p3[i,0]
    K1l_p3 += (c_rev_p3[4*4+i]*np.exp(-(L1-z1)/ew1_p3[i])+c_rev_p3[4*4+i2]*np.exp(((L1-z1)-L1)/ew1_p3[i]))*ev1_p3[i,0]

K1u_p3 = Ps1_p3*(np.sqrt(4.0*np.pi)*(K1u_p3+eps1_p3[0]*np.exp(-mut1_p3*z1)) + np.exp(-mut1_p3*z1)/mu0)
K2u_p3 = Ps1_p3*(np.sqrt(4.0*np.pi)*(K2u_p3+eps2_p3[0]*np.exp(-mut1_p3*L1-mut2_p3*z2)) + np.exp(-mut1_p3*L1-mut2_p3*z2)/mu0)
K3u_p3 = Ps1_p3*(np.sqrt(4.0*np.pi)*(K3u_p3+eps3_p3[0]*np.exp(-mut1_p3*L1-mut2_p3*L2-mut3_p3*z3)) + np.exp(-mut1_p3*L1-mut2_p3*L2-mut3_p3*z3)/mu0)
K3l_p3 = Ps2_p3*(np.sqrt(4.0*np.pi)*(K3l_p3+eps3_p3[0]*np.exp(-mut3_p3*(L3-z3))) + np.exp(-mut3_p3*(L3-z3))/mu0)
K2l_p3 = Ps2_p3*(np.sqrt(4.0*np.pi)*(K2l_p3+eps2_p3[0]*np.exp(-mut3_p3*L3-mut2_p3*(L2-z2))) + np.exp(-mut3_p3*L3-mut2_p3*(L2-z2))/mu0)
K1l_p3 = Ps2_p3*(np.sqrt(4.0*np.pi)*(K1l_p3+eps1_p3[0]*np.exp(-mut3_p3*L3-mut2_p3*L2-mut1_p3*(L1-z1))) + np.exp(-mut3_p3*L3-mut2_p3*L2-mut1_p3*(L1-z1))/mu0)

# %% PN fluence

K1u = np.zeros_like(z1)
K2u = np.zeros_like(z2)
K3u = np.zeros_like(z3)

K1l = np.zeros_like(z1)
K2l = np.zeros_like(z2)
K3l = np.zeros_like(z3)

for i in range(b):
    i2 = i + b
    K1u += (c[i]*np.exp(-z1/ew1[i])+c[i2]*np.exp((z1-L1)/ew1[i]))*ev1[i,0]
    K2u += (c[2*b+i]*np.exp(-z2/ew2[i])+c[2*b+i2]*np.exp((z2-L2)/ew2[i]))*ev2[i,0]
    K3u += (c[4*b+i]*np.exp(-z3/ew3[i])+c[4*b+i2]*np.exp((z3-L3)/ew3[i]))*ev3[i,0]
    K3l += (c_rev[i]*np.exp(-(L3-z3)/ew3[i])+c_rev[i2]*np.exp(((L3-z3)-L3)/ew3[i]))*ev3[i,0]
    K2l += (c_rev[2*b+i]*np.exp(-(L2-z2)/ew2[i])+c_rev[2*b+i2]*np.exp(((L2-z2)-L2)/ew2[i]))*ev2[i,0]
    K1l += (c_rev[4*b+i]*np.exp(-(L1-z1)/ew1[i])+c_rev[4*b+i2]*np.exp(((L1-z1)-L1)/ew1[i]))*ev1[i,0]

K1u = Ps1*(np.sqrt(4.0*np.pi)*(K1u+eps1[0]*np.exp(-mut1/mu0*z1)) + np.exp(-mut1/mu0*z1)/mu0)
K2u = Ps1*(np.sqrt(4.0*np.pi)*(K2u+eps2[0]*np.exp(-mut1/mu0*L1-mut2/mu0*z2)) + np.exp(-mut1/mu0*L1-mut2/mu0*z2)/mu0)
K3u = Ps1*(np.sqrt(4.0*np.pi)*(K3u+eps3[0]*np.exp(-mut1/mu0*L1-mut2/mu0*L2-mut3/mu0*z3)) + np.exp(-mut1/mu0*L1-mut2/mu0*L2-mut3/mu0*z3)/mu0)
K3l = Ps2*(np.sqrt(4.0*np.pi)*(K3l+eps3[0]*np.exp(-mut3/mu0*(L3-z3))) + np.exp(-mut3/mu0*(L3-z3))/mu0)
K2l = Ps2*(np.sqrt(4.0*np.pi)*(K2l+eps2[0]*np.exp(-mut3/mu0*L3-mut2/mu0*(L2-z2))) + np.exp(-mut3/mu0*L3-mut2/mu0*(L2-z2))/mu0)
K1l = Ps2*(np.sqrt(4.0*np.pi)*(K1l+eps1[0]*np.exp(-mut3/mu0*L3-mut2/mu0*L2-mut1/mu0*(L1-z1))) + np.exp(-mut3/mu0*L3-mut2/mu0*L2-mut1/mu0*(L1-z1))/mu0)

# %% Correction

corr1_p3 = (A1+B3)/np.abs(A1_p3+B3_p3)
corr2_p3 = (A2+B2)/np.abs(A2_p3+B2_p3)
corr3_p3 = (A3+B1)/np.abs(A3_p3+B1_p3)

corr1_p1 = (A1+B3)/np.abs(A1_p1+B3_p1)
corr2_p1 = (A2+B2)/np.abs(A2_p1+B2_p1)
corr3_p1 = (A3+B1)/np.abs(A3_p1+B1_p1)

z = np.concatenate([z1,L1+z2,L1+L2+z3])
abs_pn = np.concatenate([mua1*(K1u+K1l),mua2*(K2u+K2l),mua3*(K3u+K3l)])
abs_p3 = np.concatenate([mua1*(K1u_p3+K1l_p3),mua2*(K2u_p3+K2l_p3),mua3*(K3u_p3+K3l_p3)])
abs_p3_cor = np.concatenate([corr1_p3*mua1*(K1u_p3+K1l_p3),corr2_p3*mua2*(K2u_p3+K2l_p3),corr3_p3*mua3*(K3u_p3+K3l_p3)])

abs_p1 = np.concatenate([mua1*(K1u_p1+K1l_p1),mua2*(K2u_p1+K2l_p1),mua3*(K3u_p1+K3l_p1)])
abs_p1_cor = np.concatenate([corr1_p1*mua1*(K1u_p1+K1l_p1),corr2_p1*mua2*(K2u_p1+K2l_p1),corr3_p1*mua3*(K3u_p1+K3l_p1)])

# %% plot

SMALL_SIZE = 24
MEDIUM_SIZE = 28
BIGGER_SIZE = 32

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


fig = plt.figure()
ax = plt.subplot(111)

ax.plot(z,np.concatenate([K1u+K1l,K2u+K2l,K3u+K3l]),label='P201')

ax.plot(z,np.abs(np.concatenate([K1u_p3+K1l_p3,K2u_p3+K2l_p3,K3u_p3+K3l_p3])),label='P3')
ax.plot(z,np.abs(np.concatenate([corr1_p3*(K1u_p3+K1l_p3),corr2_p3*(K2u_p3+K2l_p3),corr3_p3*(K3u_p3+K3l_p3)])),label='P3 corrected')

ax.plot(z,np.abs(np.concatenate([K1u_p1+K1l_p1,K2u_p1+K2l_p1,K3u_p1+K3l_p1])),label='P1')
ax.plot(z,np.abs(np.concatenate([corr1_p1*(K1u_p1+K1l_p1),corr2_p1*(K2u_p1+K2l_p1),corr3_p1*(K3u_p1+K3l_p1)])),label='P1 corrected')

plt.xlabel('z / mm')
plt.ylabel('Fluence')

ax.set_xlim([0.0,5.0])

ax.legend(loc='upper right')

# tikz_save('skin_fluence_60.tex',strict=True)
plt.show()
