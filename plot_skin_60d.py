#!/usr/bin/env python3
# coding: utf-8

"""plot_skin_60d.py: plot temperature profile for oblique incidence"""

import numpy as np
from functools import partial

from RTE_heat_3L import Layer
from RTE_heat_3L_thick import rte_heat_3L

from numeric_transform import ILT_IFT_hyperbolic
from numeric_transform import IFT_pts

import matplotlib.pyplot as plt
#from tikzplotlib import save as tikz_save

import time

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2021, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"

# default source function

def _src_switch(s):
    return 1.0/s

# layers

layer1 = Layer(
     rho     = 1100.0,
     cp      = 3000.0,
     k       = 0.25,
     musp    = 2.0,
     mua     = 0.02,
     g       = 0.8,
     L       = 1.0
     )
layer2 = Layer(
     rho     = 900,
     cp      = 3000.0,
     k       = 0.20,
     musp    = 1.0,
     mua     = 0.003,
     g       = 0.8,
     L       = 2.0
     )
layer3 = Layer(
     rho     = 1100.0,
     cp      = 3500.0,
     k       = 0.5,
     musp    = 0.5,
     mua     = 0.04,
     g       = 0.8,
     L       = 10.0
     )

# global parameters

mu1 = np.cos(60/180*np.pi)
NN = 100

h1 = 100.0
h2 = 0.0

rte_heat = rte_heat_3L(layer1,layer2,layer3,h1,h2,mu1=mu1,NN=NN,n=1.4,Rn_file_n='Rn__1-4.npz')

calcfn_b = partial(rte_heat_3L.calc_laplace, rte_heat, z=0.0)
# calcfn_b = partial(rte_heat_3L.calc_laplace, rte_heat, z=4.0)

# Source strength
Q = 1.0

# Source beam radius
rw = 1.0

# Transform parameters
Ns = 35
Ns2 = 4
Nq = 240
Nphi = 80

# Source profile

def profile(q, phi):
    return np.exp(-0.125*q*q*rw*rw*(np.sin(phi)**2+(np.cos(phi)/mu1)**2))

Nr = 400
rm = 5.0
xx = np.linspace(-rm,rm,Nr)

q,phi = IFT_pts(Nq,Nphi)[0:2]

start = time.perf_counter()
rte_heat.calc_laplace_pre_P3(q,phi)
res = Q*ILT_IFT_hyperbolic(calcfn_b, profile, xx[:,None,None], 0.0, 15.0, Ns, Nq, Nphi)
stop = time.perf_counter()
print("P3 time: %f" % (stop-start))

start = time.perf_counter()
rte_heat.calc_laplace_pre_P1(q,phi)
res_p1 = Q*ILT_IFT_hyperbolic(calcfn_b, profile, xx[:,None,None], 0.0, 15.0, Ns, Nq, Nphi)
stop = time.perf_counter()
print("P1 time: %f" % (stop-start))

########
SMALL_SIZE = 24
MEDIUM_SIZE = 28
BIGGER_SIZE = 32

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


fig = plt.figure(dpi=300)
ax = plt.subplot(111)
##################

ax.plot(xx,res,label='analytical P3')
ax.plot(xx,res_p1,label='analytical P1')

# numdata = np.load('numerical_data/data_60d_short.npz')
numdata = np.load('numerical_data/data_60d_long.npz')

timeindex = 5
time = numdata['t'][timeindex]
print('Time: {} s'.format(time))

zindex = 0
# zindex = 40
z = numdata['z'][zindex]
print('Depth: {} mm'.format(z))

x = numdata['x'][::2]

data = numdata['data'][timeindex,zindex,40,::2]

ax.plot(x,data,'o',label='numerical',markersize=3)
ax.set_xlim([-5.0,5.0])

plt.xlabel('x / mm')
plt.ylabel('T / K')

lgd = plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.01), borderaxespad=0)
for legobj in lgd.legendHandles:
    legobj.set_linewidth(2.0)


plt.show()
