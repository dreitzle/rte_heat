# coding: utf-8

"""
RTE_planar_slab_3L.py: Planar PN boundary system for a 3-layered slab
"""

import numpy as np
from fresnel import load_R_planar

from scipy.linalg.lapack import dgesv

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2021, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"

def get_calc_c_marshak_3L(Rn_file_n, Rn_file_1, NN):

    R = load_R_planar(Rn_file_n,NN)
    T = load_R_planar(Rn_file_1,NN)

    def calc_c_marshak_3L(
            L1,mut1,ew1,ev1,eps1,
            L2,mut2,ew2,ev2,eps2,
            L3,mut3,ew3,ev3,eps3,
            mu0):
    
        d = 2*(NN+1)
    
        m1l = -2*(np.arange(d)&1)+1
    
        Ts = -np.transpose(np.transpose(T)*m1l)
    
        ex1 = np.exp(-L1/ew1)
        ex2 = np.exp(-L2/ew2)
        ex3 = np.exp(-L3/ew3)
    
        mbc = np.zeros([3*d,3*d])
    
        b = NN+1
    
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
    
        rbc = np.concatenate([
            -eps1.dot(R),
            (eps2-eps1*np.exp(-mut1/mu0*L1)).dot(T),
            (eps2-eps1*np.exp(-mut1/mu0*L1)).dot(Ts),
            (eps3-eps2*np.exp(-mut2/mu0*L2)).dot(T),
            (eps3-eps2*np.exp(-mut2/mu0*L2)).dot(Ts),
            (eps3*m1l).dot(R)*np.exp(-mut3/mu0*L3)
            ])
    
        dgesv(mbc,rbc,overwrite_a = 1, overwrite_b = 1)
    
        return rbc
    
    return calc_c_marshak_3L
