#!/usr/bin/env python3
# coding: utf-8

"""
saveR.py: generate coefficients for Marshak BC
"""

from fresnel import save_R_marshak_planar
from fresnel import save_R_marshak_planar_1_analytic

__author__ = "Dominik Reitzle"
__copyright__ = "Copyright 2021, ILM"
__credits__ = ["Dominik Reitzle", "Simeon Geiger"]
__license__ = "GPL"

save_R_marshak_planar_1_analytic('Rn__1-0_analytic.npz',500)

# save_R_marshak_planar(1.0,'Rn__1-0.npz',500)
save_R_marshak_planar(1.33,'Rn__1-33.npz',500)
save_R_marshak_planar(1.4,'Rn__1-4.npz',500)
save_R_marshak_planar(1.5,'Rn__1-5.npz',300)
save_R_marshak_planar(2.0,'Rn__2-0.npz',300)
