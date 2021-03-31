#!/usr/bin/env python3
# coding: utf-8

"""
extract_numdata.py: Extract part of the numerical results for inclusion
in the repository, since the whole data set is too large (several GB)
"""

from read_comsol_grid import read_comsol_grid
import numpy as np

resultdata, coords = read_comsol_grid("2021_rev1/comsol_sim/export.txt")
print(resultdata.shape)

t = coords[0]
z = coords[1][::100]
r = coords[2][::10]

data = resultdata[:,::100,::10]

np.savez('data_0d_short.npz',data=data, t=t, z=z, r=r)

resultdata, coords = read_comsol_grid("05_12.2019_Comsol2d_3layered_zyl_gauss_2/export.txt")
print(resultdata.shape)

t = coords[0]
z = coords[1][::100]
r = coords[2][::10]

data = resultdata[:,::100,::10]

np.savez('data_0d_long.npz',data=data, t=t, z=z, r=r)

resultdata, coords = read_comsol_grid("2021_rev1/comsol_sim2/export.txt")
newshape = (coords[0].size, coords[1].size, coords[2].size, coords[3].size)
resultdata=resultdata.reshape(newshape)
print(resultdata.shape)

t = coords[0]
z = coords[1][::10]
y = coords[2]
x = coords[3]

data = resultdata[:,::10,:,:]

np.savez('data_60d_long.npz',data=data, t=t, z=z, y=y, x=x)

resultdata, coords = read_comsol_grid("2021_rev1/comsol_sim2/export_short.txt")
newshape = (coords[0].size, coords[1].size, coords[2].size, coords[3].size)
resultdata=resultdata.reshape(newshape)
print(resultdata.shape)

t = coords[0]
z = coords[1][::10]
y = coords[2]
x = coords[3]

data = resultdata[:,::10,:,:]

np.savez('data_60d_short.npz',data=data, t=t, z=z, y=y, x=x)