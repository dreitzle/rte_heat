#!/usr/bin/env python3
# coding: utf-8

""" read_comsol_grid.py: read comsol grid data """


import numpy as np

__author__ = "Simeon Geiger"
__copyright__ = "Copyright 2021, ILM"
__credits__ = ["Simeon Geiger", "Dominik Reitzle"]
__license__ = "GPL"

def read_comsol_grid(filename):

    with open(filename, 'r') as infile:
        line = infile.readline()
        # skip comments
        while line and line.startswith('%'):
            line = infile.readline()
            
        # read coordinates
        coords=[]
        while line and not line.startswith('%'):
            coords.append(np.fromstring(line,sep=' '))
            line = infile.readline()
            
        # read data
        # skip coomments but keep last comment to get time
        times=[]
        data=[]
        while line:
            if(line.startswith('%')):
                prevline=line
                line = infile.readline()
            else:
                try:
                    times.append(float(prevline.split('=')[-1]))
                except ValueError:
                    print("cant parse time")
                tempdata=[]
                while line and not line.startswith('%'):
                    tempdata.append(np.fromstring(line,sep=' '))
                    line = infile.readline()
                data.append(tempdata)
                
        coords.append(np.array(times))
        resultdata=np.array(data)
        coords.reverse()
        
    return resultdata, coords
    