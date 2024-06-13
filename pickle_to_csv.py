#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:55:47 2024

@author: sascha
"""

import numpy as np
import glob
import pickle
import csv

files = glob.glob('/home/sascha/Desktop/vbm_torch/behav_fit/IC/*.p')

for file in files:
    print(f"Opening file {file}.")
    loglike, agent_indexer = pickle.load(open( file, "rb" ))
    # Specify the file name (you can include a path if needed)
    filename = f'{file[0:-2]}.csv'
    
    # Save the NumPy array to a CSV file
    np.savetxt(filename, loglike, delimiter=',', fmt='%.8f')
    np.savetxt(filename+'agent_indexer', agent_indexer, delimiter=',', fmt='%.8f')
    
    print(f"Data successfully written to {filename}")