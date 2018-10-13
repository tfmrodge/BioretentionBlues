# -*- coding: utf-8 -*-
"""
Load data for the 1d ADRE BC Blues model
Created on Fri Oct 12 18:10:27 2018

@author: Tim Rodgers
"""

import pandas as pd
from data_1d import makeic
params = pd.read_excel('params_1d.xlsx',index_col = 0) 
ic = makeic()

