# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:37:32 2019

@author: Tim Rodgers
"""
import pandas as pd
import numpy as np
from Hydro_veg import Hydro_veg
import pdb
params = pd.read_excel('params_hydro.xlsx',index_col = 0) 
locsumm = pd.read_excel('Hydro_veg.xlsx',index_col = 0) 
chemsumm = pd.read_excel('OPE_only_CHEMSUM_hydro.xlsx',index_col = 0)
#chemsumm = pd.read_excel('OPECHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('EHDPPCHEMSUMM.xlsx',index_col = 0)
timeseries = pd.read_excel('timeseries_wanhydro1.xlsx')
tspan = np.arange(0,250,1)
test = Hydro_veg(locsumm,chemsumm,params,6)
#pdb.set_trace()
#res_time, sols = test.ivp_hydro(locsumm,chemsumm,params,timeseries,tspan,numc=6,pp=None,outtype = 'maxi')
res_t,sols = test.ivp_hydro(locsumm,chemsumm,params,timeseries,tspan,numc=6,pp=None,outtype = 'mini')

