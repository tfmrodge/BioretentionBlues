# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:26:40 2019

@author: Tim Rodgers
"""

import time
import pandas as pd
import numpy as np
from BCBlues_1d import BCBlues_1d
import seaborn as sns; sns.set()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
#plt.style.use("ggplot")
#from data_1d import makeic

params = pd.read_excel('params_1d.xlsx',index_col = 0) 
#locsumm = pd.read_excel('Oro_Loma.xlsx',index_col = 0) 
locsumm = pd.read_excel('Oro_Loma.xlsx',index_col = 0) 
#chemsumm = pd.read_excel('OPECHEMSUMM.xlsx',index_col = 0)
chemsumm = pd.read_excel('PROBLEMCHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('EHDPPCHEMSUMM.xlsx',index_col = 0)
timeseries = pd.read_excel('timeseries_test2.xlsx')
#Truncate timeseries if you want to run fewer
totalt = 1
timeseries = timeseries[0:totalt+1]
numc = 8
pp = None
test = BCBlues_1d(locsumm,chemsumm,params,8)
#res_t, res_time = test.run_it(locsumm,chemsumm,params,numc,pp,timeseries)
ssdata = res_t[0].groupby(level = 0).sum(axis = 0)
ssdata.loc[:,'inp_1'] = res_t[0].bc_us[slice(None),0] #Set inputs to upstream equal to upstream boundary condition
SSouts = test.forward_calc_ss(ssdata,8)