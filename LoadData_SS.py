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

params = pd.read_excel('params_1d_1.xlsx',index_col = 0) 
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

"""
#Plot activity in the water compartment

#Set plotting parameters
%matplotlib inline
plttime = 799
#yvar = 'a1_t1'
yvar = 'a/ao'
res_time.loc[:,'a/ao'] = res_time.loc[:,'a1_t']/res_time.loc[:,'bc_us']
pltdata = res_time.loc[(plttime,slice('EHDPP','TCEP','TCiPP','TBEP','TDCiPP','TPhP'),slice(None)),slice(None)]
#Just at plttime
ylim = [1, 1.5]
xlim = [0, 50]
ylabel = 'Activity (a/ain)'
xlabel = 'Distance from Influent (m)'
#pltdata = res_time #All times at once
fig = plt.figure(figsize=(30,15),dpi = 500)
ax = sns.lineplot(x = 'x', y = yvar, hue = pltdata.index.get_level_values(1),data = pltdata)
#ax.set_ylim(ylim)
ax.set_xlim(xlim)
ax.set_ylabel(ylabel, fontsize=44)
ax.set_xlabel(xlabel, fontsize=44)
ax.tick_params(axis='both', which='major', labelsize=36)
plt.setp(ax.get_legend().get_texts(), fontsize='32')
"""