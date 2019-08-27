# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:12:51 2019

@author: Tim Rodgers
"""

import time
import pandas as pd
import numpy as np
from BCBlues import BCBlues
from HelperFuncs import df_sliced_index
import seaborn as sns; sns.set()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
import math
#plt.style.use("ggplot")
#from data_1d import makeic

params = pd.read_excel('params_BC.xlsx',index_col = 0) 
locsumm = pd.read_excel('Kortright_BC.xlsx',index_col = 0) 
#locsumm = pd.read_excel('Oro_Loma_1.xlsx',index_col = 0) 
chemsumm = pd.read_excel('OPE_only_CHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('OPECHEMSUMM.xlsx',index_col = 0)
#emsumm = pd.read_excel('PROBLEMCHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('EHDPPCHEMSUMM.xlsx',index_col = 0)
timeseries = pd.read_excel('timeseries_test_bc.xlsx')
#Truncate timeseries if you want to run fewer

run_period = 240 #if run_period/dt not a whole number there will be a problem
dt = timeseries.time[1] - timeseries.time[0]
totalt = int(math.ceil(run_period/dt))
if totalt <= len(timeseries):
    timeseries = timeseries[0:totalt+1]
else:
    while math.ceil(totalt/len(timeseries)) > 2.0:
        timeseries = timeseries.append(timeseries)
    totalt = totalt - len(timeseries)
    timeseries = timeseries.append(timeseries[0:totalt])
    timeseries.loc[:,'time'] = np.arange(dt,run_period+dt,dt)
    timeseries.index = range(len(timeseries))
pdb.set_trace()    
numc = np.array(np.concatenate([locsumm.index[0:2].values,locsumm.index[3:10].values]),dtype = 'str') 
pp = None
test = BCBlues(locsumm,chemsumm,params,timeseries,numc) #Leave as 9
#res = test.make_system(locsumm,params,numc)
#chemsumm = test.make_chems(chemsumm,pp=None)
#res = test.input_calc(locsumm,chemsumm,params,pp,numc)
start = time.time()
#chemsumm, res = test.sys_chem(locsumm,chemsumm,params,pp,numc)

#res = test.ic
#bc_dims = test.bc_dims(locsumm,inflow,rainrate,dt,params)
#res_time = test.flow_time(locsumm,timeseries,params)
res_time =pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/Flow_time.pkl')
res_time = df_sliced_index(res_time.loc[(slice(0,5),slice(None)),:])
#res = test.make_system(res_time,params,numc)
res_t = test.input_calc(locsumm,chemsumm,params,pp,numc,res_time) #Give entire time series - will not run flow module
#res_t = test.input_calc(locsumm,chemsumm,params,pp,numc,timeseries) #Give timeseries values - will run the flows again
#res_t, res_time = test.run_it(locsumm,chemsumm,params,numc,pp,timeseries)
#mf = test.mass_flux(res_time,numc)
#res_t, res_time = test.run_it(locsumm,chemsumm,params,1,pp,timeseries)


codetime = (time.time()-start)

yvar = 'V'
comp1 = 'water'
comp2 = 'drain_pores'
comp3 = 'pond'
pltdata = res_time.loc[(slice(None),['water','pond','drain_pores']),:] #To plot volumes
#res_time.loc[(plttime,slice(None),slice(None)),slice(None)] #Just at plttime
ylim = [0, 45]
xlim = [0, 1000]
ylabel = 'Concentration (μg/g dw)'
xlabel = 'Time'
#pltdata = res_time #All times at once
fig = plt.figure(figsize=(14,8))
#fig = plt.figure(figsize=(14,8))
ax = sns.lineplot(x = pltdata.index.get_level_values(0),hue = pltdata.index.get_level_values(1),y='V',data = pltdata)
ax.set_ylim(ylim)#
#axes[1].set_xlim(xlim)
ax.tick_params(axis='both', which='major', labelsize=15)


outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/output.pkl'
res_time.to_pickle(outpath)