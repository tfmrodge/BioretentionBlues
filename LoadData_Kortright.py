# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:12:51 2019

@author: Tim Rodgers
"""

import time
import pandas as pd
import numpy as np
from Subsurface_Sinks import Subsurface_Sinks
from Loma_Loadings import Loma_Loadings
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
timeseries = pd.read_excel('timeseries_test2.xlsx')
#Truncate timeseries if you want to run fewer
pdb.set_trace()
totalt = 100
if totalt <= len(timeseries):
    timeseries = timeseries[0:totalt+1]
else:
    while math.ceil(totalt/len(timeseries)) > 2.0:
        timeseries = timeseries.append(timeseries)
    totalt = totalt - len(timeseries)
    timeseries = timeseries.append(timeseries[0:totalt])
    timeseries.loc[:,'time'] = np.arange(1,len(timeseries)+1,timeseries.time.iloc[1]-timeseries.time.iloc[0])
    timeseries.index = range(len(timeseries))
    
numc = 9 #Change to 1 for pure advection testing of water compartment
pp = None
test = Loma_Loadings(locsumm,chemsumm,params,timeseries,9) #Leave as 9
#res = test.make_system(locsumm,params,numc)
#chemsumm = test.make_chems(chemsumm,pp=None)
#res = test.input_calc(locsumm,chemsumm,params,pp,numc)
start = time.time()
#chemsumm, res = test.sys_chem(locsumm,chemsumm,params,pp,numc)

#res = test.ic
res_t, res_time = test.run_it(locsumm,chemsumm,params,numc,pp,timeseries)
mf = test.mass_flux(res_time)
#res_t, res_time = test.run_it(locsumm,chemsumm,params,1,pp,timeseries)


codetime = (time.time()-start)
#res = test.input_calc(locsumm,chemsumm,params,None,4)
#ic = makeic()
#Matplotlib plot at a given x across time
plot = plt.plot(timeseries.time, res_time.loc[(slice(None),'EHDPP',0),'a1_t1'],\
   'r--', timeseries.time, res_time.loc[(slice(None),'EHDPP',5),'a1_t1'], 'b--',\
   timeseries.time, res_time.loc[(slice(None),'EHDPP',9),'a1_t1'], 'g--')
#Seaborn
#Set plotting parameters
plttime = 336
yvar = 'a1_t1'
pltdata = res_time.loc[(plttime,slice(None),slice(None)),slice(None)]
#res_time.loc[(plttime,slice(None),slice(None)),slice(None)] #Just at plttime
ylim = [-0.5e-15, 7e-3]
ylabel = 'Activity'
xlabel = 'Distance'
#pltdata = res_time #All times at once
fig = plt.figure(figsize=(14,8))
ax = sns.lineplot(x = 'x', y = yvar, hue = pltdata.index.get_level_values(1),data = pltdata)
#ax.set_ylim(ylim)
ax.set_ylabel(ylabel, fontsize=20)
ax.set_xlabel(xlabel, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)

outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/output.pkl'
res_time.to_pickle(outpath)