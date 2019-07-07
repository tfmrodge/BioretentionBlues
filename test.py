# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:19:07 2019
Load data for the hydroponic plant/water model
@author: Tim Rodgers
"""

import time
import pandas as pd
import numpy as np
from Hydro_veg import Hydro_veg
import seaborn as sns; sns.set()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
import math

params = pd.read_excel('params_hydro.xlsx',index_col = 0) 
locsumm = pd.read_excel('Hydro_veg.xlsx',index_col = 0) 
chemsumm = pd.read_excel('OPE_only_CHEMSUM_hydro.xlsx',index_col = 0)
#chemsumm = pd.read_excel('OPECHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('EHDPPCHEMSUMM.xlsx',index_col = 0)
timeseries_orig = pd.read_excel('timeseries_wanhydro1.xlsx')
timeseries = timeseries_orig.copy(deep=True)
#pdb.set_trace()
run_period = 250 #if run_period/dt not a whole number there will be a problem
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

test = Hydro_veg(locsumm,chemsumm,params,6)
#chemsumm, res = test.hydro_sys(locsumm,chemsumm,params,pp=None,numc=6)
#res = test.input_calc(locsumm,chemsumm,params,pp=None,numc=6)
res_t, res_time = test.run_hydro(locsumm,chemsumm,params,timeseries,numc=6,pp=None)
mass_t,mass_bal = test.mass_bal(res_time,6)

#Seaborn
#Set plotting parameters
yvar = 'a4_t'
pltdata = res_time.loc[(slice(None),slice(None),0),slice(None)]
#res_time.loc[(plttime,slice(None),slice(None)),slice(None)] #Just at plttime
ylim = [0, 0.3]
ylabel = 'Activity'
xlabel = 'Time'
#pltdata = res_time #All times at once
fig = plt.figure(figsize=(14,8))
ax = sns.lineplot(x = pltdata.index.get_level_values(0), y = yvar, hue = pltdata.index.get_level_values(1),data = pltdata)
#ax.set_ylim(ylim)
ax.set_ylabel(ylabel, fontsize=20)
ax.set_xlabel(xlabel, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)

outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/hydro_output.pkl'
res_time.to_pickle(outpath)