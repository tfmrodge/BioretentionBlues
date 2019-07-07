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
locsumm = pd.read_excel('Hydro_veg1.xlsx',index_col = 0) 
chemsumm = pd.read_excel('OPE_only_CHEMSUM_hydro.xlsx',index_col = 0)
#chemsumm = pd.read_excel('OPECHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('EHDPPCHEMSUMM.xlsx',index_col = 0)
timeseries = pd.read_excel('timeseries_wanhydro2.xlsx') #for IVP solver

tspan = np.arange(0,250,1)
test = Hydro_veg(locsumm,chemsumm,params,6)
#pdb.set_trace()

res_time, sols = test.ivp_hydro(locsumm,chemsumm,params,timeseries,tspan,numc=6,pp=None,outtype = 'maxi')
mass_t, mass_bal = test.mass_bal(res_time,6)
res_time = test.mass_conc(res_time,6)
#res_t,sols = test.ivp_hydro(locsumm,chemsumm,params,timeseries,tspan,numc=6,pp=None,outtype = 'mini')


#Plot concentration in the roots and the shoots
yvar1 = 'shoot_conc'
yvar2 = 'root_conc'
yvar3 = 'water_conc'
pltdata = res_time.loc[(slice(None),slice(None),0),slice(None)]
#res_time.loc[(plttime,slice(None),slice(None)),slice(None)] #Just at plttime
ylim = [-0.5e-15, 7e-3]
xlim = [0, 1000]
ylabel = 'Concentration (μg/g dw)'
xlabel = 'Time'
#pltdata = res_time #All times at once
f, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
#fig = plt.figure(figsize=(14,8))
sns.lineplot(x = pltdata.index.get_level_values(0), y = yvar1, hue = pltdata.index.get_level_values(1),data = pltdata, ax = axes[0])
sns.lineplot(x = pltdata.index.get_level_values(0), y = yvar2, hue = pltdata.index.get_level_values(1),data = pltdata, ax = axes[1])
sns.lineplot(x = pltdata.index.get_level_values(0), y = yvar3, hue = pltdata.index.get_level_values(1),data = pltdata, ax = axes[2])
#ax[1].set_ylim(ylim)#
#axes[1].set_xlim(xlim)
axes[0].tick_params(axis='both', which='major', labelsize=15)
axes[1].tick_params(axis='both', which='major', labelsize=15)
axes[2].tick_params(axis='both', which='major', labelsize=15)

outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/hydro_output.pkl'
res_time.to_pickle(outpath)
