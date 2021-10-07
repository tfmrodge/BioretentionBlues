# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:47:03 2021

@author: Tim Rodgers
"""

#Run the Oro Loma model in one module.
import time
import pandas as pd
import numpy as np
from Loma_Loadings import LomaLoadings
import pdb
import math
import hydroeval #For the efficiency
from HelperFuncs import df_sliced_index
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from hydroeval import kge

#plt.style.use("ggplot")
#from data_1d import makeic

params = pd.read_excel('params_1d.xlsx',index_col = 0) 
#params = pd.read_excel('params_OroLoma_CellF.xlsx',index_col = 0)
#params = pd.read_excel('params_OroLomaTracertest.xlsx',index_col = 0)
#Cell F and G
locsumm = pd.read_excel('Oro_Loma_CellF.xlsx',index_col = 0) 
#locsumm = pd.read_excel('Oro_Loma_CellG.xlsx',index_col = 0) 
#15 cm tracertest conducted with the Oro Loma system Mar. 2019
#locsumm = pd.read_excel('Oro_Loma_Brtracertest.xlsx',index_col = 0) 


#Specific Groups
chemsumm = pd.read_excel('Oro_ALLCHEMSUMMtest.xlsx',index_col = 0)
#chemsumm = pd.read_excel('Oro_ALLCHEMSUMM.xlsx',index_col = 0)

#timeseries = pd.read_excel('timeseries_OroLomaTracertest.xlsx')
#timeseries = pd.read_excel('timeseries_OroLoma_Dec_CellF.xlsx')
#timeseries = pd.read_excel('timeseries_OroLoma_Spinup_CellF_short.xlsx')
timeseries = pd.read_excel('timeseries_OroLoma_Spinup_CellF.xlsx')
pp = None
#numc = ['water', 'subsoil','topsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air']
numc = ['water', 'subsoil','topsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air']
test = LomaLoadings(locsumm,chemsumm,params,numc) 
#timeseries = timeseries.loc[timeseries.time>0,:]
pdb.set_trace()
start = time.time()
#First calculate Input calcs
res = test.input_calc(locsumm,chemsumm,params,pp,numc,timeseries)
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_inputs_spinuptest.pkl'
res.to_pickle(outpath)
#res = pd.read_pickle(outpath)

'''
ssdata = res.loc[(slice(None),slice(3),slice(None)),:].groupby(level = 0).sum()
for chem in chemsumm.index:
    #ssdata.loc[chem,'inp_1'] = timeseries.loc[:,chem+'_Cin'].mean()*timeseries.loc[:,'Qin'].mean()/chemsumm.loc[chem,'MolMass']
    #ssdata.loc[chem,'inp_1'] = timeseries.loc[:,chem+'_Cin'].mean()#*timeseries.loc[:,'Qin'].mean()/chemsumm.loc[chem,'MolMass']
    ssdata.loc[chem,'inp_1'] = timeseries.loc[:,chem+'_Cin'].mean()/chemsumm.MolMass[chem]/res.loc[(chem,slice(None),0),'Z1'].mean()
last_step = test.forward_calc_ss(ssdata,8)
'''
#pdb.set_trace()
#For a last step from a time dependent system
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_outs_spinuptest.pkl'
#outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_outs_spinup20210610test.pkl'
last_step = pd.read_pickle(outpath)
#pdb.set_trace()
#'''
try:
    last_step.index.levels[1]
    dt = timeseries.time.iloc[-1] - timeseries.time.iloc[-2]
    start = last_step.time[0] + dt
    stop = len(timeseries)*dt+start
    newtimes = pd.Series(np.arange(start,stop,dt))
    timeseries.loc[:,'time'] = newtimes
    res.loc[:,'time'] = newtimes.reindex(res.index,level=1)
except NameError:
    pass
#'''
#Main model
#res = test.run_it(locsumm,chemsumm,params,pp,numc,timeseries,res,last_step)
res = test.run_it(locsumm,chemsumm,params,pp,numc,timeseries,res)
mass_flux = test.mass_flux(res,numc) #Run to get mass flux
mbal = test.mass_balance(res,numc,mass_flux)
Couts = test.conc_out(numc,timeseries,chemsumm,res,mass_flux)
recovery = mass_flux.N_effluent.groupby(level=0).sum()/mass_flux.N_influent.groupby(level=0).sum()
#Kling-Gupta Efficiency (modified Nash-Sutcliffe) can be our measure of model performance
KGE = {}
pltnames = []
pltnames.append('time')
#Calculate performance and define what to plot
for chem in chemsumm.index:
    #If measured values provided
    try:
        KGE[chem] = hydroeval.evaluator(kge, np.array(Couts.loc[:,chem+'_Coutest']),\
                          np.array(Couts.loc[:,chem+'_Coutmeas']))
        pltnames.append(chem+'_Coutmeas')
    except KeyError:    
        pass
    pltnames.append(chem+'_Coutest')
    

#plot it    
pltdata = Couts[pltnames]
pltdata = pltdata.melt('time',var_name = 'Test_vs_est',value_name = 'Cout (mg/L)')
ylim = [0, 50]
ylabel = 'Cout (mg/L)'
xlabel = 'Time'
#pltdata = res_time #All times at once
fig = plt.figure(figsize=(14,8))
ax = sns.lineplot(x = pltdata.time, y = 'Cout (mg/L)', hue = 'Test_vs_est',data = pltdata)

#ax.set_ylim(ylim)
ax.set_ylabel(ylabel, fontsize=20)
ax.set_xlabel(xlabel, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
#Save it
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_outs_test.pkl'
#outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_outs_630max.pkl'
res.to_pickle(outpath)
#Save a new last_step
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_outs_spinuptest.pkl'
last_step = res.loc[(slice(None),max(res.index.levels[1]),slice(None)),:]
last_step.to_pickle(outpath)


