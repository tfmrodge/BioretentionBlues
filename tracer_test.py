# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:07:34 2019
Code for 2-compartment version for the tracer test
@author: Tim Rodgers
"""
import time
import pandas as pd
import numpy as np
from BCBlues import BCBlues
from HelperFuncs import df_sliced_index
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pdb
import math
import hydroeval #For the efficiency
from hydroeval import kge #Kling-Gupta efficiency (Kling-Gupta et al., 2009)
from hydroeval import *
#plt.style.use("ggplot")
#Testing slow drainage - how would this change performance? 
#params = pd.read_excel('params_BC_SlowDrain.xlsx',index_col = 0)
#locsumm = pd.read_excel('Kortright_FullBC.xlsx',index_col = 0)
#Next two are the "normal" ones
params = pd.read_excel('params_BC_5.xlsx',index_col = 0) 
#params = pd.read_excel('params_BC_synthetic.xlsx',index_col = 0)
locsumm = pd.read_excel('Kortright_BC.xlsx',index_col = 0)
#
locsumm.iloc[:,slice(0,14)] = locsumm.astype('float') #Convert any ints to floats 
#locsumm = pd.read_excel('Oro_Loma_1.xlsx',index_col = 0) 
#All chemicals, including OPEs
chemsumm = pd.read_excel('Kortright_ALLCHEMSUMM.xlsx',index_col = 0)
#Synthetic chemicals for exploring chemical space
#chemsumm = pd.read_excel('Kortright_KowCHEMSUMM.xlsx',index_col = 0)
#Not including OPEs
#chemsumm = pd.read_excel('Kortright_CHEMSUMM.xlsx',index_col = 0)
#Specific Groups
#chemsumm = pd.read_excel('TPhP_CHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('Kortright_OPECHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('Kortright_TCEPCHEMSUMM.xlsx',index_col = 0)
#timeseries = pd.read_excel('timeseries_tracertest_Kortright_valve.xlsx')
#timeseries = pd.read_excel('timeseries_tracertestExtended_Kortright_AllChems.xlsx')
#***SYNTHETIC EVENT***
#timeseries = pd.read_excel('timeseries_synthetic.xlsx')
timeseries = pd.read_excel('timeseries_tracertest_Kortright_Short.xlsx')
#timeseries = pd.read_excel('timeseries_tracertest_Kortright_extended.xlsx')
#timeseries = pd.read_excel('timeseries_tracertestExtended_Kortright_SlowDrain.xlsx')
#timeseries = pd.read_excel('timeseries_tracertest630Max_Test.xlsx')
#timeseries = pd.read_excel('timeseries_tracertest630Max_Kortright_AllChems.xlsx')
#numc = np.array(np.concatenate([locsumm.index[0:2].values]),dtype = 'str')  #Change to 1 for pure advection testing of water compartment
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air', 'pond']

pp = None
test = BCBlues(locsumm,chemsumm,params,timeseries,numc) #Leave as 9
#Truncate time series if you want to run fewer
#maxtime =7 #710# #hrs, after injection. 7 is the tracer test, 708.98 is the entire series.
#timeseries = pd.DataFrame(timeseries[timeseries.time<=maxtime])
#pdb.set_trace()
''' 
run_period = 10.5 #if run_period/dt not a whole number there will be a problem
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
''' 




#res_time =pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/Flow_time_tracertest.pkl')
#Extended
pdb.set_trace()
res =pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_input_calcs_extended.pkl')
#630 max
#res =pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_input_calcs_630max.pkl')
#mask = input_calcs.time>=0 #Find all the positive values
#mask = mask == False 
#minslice = np.min(np.where(mask))
#maxslice = np.max(np.where(mask))#minslice + 5 #
#input_calcs = df_sliced_index(input_calcs.iloc[slice(minslice,maxslice),:])#Quite slow function, may be better to slice smaller ones
#res = test.make_system(res_time,params,numc)
#res_t = test.input_calc(locsumm,chemsumm,params,pp,numc,res_time) #Give entire time series - will not run flow module
#mf = test.mass_flux(res_time,numc)
#Start with initial conditions from a previous model run.
#oldres = pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_outs_extended1.pkl')
#last_step = oldres.loc[(slice(None),oldres.index.levels[1][-1],slice(None)),:]
#outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_last_step_630.pkl'
#last_step = pd.read_pickle(outpath)
#If you want to cut it down e.g. to run from a previous file, do so here. For some reason takes a while so save after
'''
mintime = 630
maxtime = res.index.levels[1].max()
res = df_sliced_index(res.loc[(slice(None),slice(mintime,maxtime),slice(None)),:]) #Cut off for testing
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_input_calcs_630.pkl'
res.to_pickle(outpath)
'''
#Then, run it.
start = time.time()
res = test.run_it(locsumm,chemsumm,params,pp,numc,timeseries,res)
codetime = (time.time()-start)
#res = test.run_it(locsumm,chemsumm,params,pp,numc,timeseries,res,last_step)
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
    KGE[chem] = hydroeval.evaluator(kge, np.array(Couts.loc[:,chem+'_Coutest']),\
                      np.array(Couts.loc[:,chem+'_Coutmeas']))
    pltnames.append(chem+'_Coutest')
    pltnames.append(chem+'_Coutmeas')

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
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_outs_extended.pkl'
#outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_outs_630max.pkl'
res.to_pickle(outpath)