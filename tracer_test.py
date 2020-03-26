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

params = pd.read_excel('params_BC_3.xlsx',index_col = 0) 
locsumm = pd.read_excel('Kortright_BC.xlsx',index_col = 0)
locsumm.iloc[:,slice(0,14)] = locsumm.astype('float') #Convert any ints to floats 
#locsumm = pd.read_excel('Oro_Loma_1.xlsx',index_col = 0) 
chemsumm = pd.read_excel('Kortright_CHEMSUMM.xlsx',index_col = 0)
timeseries = pd.read_excel('timeseries_tracertest_Kortright.xlsx')
numc = np.array(np.concatenate([locsumm.index[0:2].values]),dtype = 'str')  #Change to 1 for pure advection testing of water compartment
pp = None

#Truncate timeseries if you want to run fewer
pdb.set_trace()
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

test = BCBlues(locsumm,chemsumm,params,timeseries,numc) #Leave as 9

start = time.time()

#res_time =pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/Flow_time_tracertest.pkl')
input_calcs =pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_input_calcs.pkl')
mintime = input_calcs.index.levels[1][0]
maxtime = mintime+5#input_calcs.index.levels[1][-1]
#input_calcs = df_sliced_index(input_calcs.loc[(slice(None),slice(mintime,maxtime),slice(None)),:]) #Cut off for testing
#mask = input_calcs.time>=0 #Find all the positive values
#mask = mask == False 
#minslice = np.min(np.where(mask))
#maxslice = np.max(np.where(mask))#minslice + 5 #
#input_calcs = df_sliced_index(input_calcs.iloc[slice(minslice,maxslice),:])#Quite slow function, may be better to slice smaller ones
#res = test.make_system(res_time,params,numc)
#res_t = test.input_calc(locsumm,chemsumm,params,pp,numc,res_time) #Give entire time series - will not run flow module
#mf = test.mass_flux(res_time,numc)

res = test.run_it(locsumm,chemsumm,params,pp,numc,timeseries,input_calcs)
mass_flux = test.mass_flux(res,numc) #Run to get mass flux
numx = mass_flux.index.levels[2][-1]#Final cell
Couts = pd.DataFrame(timeseries.time,index = timeseries[timeseries.time>=0].index)
shiftdist = 10 #Set to equal shift in the optimizations. Shifts modeled values to account for pipe length
Couts.loc[:,'Q_out'] = np.array(res.loc[('Bromide',slice(None),numx),'Qout'])
Couts.loc[:,'Br_meas'] = timeseries[timeseries.time>=0].loc[:,'Bromide_Cout (measured)'] #g/m³
Couts.loc[:,'Benz_meas']=timeseries[timeseries.time>=0].loc[:,'Benzotriazole_Cout (measured)'] #g/m³
Couts.loc[:,'Rho_meas']=timeseries[timeseries.time>=0].loc[:,'Rhodamine_Cout (measured)'] #g/m³
Couts.loc[:,'Br_est'] = np.array(mass_flux.loc[('Bromide',slice(None),numx),'N_effluent'])\
/np.array(res.loc[('Bromide',slice(None),numx),'Qout'])*np.array(chemsumm.MolMass.Bromide)
Couts.loc[:,'Benz_est'] = np.array(mass_flux.loc[('Benzotriazole',slice(None),numx),'N_effluent'])\
/np.array(res.loc[('Benzotriazole',slice(None),numx),'Qout'])*np.array(chemsumm.MolMass.Benzotriazole)
Couts.loc[:,'Rho_est'] = np.array(mass_flux.loc[('Rhodamine',slice(None),numx),'N_effluent'])\
/np.array(res.loc[('Rhodamine',slice(None),numx),'Qout'])*np.array(chemsumm.MolMass.Rhodamine)
Couts.loc[:,slice('Br_est','Rho_est')] = Couts.loc[:,slice('Br_est','Rho_est')].shift(shiftdist)
Couts.loc[:,'Q_out'] = Couts.loc[:,'Q_out'].shift(shiftdist)
Couts[np.isnan(Couts)] = 0
Couts[np.isinf(Couts)] = 0 #if no flow, 0
#Kling-Gupta Efficiency (modified Nash-Sutcliffe) can be our measure of model performance
KGE_Br = hydroeval.evaluator(kge, np.array(Couts.loc[:,'Br_est']),\
                      np.array(Couts.loc[:,'Br_meas']))
KGE_Rho = hydroeval.evaluator(kge, np.array(Couts.loc[:,'Rho_est']),\
                      np.array(Couts.loc[:,'Rho_meas']))
KGE_Benz = hydroeval.evaluator(kge, np.array(Couts.loc[:,'Benz_est']),\
                      np.array(Couts.loc[:,'Benz_meas']))

codetime = (time.time()-start)
pltdata = Couts[['time','Br_meas','Br_est','Benz_meas','Benz_est','Rho_meas','Rho_est']]
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
#For the input calcs
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_outs.pkl'
res.to_pickle(outpath)