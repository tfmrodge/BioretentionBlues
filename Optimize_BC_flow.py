# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:30:25 2019

@author: Tim Rodgers
"""
import pandas as pd
import numpy as np
from BCBlues import BCBlues
import seaborn as sns; sns.set()
import pdb
import math
import hydroeval #For the nash-sutcliffe efficiency
from hydroeval import nse
from scipy.optimize import minimize
pdb.set_trace()
params = pd.read_excel('params_BC.xlsx',index_col = 0) 
locsumm = pd.read_excel('Kortright_BC.xlsx',index_col = 0)
locsumm.iloc[:,slice(0,14)] = locsumm.astype('float') #Convert any ints to floats 
chemsumm = pd.read_excel('OPE_only_CHEMSUMM.xlsx',index_col = 0)
timeseries = pd.read_excel('timeseries_tracertest_Kortright.xlsx')

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
    
numc = np.array([locsumm.index[0:2].values]) #Change to 1 for pure advection testing of water compartment
pp = None

def optBC_flow(Kf):
    params.loc['Kf','val'] = Kf #Update the value of Kf
    test = BCBlues(locsumm,chemsumm,params,timeseries,9)
    res_time = test.flow_time(locsumm,timeseries,params)
    timeseries.loc[:,'Q_drainout']= np.array(res_time.loc[(slice(None),'drain'),'Q_out'])
    NSE = hydroeval.evaluator(nse, np.array(timeseries.loc[timeseries.time>0,'Q_drainout']),\
                          np.array(timeseries.loc[timeseries.time>0,'Qout (measured)']))
    #Now, since we are going to minimize we will use the inverse of the NSE as our 
    #objective function. 
    if NSE <= 0:
        NSE = 1e-6
    return 1/NSE

Kf0 = params.val.Kf
res = minimize(optBC_flow,Kf0,method='nelder-mead',options={'xtol': 1e-8, 'disp': True})