# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:30:25 2019

@author: Tim Rodgers
"""
#We can optimize the flow module using an estimate of model accuracy - nash-sutcliffe coefficient or other (kling-gupta)
import pdb
from hydroeval import kge #Kling-Gupta efficiency (Kling-Gupta et al., 2009)
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import numpy as np
import pandas as pd
#Load the model parameterization files
params = pd.read_excel('params_BC_3.xlsx',index_col = 0) 
locsumm = pd.read_excel('Kortright_BC.xlsx',index_col = 0)
locsumm.iloc[:,slice(0,14)] = locsumm.astype('float') #Convert any ints to floats 
chemsumm = pd.read_excel('Kortright_CHEMSUMM.xlsx',index_col = 0)
timeseries = pd.read_excel('timeseries_tracertest_Kortright.xlsx') #Runs for 3.5 hrs before the model starts
#Then, initialize the rest of the model parameters
dt = timeseries.time[1] - timeseries.time[0]#Define the timestep    
numc = np.array(np.concatenate([locsumm.index[0:2].values]),dtype = 'str') #Change to 1 for pure advection testing of water compartment
shiftdist = 10 #~length of time that it takes to get through the pipe! Should be ~ 50 minutes for ~75m of pipe and an average flow of 0.7m³/hr
#Now we define the objective function - we want a minimization problem with the only input Kf
#(in this case we are doing univariate optimization)
def optBC_flow(Kf):
    #pdb.set_trace()
    params.loc['Kf','val'] = Kf #Update the value of Kf for each time step
    test = BCBlues(locsumm,chemsumm,params,timeseries,9)
    res_time = test.flow_time(locsumm,params,numc,timeseries)
    timeseries.loc[:,'Q_drainout']= np.array(res_time.loc[(slice(None),'drain'),'Q_out'])
    timeseries.loc[:,'Q_drainout'] = timeseries.loc[:,'Q_drainout'].shift(shiftdist)
    timeseries.loc[np.isnan(timeseries.Q_drainout),'Q_drainout'] = 0
    #Kling-Gupta Efficiency (modified Nash-Sutcliffe) can be our measure of model performance
    eff = hydroeval.evaluator(kge, np.array(timeseries.loc[timeseries.time>0,'Q_drainout']),\
                          np.array(timeseries.loc[timeseries.time>0,'Qout (measured)']))
    #An efficiency of 1 is ideal, therefore we want to see how far it is from 1
    obj = 1-eff[0] 
    return obj

Kf0 = 0.1 #Define initial value of Kf
#Now, we use the scipy minimize function to optimize. 
#For now, using the nelder-mead parameterization as it doesn't take too long
res = minimize(optBC_flow,Kf0,method='nelder-mead',options={'xtol': 1e-2, 'disp': True})
#res = minimize_scalar(optBC_flow,method='Brent',options={'xtol': 1e-5, 'disp': True})
res
#20200324 Kf = 0.36992676 with Nelder-mead, starting at 0.4, shiftdist = 12
#0.23042952 with Brent, shiftdist = 12
#0.2675 Nelder-Mead, starting @ 0.4 using BC_Params_2, shiftdist = 10
