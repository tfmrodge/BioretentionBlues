# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:31:52 2019

@author: Tim Rodgers
"""

#So lets go ahead and do the same thing again, targeting alpha (longitudinal dispersivity) via nash-sutcliffe coefficient
#or other (kling-gupta)
import pdb
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
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
#Load the model parameterization files
params = pd.read_excel('params_BC_3.xlsx',index_col = 0) 
locsumm = pd.read_excel('Kortright_BC.xlsx',index_col = 0)
locsumm.iloc[:,slice(0,14)] = locsumm.astype('float') #Convert any ints to floats 
chemsumm = pd.read_excel('Bromide_CHEMSUMM.xlsx',index_col = 0)
timeseries = pd.read_excel('timeseries_tracertest_Kortright.xlsx') #Runs for 3.5 hrs before the model starts
pp = None
#Then, initialize the rest of the model parameters
dt = timeseries.time[1] - timeseries.time[0]#Define the timestep    
numc = np.array(np.concatenate([locsumm.index[0:2].values]),dtype = 'str') #Change to 1 for pure advection testing of water compartment
#Now we define the objective function - we want a minimization problem with the only input alpha
#for this we will need to run the input calcs and the model itself
#Import the flows - these will be constant
res_time = pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/Flow_time_tracertest.pkl')
mask = timeseries.time>=0 #Find all the positive values
minslice = np.min(np.where(mask))
maxslice = np.max(np.where(mask))#minslice + 5 #
res_time = df_sliced_index(res_time.loc[(slice(minslice,maxslice),slice(None)),:])#Slice inputs to only the relevant section
shiftdist = 12 #Set to the shift from the hydrology estimation
#pdb.set_trace()
def optBC(param): #param is the parameter that is being updated
    #pdb.set_trace()
    params.loc['alpha','val'] = param #Update the value of alpha for each time step
    kortright_bc = BCBlues(locsumm,chemsumm,params,timeseries,numc)
    input_calcs = kortright_bc.input_calc(locsumm,chemsumm,params,pp,numc,res_time)
    model_outs = kortright_bc.run_it(locsumm,chemsumm,params,pp,numc,timeseries,input_calcs)
    mass_flux = kortright_bc.mass_flux(model_outs,numc) #Run to get mass flux
    numx = mass_flux.index.levels[2][-1]#Final cell
    Couts = pd.DataFrame(timeseries.time,index = timeseries[timeseries.time>=0].index)
    Couts.loc[:,'Q_out'] = np.array(model_outs.loc[('Bromide',slice(None),numx),'Qout'])
    Couts.loc[:,'Br_meas'] = timeseries[timeseries.time>=0].loc[:,'Bromide_Cout (measured)']
    Couts.loc[:,'Br_est'] = np.array(mass_flux.loc[('Bromide',slice(None),numx),'N_effluent'])\
    /np.array(model_outs.loc[('Bromide',slice(None),numx),'Qout'])*np.array(chemsumm.MolMass.Bromide)
    Couts.loc[:,'Br_est'] = Couts.loc[:,'Br_est'].shift(shiftdist)
    Couts[np.isnan(Couts)] = 0
    Couts[np.isinf(Couts)] = 0 #if no flow, 0
    #Kling-Gupta Efficiency (modified Nash-Sutcliffe) can be our measure of model performance
    eff = hydroeval.evaluator(kge, np.array(Couts.loc[:,'Br_est']),\
                          np.array(Couts.loc[:,'Br_meas']))
    #An efficiency of 1 is ideal, therefore we want to see how far it is from 1
    obj = np.abs(1-eff[0] )
    return obj

alpha0 = 0.05 #Define initial value
#Now, we use the scipy minimize function to optimize. 
#For now, using the nelder-mead parameterization as it doesn't take too long
#res = minimize(optBC,alpha0,method='nelder-mead',options={'xtol': 1e-2, 'disp': True})
res = minimize(optBC,alpha0,method='nelder-mead',options={'xtol': 1e-3, 'disp': True})
#res = minimize_scalar(optBC,method='Brent',options={'xtol': 1e-2, 'disp': True})
res
#alpha = 0.060625 with thetam = 0.4