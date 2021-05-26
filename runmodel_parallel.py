# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:50:20 2021
Run model in parallel

@author: Tim Rodgers
"""

#Now we are going to make it so it actually runs the code. 
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
from joblib import Parallel, delayed
#def runEE(params,traj,EE,y,j):
#objparams = []
def run_BC(modparams):
    #pdb.set_trace()
    #EE = []
    kortright_BC = BCBlues(modparams[0],modparams[1],modparams[5],modparams[2],modparams[4])
    #Run the model
    #pdb.set_trace()
    flow_time = kortright_BC.flow_time(modparams[0],modparams[5],['water','subsoil'],modparams[2])
    mask = modparams[2].time>=0
    minslice = np.min(np.where(mask))
    maxslice = np.max(np.where(mask))#minslice + 5 #
    flow_time = df_sliced_index(flow_time.loc[(slice(minslice,maxslice),slice(None)),:])
    input_calcs = kortright_BC.input_calc(modparams[0],modparams[1],modparams[5],modparams[3],modparams[4],flow_time)
    model_outs = kortright_BC.run_it(modparams[0],modparams[1],modparams[5],modparams[3],modparams[4],
                                     modparams[2],input_calcs)
    #Set what the tracked output variable(s) are
    #Lets set to proportion of mass flux out the end.
    
    return model_outs

params = pd.read_excel('params_BC_synthetic.xlsx',index_col = 0)
locsumm = pd.read_excel('Kortright_BC.xlsx',index_col = 0)
chemsumm = pd.read_excel('Kortright_KowCHEMSUMM.xlsx',index_col = 0)
timeseries = pd.read_excel('timeseries_synthetic.xlsx')
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air', 'pond']
pp = None

modparams = [locsumm,chemsumm,timeseries,pp,numc,params]
start = time.time()
#res = Parallel(n_jobs=2)(delayed(run_BC)(modparams) for j in [0])
res = run_BC(modparams)
codetime = (time.time()-start)
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_outs_synthetic.pkl'
res.to_pickle(outpath)