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

#params = pd.read_excel('params_BC_synthetic.xlsx',index_col = 0)
params = pd.read_excel('params_BC_5.xlsx',index_col = 0) 
#params = pd.read_excel('params_BC_highplant.xlsx',index_col = 0)
locsumm = pd.read_excel('Kortright_BC.xlsx',index_col = 0)
chemsumm = pd.read_excel('Kortright_KowCHEMSUMM_7.xlsx',index_col = 0)
#chemsumm = pd.read_excel('Kortright_ALLCHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('Kortright_benzCHEMSUMM.xlsx',index_col = 0)
pdb.set_trace()
#chemsumm.loc[:,'VegHL'] = params.val.VegHL*chemsumm.VegHL
#timeseries = pd.read_excel('timeseries_synthetic.xlsx')
#timeseries = pd.read_excel('timeseries_tracertest_synthetic.xlsx')
timeseries = pd.read_excel('timeseries_tracertest_Kortright_extended.xlsx')
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air', 'pond']
pp = None

modparams = [locsumm,chemsumm,timeseries,pp,numc,params]
kortright_BC = BCBlues(modparams[0],modparams[1],modparams[5],modparams[2],modparams[4])
start = time.time()
#res = Parallel(n_jobs=2)(delayed(run_BC)(modparams) for j in [0])
res = run_BC(modparams)
#mbal = kortright_BC.mass_balance(res,numc)
#mbal_cum = kortright_BC.mass_balance_cumulative(numc, mass_balance = mbal,normalized=True)
#t = res.index.levels[1][-1] # 630 #6356#
#Total mass out through water
'''
df_chemspace=pd.DataFrame(index=mbal.index.levels[0]) 
df_chemspace.loc[:,'watfrac'] = np.array(mbal_cum.loc[(slice(None),t),['Madvpond','Mexf','Meff']].sum(axis=1))*100
df_chemspace.loc[:,'soilfrac'] = np.array(mbal_cum.loc[(slice(None),t),['Msubsoil']])*100
df_chemspace.loc[:,'airfrac'] = np.array(mbal_cum.loc[(slice(None),t),['Madvair']])*100
df_chemspace.loc[:,'rxnfrac'] = np.array(mbal_cum.loc[(slice(None),t),['Mrwater','Mrsubsoil','Mrrootbody','Mrrootxylem',
                                                                         'Mrrootcyl','Mrshoots','Mrair','Mrpond']].sum(axis=1))*100
df_chemspace.loc[:,'soilrxnfrac'] = np.array(mbal_cum.loc[(slice(None),t),'Mrsubsoil'])*100
df_chemspace.loc[:,'vegrxnfrac'] = np.array(mbal_cum.loc[(slice(None),t),['Mrrootbody','Mrrootxylem',
                                                                         'Mrrootcyl','Mrshoots']].sum(axis=1))*100
'''
codetime = (time.time()-start)
#outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_outs_SUBMISSION20210826.pkl'
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_outs_synthetic_7large.pkl'
res.to_pickle(outpath)