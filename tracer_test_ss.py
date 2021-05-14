# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:23:18 2021

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
#params = pd.read_excel('params_BC_6.xlsx',index_col = 0) 
params = pd.read_excel('params_BC_5.xlsx',index_col = 0) 
#params = pd.read_excel('params_BC_synthetic.xlsx',index_col = 0)
#locsumm = pd.read_excel('Kortright_BC.xlsx',index_col = 0)
locsumm = pd.read_excel('Kortright_BC_test.xlsx',index_col = 0)

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
#res =pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_input_calcs.pkl')
#res = test.input_calc(locsumm,chemsumm,params,pp,numc,timeseries)

ssdata = res.loc[(slice(None),slice(330),slice(None)),:].groupby(level = 0).sum()

for chem in chemsumm.index:
    ssdata.loc[chem,'inp_1'] = timeseries.loc[330,chem+'_Min']
pdb.set_trace()
xxx = 'y'
#No ponding zone in steady state version
SSouts = test.forward_calc_ss(ssdata,7)
#Add back ponding zone, set initial activity to zero.
SSouts.loc[:,'f8'] = 0
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_outs_steady.pkl'
#outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_outs_630max.pkl'
SSouts.to_pickle(outpath)
