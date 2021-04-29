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
#plt.style.use("ggplot")
#Testing slow drainage - how would this change performance? 
#params = pd.read_excel('params_BC_SlowDrain.xlsx',index_col = 0) 
params = pd.read_excel('params_BC_5.xlsx',index_col = 0) 
#params = pd.read_excel('params_BC_synthetic.xlsx',index_col = 0)
locsumm = pd.read_excel('Kortright_BC.xlsx',index_col = 0)
#Assuming the entire bioretention cell area is utilized
#locsumm = pd.read_excel('Kortright_FullBC.xlsx',index_col = 0)
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
#chemsumm = pd.read_excel('Kortright_BRCHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('Kortright_OPECHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('Kortright_TCEPCHEMSUMM.xlsx',index_col = 0)
#emsumm = pd.read_excel('PROBLEMCHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('EHDPPCHEMSUMM.xlsx',index_col = 0)
#timeseries = pd.read_excel('timeseries_tracertest_Kortright_valve.xlsx')
#***NORMAL ONE***
#timeseries = pd.read_excel('timeseries_tracertest_Kortright_extended.xlsx')
timeseries = pd.read_excel('timeseries_tracertest_Kortright_Short.xlsx')

#***SYNTHETIC EVENT***
#timeseries = pd.read_excel('timeseries_synthetic.xlsx')

#timeseries = pd.read_excel('timeseries_tracertestExtended_Kortright_SlowDrain.xlsx')
#timeseries = pd.read_excel('timeseries_tracertest630Max_Kortright_AllChems.xlsx')
#timeseries = pd.read_excel('timeseries_tracertest630Max_Test.xlsx')
#2-compartment version
#numc = np.array(np.concatenate([locsumm.index[0:2].values]),dtype = 'str')  #Change to 1 for pure advection testing of water compartment
#All full compartments - drain, topsoil included in "subsoil" compartment.
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air', 'pond']
pp = None
test = BCBlues(locsumm,chemsumm,params,timeseries,numc)

#Truncate timeseries if you want to run fewer
pdb.set_trace()
'''
run_period = 712.4833 #if run_period/dt not a whole number there will be a problem
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


start = time.time()
#res_time =pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/Flow_time_tracertest_synthetic20210310.pkl')
res_time =pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/Flow_time_tracertest_extended.pkl')
#res_time =pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/Flow_time_tracertest_630max.pkl')
mask = timeseries.time>=0 #Find all the positive values
#mask = mask == False 
minslice = np.min(np.where(mask))
maxslice = np.max(np.where(mask))#minslice + 5 #
res_time = df_sliced_index(res_time.loc[(slice(minslice,maxslice),slice(None)),:])
#res = test.make_system(res_time,params,numc)
res_t = test.input_calc(locsumm,chemsumm,params,pp,numc,res_time) #Give entire time series - will not run flow module
#mf = test.mass_flux(res_time,numc)
#res_t, res_time = test.run_it(locsumm,chemsumm,params,1,pp,timeseries)


codetime = (time.time()-start)
#For the input calcs
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_input_calcs_extended.pkl'
#outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_input_calcs_630max.pkl'
#outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_input_calcs.pkl'
res_t.to_pickle(outpath)