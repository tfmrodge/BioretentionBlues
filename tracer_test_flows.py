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
#BC_2 Being used to test immobile fraction impact
#params = pd.read_excel('params_BC_5.xlsx',index_col = 0)
#params = pd.read_excel('params_BC_6.xlsx',index_col = 0)
#params = pd.read_excel('params_BC_synthetic.xlsx',index_col = 0)
#params = pd.read_excel('params_BC_highplant.xlsx',index_col = 0)
#Testing slow drainage - how would this change performance? 
params = pd.read_excel('params_BC_SlowDrain.xlsx',index_col = 0) 
#locsumm = pd.read_excel('Kortright_BC.xlsx',index_col = 0)
locsumm = pd.read_excel('Kortright_FullBC.xlsx',index_col = 0)
#Assuming the entire bioretention cell area is utilized
#locsumm = pd.read_excel('Kortright_FullBC.xlsx',index_col = 0)
locsumm.iloc[:,slice(0,14)] = locsumm.astype('float') #Convert any ints to floats 
#locsumm = pd.read_excel('Oro_Loma_1.xlsx',index_col = 0) 
#chemsumm = pd.read_excel('Kortright_CHEMSUMM.xlsx',index_col = 0)
chemsumm = pd.read_excel('OPECHEMSUMM.xlsx',index_col =0)
#emsumm = pd.read_excel('PROBLEMCHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('EHDPPCHEMSUMM.xlsx',index_col = 0)
#timeseries = pd.read_excel('timeseries_tracertest_Kortright_extended.xlsx')

#timeseries = pd.read_excel('timeseries_tracertest_Kortright_Short.xlsx')
#timeseries = pd.read_excel('timeseries_tracertest630Max_Kortright_AllChems.xlsx')
#timeseries = pd.read_excel('timeseries_tracertest630Max_Test.xlsx')
#timeseries = pd.read_excel('timeseries_tracertestExtended_Kortright_AllChems.xlsx')
#timeseries = pd.read_excel('timeseries_tracertestExtended_Kortright_SlowDrain.xlsx')
#***SYNTHETIC EVENT***
timeseries = pd.read_excel('timeseries_tracertest_Kortright_extended.xlsx')
#timeseries = pd.read_excel('timeseries_synthetic.xlsx')
#timeseries = pd.read_excel('timeseries_LatterEvent_Kortright.xlsx')
#Truncate timeseries if you want to run fewer
numc = np.array(np.concatenate([locsumm.index[0:2].values]),dtype = 'str')
pdb.set_trace()
#run_period = 708.98+3.5 #if run_period/dt not a whole number there will be a problem
''' To truncate the timeseries
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
pp = None
test = BCBlues(locsumm,chemsumm,params,timeseries,9) #Leave as 9
#res = test.make_system(locsumm,params,numc)
#chemsumm = test.make_chems(chemsumm,pp=None)
#res = test.input_calc(locsumm,chemsumm,params,pp,numc)
start = time.time()
#chemsumm, res = test.sys_chem(locsumm,chemsumm,params,pp,numc)

#res = test.ic
#bc_dims = test.bc_dims(locsumm,inflow,rainrate,dt,params)
#res_time = test.flow_time(locsumm,timeseries,params)
#chemsumm, res = test.sys_chem(locsumm,chemsumm,params,pp,numc)

#res = test.ic
#bc_dims = test.bc_dims(locsumm,inflow,rainrate,dt,params)
res_time = test.flow_time(locsumm,params,numc,timeseries)
#res_time =pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/Flow_time.pkl')
#mask = timeseries.time>0
#mask = mask == False #Find all the positive values
#minslice = np.min(np.where(mask))
#maxslice = 5#np.max(np.where(mask))
#res_time = df_sliced_index(res_time.loc[(slice(minslice,maxslice),slice(None)),:])
#res = test.make_system(res_time,params,numc)
#res_t = test.input_calc(locsumm,chemsumm,params,pp,numc,res_time) #Give entire time series - will not run flow module
#mf = test.mass_flux(res_time,numc)
#res_t, res_time = test.run_it(locsumm,chemsumm,params,1,pp,timeseries)


codetime = (time.time()-start)

yvar = 'Q_out'
comp1 = 'water'
comp2 = 'drain'
comp3 = 'pond'
shiftdist = 0
pltdata = res_time.loc[(slice(210,timeseries.index[-1]),comp2),:] #To plot 
pltdata2 = res_time.loc[(slice(210,timeseries.index[-1]),comp3),:]
#res_time.loc[(plttime,slice(None),slice(None)),slice(None)] #Just at plttime
ylim = [0, 4]
xlim = [0,48]
ylabel = 'Concentration (μg/g dw)'
xlabel = 'Time'
#pltdata = res_time #All times at once
fig = plt.figure(figsize=(14,8))
#fig = plt.figure(figsize=(14,8))
#ax = sns.lineplot(x = pltdata.index.get_level_values(0),hue = pltdata.index.get_level_values(1),y=yvar,data = pltdata)
ax = sns.lineplot(x = pltdata.loc[(slice(None),'drain'),'time'],hue = pltdata.index.get_level_values(1),y=yvar,data = pltdata)
#ax2 = sns.lineplot(x = pltdata2.loc[(slice(None),'pond'),'time'],hue = pltdata2.index.get_level_values(1),y='Depth',data = pltdata2)
#ax2.set_xlim(xlim)
ax.set_ylim(ylim)#KGE
#ax.set_xlim(xlim)
ax.tick_params(axis='both', which='major', labelsize=15)
#Calculate the draindown time - this isn't quite accurate as it should measure from the end of the influent.
#timeseries.loc[630,'time'] is the last influent during the tracer test.
draindownt = max(test.draintimes(timeseries,res_time)[0])
        
        
#Finally, let's see how good this flow-routing has been
timeseries.loc[:,'Q_drainout']= np.array(res_time.loc[(slice(None),'drain'),'Q_out'].shift(shiftdist))
timeseries.loc[np.isnan(timeseries.Q_drainout),'Q_drainout'] = 0
KGE_hydro = hydroeval.evaluator(kge, np.array(timeseries.loc[timeseries.time>0,'Q_drainout']),\
                          np.array(timeseries.loc[timeseries.time>0,'Qout_meas']))
#outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/Flow_time_tracertest_extended.pkl'
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/Flow_time_tracertest_highplant.pkl'
res_time.to_pickle(outpath)