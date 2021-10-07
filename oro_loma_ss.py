# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:23:18 2021

@author: Tim Rodgers
"""

import time
import pandas as pd
import numpy as np
from Loma_Loadings import LomaLoadings
import pdb
import math
#plt.style.use("ggplot")
#from data_1d import makeic

params = pd.read_excel('params_1d.xlsx',index_col = 0) 
#params = pd.read_excel('params_OroLoma_CellF.xlsx',index_col = 0)
#params = pd.read_excel('params_OroLomaTracertest.xlsx',index_col = 0)
#Cell F and G
locsumm = pd.read_excel('Oro_Loma_CellF.xlsx',index_col = 0) 
#locsumm = pd.read_excel('Oro_Loma_CellG.xlsx',index_col = 0) 
#15 cm tracertest conducted with the Oro Loma system Mar. 2019
#locsumm = pd.read_excel('Oro_Loma_Brtracertest.xlsx',index_col = 0) 
#chemsumm = pd.read_excel('Kortright_OPECHEMSUMM.xlsx',index_col = 0)

#Specific Groups
#chemsumm = pd.read_excel('Kortright_BRCHEMSUMM.xlsx',index_col = 0)
chemsumm = pd.read_excel('Oro_FAVCHEMSUMM.xlsx',index_col = 0)

#timeseries = pd.read_excel('timeseries_OroLomaTracertest.xlsx')
timeseries = pd.read_excel('timeseries_OroLoma_Dec_CellF.xlsx')
#timeseries = pd.read_excel('timeseries_OroLoma_Spinup_CellF.xlsx')
#Truncate timeseries if you want to run fewer
pdb.set_trace()
totalt = len(timeseries.index) #100
if totalt <= len(timeseries):
    timeseries = timeseries[0:totalt+1]
else:
    while math.ceil(totalt/len(timeseries)) > 2.0:
        timeseries = timeseries.append(timeseries)
    totalt = totalt - len(timeseries)
    timeseries = timeseries.append(timeseries[0:totalt])
    timeseries.loc[:,'time'] = np.arange(1,len(timeseries)+1,timeseries.time.iloc[1]-timeseries.time.iloc[0])
    timeseries.index = range(len(timeseries))
    
pp = None
#numc = ['water', 'subsoil','topsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air']
numc = ['water', 'subsoil','topsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air']
test = LomaLoadings(locsumm,chemsumm,params,numc) 
#Run or load inputs
pdb.set_trace()
res =pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_input_calcs.pkl')
#res = test.input_calc(locsumm,chemsumm,params,pp,numc,timeseries)

ssdata = res.loc[(slice(None),slice(3),slice(None)),:].groupby(level = 0).sum()

for chem in chemsumm.index:
    #ssdata.loc[chem,'inp_1'] = timeseries.loc[:,chem+'_Cin'].mean()*timeseries.loc[:,'Qin'].mean()/chemsumm.loc[chem,'MolMass']
    #ssdata.loc[chem,'inp_1'] = timeseries.loc[:,chem+'_Cin'].mean()#*timeseries.loc[:,'Qin'].mean()/chemsumm.loc[chem,'MolMass']
    ssdata.loc[chem,'inp_1'] = timeseries.loc[:,chem+'_Cin'].mean()/chemsumm.MolMass[chem]/res.loc[(chem,slice(None),0),'Z1'].mean()
SSouts = test.forward_calc_ss(ssdata,8)
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_outs_steady.pkl'
#outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_outs_630max.pkl'
SSouts.to_pickle(outpath)
