# -*- coding: utf-8 -*-
"""
Load data for the 1d ADRE BC Blues model
Created on Fri Oct 12 18:10:27 2018

@author: Tim Rodgers
"""
import time
import pandas as pd
from BCBlues_1d import BCBlues_1d
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
#from data_1d import makeic

params = pd.read_excel('params_1d.xlsx',index_col = 0) 
locsumm = pd.read_excel('Oro_Loma.xlsx',index_col = 0) 
chemsumm = pd.read_excel('OPECHEMSUMM.xlsx',index_col = 0)
timeseries = pd.read_excel('timeseries_test1.xlsx')
numc = 8
pp = None
test = BCBlues_1d(locsumm,chemsumm,params,8)
#res = test.make_system(locsumm,params,numc)
#chemsumm = test.make_chems(chemsumm,pp=None)
#res = test.input_calc(locsumm,chemsumm,params,pp,numc)
start = time.time()
#chemsumm, res = test.sys_chem(locsumm,chemsumm,params,pp,numc)

#res = test.ic
res_t, res_time = test.run_it(locsumm,chemsumm,params,numc,pp,timeseries)

codetime = (time.time()-start)
#res = test.input_calc(locsumm,chemsumm,params,None,4)
#ic = makeic()
plot = plt.plot(timeseries.time, res_time.loc[(slice(None),'EHDPP',0),'a1_t1'],\
   'r--', timeseries.time, res_time.loc[(slice(None),'EHDPP',20),'a1_t1'], 'b--',\
   timeseries.time, res_time.loc[(slice(None),'EHDPP',30),'a1_t1'], 'g--')

sns.lineplot()
