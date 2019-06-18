# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 10:38:36 2019

@author: Tim Rodgers
"""
from BCBlues_1d import BCBlues_1d
import pdb
import pandas as pd
import numpy as np
pdb.set_trace()
#Calculate porosity required to give a target residence time for the system
#res = res_t[0]
#res.groupby(level = 0)['del_0'].sum()
#Load datafiles
timeseries = pd.read_excel('timeseries_test2.xlsx')
chemsumm = pd.read_excel('OPE_only_CHEMSUMM.xlsx',index_col = 0)
params = pd.read_excel('params_1d.xlsx',index_col = 0) 
locsumm = pd.read_excel('Oro_Loma_5cm topsoil.xlsx',index_col = 0) 
totalt = 1
timeseries = timeseries[0:totalt+1]


#initialize variables
HRT_targ = 14*24 #14 days in hour, what we want HRT to be
outs = pd.DataFrame(columns = ['Porosity','thetam','HRT']) #Hydraulic residence time
porosity = locsumm.loc['Water','Porosity']#Initial test porosity
thetam = params.val.thetam #Initial test mobile fraction of water
HRT = 261.24589
outs.loc[0,:] = porosity,thetam,HRT

#switch = 'porosity'
switch = 'thetam'
counter = 0
while HRT_targ > HRT:
    if switch == 'porosity': #To find optimal porosity
        porosity += 0.01
        outs.loc[counter,'Porosity'] = locsumm.loc['Water','Porosity'] = porosity
        locsumm.loc['SubSoil','Porosity'] = 1 - porosity
    else:
        thetam += 0.01
        outs.loc[counter,'thetam'] = params.loc['thetam','val'] = thetam    
    numc = 8
    pp = None
    test = BCBlues_1d(locsumm,chemsumm,params,8)
    res_t, res_time = test.run_it(locsumm,chemsumm,params,numc,pp,timeseries)
    res = res_t[0]
    HRT = res.groupby(level = 0)['del_0'].sum()['EHDPP']
    outs.loc[counter,'HRT'] = HRT
    counter += 1
    

"""
time = 0
numc=1
dt = 1
res = res_t[time]
lastcell = 9
res.loc[:,'Mtot_i'] = 0.
res.loc[:,'Mtot_t'] = 0.
for i in range(numc):
    M1,Mi,Vi,Zi, a_val,a_val1 = 'M' + str(i+1),'Mi' + str(i+1),'V' + str(i+1),'Z' + str(i+1), 'a'+str(i+1) + '_t', 'a'+str(i+1) + '_t1'
    Mcumi, Mcumt = 'Mcumi' + str(i+1),'Mcumt' + str(i+1),
    res.loc[:,Mi] = res.loc[:,a_val]*res.loc[:,Vi]*res.loc[:,Zi] #Mass at beginning of time step
    res.loc[:,M1] = res.loc[:,a_val1]*res.loc[:,Vi]*res.loc[:,Zi] #Mass at end of time step
    res.loc[:,Mcumi] = res.groupby(level = 0)[Mi].cumsum()
    res.loc[:,Mcumt] = res.groupby(level = 0)[M1].cumsum()
    res.loc[:,'Mtot_i'] = res.Mtot_i+res.loc[:,Mi]
    res.loc[:,'Mtot_t'] = res.Mtot_t+res.loc[:,M1]
#Now, cumulative mass in all compartments
res.loc[:,'Mcumtot_i'] = res.groupby(level = 0)['Mtot_i'].cumsum()
res.loc[:,'Mcumtot_t'] = res.groupby(level = 0)['Mtot_t'].cumsum()
Mis = np.array(res.loc[(slice(None),lastcell),'Mcumtot_i'])
Ms = np.array(res.loc[(slice(None),lastcell),'Mcumtot_t'])
Min = timeseries.Qin[time]*res.bc_us[slice(None),0]*dt*res.Z1[slice(None),0]
delM = Min-Ms
Mout = timeseries.Qout[time]*res.Z1[0]*res.Z1[0]*dt*res.bc_us[time]
"""

"""
yvar = 'Mcumtot_t'
#compound = 'EHDPP
pltdata = res.loc[(slice(None),slice(None)),slice(None)]
#res_time.loc[(plttime,slice(None),slice(None)),slice(None)] #Just at plttime
ylim = [0, 5e-3]
xlim = [0,10]
ylabel = 'Cumulative Mass'
xlabel = 'Distance'

fig = plt.figure(figsize=(14,8))
ax = sns.lineplot(x = 'x', y = yvar, hue = pltdata.index.get_level_values(0),data = pltdata)
ax.set_ylim(ylim)
ax.set_xlim(xlim)
ax.set_ylabel(ylabel, fontsize=20)
ax.set_xlabel(xlabel, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
"""