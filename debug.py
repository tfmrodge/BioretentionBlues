# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 10:38:36 2019

@author: Tim Rodgers
"""
import pdb
#pdb.set_trace()
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