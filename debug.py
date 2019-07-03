# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 10:38:36 2019

@author: Tim Rodgers
"""
import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ode_helpers import state_plotter
# Define derivative function
def f(t, y, c):
    #pdb.set_trace()
    if t % 24 <= 14:
        dydt = [c[0,0]*y[0]+c[0,1]*y[1]+c[0,2]*y[2]+c[0,3]*y[3]+c[0,4]*y[4]+c[0,5]*y[5],\
                c[1,0]*y[0]+c[1,1]*y[1]+c[1,2]*y[2]+c[1,3]*y[3]+c[1,4]*y[4]+c[1,5]*y[5],\
                c[2,0]*y[0]+c[2,1]*y[1]+c[2,2]*y[2]+c[2,3]*y[3]+c[2,4]*y[4]+c[2,5]*y[5],\
                c[3,0]*y[0]+c[3,1]*y[1]+c[3,2]*y[2]+c[3,3]*y[3]+c[3,4]*y[4]+c[3,5]*y[5],\
                c[4,0]*y[0]+c[4,1]*y[1]+c[4,2]*y[2]+c[4,3]*y[3]+c[4,4]*y[4]+c[4,5]*y[5],\
                c[5,0]*y[0]+c[5,1]*y[1]+c[5,2]*y[2]+c[5,3]*y[3]+c[5,4]*y[4]+c[5,5]*y[5]]
    else:
        dydt = [c1[0,0]*y[0]+c1[0,1]*y[1]+c1[0,2]*y[2]+c1[0,3]*y[3]+c1[0,4]*y[4]+c1[0,5]*y[5],
                c1[1,0]*y[0]+c1[1,1]*y[1]+c1[1,2]*y[2]+c1[1,3]*y[3]+c1[1,4]*y[4]+c1[1,5]*y[5],
                c1[2,0]*y[0]+c1[2,1]*y[1]+c1[2,2]*y[2]+c1[2,3]*y[3]+c1[2,4]*y[4]+c1[2,5]*y[5],
                c1[3,0]*y[0]+c1[3,1]*y[1]+c1[3,2]*y[2]+c1[3,3]*y[3]+c1[3,4]*y[4]+c1[3,5]*y[5],
                c1[4,0]*y[0]+c1[4,1]*y[1]+c1[4,2]*y[2]+c1[4,3]*y[3]+c1[4,4]*y[4]+c1[4,5]*y[5],
                c1[5,0]*y[0]+c1[5,1]*y[1]+c1[5,2]*y[2]+c1[5,3]*y[3]+c1[5,4]*y[4]+c1[5,5]*y[5]]
    return dydt

tspan =  np.arange(0,5000,1)
compound = 5
yinit = [tester.a1_t[compound],0,0,0,0,0]
#ryinit = -1*inp[compound,:]
c = mat[compound,:,:] #For one compound
c1 = mat1[compound,:,:]

# Solve differential equation

sol = solve_ivp(lambda t, y: f(t, y, c), 
                [tspan[0], tspan[-1]], yinit,method = 'Radau', t_eval=tspan)

# Plot states
state_plotter(sol.t, sol.y, 1)

#Check the mass balance
numc = 6
compname = res_t[0].index.get_level_values(0)
mass_bal = np.zeros([1,sol.y.shape[1]])
mass_t = np.zeros([sol.y.shape[0]+1,sol.y.shape[1]])
for j in range(numc):
            Vj,Zj, inp_mass = 'V'+str(j+1),'Z'+str(j+1),'inp_mass'+str(j+1)
            mass_t[j,:] = sols[compname].y[j,:]*np.array(res.loc[compname,Vj]*res.loc[compname,Zj])
mass_t[numc,:]= mass_t.sum(axis = 0)
mass_bal = np.diff(mass_t[numc,:])

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