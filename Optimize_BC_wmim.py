# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:56:23 2020

@author: Tim Rodgers
"""

#Next, lets see what we can do by targeting the mass removal of rhodamine, our sorptive tracer, using the mobile/immobile 
#mixing rate. This can't use the function below as we are not targeting the KGE, just the mass removal
#Load the model parameterization files
params = pd.read_excel('params_BC_2.xlsx',index_col = 0) 
locsumm = pd.read_excel('Kortright_BC.xlsx',index_col = 0)
locsumm.iloc[:,slice(0,14)] = locsumm.astype('float') #Convert any ints to floats 
chemsumm = pd.read_excel('Rhodamine_CHEMSUMM.xlsx',index_col = 0)
timeseries = pd.read_excel('timeseries_tracertest_Kortright.xlsx') #Runs for 3.5 hrs before the model starts
pp = None
#Then, initialize the rest of the model parameters
dt = timeseries.time[1] - timeseries.time[0]#Define the timestep    
numc = np.array(np.concatenate([locsumm.index[0:2].values]),dtype = 'str') #Change to 1 for pure advection testing of water compartment
#Define the funtion
def optBC(paramval): #param is the parameter that is being updated
    if paramval < 0:
        obj = 999
    else:
        #param should be a string giving the name of the parameter in the params dataframe, paramval is the starting guess
        #pdb.set_trace()
        target = 63.5979240213253/100
        params.loc['wmim','val'] = paramval #Update the value of alpha for each time step
        kortright_bc = BCBlues(locsumm,chemsumm,params,timeseries,numc)
        input_calcs = kortright_bc.input_calc(locsumm,chemsumm,params,pp,numc,res_time)
        model_outs = kortright_bc.run_it(locsumm,chemsumm,params,pp,numc,timeseries,input_calcs)
        mass_flux = kortright_bc.mass_flux(model_outs,numc) #Run to get mass flux
        numx = mass_flux.index.levels[2][-1]#Final cell
        Madv = mass_flux.N_effluent.groupby(level=0).sum()*dt/model_outs.Min.groupby(level=0).sum()
        obj = np.array(abs(target-Madv))
        #pdb.set_trace()
    return obj

res_time =pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/Flow_time_tracertest.pkl')
mask = timeseries.time>=0 #Find all the positive values
minslice = np.min(np.where(mask))
maxslice = np.max(np.where(mask))#minslice + 5 #
res_time = df_sliced_index(res_time.loc[(slice(minslice,maxslice),slice(None)),:])#Slice inputs to only the relevant section
param = 'wmim'
paramval = 5 #0.0000294*60 #Define initial value
#Now, we use the scipy minimize function to optimize. 
#For now, using the nelder-mead parameterization as it doesn't take too long
res = minimize(optBC,paramval,method='nelder-mead',options={'xtol': 1e-5, 'disp': True})
res