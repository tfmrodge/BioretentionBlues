# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:52:52 2019

@author: Tim Rodgers
"""

from FugModel import FugModel #Import the parent FugModel class
from Subsurface_Sinks import SubsurfaceSinks
from HelperFuncs import vant_conv, arr_conv #Import helper functions
from scipy.integrate import solve_ivp
from scipy import optimize
from ode_helpers import state_plotter
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from collections import OrderedDict
#import time
import pdb #Turn on for error checking

class BCBlues(SubsurfaceSinks):
    """Bioretention cell implementation of the Subsurface_Sinks model.
    This model represents a vertically flowing stormwater infiltration low impact 
    development (LID) technology. It is intended to be solved as a Level V 
    1D ADRE, across space and time, although it can be modified to make a Level III or
    Level IV multimedia model
        
    Attributes:
    ----------
            
            locsumm (df): physical properties of the systmem
            chemsumm (df): physical-chemical properties of modelled compounds
            params (df): Other parameters of the model
            timeseries (df): Timeseries values providing the time-dependent inputs
            to the model.
            num_compartments (int): (optional) number of non-equilibirum 
            compartments
            name (str): (optional) name of the BC model 
            pplfer_system (df): (optional) input ppLFERs to use in the model
    """
    
    def __init__(self,locsumm,chemsumm,params,timeseries,num_compartments = 9,name = None,pplfer_system = None):
        FugModel. __init__(self,locsumm,chemsumm,params,num_compartments,name)
        self.pp = pplfer_system
        #self.ic = self.input_calc(self.locsumm,self.chemsumm,self.params,self.pp,self.numc) 
        
    def make_system(self,locsumm,params,numc,timeseries,dx = None):
        """
        This function will build the dimensions of the 1D system based on the "locsumm" input file.
        If you want to specify more things you can can just skip this and input a dataframe directly
        
        Since the hydrology and the contaminant transport are calculated independently,
        we can call them independently to save time & memory
        Either input a raw summary of the BC dimensions, and we will run the hydrology,
        or input the timeseries of BC flows etc. as a locsumm with the flows
        If the input timeseries has the same index as locsumm, it assumes that
        the hydrology has been pre-calculated
        """
        #pdb.set_trace()
        try: #See if there is a compartment index in the timeseries
            timeseries.index.levels[1]
        except AttributeError:
            timeseries = self.flow_time(locsumm,params,numc,timeseries)
        
        #Set up our control volumes
        L = locsumm.loc['subsoil','Depth']+locsumm.loc['topsoil','Depth']
        params.loc['L','val'] = L
        #dx = 0.1
        if dx == None:
            dx = params.val.dx
        #Set up results dataframe - for discretized compartments this is the length of the flowing water
        res = pd.DataFrame(np.arange(0.0+dx/2.0,L,dx),columns = ['x'])
        if len(numc) > 2: #Don't worry about this for the 2-compartment version
            res = res.append(pd.DataFrame([999],columns = ['x'])) #Add a final x for those that aren't discretized
        res = pd.DataFrame(np.array(res),columns = ['x'])
        res.loc[:,'dx'] = dx
        #pdb.set_trace()
        #Control volume length dx - x is in centre of each cell.
        #res.iloc[-2,1] = res.iloc[-3,1]/2+L-res.iloc[-2,0]
        
        #This code will give another index of the compartments to our dataframe, but it makes things overly complicated
        """
        res_j = OrderedDict.fromkeys(locsumm.index.levels[1][[0,1,2,4,5,6,7,8,9]],[])
        for j in res_j:
            #pdb.set_trace()
            if j in locsumm.index.levels[1][[0,1,5,6,7]]:
                res_j[j] = res.copy(deep=True)
            else: #For the other compartments we will just give the length of the compartment at t = 0 as dx
                res_j[j] = pd.DataFrame([0],columns = ['x'])
                res_j[j].loc[:,'dx'] = res_j[j].loc[:,'dx'] = locsumm.loc[(0,j),'Length']
        res = pd.concat(res_j)
        """
        #Then, we put the times in by copying res across them
        res_t = dict.fromkeys(timeseries.index.levels[0],[]) 
        for t in timeseries.index.levels[0]:
            res_t[t] = res.copy(deep=True)
        res = pd.concat(res_t)
        #Add the 'time' column to the res dataframe
        res.loc[:,'time'] = timeseries.loc[:,'time'].reindex(res.index,method = 'bfill')

        #Set up advection between compartments.
        #The non-discretized compartments don't need to worry about flow the same way - define a mask "discretized mask" or dm
        res.loc[:,'dm'] = res.x!=999
        #Now, we are going to define the submerged zone as a single compartment containing the topsoil and filter
        res.loc[:,'maskts'] = res.x < timeseries.loc[(slice(None),'topsoil'),'Depth'].reindex(res.index,method = 'bfill')
        res.loc[:,'maskss'] = (res.maskts ^ res.dm)
        res.loc[res.maskts,'porositywater'] = timeseries.loc[(slice(None),'topsoil'),'Porosity'].reindex(res.index,method = 'bfill')
        res.loc[res.maskss,'porositywater'] = timeseries.loc[(slice(None),'subsoil'),'Porosity']\
        .reindex(res.index,method = 'bfill') #added so that porosity can vary with x
        #Now we define the flow area as the area of the compartment * porosity * mobile fraction water
        res.loc[res.maskts,'Awater'] = timeseries.loc[(slice(None),'subsoil'),'Area'].reindex(res.index,method = 'bfill')\
        * res.loc[res.maskts,'porositywater']* params.val.thetam #right now not worrying about having different areas
        res.loc[res.maskss,'Awater'] = timeseries.loc[(slice(None),'subsoil'),'Area'].reindex(res.index,method = 'bfill')\
        * res.loc[res.maskss,'porositywater']* params.val.thetam
        #Now we calculate the volume of the soil
        res.loc[res.dm,'Vsubsoil'] = (timeseries.loc[(slice(None),'subsoil'),'Area'].reindex(res.index,method = 'bfill')\
        - res.loc[res.dm,'Awater'])*res.dx
        res.loc[res.dm,'V2'] = res.loc[res.dm,'Vsubsoil'] 
        #Subsoil area - surface area of contact with the water, based on the specific surface area per m³ soil and water
        res.loc[res.dm,'Asubsoil'] = timeseries.loc[(slice(None),'subsoil'),'Area']\
        .reindex(res.index,method = 'bfill')*res.dm*params.val.AF_soil
        #For the water compartment assume a linear flow gradient from Qin to Qout - so Q can be increasing as we go down
        res.loc[res.dm,'Qwater'] = timeseries.loc[(slice(None),'pond'),'Q_towater'].reindex(res.index,method = 'bfill')-\
            (timeseries.loc[(slice(None),'pond'),'Q_towater'].reindex(res.index,method = 'bfill')\
                         -timeseries.loc[(slice(None),'water'),'Q_todrain'].reindex(res.index,method = 'bfill'))/L*res.x
        """
        #determine the influent velocity, to determine how far the water from the pond will go into the cell in one time step
        vin = timeseries.loc[(slice(None),'pond'),'Q_towater'] / np.array(res.loc[(slice(None),0),'Awater'])#Assuming flow area is constant
        
        dxwater = (vin*(dt)).reindex(res.index,method = 'bfill')
        mask = dxwater > (res.x-res.dx/2) #Compartments water flows into.
        """
        res.loc[res.dm,'Qin'] = timeseries.loc[(slice(None),'pond'),'Q_towater'].reindex(res.index,method = 'bfill')-\
            (timeseries.loc[(slice(None),'pond'),'Q_towater'].reindex(res.index,method = 'bfill')\
                         -timeseries.loc[(slice(None),'water'),'Q_todrain'].reindex(res.index,method = 'bfill'))/L*(res.x - res.dx/2)
        #Out of each cell
        numx = res[res.dm].index.levels[1] #count 
        res.loc[res.dm,'Qout'] = timeseries.loc[(slice(None),'pond'),'Q_towater'].reindex(res.index,method = 'bfill')-\
            (timeseries.loc[(slice(None),'pond'),'Q_towater'].reindex(res.index,method = 'bfill')\
                         -timeseries.loc[(slice(None),'water'),'Q_todrain'].reindex(res.index,method = 'bfill'))/L*(res.x + res.dx/2)
        res.loc[(slice(None),numx[-1]),'Qout'] = timeseries.loc[(slice(None),'water'),'Q_todrain'].reindex(res.index,method = 'bfill')
        #Assume ET flow from filter zone only, assume proportional to water flow
        #To calculate the proportion of ET flow in each cell, divide the total ETflow for the timestep
        #by the average of the inlet and outlet flows, then divide evenly across the x cells (i.e. divide by number of cells)
        #to get the proportion in each cell, and multiply by Qwater
        ETprop = timeseries.loc[(slice(None),'water'),'QET'].reindex(res.index,method = 'bfill')/(\
                         (timeseries.loc[(slice(None),'water'),'Q_todrain'].reindex(res.index,method = 'bfill')\
                          +timeseries.loc[(slice(None),'pond'),'Q_towater'].reindex(res.index,method = 'bfill'))/2)
        ETprop[np.isnan(ETprop)] = 0 #returns nan if there is no flow so replace with 0
        res.loc[res.dm,'Qet'] = ETprop/len(res.index.levels[1]) * res.Qwater
        res.loc[res.dm,'Qetsubsoil'] = (1-params.val.froot_top)*res.Qet
        res.loc[res.dm,'Qettopsoil'] = (params.val.froot_top)*res.Qet
        exfprop = timeseries.loc[(slice(None),'water'),'Q_exf'].reindex(res.index,method = 'bfill')/(\
                         (timeseries.loc[(slice(None),'water'),'Q_todrain'].reindex(res.index,method = 'bfill')\
                          +timeseries.loc[(slice(None),'pond'),'Q_towater'].reindex(res.index,method = 'bfill'))/2)
        exfprop[np.isnan(exfprop)] = 0
        res.loc[res.dm,'Qwaterexf'] = exfprop/len(res.index.levels[1]) * res.Qwater #Amount of exfiltration from the system, for unlined systems
        #Capillary water enters the last cell - need to deal with in D values or as a boundary condition
        #res.loc[(slice(None),numx[-1]),'Qcap'] = timeseries.loc[(slice(None),'drain'),'Q_towater'].reindex(res.index,method = 'bfill') #Capillary rise
        res.loc[res.dm,'Qcap'] = timeseries.loc[(slice(None),'drain'),'Q_towater'].reindex(res.index,method = 'bfill')/len(res.index.levels[1]) #Capillary rise
        res.loc[np.isnan(res.Qcap),'Qcap'] = 0 #Distributed through all cells for mass balance reasons
        if any(res.dm==False):
            res.loc[res.dm==False,'Qpondexf'] = timeseries.loc[(slice(None),'pond'),'Q_exf'].reindex(res.index,method = 'bfill')
            res.loc[res.dm==False,'Qdrainexf'] = timeseries.loc[(slice(None),'pond'),'Q_exf'].reindex(res.index,method = 'bfill')
        #Then water volume and velocity
        #For the water volume we need to do a mass balance on each cell at each timestep, discretizing the overall picture.
        #We do this by taking the initial volume - from the initial moisture content - and determining the change in volume at
        #each timestep with the flows we have already allocated to each cell. Since we know that (for the whole compartment):
        #dV = V + Qin - Qout - Qet - Qexf it follows that (for the sum of cells, n):
        #sum(dV,n) = sum(V,n) + sum(Qin,n) - sum(Qout,n) -sum(Qet,n) - sum(Qexf,n)
        #and therefore applying our allocated flows across the x values will give a mass conservative result.
        dt = timeseries.loc[(slice(None),'pond'),'time'] - timeseries.loc[(slice(None),'pond'),'time'].shift(1)
        dt[np.isnan(dt)] = dt[1]
        for t in timeseries.index.levels[0]: #This step is slow, as it needs to loop through the entire timeseries.
            if t == timeseries.index.levels[0][0]:
                res.loc[(t,numx),'V1'] = timeseries.loc[(slice(None),'water'),'V'].reindex(res.index,method = 'bfill')\
                        /(len(res[res.dm].index.levels[1])) #First time step just assume that it is uniformly saturated
            else:
                res.loc[(t,numx),'V1'] = np.array(res.loc[(t-1,numx),'V1']) + (res.loc[(t,numx),'Qin']+res.loc[(t,numx),'Qcap']-res.loc[(t,numx),'Qet']\
                       -res.loc[(t,numx),'Qwaterexf']-res.loc[(t,numx),'Qout'])*np.array(dt[t])                                        
        res.loc[res.dm,'Vwater'] = res.loc[res.dm,'V1'] #Bad code but who cares!
        res.loc[res.dm,'v1'] = res.Qwater/res.Awater #velocity [L/T] at every x
        #Root volumes & area based off of soil volume fraction
        res.loc[res.dm,'Vroot'] = params.val.VFroot*timeseries.loc[(slice(None),'subsoil'),'Area']\
        .reindex(res.index,method = 'bfill')/len(res.index.levels[0])  #Total root volume per m² ground area
        res.loc[res.dm,'Aroot'] = params.val.Aroot*timeseries.loc[(slice(None),'subsoil'),'Area']\
        .reindex(res.index,method = 'bfill')/len(res.index.levels[0]) #Total root area per m² ground area
        #Now we loop through the compartments to set everything else.
        #pdb.set_trace()
        for j in numc:
            #Area (A), Volume (V), Density (rho), organic fraction (foc), ionic strength (Ij)
            #water fraction (fwat), air fraction (fair), temperature (tempj), pH (phj)
            jind = np.where(numc==j)[0]+1 #The compartment number, for the advection term
            Aj, Vj, Vjind, rhoj, focj, Ij = 'A' + str(j), 'V' + str(j), 'V' + str(jind[0]),'rho' + str(j),'foc' + str(j),'I' + str(j)
            fwatj, fairj, tempj, pHj = 'fwat' + str(j), 'fair' + str(j),'temp' + str(j), 'pH' + str(j)
            rhopartj, fpartj, advj = 'rhopart' + str(j),'fpart' + str(j),'adv' + str(j)
            compartment = j
            if compartment not in ['topsoil', 'drain_pores', 'filter']:#These compartments are ignored
                if compartment in ['rootbody', 'rootxylem', 'rootcyl']: #roots are discretized
                    mask= res.dm
                    res.loc[mask,Aj] = timeseries.loc[(slice(None),compartment),'Area'].reindex(res.index,method = 'bfill')\
                            /len(res.index.levels[1])-1    
                    if compartment in ['rootbody']:
                        res.loc[mask,Vj] = (params.val.VFrootbody+params.val.VFapoplast)*res.Vroot #Main body consists of apoplast and cytoplasm
                    else:
                        VFrj = 'VF' + str(j)
                        res.loc[mask,Vj] = params.loc[VFrj,'val']*res.Vroot
                    res.loc[mask,Vjind] = res.loc[mask,Vj] #Needed for the FugModel module
                elif compartment in ['water','subsoil']: #water and subsoil
                    mask = res.dm
                else:#Other compartments aren't discretized
                    mask = res.dm==False
                    res.loc[mask,Vj] = timeseries.loc[(slice(None),compartment),'V'].reindex(res.index,method = 'bfill')
                    res.loc[mask,Vjind] = res.loc[mask,Vj] #Needed for FugModel ABC
                    res.loc[mask,Aj] = timeseries.loc[(slice(None),compartment),'Area'].reindex(res.index,method = 'bfill')
                res.loc[mask,focj] = timeseries.loc[(slice(None),compartment),'FrnOC'].reindex(res.index,method = 'bfill') #Fraction organic matter
                res.loc[mask,Ij] = timeseries.loc[(slice(None),compartment),'cond'].reindex(res.index,method = 'bfill')*1.6E-5 #Ionic strength from conductivity #Plants from Trapp (2000) = 0.5
                res.loc[mask,fwatj] = timeseries.loc[(slice(None),compartment),'FrnWat'].reindex(res.index,method = 'bfill') #Fraction water
                res.loc[mask,fairj] = timeseries.loc[(slice(None),compartment),'Frnair'].reindex(res.index,method = 'bfill') #Fraction air
                res.loc[mask,fpartj] = timeseries.loc[(slice(None),compartment),'FrnPart'].reindex(res.index,method = 'bfill') #Fraction particles
                res.loc[mask,tempj] = timeseries.loc[(slice(None),compartment),'Temp'].reindex(res.index,method = 'bfill') + 273.15 #Temperature [K]
                res.loc[mask,pHj] = timeseries.loc[(slice(None),compartment),'pH'].reindex(res.index,method = 'bfill') #pH
                res.loc[mask,rhopartj] = timeseries.loc[(slice(None),compartment),'PartDensity'].reindex(res.index,method = 'bfill') #Particle density
                res.loc[mask,rhoj] = timeseries.loc[(slice(None),compartment),'Density'].reindex(res.index,method = 'bfill') #density for every x [M/L³]
                res.loc[mask,advj] = timeseries.loc[(slice(None),compartment),'Q_out'].reindex(res.index,method = 'bfill')
                if compartment == 'air': #Set air density based on temperature
                    res.loc[mask,rhoj] = 0.029 * 101325 / (params.val.R * res.loc[:,tempj])
                

        #For area of roots in contact with sub and topsoil assume that roots in both zones are roughly cylindrical
        #with the same radius. SA = pi r² 
        res.loc[res.dm,'Arootsubsoil'] = res.Aroot #Area of roots in direct contact with subsoil
        res.loc[res.dm,'AsoilV'] = timeseries.loc[(slice(None),'subsoil'),'Area']\
        .reindex(res.index,method = 'bfill')
        res.loc[res.maskts,'Asoilair'] = timeseries.loc[(slice(None),'topsoil'),'Area']\
        .reindex(res.index,method = 'bfill') #Only the "topsoil" portion of the soil interacts with the air
        #Shoot area based off of leaf area index (LAI) 
        if any(res.dm==False): #skip if no undiscretized compartments
            res.loc[res.dm==False,'A_shootair'] = params.val.LAI*timeseries.loc[(slice(None),'topsoil'),'Area']\
            .reindex(res.index,method = 'bfill') #Total root volume per m² ground area

        #Longitudinal Dispersivity. Calculate using relationship from Schulze-Makuch (2005) 
        #for unconsolidated sediment unless a value of alpha [L] is given
        if 'alpha' not in params.index:
            params.loc['alpha','val'] = 0.2 * L**0.44 #alpha = c(L)^m, c = 0.2 m = 0.44
        res.loc[res.dm,'ldisp'] = params.val.alpha * res.v1 #Resulting Ldisp is in [L²/T]
        #Replace nans with 0s for the next step
        res = res.fillna(0)
        return res
    
    def bc_dims(self,locsumm,inflow,rainrate,dt,params):
        """
        Calculate BC dimension & compartment information for a given time step.
        Forward calculation of t(n+1) from inputs at t(n)
        The output of this will be a "locsumm" file which can be fed into the rest of the model.
        These calculations do not depend on the contaminant transport calculations.
        
        This module includes particle mass balances, where particles are
        advective transfer media for compounds in the model.
        Particle model based off of 
        water flow modelling based on Randelovic et al (2016)
        
        locsumm gives the conditions at t(n). 
        Inflow in m³/s, rainrate in mm/h, dt in s
        Initial conditions includes height of water (h), saturation(S)
        Could make this just part of locsumm?
        """
        #pdb.set_trace()
        res = locsumm.copy(deep=True)
        
        #For first round, calculate volume
        if 'V' not in res.columns:
            res.loc[:,'V'] = res.Area * res.Depth #Volume m³
            #Now we are going to make a filter zone, which consists of the volume weighted
            #averages of he topsoil and the subsoil.
            res.loc['filter',:] = (res.V.subsoil*res.loc['subsoil',:]+res.V.topsoil*\
                   res.loc['topsoil',:])/(res.V.subsoil+res.V.topsoil)
            res.loc['filter','Depth'] = res.Depth.subsoil+res.Depth.topsoil
            res.loc['filter','V'] = res.V.subsoil+res.V.topsoil
            res.Porosity['filter'] = res.Porosity['filter']*params.val.thetam #Effective porosity - the rest is the immobile water fraction
            res.loc['water','V'] = res.V['filter']*res.FrnWat['filter'] #water fraction in subsoil - note this is different than saturation
            res.loc['drain_pores','V'] = res.V.drain*res.FrnWat.drain
            res.loc[:,'P'] = 2 * (res.Area/res.Depth + res.Depth) #Perimeter, m ## Need to make into hydraulic perimeter##
            res.loc[np.isnan(res.loc[:,'P']),'P'] = 0 #If the compartment doesn't exist
            res.loc[np.isinf(res.loc[:,'P']),'P'] = 0
            res.loc['filter','Discrete'] = 1.
        #Define the BC geometry
        pondV = np.array(params.val.BC_Volume_Curve.split(","),dtype='float')
        pondH = np.array(params.val.BC_Depth_Curve.split(","),dtype='float')
        pondA = np.array(params.val.BC_pArea_curve.split(","),dtype='float')

        #filter saturation
        Sss = res.V.water /(res.V['filter'] * (res.Porosity['filter']))
        #ponding Zone
        #Convert rain to flow rate (m³/hr). Direct to ponding zone
        Qr_in = res.Area.air*rainrate/1E3 #m³/hr
        #Infiltration Kf is in m/hr
        #Potential
        Qinf_poss = params.val.Kf * (res.Depth.pond + res.Depth['filter'])\
        /res.Depth['filter']*res.Area.pond
        #Upstream max flow (from volume)
        Qinf_us = 1/dt * (res.V.pond+(Qr_in+inflow)*dt)
        #Downstream capacity
        #Maximum infiltration to native soils through filter and submerged zones, Kn is hydr. cond. of native soil
        Q_max_inf = params.val.Kn * (res.Area['filter'] + res.P['drain_pores']*params.val.Cs)
        Qinf_ds= 1/dt * ((1-Sss) * res.Porosity.subsoil * res.V.subsoil) +Q_max_inf #Why not Qpipe here?
        #FLow from pond to subsoil zone
        Q26 = max(min(Qinf_poss,Qinf_us,Qinf_ds),0)
        #Overflow from system - everything over top of cell
        if res.V.pond > pondV[-1]:
            Qover = (1/dt)*(res.V.pond-pondV[-1])
            res.loc['pond','V'] += -Qover*dt
            res.loc['pond','Depth'] = np.interp(res.V.pond,pondV,pondH, left = 0, right = pondH[-1])
            res.loc['pond','Area'] = np.interp(res.V.pond,pondV,pondA, left = 0, right = pondA[-1]) 
        else:
            Qover = 0
        #Flow over weir
        if res.Depth.pond > params.val.Hw:
            #Physically possible
            def pond_depth(pond_depth):
                Q2_wp = params.val.Cq * params.val.Bw * np.sqrt(2.*9.81*(res.Depth.pond - params.val.Hw)**3.)
                dVp = (inflow + Qr_in - Q26 - Q2_wp)*dt
                res.loc['pond','V'] += dVp
                old_depth = pond_depth
                res.loc['pond','Depth'] = np.interp(res.V.pond,pondV,pondH, left = 0, right = pondH[-1])
                minimizer = abs(old_depth - res.Depth.pond)
                return minimizer
            res.loc['pond','Depth'] = optimize.newton(pond_depth,res.Depth.pond,tol=1e-3)
            if res.Depth.pond < params.val.Hw:
                res.loc['pond','Depth'] = params.val.Hw
            Q2_wp = params.val.Cq * params.val.Bw * np.sqrt(2.*9.81*(res.Depth.pond - params.val.Hw)**3.)
            #Upstream Control
            Q2_wus = 1/dt * (res.Depth.pond - params.val.Hw)*res.Area.pond + (Qr_in+inflow)*dt - Q26*dt
            Q2_w = max(min(Q2_wp,Q2_wus),0)
        else:
            Q2_w = 0
        #Exfiltration to surrounding soil from pond
        #Maximum possible
        if  res.Area.pond > res.Area['filter']: #If pond is over surrounding soil
            Qpexf_poss = params.val.Kn*((res.Area.pond-res.Area['filter']) + params.val.Cs*res.P['pond'])
        else:
            Qpexf_poss = 0 #If the pond is only over the filter it drains to the filter
        #Upstream availability, no need for downstream as it flows out of the system
        Qpexf_us = 1/dt*(res.V.pond) + (Qr_in+inflow-Q26-Q2_w)*dt
        Q2_exf = max(min(Qpexf_poss,Qpexf_us),0) #Actual exfiltration
        #pond Volume from mass balance
        #Change in pond volume dVp at t
        dVp = (inflow + Qr_in - Q26 - Q2_w - Q2_exf)*dt
        res.loc['pond','V'] += dVp
        res.loc['pond','Depth'] = np.interp(res.V.pond,pondV,pondH, left = 0, right = pondH[-1])  
        #Area of pond/soil surface m² at t+1
        res.loc['pond','Area'] = np.interp(res.V.pond,pondV,pondA, left = 0, right = pondA[-1])  
        res.loc['air','Area'] = pondA[-1] - res.Area.pond #Just subtract the area that is submerged
    
                
        #Pore water Flow - filter zone
        #Capillary Rise - from drain/submerged zone to subsoil zone.
        if Sss > params.val.Ss and Sss < params.val.Sfc:
            Cr = 4 * params.val.Emax/(2.5*(params.val.Sfc - params.val.Ss)**2)
            Q10_cp = res.Area['filter'] * Cr * (Sss-params.val.Ss)*(params.val.Sfc - Sss)
            #Upstream volume available (in drain layer)
            Q10_cus = (res.V.drain_pores)/dt
            #Space available in pore_filt
            Q10_cds = 1/dt * ((1 - Sss)*(res.V['filter'] - Q26*dt))
            Q106 = max(min(Q10_cp,Q10_cus,Q10_cds),0)
        else: 
            Q106 = 0
        #Estimated saturation at time step t+1
        S_est = min(1.0,Sss+Q26*dt/(res.V['filter']*res.Porosity['filter']))
        #Infiltration from filter to drainage layer
        Q6_infp = res.Area.subsoil*params.val.Kf*S_est*(res.Depth.pond + res.Depth['filter'])/res.Depth['filter']
        if Sss < params.val.Sh: #No volume available in filter zone if at hygroscopic point
            Q6_inf_us = (Q26+Q106)*dt
        else:
            Q6_inf_us = 1/dt*((Sss-params.val.Sh)*res.Porosity['filter']*res.V['filter']+(Q26+Q106)*dt)
        Q610 = max(min(Q6_infp,Q6_inf_us),0)
        #Flow due to evapotranspiration. Some will go our the air, some will be transferred to the plants for cont. transfer?
        if S_est <= params.val.Sh:
            Q6_etp = 0
        elif S_est <= params.val.Sw:
            Q6_etp = res.Area['filter'] * params.val.Ew*(Sss-params.val.Sh)\
            /(params.val.Sw - params.val.Sh)
        elif S_est <= params.val.Ss:
            Q6_etp = res.Area['filter'] * (params.val.Ew +(params.val.Emax - params.val.Ew)\
            *(Sss-params.val.Sw)/(params.val.Ss - params.val.Sw))
        else:
            Q6_etp = res.Area['filter']*params.val.Emax
        #Upstream available - should Q610 be subtracted maybe?
        Q6_etus = 1/dt* ((Sss-params.val.Sh)*res.V['filter']*res.Porosity['filter'] +(Q26+Q106-Q610)*dt)
        Q6_et = max(min(Q6_etp,Q6_etus),0)
        
        #topsoil and subsoil pore water - water
        #going to try a single unified water compartment, with velocities that 
        #change depending on the zone. Might mess some things up if drain zone is fast?
        #Change in pore water volume dVf at t
        dVf = (Q26 + Q106 - Q610 - Q6_et)*dt
        res.loc['water','V'] += dVf
        #subsoil Saturation (in the water depth column) at t+1
        Sss = res.V.water /(res.V['filter'] * (res.Porosity['filter'])) 
        
        #Water flow - drain/submerged zone
        #So this is a "pseudo-compartment" which will be incorporated into the 
        #subsoil compartment for the purposes of contaminant transport hopefully.
        #Exfiltration from drain zone to native soil
        Q10_exfp = params.val.Kn * (res.Area.drain + params.val.Cs*\
                   res.P.drain*res.Depth.drain_pores/res.Depth.drain)
        Q10_exfus = 1/dt * ((1-res.Porosity.drain)*(res.Depth.drain_pores-params.val.hpipe)*res.Area.drain + (Q610-Q106)*dt)
        Q10_exf = max(min(Q10_exfp,Q10_exfus),0)
        dVdest = (Q610 - Q106 - Q10_exf)*dt #Estimate the height for outflow
        draindepth_est =  (res.loc['drain_pores','V'] + dVdest) /(res.Area.drain * (res.Porosity.drain))
        #drain through pipe - this is set to just remove everything each time step, probably too fast
        if draindepth_est >= params.val.hpipe: #Switched to estimated drainage depth 20190912
            piperem = params.val.hpipe/100 #Keep the level above the top of the pipe so that flow won't go to zero
            Q10_pipe = 1/dt * ((draindepth_est-(params.val.hpipe+piperem))*(1-res.Porosity.drain)\
            *res.Area.drain)
            #Q10_pipe = 1/dt * ((res.loc['drain_pores','Depth']-params.val.hpipe)*(1-res.Porosity.drain)\
            #*res.Area.drain + (Q610-Q106-Q10_exf)*dt)
        else: Q10_pipe = 0;


        #drain Pore water Volume from mass balance\
        #I think this will have to be a separate compartment.
        #Change in drain pore water volume dVd at t
        dVd = (Q610 - Q106 - Q10_exf - Q10_pipe)*dt
        res.loc['drain_pores','V'] += dVd
        #Height of submerged zone - control variable for flow leaving SZ
        res.loc['drain_pores','Depth'] = res.V.drain_pores /(res.Area.drain * (res.Porosity.drain))
        
        #Put final flows into the res df. Flow rates are given as flow from a compartment (row)
        #to another compartment (column). Flows out of the system have their own
        #columns (eg exfiltration, ET, outflow), as do flows into the system.
        res.loc['pond','Q_towater'] = Q26 #Infiltration to subsoil
        res.loc['pond','Q_out'] = Q2_w +Qover #Weir + overflow
        res.loc['pond','Q_exf'] = Q2_exf #exfiltration from pond
        res.loc['pond','Q_in'] = inflow + Qr_in#From outside system
        res.loc['water','Q_todrain'] = Q610 #Infiltration to drain layer
        res.loc['water','QET'] = Q6_et #Infiltration to drain layer
        res.loc['water','Q_exf'] = 0 #Need to set exfiltration from water
        res.loc['drain','Q_towater'] = Q106 #Capillary rise
        res.loc['drain','Q_exf'] = Q10_exf #exfiltration from drain layer
        res.loc['drain','Q_out'] = Q10_pipe #Drainage from system
        #Calculate VFwater based on saturation
        res.loc['subsoil','FrnWat'] = Sss*res.Porosity['filter']
        res.loc['topsoil','FrnWat'] = Sss*res.Porosity['filter']
        res.loc['drain','FrnWat'] = res.V.drain_pores/res.V.drain
        #Calculate VFair for drain and subsoil zones based on saturation
        res.loc['subsoil','Frnair'] = res.Porosity['filter'] - Sss
        res.loc['topsoil','Frnair'] = res.Porosity['filter'] - Sss
        res.loc['drain','Frnair'] = res.Porosity.drain - res.FrnWat.drain/res.Porosity.drain
        res.loc[:,'P'] = 2 * (res.Area/res.Depth + res.Depth) #Perimeter, m ## Need to make into hydraulic perimeter##
        res.loc[np.isnan(res.loc[:,'P']),'P'] = 0 #If the compartment has no volume
        res.loc[np.isinf(res.loc[:,'P']),'P'] = 0 #If the compartment has no volume
        return res
    
    def flow_time(self,locsumm,params,numc,timeseries):
        """
        Step through the flow calculations in time.             
        
        locsumm (df) gives the dimensions of the system at the initial conditions
        Timeseries (df) needs to contain inflow in m³/h, rainrate in mm/h, dt in h
        The initial conditions must include height of water in pond and drain zones (h)
        volume fractions of water in the filter zone
        """
        #dt = timeseries.time[1]-timeseries.time[0] #Set this so it works
        res = locsumm.copy(deep=True)
        ntimes = len(timeseries['time'])
        #Set up 4D output dataframe by adding time as the third multi-index
        #Index level 0 = time, level 1 = chems, level 2 = cell number
        times = timeseries.index
        res_t = dict.fromkeys(times,[]) 
        #pdb.set_trace()
        for t in range(ntimes):
            if t == 0:
                dt = timeseries.time[1]-timeseries.time[0]
            else:                
                dt = timeseries.time[t]-timeseries.time[t-1] #For adjusting step size
            #First, update params. Updates:
            #Qin, Qout, RainRate, WindSpeed, 
            #Next, update locsumm
            rainrate = timeseries.RainRate[t] #m³/h
            inflow = timeseries.Qin[t] #m³/h
            if t == 631:#Break at specific timing
                yomama = 'great'
            res = self.bc_dims(res,inflow,rainrate,dt,params)
            res_t[t] = res.copy(deep=True)
            #Add the 'time' to the resulting dataframe
            res_t[t].loc[:,'time'] = timeseries.loc[t,'time']
            for j in numc:
                Tj,pHj = 'T'+str(j),'pH'+str(j)
                res_t[t].loc[j,'Temp'] = timeseries.loc[t,Tj] #compartment temperature
                res_t[t].loc[j,'pH'] = timeseries.loc[t,pHj] #Compartment pH
            #For now, assume that inflow only to the water compartm
            
    
        res_time = pd.concat(res_t)
        
        
        return res_time