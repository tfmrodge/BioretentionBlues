# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 13:38:57 2019

@author: Tim Rodgers
"""
from FugModel import FugModel #Import the parent FugModel class
from Subsurface_Sinks import SubsurfaceSinks
from HelperFuncs import vant_conv, arr_conv #Import helper functions
from scipy.integrate import solve_ivp
from ode_helpers import state_plotter
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
#import time
import pdb #Turn on for error checking

class LomaLoadings(SubsurfaceSinks):
    """Wastewater treatment wetland implementation of the Subsurface_Sinks model.
    Created for the Oro Loma Horizontal Levee, hence the name. This model represents
    a horizontally flowing, planted wetland. It is intended to be solved as a Level V 
    1D ADRE, across space and time, although it can be modified to make a Level III or
    Level IV multimedia model
        
    Attributes:
    ----------
            
            locsumm (df): physical properties of the system
            chemsumm (df): physical-chemical properties of modelled compounds
            params (df): Other parameters of the model
            timeseries (df): Timeseries values providing the time-dependent inputs
            to the model.
            num_compartments (int): (optional) number of non-equilibirum 
            compartments
            name (str): (optional) name of the BC model 
            pplfer_system (df): (optional) input ppLFERs to use in the model
    """
    
    def __init__(self,locsumm,chemsumm,params,timeseries,num_compartments = 8,name = None,pplfer_system = None):
        FugModel. __init__(self,locsumm,chemsumm,params,num_compartments,name)
        self.pp = pplfer_system
        #self.ic = self.input_calc(self.locsumm,self.chemsumm,self.params,self.pp,self.numc) 
        
    def make_system(self,locsumm,params,numc,dx = None):
        #This function will build the dimensions of the 1D system based on the "locsumm" input file.
        #If you want to specify more things you can can just skip this and input a dataframe directly
        L = locsumm.Length.Water
        params.loc['L','val'] = L
        #dx = 0.1
        if dx == None:
            dx = params.val.dx
        #Smaller cells at influent - testing turn on/off
        #pdb.set_trace()
        samegrid = True
        if samegrid == True:
            res = pd.DataFrame(np.arange(0.0+dx/2.0,L,dx),columns = ['x'])
            res.loc[:,'dx'] = dx
        else:
            dx_alpha = 0.5
            dx_in = params.val.dx/10
            res = pd.DataFrame(np.arange(0.0+dx_in/2.0,L/10.0,dx_in),columns = ['dx'])
            res = pd.DataFrame(np.arange(0.0+dx_in/2.0,L/10.0,dx_in),columns = ['x'])
            lenin = len(res) #Length of the dataframe at the inlet resolution
            res = res.append(pd.DataFrame(np.arange(res.iloc[-1,0]+dx_in/2+dx/2,L,dx),columns = ['x']))
            res = pd.DataFrame(np.array(res),columns = ['x'])
            res.loc[0:lenin-1,'dx'] = dx_in
            res.loc[lenin:,'dx'] = dx
        #pdb.set_trace()
        #Control volume length dx - x is in centre of each cell.
        res.iloc[-1,1] = res.iloc[-2,1]/2+L-res.iloc[-1,0]
        #Integer cell number is the index, columns are values, 'x' is the centre of each cell
        #res = pd.DataFrame(np.arange(0+dx/2,L,dx),columns = ['x'])
        #Set up the water compartment
        res.loc[:,'Q1'] = params.val.Qin - (params.val.Qin-params.val.Qout)/L*res.x 
        res.loc[:,'Qet'] = -1*res.Q1.diff() #ET flow 
        res.loc[0,'Qet'] = params.val.Qin - res.Q1[0] #Upstream boundary
        res.loc[:,'Qet2'] = res.Qet*params.val.fet2
        res.loc[:,'Qet4'] = res.Qet*params.val.fet4
        res.loc[:,'Q_exf'] = 0 #Amount of exfiltration from the system, for unlined systems
        res.loc[:,'q1'] = res.Q1/(locsumm.Depth[0] * locsumm.Width[0])  #darcy flux [L/T] at every x
        res.loc[:,'porosity1'] = locsumm.Porosity[0] #added so that porosity can vary with x
        res.loc[:,'porosity2'] = locsumm.Porosity[1] #added so that porosity can vary with x
        res.loc[:,'porosity4'] = locsumm.Porosity[3]
        #Define the geometry of the Oro Loma system
        oro_x = [0,1.5239,1.524,15.4305,15.4306,16.1163,16.1164,30.0228,30.0229,30.7086,30.7087,45.0342,45.0343,45.72] #x-coordinates of the Oro Loma design drawing taper and mixing wells
        #With Mixing Wells
        #oro_dss = [0.3048,0.3048,0.3048,0.3048,0.6096,0.9144,0.3048,0.3048,0.9144,0.9144,0.3048,0.3048,0.762,0.762] #Subsoil depths from the Oro Loma design drawing. Mixing wells are represented by 3' topsoil depths
        #oro_dts = [0.,0.,0.6096,0.6096,0.,0.,0.6096,0.6096,0.,0.,0.6096,0.4572,0.,0.] #Topsoil depths. Mixing wells do not have a topsoil layer
        #Without Mixing Wells
        #For topsoil as just surface layer
        oro_dss = [0.2548, 0.2548, 0.8644, 0.8644, 0.8644, 0.8644, 0.8644, 0.8644,0.8644, 0.8644, 0.8644, 0.712 , 0.712 , 0.712 ]
        oro_dts = [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]
        #oro_dss = [0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048] #Subsoil depths from the Oro Loma design drawing. Mixing wells are represented by 3' topsoil depths
        #oro_dts = [0.,0.,0.6096,0.6096,0.6096,0.6096,0.6096,0.6096,0.6096,0.6096,0.6096,0.4572,0.4572,0.4572]
        f_dss = interp1d(oro_x,oro_dss,'linear') #Function to define subsoil depth
        f_dts = interp1d(oro_x,oro_dts,'linear') #Function to define topsoil depth 
        res.loc[:,'depth_ss'] = f_dss(res.x)#Depth of the subsoil
        res.loc[:,'depth_ts'] = f_dts(res.x)#Depth of the topsoil
        #Include immobile phase water content, so that Vw is only mobile phase & V2 includes immobile phase
        #Areas for each compartment are defined as the cross sectional "flow area"
        res.loc[:,'A1'] = locsumm.Width[0] * res.depth_ss * res.porosity1 * params.val.thetam
        res.loc[:,'A2'] =  locsumm.Width[0] * res.depth_ss * (res.porosity2 + res.porosity1*(1-params.val.thetam))
        res.loc[:,'v1'] = res.Q1/res.A1 #velocity [L/T] at every x - velocity is eq
        params.loc['vin','val'] = params.val.Qin/(res.A1[0])
        params.loc['vout','val'] = params.val.Qout/(res.A1[0])
        #For the topsoil compartment there is a taper in the bottom 2/3 of the cell
        #res.loc[:,'depth_ts'] = 0.6096
        #res.loc[res.x<2.1336,'depth_ts'] = 0 #No topsoil compartment in the first 7 feet
        #res.loc[res.x>30.7848,'depth_ts'] = 0.6096-(res.x-30.7848)*(0.5/47) #Topsoil tapers in bottom third from 2' to 1.5'
        #res.loc[res.x>45.1104,'depth_ts'] = 0 #Bottom 2' is just gravel drain
        res.loc[:,'A4'] = locsumm.Width[0] * res.depth_ts
        #Now loop through the columns and set the values
        #pdb.set_trace()
        for j in range(numc):
            #Area (A), Volume (V), Density (rho), organic fraction (foc), ionic strength (Ij)
            #water fraction (fwat), air fraction (fair), temperature (tempj), pH (phj)
            Aj, Vj, rhoj, focj, Ij = 'A' + str(j+1), 'V' + str(j+1),'rho' + str(j+1),'foc' + str(j+1),'I' + str(j+1)
            fwatj, fairj, tempj, pHj = 'fwat' + str(j+1), 'fair' + str(j+1),'temp' + str(j+1), 'pH' + str(j+1)
            rhopartj, fpartj, advj = 'rhopart' + str(j+1),'fpart' + str(j+1),'adv' + str(j+1)
            if j <= 1 or j == 3: #done above, water and subsoil as 0 and 1, topsoil as 3
                pass
            else: #Other compartments don't share the same CV
                res.loc[:,Aj] = locsumm.Width[j] * locsumm.Depth[j]
            res.loc[:,Vj] = res.loc[:,Aj] * res.dx #volume at each x [L³]
            res.loc[:,focj] = locsumm.FrnOC[j] #Fraction organic matter
            res.loc[:,Ij] = locsumm.cond[j]*1.6E-5 #Ionic strength from conductivity #Plants from Trapp (2000) = 0.5
            res.loc[:,fwatj] = locsumm.FrnWat[j] #Fraction water
            res.loc[:,fairj] = locsumm.FrnAir[j] #Fraction air
            res.loc[:,fpartj] = locsumm.FrnPart[j] #Fraction particles
            res.loc[:,tempj] = locsumm.Temp[j] + 273.15 #Temperature [K]
            res.loc[:,pHj] = locsumm.pH[j] #pH
            res.loc[:,rhopartj] = locsumm.PartDensity[j] #Particle density
            res.loc[:,rhoj] = locsumm.Density[j] #density for every x [M/L³]
            res.loc[:,advj] = locsumm.Advection[j]
            if locsumm.index[j] == 'Air': #Set air density based on temperature
                res.loc[:,rhoj] = 0.029 * 101325 / (params.val.R * res.loc[:,tempj])
                
        #Root volumes & area based off of soil volume fraction
        res.loc[:,'Vroot'] = params.val.VFroot*locsumm.Width[0]*res.dx #Total root volume per m² ground area
        res.loc[:,'Aroot'] = params.val.Aroot*locsumm.Width[0]*res.dx #Need to define how much is in each section top and sub soil
        #For area of roots in contact with sub and topsoil assume that roots in both zones are roughly cylindrical
        #with the same radius. SA = pi r² 
        res.loc[:,'A62'] = (1-params.val.froot_top) * params.val.Aroot*locsumm.Width[0]*res.dx #Area of roots in direct contact with subsoil
        res.loc[:,'A64'] = params.val.froot_top * params.val.Aroot*locsumm.Width[0]*res.dx #Area of roots in contact with topsoil
        res.loc[:,'A4V'] = locsumm.Width[0]*res.dx #Vertical direction area of the topsoil compartment
        #Shoot area based off of leaf area index (LAI) 
        res.loc[:,'A35'] = params.val.LAI*res.dx*locsumm.Width[0]
        res.loc[res.depth_ts==0,'A4V'] = 0
        res.loc[res.depth_ts==0,'A64'] = 0
        #Roots are broken into the body, the xylem and the central cylinder.
        res.loc[:,'V6'] = (params.val.VFrootbody+params.val.VFapoplast)*res.Vroot #Main body consists of apoplast and cytoplasm
        res.loc[:,'V7'] = params.val.VFrootxylem*res.Vroot #Xylem
        res.loc[:,'V8'] = params.val.VFrootcylinder*res.Vroot #Central cylinder
        

        #Longitudinal Dispersivity. Calculate using relationship from Schulze-Makuch (2005) 
        #for unconsolidated sediment unless a value of alpha [L] is given
        if 'alpha' not in params.index:
            params.loc['alpha','val'] = 0.2 * L**0.44 #alpha = c(L)^m, c = 0.2 m = 0.44
        res.loc[:,'ldisp'] = params.val.alpha * res.v1 #Resulting Ldisp is in [L²/T]
        return res