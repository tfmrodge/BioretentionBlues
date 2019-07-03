# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:04:52 2019

@author: Tim Rodgers
"""

from FugModel import FugModel #Import the parent FugModel class
from BCBlues_1d import BCBlues_1d
from HelperFuncs import vant_conv, arr_conv #Import helper functions
from scipy.integrate import solve_ivp
from ode_helpers import state_plotter
import numpy as np
import pandas as pd
#import time
import pdb #Turn on for error checking

class Hydro_veg(BCBlues_1d):
    """USS  Model of CSTR contaminant transport in hydroponic system. Goal is 
    to compare with hydroponic plant data from: DOI's 10.1021/acs.est.8b07189 (liu 2019)
    10.1021/acs.est.7b01758 (Wan 2017)
        
    Attributes:
    ----------
            
            locsumm (df): physical properties of the systmem
            chemsumm (df): physical-chemical properties of modelled compounds
            params (df): Other parameters of the model
            results (df): Results of the model
            num_compartments (int): (optional) number of non-equilibirum 
            compartments
            name (str): (optional) name of the BC model 
            pplfer_system (df): (optional) input ppLFERs to use in the model
    """
    
    def __init__(self,locsumm,chemsumm,params,num_compartments = 6,name = None,pplfer_system = None):
        FugModel. __init__(self,locsumm,chemsumm,params,num_compartments,name)
        self.pp = pplfer_system
        #self.ic = self.input_calc(self.locsumm,self.chemsumm,self.params,self.pp,self.numc)  
        
    def hydro_sys(self,locsumm,chemsumm,params,numc,pp):
        """
        Make the hydroponic system for the given chemicals with the inputs 
        provided
        This is just a single compartment CSTR model. We use the same construction
        as the spatially resolved case for consistency of code, but it makes for
        a very long dataframe ¯\_(ツ)_/¯
        """
        #pdb.set_trace()
        chemsumm = self.make_chems(chemsumm,pp)
        res = pd.DataFrame([locsumm.Length.Water/2],columns = ['x']) #Middle of cell
        res.loc[:,'dx'] = locsumm.Length.Water
        for j in range(numc):
            #Area (A), Volume (V), Density (rho), organic fraction (foc), ionic strength (Ij)
            #water fraction (fwat), air fraction (fair), temperature (tempj), pH (phj)
            Aj, Vj, rhoj, focj, Ij = 'A' + str(j+1), 'V' + str(j+1),'rho' + str(j+1),'foc' + str(j+1),'I' + str(j+1)
            fwatj, fairj, tempj, pHj = 'fwat' + str(j+1), 'fair' + str(j+1),'temp' + str(j+1), 'pH' + str(j+1)
            rhopartj, fpartj, advj = 'rhopart' + str(j+1),'fpart' + str(j+1),'adv' + str(j+1)
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
        res.loc[:,'Vroot'] = params.val.VFroot*locsumm.Width[0]*res.dx #Total root volume
        res.loc[:,'A12'] = params.val.Aroot*locsumm.Width[0]*res.dx #Total root area
        res.loc[:,'A16'] = params.val.A_wa #Water/Air interfacial area
        res.loc[:,'A56'] = params.val.LAI*res.dx*locsumm.Width[0] #Shoot area based off of leaf area index (LAI) 
        #Roots are broken into the body, the xylem and the central cylinder.
        res.loc[:,'V2'] = (params.val.VFrootbody+params.val.VFapoplast)*res.Vroot #Main body consists of apoplast and cytoplasm
        res.loc[:,'V3'] = params.val.VFrootxylem*res.Vroot #Xylem
        res.loc[:,'V4'] = params.val.VFrootcylinder*res.Vroot #Central cylinder
        res.loc[:,'V5'] = params.val.VFshoot*locsumm.Width[0]*res.dx
        
        #Now we set up the stuff that varies by compound and by compartment
        chems = chemsumm.index
        numchems = len(chems)
        resi = dict.fromkeys(chems,[])
        #Using the chems as the keys of the dict(resi) then concatenate
        for i in range(numchems):
            resi[chems[i]] = res.copy(deep=True)
        res = pd.concat(resi)
        #Add a dummy variable as mul is how I am sorting by levels w/e
        res.loc[:,'dummy'] = 1
        res.loc[:,'Deff1'] = res['dummy'].mul(chemsumm.WatDiffCoeff, level = 0) #Effective water diffusion coefficient 
        #Read dU values in for temperature conversions. Probably there is a better way to do this.
        res.loc[:,'dUoa'] = res['dummy'].mul(chemsumm.dUoa,level = 0)
        res.loc[:,'dUow'] = res['dummy'].mul(chemsumm.dUow,level = 0)
        res.loc[:,'dUslw'] = res['dummy'].mul(chemsumm.dUslW,level = 0)
        res.loc[:,'dUaw'] = res['dummy'].mul(chemsumm.dUaw,level = 0)
        #Equilibrium constants 
        for j in range(numc):
            Kdj, Kdij, focj, tempj = 'Kd' +str(j+1),'Kdi' +str(j+1),'foc' +str(j+1),'temp' +str(j+1)
            Kawj = 'Kaw' +str(j+1)
            #Kaw is only neutral
            res.loc[:,Kawj] = vant_conv(res.dUaw,res.loc[:,tempj],res['dummy'].mul(10**chemsumm.LogKaw,level = 0))
            #Kd neutral and ionic
            if any( [j == 0, j == 1, j == 3]): #for water, subsoil and topsoil if not directly input
                if 'Kd' in chemsumm.columns: #If none given leave blank, will estimate from Koc & foc
                    res.loc[:,Kdj] = res['dummy'].mul(chemsumm.Kd,level = 0)
                    mask = np.isnan(res.loc[:,Kdj])
                else:
                    mask = res.dummy==1
                if 'Kdi' in chemsumm.columns:#If none given leave blank, will estimate from Koc & foc
                    res.loc[:,Kdij] = res['dummy'].mul(chemsumm.Kdi,level = 0)
                    mask = np.isnan(res.loc[:,Kdij])
                else:
                    mask = res.dummy==1
                res.loc[mask,Kdj] = res.loc[:,focj].mul(10**chemsumm.LogKocW, level = 0)
                res.loc[mask,Kdij] = res.loc[mask,Kdj] #Assume same sorption if not given
                res.loc[:,Kdj] = vant_conv(res.dUow,res.loc[:,tempj],res.loc[:,Kdj]) #Convert with dUow
                #The ionic Kd value is based off of pKa and neutral Kow
                res.loc[:,Kdij] = vant_conv(res.dUow,res.loc[:,tempj],res.loc[:,Kdij])
        #Roots&shoots based on Kslw, air on Kqa
        res.loc[:,'Kd2'] = vant_conv(res.dUslw,res.temp3,res.loc[:,'foc2'].mul(10**chemsumm.LogKslW, level = 0))
        res.loc[:,'Kdi2'] = res.loc[:,'Kd2'] #Fix later if it is different
        res.loc[:,'Kd3'] = vant_conv(res.dUoa,res.temp5,res.loc[:,'foc3'].mul(10**chemsumm.LogKslW, level = 0))
        res.loc[:,'Kdi3'] = res.loc[:,'Kd3'] #Fix later if it is different
        res.loc[:,'Kd4'] = vant_conv(res.dUslw,res.temp3,res.loc[:,'foc4'].mul(10**chemsumm.LogKslW, level = 0))
        res.loc[:,'Kdi4'] = res.loc[:,'Kd4'] #Fix later if it is different
        res.loc[:,'Kd5'] = vant_conv(res.dUslw,res.temp3,res.loc[:,'foc5'].mul(10**chemsumm.LogKslW, level = 0))
        res.loc[:,'Kdi5'] = res.loc[:,'Kd5'] #Fix later if it is different
        #Air based on ppLFER
        res.loc[:,'Kd6'] = vant_conv(res.dUoa,res.temp5,res.loc[:,'foc6'].mul(10**chemsumm.LogKqa, level = 0)) #Air
        res.loc[:,'Kdi6'] = res.loc[:,'Kd6'] #Fix later if it is different
        #Calculate temperature-corrected media reaction rates (/h)
        #These are all set so that they can vary in x, even though for now they do not
        #1 Water
        res.loc[:,'rrxn1'] = res['dummy'].mul(np.log(2)/chemsumm.WatHL,level = 0)
        res.loc[:,'rrxn1'] = arr_conv(params.val.Ea,res.temp1,res.rrxn1)
        #2-5 Veg - Assume same for shoots and roots
        if 'VegHL' in res.columns:
            res.loc[:,'rrxn2'] = res['dummy'].mul(np.log(2)/chemsumm.VegHL,level = 0)
        else:#If no HL for vegetation specified, assume 0.1 * wat HL - based on Wan (2017) wheat plants?
            #res.loc[:,'rrxn2'] = res['dummy'].mul(np.log(2)/chemsumm.WatHL*0.1,level = 0) #0.1 * watHL
            res.loc[:,'rrxn2'] = res['dummy'].mul(np.log(2)/(chemsumm.WatHL*0.1),level = 0) #Testing shorter HL
        res.loc[:,'rrxn3'] = arr_conv(params.val.Ea,res.temp3,res.rrxn2) #RootXylem
        res.loc[:,'rrxn4'] = arr_conv(params.val.Ea,res.temp4,res.rrxn2) #Root Cylinder
        res.loc[:,'rrxn5'] = arr_conv(params.val.Ea,res.temp5,res.rrxn2) #Shoots
        res.loc[:,'rrxn2'] = arr_conv(params.val.Ea,res.temp2,res.rrxn2) #RootBody
        # Air (air_rrxn /s)
        res.loc[:,'rrxn6'] = res['dummy'].mul(chemsumm.AirOHRateConst, level = 0)  * params.val.OHConc    
        res.loc[:,'rrxn6'] = arr_conv(params.val.EaAir,res.temp6,res.rrxn6)
        #Air Particles (airq_rrxn) use 10% of AirOHRateConst if not present (as in Rodgers 2018)
        if 'AirQOHRateConst' not in res.columns:
            res.loc[:,'airq_rrxn'] = 0.1 * res.rrxn1
        else:
            res.loc[:,'airq_rrxn'] = res['dummy'].mul(chemsumm.AirOHRateConst*0.1, level = 0)*params.val.OHConc    
            res.loc[:,'airq_rrxn'] = arr_conv(params.val.EaAir,res.temp5,res.airq_rrxn)
        #pdb.set_trace()
        #Mass transfer coefficients (MTC) [l]/[T]
        #Chemical but not location specific mass transport values
        #Membrane neutral and ionic mass transfer coefficients, Trapp 2000
        res.loc[:,'kmvn'] = 10**(1.2*res['dummy'].mul(chemsumm.LogKow, level = 0) - 7.5) * 3600 #Convert from m/s to m/h
        res.loc[:,'kmvi'] = 10**(1.2*(res['dummy'].mul(chemsumm.LogKow, level = 0) -3.5) - 7.5)* 3600 #Convert from m/s to m/h
        res.loc[:,'kspn'] = 1/(1/params.val.kcw + 1/res.kmvn) #Neutral MTC between soil and plant. Assuming that there is a typo in Trapp (2000)
        res.loc[:,'kspi'] = 1/(1/params.val.kcw + 1/res.kmvi)
        #Correct for kmin = 10E-10 m/s for ions
        kspimin = (10e-10)*3600
        res.loc[res.kspi<kspimin,'kspi'] = kspimin
        #Air side MTC for veg (from Diamond 2001)
        delta_blv = 0.004 * ((0.07 / params.val.WindSpeed) ** 0.5) #leaf boundary layer depth, windspeed in m/s
        res.loc[:,'AirDiffCoeff'] = res['dummy'].mul(chemsumm.AirDiffCoeff, level = 0)
        res.loc[:,'kav'] = res.AirDiffCoeff/delta_blv #m/h
        #Veg side (veg-air) MTC from Trapp (2007). Consists of stomata and cuticles in parallel
        #Stomata - First need to calculate saturation concentration of water
        C_h2o = (610.7*10**(7.5*(res.temp3-273.15)/(res.temp3-36.15)))/(461.9*res.temp3)
        g_h2o = params.val.Qet/(res.A3*(C_h2o-params.val.RH/100*C_h2o)) #MTC for water
        g_s = g_h2o*np.sqrt(18)/np.sqrt(res['dummy'].mul(chemsumm.MolMass, level = 0))
        res.loc[:,'kst'] = g_s * res['dummy'].mul((10**chemsumm.LogKaw), level = 0) #MTC of stomata [L/T] (defined by Qet so m/h)
        #Cuticle
        Pcut = 10**(0.704*res['dummy'].mul((chemsumm.LogKow), level = 0)-11.2)*3600 #m/h
        res.loc[:,'kcut'] = 1/(1/Pcut + 1*res['dummy'].mul((10**chemsumm.LogKaw), level = 0)/(res.kav)) #m/h
        res.loc[:,'kvv'] = res.kcut+res.kst #m/h
                
        
        return chemsumm, res
        
        
    def input_calc(self,locsumm,chemsumm,params,pp,numc):
        """Calculate Z, D and inp values using the compartment parameters from
        bcsumm and the chemical parameters from chemsumm, along with other 
        parameters in params.
        """
        #Initialize the results data frame as a pandas multi indexed object with
        #indices of the compound names and cell numbers
        #pdb.set_trace()
        try:
            print(res.head())
        except NameError:
            chemsumm, res = self.hydro_sys(locsumm,chemsumm,params,numc,pp)
            
        #Declare constants
        Ifd = 1 - np.exp(-2.8 * params.val.Beta) #Vegetation dry deposition interception fraction
        
        #Calculate activity-based Z-values (m³/m³). This is where things start
        #to get interesting if compounds are not neutral. Z(j) is the bulk Z value
        #Refs - Csiszar et al (2011), Trapp, Franco & MacKay (2010), Mackay et al (2011)
        #pdb.set_trace()
        res.loc[:,'pKa'] = res['dummy'].mul(chemsumm.pKa, level = 0) #999 = neutral
        if 'pKb' in chemsumm.columns: #Check for zwitters
            res.loc[:,'pKb'] = res['dummy'].mul(chemsumm.pKb, level = 0) #Only fill in for zwitterionic compounds
        else:
            res.loc[:,'pKb'] = np.nan
        res.loc[:,'chemcharge'] = res['dummy'].mul(chemsumm.chemcharge, level = 0) #0 = neutral, -1 acid first, 1 - base first
        for j in range(numc): #Loop through compartments
            dissi_j, dissn_j, pHj, Zwi_j = 'dissi_' + str(j+1),'dissn_' + str(j+1),\
            'pH' + str(j+1),'Zwi_' + str(j+1)
            gammi_j, gammn_j, Ij, Zwn_j = 'gammi_' + str(j+1),'gammn_' + str(j+1),\
            'I' + str(j+1),'Zwn_' + str(j+1)
            Zqi_j, Zqn_j, Kdj, Kdij,rhopartj = 'Zqi_' + str(j+1),'Zqn_' + str(j+1),\
            'Kd' +str(j+1),'Kdi' +str(j+1),'rhopart' +str(j+1)
            #Dissociation of compounds in environmental media using Henerson-Hasselbalch equation
            #dissi_j - fraction ionic, dissn_j - fraction neutral. A pka of 999 = neutral
            #Multiplying by chemcharge takes care of the cations and the anions
            res.loc[:,dissi_j] = 1/(1+10**(res.chemcharge*(res.pKa-res.loc[:,pHj])))
            #Deal with the amphoterics
            mask = np.isnan(res.pKb) == False
            if mask.sum() != 0:
                res.loc[mask,dissi_j] = 1/(1+10**(res.pKa-res.loc[:,pHj])\
                       + 10**(res.loc[:,pHj]-res.pKb))
            #Deal with the neutrals
            mask = res.chemcharge == 0
            res.loc[mask,dissi_j] = 0
            #Then set the neutral fraction
            res.loc[:,dissn_j] = 1-res.loc[:,dissi_j]
            #Now calculate the activity of water in each compartment or sub compartment
            #From Trapp (2010) gamman = 10^(ks*I), ks = 0.3 /M Setchenov approximation
            res.loc[:,gammn_j] = 10**(0.3*res.loc[:,Ij])
            #Trapp (2010) Yi = -A*Z^2(sqrt(I)/(1+sqrt(I))-0.3I), A = 0.5 @ 15-20°C Davies approximation
            res.loc[:,gammi_j] = 10**(-0.5*res.chemcharge**2*(np.sqrt(res.loc[:,Ij])/\
                   (1+np.sqrt(res.loc[:,Ij]))-0.3*res.loc[:,Ij]))
            #Now, we can calculate Zw for every compartment based on the above
            #Z values for neutral and ionic 
            res.loc[:,Zwi_j] =  res.loc[:,dissi_j]/res.loc[:,gammi_j]
            res.loc[:,Zwn_j] =  res.loc[:,dissn_j]/res.loc[:,gammn_j]
            #Now we can calculate the solids based on Kd, water diss and gamma for that compartment
            res.loc[:,Zqi_j] =  res.loc[:,Kdij] * res.loc[:,rhopartj]/1000 * res.loc[:,Zwi_j]
            res.loc[:,Zqn_j] =  res.loc[:,Kdj] * res.loc[:,rhopartj]/1000 * res.loc[:,Zwn_j]
        
        #1 Water - Consists of suspended solids and pure water
        res.loc[:,'Zi1'] = (1-res.fpart1) * (res.Zwi_1) + res.fpart1 * (res.Zqi_1)
        res.loc[:,'Zn1'] = (1-res.fpart1) * (res.Zwn_1) + res.fpart1 * (res.Zqn_1)
        res.loc[:,'Zw1'] = (1-res.fpart1)*(res.Zwi_1) + (1-res.fpart1)*(res.Zwn_1)
        res.loc[:,'Z1'] = res.Zi1+res.Zn1
        
        #2 Root Body - main portion of the root. Consists of "free space" 
        #(soil pore water), and cytoplasm - could add vaccuol
        res.loc[:,'Zi2'] = res.fwat2*(res.Zwi_2) + res.Zqi_2
        res.loc[:,'Zn2'] = res.fwat2*(res.Zwn_2) + res.Zqn_2 + res.fair2 * res.Kaw2 
        res.loc[:,'Zw2'] = res.fwat2*(res.Zwi_2) + res.fwat2*(res.Zwn_2)
        res.loc[:,'Z2'] = res.Zi2 + res.Zn2
        
        #3 Root xylem
        res.loc[:,'Zi3'] = res.fwat3*(res.Zwi_3) + res.Zqi_3
        res.loc[:,'Zn3'] = res.fwat3*(res.Zwn_3) + res.Zqn_3 + res.fair3 * res.Kaw3 
        res.loc[:,'Zw3'] = res.fwat3*(res.Zwi_3) + res.fwat3*(res.Zwn_3)
        res.loc[:,'Z3'] = res.Zi3+res.Zn3

        #4 Root central cylinder
        res.loc[:,'Zi4'] = res.fwat4*(res.Zwi_4) + res.Zqi_4
        res.loc[:,'Zn4'] = res.fwat4*(res.Zwn_4) + res.Zqn_4 + res.fair4 * res.Kaw4
        res.loc[:,'Zw4'] = res.fwat4*(res.Zwi_4) + res.fwat4*(res.Zwn_4)
        res.loc[:,'Z4'] = res.Zi4+res.Zn4

        #5 Shoots - Water, lipid and air
        res.loc[:,'Zi5'] = res.fwat5*(res.Zwi_5) + res.Zqi_5 
        res.loc[:,'Zn5'] = res.fwat5*(res.Zwn_5) + res.Zqn_5 + res.fair5*res.Kaw5 
        res.loc[:,'Z5'] = res.Zi5+res.Zn5
        
        #6 Air - water, aerosol, air 
        #Aerosol particles - composed of water and particle, with the water fraction defined
        #by hygroscopic growth of the aerosol. Growth is defined as per the Berlin Spring aerosol from Arp et al. (2008)
        if params.val.RH > 100: #maximum RH = 100%
            params.val.RH = 100
        #Hardcoded hygroscopic growth factor (GF) not ideal but ¯\_(ツ)_/¯
        GF = np.interp(params.val.RH/100,xp = [0.12,0.28,0.77,0.92],fp = \
                [1.0,1.08,1.43,2.2],left = 1.0,right = params.val.RH/100*5.13+2.2)
        #Volume fraction of water in aerosol 
        VFQW_a = (GF - 1) * locsumm.Density.Water / ((GF - 1) * \
                  locsumm.Density.Water + locsumm.loc['Air','PartDensity'])
        res.loc[:,'fwat6'] = res.fwat6 + res.fpart6*VFQW_a #add cloud water from locsumm
        res.loc[:,'Zi6'] = res.fwat6*(res.Zwi_6) + (res.fpart6) * res.Zqi_6
        res.loc[:,'Zn6'] = res.fwat6*(res.Zwn_6) + (res.fpart6) * res.Zqn_6 + \
        (1- res.fwat6-res.fpart6)*res.Kaw6 
        res.loc[:,'Z6'] = res.Zi6+res.Zn6
        res.loc[:,'Zw6'] = res.fwat6*(res.Zwi_6)+res.fwat6*(res.Zwn_6)
        res.loc[:,'Zq6'] = res.fpart6*(res.Zqi_6 + res.Zqn_6)
        res.loc[:,'phi6'] = res.fpart6*(res.Zqi_6 + res.Zqn_6)/res.Z6 #particle bound fraction
        
        #D values (m³/h), N (mol/h) = a*D (activity based)
        #Loop through compartments to set reactive and out of system advective D values
        for j in range(numc): #Loop through compartments
            Drj, Dadvj, Zj, rrxnj, Vj= 'Dr' + str(j+1),'Dadv' + str(j+1),'Z' + \
            str(j+1),'rrxn' + str(j+1),'V' + str(j+1)
            advj = 'adv' + str(j+1)
            #Assuming that degradation is not species specific and happends on 
            #the bulk medium (unless over-written)
            res.loc[:,Drj] = 0#res.loc[:,Zj] * res.loc[:,Vj] * res.loc[:,rrxnj] 
            res.loc[:,Dadvj] = res.loc[:,Zj] * res.loc[:,Vj] * res.loc[:,advj]
        #For air, different reactive rate for the particle and bulk
        res.loc[:,'Dr6'] = 0#(1-res.phi6) * res.loc[:,'V6'] * res.loc[:,'rrxn6']\
        #+ res.phi6 * res.airq_rrxn
        
        #1 Water - roots and air
        #pdb.set_trace()
        #Water to root body (2)
        #Plant uptake - depends on neutral and ionic processes
        #First, calculate the value of N =zeF/RT
        res.loc[:,'N1'] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.temp2) #Water
        res.loc[:,'N2'] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.temp6) #Root body
        #Water-root body
        res.loc[:,'Dwr_n1'] = res.A12*(res.kspn*res.Zn1) #100% Water so no need for fwat1
        res.loc[:,'Dwr_i1'] = res.A12*(res.kspi*res.Zi1*res.N1/(np.exp(res.N1)-1)) #100% Water so no need for fwat1
        res.loc[:,'D_apo1'] = params.val.Qet*(params.val.f_apo)*(res.Zw1) #Apoplast bypass
        #res.loc[:,'Drw_n1'] = res.A12*(res.kspn*res.Zwn_2) #fwat2 * Zwn_2 is the free water concentration
        #res.loc[:,'Drw_i1'] = res.A12*(res.kspi*res.Zwi_2*res.N2/(np.exp(res.N2)-1))
        res.loc[:,'Drw_n1'] = res.A12*(res.kspn*res.fwat2*res.Zwn_2) #fwat2 * Zwn_2 is the free water concentration
        res.loc[:,'Drw_i1'] = res.A12*(res.kspi*res.fwat2*res.Zwi_2*res.N2/(np.exp(res.N2)-1))
        res.loc[mask,'Dwr_i1'], res.loc[mask,'Drw_i1'] = 0,0 #Set neutral to zero
        res.loc[:,'D_rd21'] = res.V2 * res.Z2 * params.val.k_rd  #Root death
        res.loc[:,'D_rd31'] = res.V3 * res.Z3 * params.val.k_rd  #Root death
        res.loc[:,'D_rd41'] = res.V4 * res.Z4 * params.val.k_rd  #Root death
        #Water - air - Diffusion & aerosol deposition
        res.loc[:,'D_vw'] =  1 / (1 / (params.val.kma * res.A16 \
                  * res.Z6) + 1 / (params.val.kmw * res.A16 * res.Zn1)) #Air/water gaseous diffusion
        res.loc[:,'D_dw'] = res.A16 * params.val.Up * res.Zq6 #dry dep of aerosol
        #Inter-Compartmental D Values
        res.loc[:,'D_12'] = (res.Dwr_n1 + res.Dwr_i1) /10
        res.loc[:,'D_21'] = res.Drw_n1 + res.Drw_i1 + res.D_rd21
        res.loc[:,'D_13'] = res.D_apo1 #Apoplast bypass straight to xylem
        res.loc[:,'D_31'] = res.D_rd31
        res.loc[:,'D_41'] = res.D_rd41
        res.loc[:,'D_16'] = res.D_vw
        res.loc[:,'D_61'] = res.D_vw + res.D_dw
        #Water does not go to central cylinder (4), or shoots (5). Explicit for error checking.
        res.loc[:,'D_14'] = 0
        res.loc[:,'D_15'] = 0
        res.loc[:,'DT1'] = res.D_12+res.D_13+res.D_14+res.D_15+res.D_16+res.Dadv1+res.Dr1 #Total D value
        
        #2 Root Body - water, xylem
        #roots-xylem froot_tip is the fraction of root without secondary epothileum
        res.loc[:,'N3'] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.temp3)#Xylem (3)
        res.loc[:,'Drx_n'] = (params.val.froot_tip)*res.A3*(res.kspn*res.fwat2*res.Zwn_2) #A3 is root body/xylem interface
        res.loc[:,'Drx_i'] = (params.val.froot_tip)*res.A3*(res.kspi*res.fwat2*res.Zwi_2*res.N2/(np.exp(res.N2)-1))
        res.loc[:,'Dxr_n'] = (params.val.froot_tip)*res.A3*(res.kspn*res.fwat3*res.Zwn_2)
        res.loc[:,'Dxr_i'] = (params.val.froot_tip)*res.A3*(res.kspi*res.fwat3*res.Zwi_2*res.N3/(np.exp(res.N3)-1))
        res.loc[mask,'Drx_i'], res.loc[mask,'Dxr_i'] = 0,0 #Set neutral to zero
        res.loc[:,'D_rg2'] = params.val.k_rg*res.V2*res.Z2 #root growth
        #Inter-Compartmental D Values
        res.loc[:,'D_23'] = res.Drx_n+res.Drx_i
        res.loc[:,'D_32'] = res.Dxr_n+res.Dxr_i
        #Root body does not go to central cylinder (4), shoots (5) or air (6). Explicit for error checking.
        res.loc[:,'D_24'] = 0
        res.loc[:,'D_25'] = 0
        res.loc[:,'D_26'] = 0
        res.loc[:,'DT2'] = res.D_21+res.D_23+res.D_24+res.D_25+res.D_26+res.D_rg2+res.Dadv2+res.Dr2 #Total D val
        
        #3 Xylem - root body, central cylinder
        #xylem-central cylinder - just advection
        res.loc[:,'D_et3'] = params.val.Qet*res.Zw3
        res.loc[:,'D_rg3'] = params.val.k_rg*res.V3*res.Z3 #root growth
        #Inter-Compartmental D Values
        res.loc[:,'D_34'] = res.D_et3
        res.loc[:,'D_87'] = 0
        #Xylem does not go to shoots (5), air (6). Explicit for error checking.
        res.loc[:,'D_35'] = 0
        res.loc[:,'D_36'] = 0
        res.loc[:,'DT3'] = res.D_31+res.D_32+res.D_34+res.D_35+res.D_36+res.D_rg3+res.Dadv3+res.Dr3 #Total D val
        
        #4 Root central cylinder - shoots, xylem
        res.loc[:,'D_et4'] = params.val.Qet*res.Zw4
        res.loc[:,'D_rg4'] = params.val.k_rg*res.V4*res.Z4 #root growth
        #Inter-Compartmental D Values
        res.loc[:,'D_45'] = res.D_et4
        #RCC does not go to root body (2), xylem (3) air (6)
        res.loc[:,'D_42'] = 0
        res.loc[:,'D_43'] = 0 #assume no phloem flow - this is false but ~ 1-5% of xylem flow
        res.loc[:,'D_46'] = 0
        res.loc[:,'DT4'] = res.D_41+res.D_42+res.D_43+res.D_45+res.D_46+res.D_rg4+res.Dadv4+res.Dr4 #Total D val
    
        #5 Shoots - interacts with central cylinder, air, topsoil
        #Shoots-air (Trapp 2007, Diamond 2001) see calcs in the calculation of kvv, it includes Qet in stomatal pathway
        res.loc[:,'D_d56'] = res.kvv*res.A56*res.Zn5 #Volatilization to air, only neutral species
        res.loc[:,'D_dv'] = res.A56* res.Zq6 * params.val.Up *Ifd  #dry dep of aerosol
        #Shoots-water - Diamond (2001) - all bulk
        res.loc[:,'D_we'] = res.A5 * params.val.kwe * res.Z5   #Wax erosion
        res.loc[:,'D_lf'] = res.V5 * res.Z5 * params.val.Rlf  #litterfall & plant death?
        #Shoot growth - Modelled as first order decay
        res.loc[:,'D_sg'] = params.val.k_sg*res.V3*res.Z3
        #Inter-Compartmental D Values
        res.loc[:,'D_56'] = res.D_d56
        res.loc[:,'D_65'] = res.D_d56+res.D_dv 
        res.loc[:,'D_51'] = res.D_we + res.D_lf
        #Shoots do not go to water (1), subsoil (2), roots (6-8, see above for 8). Explicit for error checking.
        res.loc[:,'D_52'] = 0
        res.loc[:,'D_53'] = 0
        res.loc[:,'D_54'] = 0
        res.loc[:,'DT5'] = res.D_51+res.D_52+res.D_53+res.D_54+res.D_56+res.D_sg+res.Dadv5+res.Dr5 #Total D value
        
        #6 Air - shoots, water
        #Air does not go to roots (2-4). Explicit for error checking.
        res.loc[:,'D_62'] = 0
        res.loc[:,'D_63'] = 0
        res.loc[:,'D_64'] = 0
        res.loc[:,'DT6'] = res.D_61+res.D_62+res.D_63+res.D_64+res.D_65+res.Dadv6 + res.Dr6
        
        return res
    
    def run_hydro(self,locsumm,chemsumm,params,timeseries,numc,pp):
        """Feed the calculated values into the ODE over time.
        timeseries is the timeseries data of temperatures, rainfall, influent 
        concentrations, influent volumes, redox conditions? What else?
        
        timeseries(df): Contains all of the time-varying parameters. 2d dataframe
        with a "time" column. Must contain Qet & input concentrations for each
        compound at each time step, others optional
        """
        #pdb.set_trace()
        dt = timeseries.time[0] #Set this so it works
        ntimes = len(timeseries['time'])
        #Set up 4D output dataframe by adding time as the third multi-index
        #Index level 0 = time, level 1 = chems, level 2 = cell number
        times = timeseries.index
        res_t = dict.fromkeys(times,[]) #This dict will contain the outputs
        #pdb.set_trace()
        for t in range(ntimes):
            #First, update params. Updates:
            #Qet
            params.loc['Qet','val'] = timeseries.Qet[t] #m³/s
            #Next, update locsumm
            #Temperature (Tj), pHj, condj
            comps = locsumm.index
            for j in range(numc):
                Tj, pHj, condj = 'T' + str(j+1), 'pH' + str(j+1), 'cond'+ str(j+1)
                if Tj in timeseries.columns: #Only update if this is a time-varying parameter
                      locsumm.loc[comps[j],'Temp'] = timeseries.loc[t,Tj]
                if pHj in timeseries.columns:
                      locsumm.loc[comps[j],'pH'] = timeseries.loc[t,pHj]
                if condj in timeseries.columns:
                          locsumm.loc[comps[j],'cond'] = timeseries.loc[t,condj]
            res = self.input_calc(locsumm,chemsumm,params,pp,6) #Calculate D Values, etc.
            #Run a timestep
            params.loc['Time','val']=t #For tracking what timestep inside function
            if t is 0: #Set initial conditions
                res.loc[(slice(None),0),'a1_t'] = np.array(chemsumm.InitialWaterConc/\
                        chemsumm.MolMass/(res.Zwn_1[slice(None),0] + res.Zwi_1[slice(None),0]))#Initial concentration
                for j in range(1,numc): #Skip water compartment
                    a_val = 'a'+str(j+1) + '_t'
                    res.loc[:,a_val] = 0#res.a1_t/100 #Set inital values as not zero?
                dt = timeseries.time[1]-timeseries.time[0]
                res = self.forward_step_euler(res,params,numc,dt)
            else: #Set the previous solution aj_t1 to the inital condition (aj_t)
                for j in range(0,numc):
                    a_val, a_valt1 = 'a'+str(j+1) + '_t', 'a'+str(j+1) + '_t1'
                    res.loc[:,a_val] = res_t[t-1].loc[:,a_valt1]
                dt = timeseries.time[t] - timeseries.time[t-1] #timestep can vary
                res = self.forward_step_euler(res,params,numc,dt)
            res_t[t] = res.copy(deep=True)
        #Once we are out of the time loop, put the whole dataframe together
        res_time = pd.concat(res_t)
        
        return res_t, res_time
    
    def ivp_hydro(self,locsumm,chemsumm,params,timeseries,tspan,numc,pp,outtype = 'MAXI'):
        """Solve the initial value problem of the vegetation system using the scipy
        solve_ivp function. Currently uses 'radau' integration, change this in the
        code itself
        
        timeseries(df): Contains all of the time-varying parameters, and when they 
        vary in a day. Since we are just doing day/night cycle only need to have 2 entries
        
        tspan(array): Numpy array with the output timespan you want to display 
        (e.g. every hour for 1,000 hours)
        """
        ntimes = timeseries.shape[0]
        numchems = len(chemsumm)
        #Find the times when the input calcs will be different, and set up a 
        #dict to contain those. This will be 2 but if you want to have more that
        #is also possible
        time_changes = timeseries.index
        res_t = dict.fromkeys(time_changes,[])
        #pdb.set_trace()
        for t in range(ntimes):
            #First, update params & locsumm. Updates:
            #Qet, Ts
            params.loc['Qet','val'] = timeseries.Qet[t] #m³/s
            locsumm.loc['Temp',:] = timeseries.T1[t]
            res_t[t] = self.input_calc(locsumm,chemsumm,params,pp,6) #Calculate D Values, etc.
            
        #Set the initial conditions in the first res
        res_t[0].loc[(slice(None),0),'a1_t'] = np.array(chemsumm.InitialWaterConc/\
                chemsumm.MolMass/(res_t[0].Zwn_1[slice(None),0] + res_t[0].Zwi_1[slice(None),0])) #Initial concentration
        chems = res_t[0].index.get_level_values(0)
        #Now, for each compound run the solver
        sols = dict.fromkeys(chems,[])
        soly_dict = dict.fromkeys(chems,[])
        c = self.IVP_matrix(res_t[0],numc) #Determine the matrix equations for all compounds
        c1 = self.IVP_matrix(res_t[1],numc)
        for i in range(0,numchems): #Then loop through the compounds
            compound = chems[i]
            def f(t, y, c):
                #pdb.set_trace()
                if t % 24 <= timeseries.time[1]: #Check if the hour is day or night
                    dydt = [c[i,0,0]*y[0]+c[i,0,1]*y[1]+c[i,0,2]*y[2]+c[i,0,3]*y[3]+c[i,0,4]*y[4]+c[i,0,5]*y[5],
                            c[i,1,0]*y[0]+c[i,1,1]*y[1]+c[i,1,2]*y[2]+c[i,1,3]*y[3]+c[i,1,4]*y[4]+c[i,1,5]*y[5],
                            c[i,2,0]*y[0]+c[i,2,1]*y[1]+c[i,2,2]*y[2]+c[i,2,3]*y[3]+c[i,2,4]*y[4]+c[i,2,5]*y[5],
                            c[i,3,0]*y[0]+c[i,3,1]*y[1]+c[i,3,2]*y[2]+c[i,3,3]*y[3]+c[i,3,4]*y[4]+c[i,3,5]*y[5],
                            c[i,4,0]*y[0]+c[i,4,1]*y[1]+c[i,4,2]*y[2]+c[i,4,3]*y[3]+c[i,4,4]*y[4]+c[i,4,5]*y[5],
                            c[i,5,0]*y[0]+c[i,5,1]*y[1]+c[i,5,2]*y[2]+c[i,5,3]*y[3]+c[i,5,4]*y[4]+c[i,5,5]*y[5]]
                else:
                    dydt = [c1[i,0,0]*y[0]+c1[i,0,1]*y[1]+c1[i,0,2]*y[2]+c1[i,0,3]*y[3]+c1[i,0,4]*y[4]+c1[i,0,5]*y[5],
                            c1[i,1,0]*y[0]+c1[i,1,1]*y[1]+c1[i,1,2]*y[2]+c1[i,1,3]*y[3]+c1[i,1,4]*y[4]+c1[i,1,5]*y[5],
                            c1[i,2,0]*y[0]+c1[i,2,1]*y[1]+c1[i,2,2]*y[2]+c1[i,2,3]*y[3]+c1[i,2,4]*y[4]+c1[i,2,5]*y[5],
                            c1[i,3,0]*y[0]+c1[i,3,1]*y[1]+c1[i,3,2]*y[2]+c1[i,3,3]*y[3]+c1[i,3,4]*y[4]+c1[i,3,5]*y[5],
                            c1[i,4,0]*y[0]+c1[i,4,1]*y[1]+c1[i,4,2]*y[2]+c1[i,4,3]*y[3]+c1[i,4,4]*y[4]+c1[i,4,5]*y[5],
                            c1[i,5,0]*y[0]+c1[i,5,1]*y[1]+c1[i,5,2]*y[2]+c1[i,5,3]*y[3]+c1[i,5,4]*y[4]+c1[i,5,5]*y[5]]
                return dydt
            yinit = [res_t[0].loc[(compound,0),'a1_t'],0.,0.,0.,0.,0.]
            sol = solve_ivp(lambda t, y: f(t, y, c), 
                [tspan[0], tspan[-1]], yinit,method = 'Radau', t_eval=tspan)
            sols[compound] = sol
            if outtype.lower() == 'maxi':
                soli = pd.DataFrame(sol.y)
                soly_dict[compound] = soli
        if outtype.lower() == 'maxi':
            sol_y = pd.concat(soly_dict)
            resi = dict.fromkeys(tspan,[])
            for t in range(len(tspan)): #Put into an overall dataframe, of the same form as the old ones (not efficient but old code will work)
                if t % 24 <= timeseries.time[1]:
                    res = res_t[0]
                else: 
                    res = res_t[1]
                for j in range(0,numc): 
                    a_val = 'a'+str(j+1) + '_t'
                    res.loc[:,a_val] = np.array(sol_y.loc[(slice(None),j),t]) #Skip water compartment
                    resi[t] = res
            res_time = pd.concat(resi)
        else: res_time = res_t
        
        return res_time, sols
    
    def mass_bal(self,res_t,sols,locsumm,chemsumm,numc):
        """Calculate the mass in each of the compartments using the IVP_Hydro
        solution. 
        
        res_t(dict):Contains the initial calculations at 
        
        sols(dict): Contains the solutions (sol) from the IVP ODE solver for each
        compound. sol.y is the activity in each compartment, sol.t is the time
        """
        pdb.set_trace()
        chems = res_t[0].index.get_level_values(0)
        numchems = len(chems)
        mass_bal = np.zeros([len(chems),sols[chems[0]].y.shape[1]]) #mass balance for each compound at each time step
        #comps = locsumm.index.get_level_values(0)
        mass_t = np.zeros([len(chems),sols[chems[0]].y.shape[0]+1,sols[chems[0]].y.shape[1]]) #Mass in each compartment & overall mass at each timestep     
        for i in range(numchems): #loop through chemicals
            chem = chems[i]
            for j in range(numc):
                Vj,Zj = 'V'+str(j+1),'Z'+str(j+1)
                mass_t[j,:] = sols[chem].y[j,:]*np.array(res_t[0].loc[chem,Vj]*res_t[0].loc[chem,Zj])
        mass_t[numc,:]= mass_t.sum(axis = 0)
        mass_bal = np.diff(mass_t[numc,:])
        
        return mass_t, mass_bal
        
        
        
    