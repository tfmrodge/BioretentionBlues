# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:52:42 2018

@author: Tim Rodgers
"""
from FugModel import FugModel #Import the parent FugModel class
from HelperFuncs import ppLFER, vant_conv, arr_conv, make_ppLFER #Import helper functions
import numpy as np
import pandas as pd
#import time
import pdb #Turn on for error checking

class BCBlues_1d(FugModel):
    """ Model of 1D contaminant transport in a vegetated, flowing system.
    This is a modification of the original BCBlues model to work with a 1D ADRE
    BCBlues_1d objects have the following properties:
        
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
    
    def __init__(self,locsumm,chemsumm,params,num_compartments = 8,name = None,pplfer_system = None):
        FugModel. __init__(self,locsumm,chemsumm,params,num_compartments,name)
        self.pp = pplfer_system
        self.ic = self.input_calc(self.locsumm,self.chemsumm,self.params,self.pp,self.numc)
        
    def make_system(self,locsumm,params,numc,dx = None):
        #This function will build the dimensions of the 1D system based on the "locsumm" input file.
        #If you want to specify more things you can can just skip this and input a dataframe directly
        L = locsumm.Length.Water
        if dx == None:
            dx = params.val.dx
        #Integer cell number is the index, columns are values, 'x' is the centre of each cell
        res = pd.DataFrame(np.arange(0+dx/2,L,dx),columns = ['x'])
        #Set up the water compartment
        res.loc[:,'Q1'] = params.val.Qin - (params.val.Qin-params.val.Qout)/L*res.x 
        res.loc[:,'Qet'] = -1*res.Q1.diff() #ET flow 
        res.loc[0,'Qet'] = params.val.Qin - res.Q1[0] #Upstream boundary
        res.loc[:,'Qet2'] = res.Qet*params.val.fet2
        res.loc[:,'Qet4'] = res.Qet*params.val.fet4
        res.loc[:,'q1'] = res.Q1/(locsumm.Depth[0] * locsumm.Width[0])  #darcy flux [L/T] at every x
        res.loc[:,'porosity1'] = locsumm.Porosity[0] #added so that porosity can vary with x
        res.loc[:,'porosity2'] = locsumm.Porosity[1] #added so that porosity can vary with x
        res.loc[:,'porosity4'] = locsumm.Porosity[3]
        #Include immobile phase water content, so that Vw is only mobile phase & V2 includes immobile phase
        res.loc[:,'A1'] = locsumm.Width[0] * locsumm.Depth[0] * res.porosity1 * params.val.thetam
        res.loc[:,'A2'] = locsumm.Width[0] * locsumm.Depth[0] * (res.porosity2 + res.porosity1*(1-params.val.thetam))
        res.loc[:,'v1'] = res.q1/res.porosity1 #velocity [L/T] at every x
        #Now loop through the columns and set the values
        #pdb.set_trace()
        for j in range(numc):
            #Area (A), Volume (V), Density (rho), organic fraction (foc), ionic strength (Ij)
            #water fraction (fwat), air fraction (fair), temperature (tempj), pH (phj)
            Aj, Vj, rhoj, focj, Ij = 'A' + str(j+1), 'V' + str(j+1),'rho' + str(j+1),'foc' + str(j+1),'I' + str(j+1)
            fwatj, fairj, tempj, pHj = 'fwat' + str(j+1), 'fair' + str(j+1),'temp' + str(j+1), 'pH' + str(j+1)
            rhopartj, fpartj, advj = 'rhopart' + str(j+1),'fpart' + str(j+1),'adv' + str(j+1)
            if j <= 1: #done above, assuming water and subsoil as 1 and 2
                pass
            else: #Other compartments don't share the same CV
                res.loc[:,Aj] = locsumm.Width[j] * locsumm.Depth[j]
            res.loc[:,Vj] = res.loc[:,Aj] * dx #volume at each x [L³]
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
                
        #Root volumes & area based off of volume fraction of soil
        res.loc[:,'Vroot'] = params.val.VFroot*locsumm.Width[0]*dx #Total root volume per m² ground area
        res.loc[:,'Aroot'] = params.val.Aroot*locsumm.Width[0]*dx #Need to define how much is in each section top and sub soil
        res.loc[:,'A62'] = 0.1 * params.val.Aroot*locsumm.Width[0]*dx #Volume of roots in direct contact with subsoil
        res.loc[:,'A62'] = 0.9 * params.val.Aroot*locsumm.Width[0]*dx #Volume of roots in contact with topsoil
        #Roots are broken into the body, the xylem and the central cylinder.
        res.loc[:,'V6'] = (params.val.VFrootbody+params.val.VFapoplast)*res.Vroot #Main body consists of apoplast and cytoplasm
        res.loc[:,'V7'] = params.val.VFrootxylem*res.Vroot #Xylem
        res.loc[:,'V8'] = params.val.VFrootcylinder*res.Vroot #Central cylinder
        

        #Longitudinal Dispersivity. Calculate using relationship from Schulze-Makuch (2005) 
        #for unconsolidated sediment unless a value of alpha [L] is given
        if 'alpha' not in params.index:
            params.loc['alpha','val'] = 0.2 * L**0.44 #alpha = c(L)^m, c = 0.2 m = 0.44
        res.loc[:,'ldisp'] = params.val.alpha * res.v1
        return res
                    
    def make_chems(self,chemsumm,pp):
        """If chemsumm relies on ppLFERs, fill it in. All chemical specific
        information that doesn't vary with x should be in this method
        """
        res = chemsumm.copy(deep=True)
        R = 8.3144598
        
        #ppLFER system parameters - initialize defaults if not there already
        if pp is None:
            pp = pd.DataFrame(index = ['l','s','a','b','v','c'])
            pp = make_ppLFER(pp)
        
        #Check if partition coefficients & dU values have been provided, or only solute descriptors
        #add based on ppLFER if not, then adjust partition coefficients to 298.15K if they aren't already
        #Aerosol-Air (Kqa), use octanol-air enthalpy
        if 'LogKqa' not in res.columns:
            res.loc[:,'LogKqa'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKqa.l,pp.logKqa.s,pp.logKqa.a,pp.logKqa.b,pp.logKqa.v,pp.logKqa.c)
        if 'dUoa' not in res.columns: #!!!This might be broken - need to check units & sign!!!
            res.loc[:,'dUoa'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUoa.l,pp.dUoa.s,pp.dUoa.a,pp.dUoa.b,pp.dUoa.v,pp.dUoa.c)
        res.loc[:,'LogKqa'] = np.log10(vant_conv(res.dUoa,298.15,10**res.LogKqa,T1 = 288.15))
        #Organic carbon-water (KocW), use octanol-water enthalpy (dUow)
        if 'LogKocW' not in res.columns:
            res.loc[:,'LogKocW'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKocW.l,pp.logKocW.s,pp.logKocW.a,pp.logKocW.b,pp.logKocW.v,pp.logKocW.c)
        if 'dUow' not in res.columns: #!!!This might be broken - need to check units & sign!!!
            res.loc[:,'dUow'] = 1000 * ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUow.l,pp.dUow.s,pp.dUow.a,pp.dUow.b,pp.dUow.v,pp.dUow.c)
        #Storage Lipid Water (KslW), use ppLFER for dUslW (kJ/mol) convert to J/mol/K
        if 'LogKslW' not in res.columns:
            res.loc[:,'LogKslW'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKslW.l,pp.logKslW.s,pp.logKslW.a,pp.logKslW.b,pp.logKslW.v,pp.logKslW.c)
        if 'dUslW' not in res.columns:
            res.loc[:,'dUslW'] = 1000 * ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUslW.l,pp.dUslW.s,pp.dUslW.a,pp.dUslW.b,pp.dUslW.v,pp.dUslW.c)
        res.loc[:,'LogKslW'] = np.log10(vant_conv(res.dUslW,298.15,10**res.LogKslW,T1 = 310.15))
        #Air-Water (Kaw) use dUaw
        if 'LogKaw' not in res.columns:
            res.loc[:,'LogKaw'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKaw.l,pp.logKaw.s,pp.logKaw.a,pp.logKaw.b,pp.logKaw.v,pp.logKaw.c)
        if 'dUaw' not in res.columns: #!!!This might be broken - need to check units & sign!!!
            res.loc[:,'dUaw'] = 1000 * ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUaw.l,pp.dUaw.s,pp.dUaw.a,pp.dUaw.b,pp.dUaw.v,pp.dUaw.c)
                    
        #Define storage lipid-air (KslA) and organic carbon-air (KocA) using the thermodynamic cycle
        res.loc[:,'LogKslA'] = np.log10(10**res.LogKslW / 10**res.LogKaw)
        res.loc[:,'LogKocA'] = np.log10(10**res.LogKocW / 10**res.LogKaw)
        #Calculate Henry's law constant (H, Pa m³/mol) at 298.15K
        res.loc[:,'H'] = 10**res.LogKaw * R * 298.15
 
        return res
    
    def sys_chem(self,locsumm,chemsumm,params,pp,numc):
        """Put together the system and the chemical parameters into the 3D dataframe
        that will be used to calculate Z and D values. Basically just tidies things
        up a bit, might not be good practice to make this a seperate function
        """

        #Set up the output dataframe, res, a multi indexed pandas dataframe with the 
        #index level 0 as the chemical names, 1 as the integer cell number along x
        #First, call make_system if a full system hasn't been given
        res = self.make_system(locsumm,params,numc,params.val.dx)
        #Then, fill out the chemsumm file
        chemsumm = self.make_chems(chemsumm,pp)
        #add the chemicals as level 0 of the multi index
        chems = chemsumm.index
        numchems = len(chems)
        resi = dict.fromkeys(chems,[])
        #Using the chems as the keys of the dict(resi) then concatenate
        for i in range(numchems):
            resi[chems[i]] = res.copy(deep=True)
        res = pd.concat(resi)
        
        #Parameters that vary by chem and x
        #Add a dummy variable as mul is how I am sorting by levels w/e
        res.loc[:,'dummy'] = 1
        #Deff = 1/tortuosity^2, tortuosity(j)^2 = 1-2.02*ln(porosity) (Shen and Chen, 2007)
        res.loc[:,'tausq1'] = 1/(1-2.02*np.log(res.porosity1))
        res.loc[:,'Deff1'] = res['tausq1'].mul(chemsumm.WatDiffCoeff, level = 0) #Effective water diffusion coefficient 
        res.loc[:,'Deff4'] = res['dummy'].mul(chemsumm.WatDiffCoeff, level = 0)*\
        res.fwat4**(10/3)/(res.fair4 +res.fwat4)**2 #Effective water diffusion coefficient 
        res.loc[:,'Bea4'] = res['dummy'].mul(chemsumm.AirDiffCoeff, level = 0)*\
        res.fair4**(10/3)/(res.fair4 +res.fwat4)**2 #Effective air diffusion coefficient 
        #Dispersivity as the sum of the effective diffusion coefficient (Deff) and ldisp.
        res.loc[:,'disp'] = res.ldisp + res.Deff1
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
        res.loc[:,'Kd3'] = vant_conv(res.dUslw,res.temp3,res.loc[:,'foc3'].mul(10**chemsumm.LogKslW, level = 0))
        res.loc[:,'Kdi3'] = res.loc[:,'Kd3'] #Fix later if it is different
        res.loc[:,'Kd5'] = vant_conv(res.dUoa,res.temp5,res.loc[:,'foc5'].mul(10**chemsumm.LogKqa, level = 0))
        res.loc[:,'Kdi5'] = res.loc[:,'Kd5'] #Fix later if it is different
        res.loc[:,'Kd6'] = vant_conv(res.dUslw,res.temp3,res.loc[:,'foc6'].mul(10**chemsumm.LogKslW, level = 0))
        res.loc[:,'Kdi6'] = res.loc[:,'Kd6'] #Fix later if it is different
        res.loc[:,'Kd7'] = vant_conv(res.dUslw,res.temp3,res.loc[:,'foc7'].mul(10**chemsumm.LogKslW, level = 0))
        res.loc[:,'Kdi7'] = res.loc[:,'Kd7'] #Fix later if it is different
        res.loc[:,'Kd8'] = vant_conv(res.dUslw,res.temp3,res.loc[:,'foc7'].mul(10**chemsumm.LogKslW, level = 0))
        res.loc[:,'Kdi8'] = res.loc[:,'Kd8'] #Fix later if it is different
        #Calculate temperature-corrected media reaction rates (/s)
        #These are all set so that they can vary in x, even though for now they do not
        #1 Water, Convert to per second
        res.loc[:,'rrxn1'] = res['dummy'].mul(np.log(2)/chemsumm.WatHL,level = 0)/3600
        res.loc[:,'rrxn1'] = arr_conv(params.val.Ea,res.temp1,res.rrxn1)
        #2 Subsoil
        res.loc[:,'rrxn2'] = res['dummy'].mul(np.log(2)/chemsumm.SoilHL,level = 0)/3600
        res.loc[:,'rrxn2'] = arr_conv(params.val.Ea,res.temp2,res.rrxn2)
        #3 Veg - Assume same for shoots and roots
        if 'VegHL' in res.columns:
            res.loc[:,'rrxn3'] = res['dummy'].mul(np.log(2)/chemsumm.VegHL,level = 0)/3600
        else:#If no HL for vegetation specified, assume 0.1 * wat HL (maybe?)
            res.loc[:,'rrxn3'] = res['dummy'].mul(np.log(2)/chemsumm.WatHL,level = 0)/3600*0.1
        res.loc[:,'rrxn3'] = arr_conv(params.val.Ea,res.temp3,res.rrxn3) #Shoots
        res.loc[:,'rrxn6'] = arr_conv(params.val.Ea,res.temp6,res.rrxn3) #RootBody
        res.loc[:,'rrxn7'] = arr_conv(params.val.Ea,res.temp7,res.rrxn3) #RootXlem
        res.loc[:,'rrxn8'] = arr_conv(params.val.Ea,res.temp8,res.rrxn3) #Root Cylinder
        #4 Topsoil
        res.loc[:,'rrxn4'] = res['dummy'].mul(np.log(2)/chemsumm.SoilHL,level = 0)/3600
        res.loc[:,'rrxn4'] = arr_conv(params.val.Ea,res.temp4,res.rrxn4)
        #5 Air (air_rrxn /s)
        res.loc[:,'rrxn5'] = res['dummy'].mul(chemsumm.AirOHRateConst, level = 0)  * params.val.OHConc    
        res.loc[:,'rrxn5'] = arr_conv(params.val.EaAir,res.temp5,res.rrxn4)
        #Air Particles (airq_rrxn) use 10% of AirOHRateConst if not present
        if 'AirQOHRateConst' not in res.columns:
            res.loc[:,'airq_rrxn'] = 0.1 * res.rrxn1
        else:
            res.loc[:,'airq_rrxn'] = res['dummy'].mul(chemsumm.AirOHRateConst*0.1, level = 0)*params.val.OHConc    
            res.loc[:,'airq_rrxn'] = arr_conv(params.val.EaAir,res.temp5,res.airq_rrxn)
        
        #Mass transfer coefficients (MTC) [l]/[T]
        #Chemical but not location specific mass transport values
        #Membrane neutral and ionic mass transfer coefficients, Trapp 2000
        res.loc[:,'kmvn'] = 10**(1.2*res['dummy'].mul(chemsumm.LogKow, level = 0) - 7.5)
        res.loc[:,'kmvi'] = 10**(1.2*(res['dummy'].mul(chemsumm.LogKow, level = 0) -3.5) - 7.5)
        res.loc[:,'kspn'] = 1/(1/params.val.kcw + res.kmvn) #Neutral MTC between soil and plant
        res.loc[:,'kspi'] = 1/(1/params.val.kcw + res.kmvi)
        #Correct for kmin = 10E-10 m/s
        res.loc[res.kspn<10E-10,'kspn'] = 10E-10
        res.loc[res.kspi<10E-10,'kspi'] = 10E-10
        #Air side MTC for veg (from Diamond 2001)
        delta_blv = 0.004 * ((0.07 / params.val.WindSpeed) ** 0.5) #leaf boundary layer depth
        res.loc[:,'AirDiffCoeff'] = res['dummy'].mul(chemsumm.AirDiffCoeff, level = 0)
        res.loc[:,'kav'] =  res.AirDiffCoeff/ delta_blv
        #Veg side MTC from Trapp (2007). Consists of stomata and cuticles in parallel
        #Stomata - First need to calculate saturation concentration of water
        C_h2o = (610.7*10**(7.5*(res.temp3-273.15)/(res.temp3-36.15)))/(461.9*res.temp3)
        g_h2o = res.Qet/(res.A3*(C_h2o-params.val.RH/100*C_h2o)) #MTC for water
        g_s = g_h2o*np.sqrt(18)/np.sqrt(res['dummy'].mul(chemsumm.MolMass, level = 0))
        res.loc[:,'kst'] = g_s * res['dummy'].mul((10**chemsumm.LogKaw), level = 0) #MTC of stoata m/d
        #Cuticle
        Pcut = 10**(0.704*res['dummy'].mul((chemsumm.LogKow), level = 0)-11.2) #m/s
        res.loc[:,'kcut'] = 1/(1/Pcut + 1/(res.kav*res['dummy'].mul((10**chemsumm.LogKaw), level = 0)))*86400 #m/d
        res.loc[:,'kvv'] = res.kcut+res.kst
        
       
       
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
            chemsumm, res = self.sys_chem(self.locsumm,self.chemsumm,self.params,self.pp,self.numc)
        
        #Declare constants
        #chems = chemsumm.index
        #numchems = len(chems)
        #R = params.val.R #Ideal gas constant, J/mol/K
        #Ifd = 1 - np.exp(-2.8 * params.Value.Beta) #Vegetation dry deposition interception fraction
        Y2 = 1e-6 #Diffusion path length from mobile to immobile flowJust guessing here
        Y24 = locsumm.Depth[1]/2 #Half the depth of the mobile phase
        Y4 = locsumm.Depth[3]/2 #Diffusion path is half the depth. Probably should make vary in X
        Ifd = 1 - np.exp(-2.8 * params.val.Beta) #Vegetation dry deposition interception fraction
        
        #Calculate activity-based Z-values (m³/m³). This is where things start
        #to get interesting if compounds are not neutral. Z(j) is the bulk Z value
        #Refs - Csiszar et al (2011), Trapp, Franco & MacKay (2010), Mackay et al (2011)
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
        res.loc[:,'Z1'] = res.Zi1+res.Zn1
        
        #2 Subsoil - Immobile-phase water and soil particles
        res.loc[:,'Zi2'] = res.fwat2*(res.Zwi_2) + (res.fpart2)*(res.Zqi_2)
        res.loc[:,'Zn2'] = res.fwat2*(res.Zwn_2) + (res.fpart2)*(res.Zqn_2)
        res.loc[:,'Zw2'] = res.fwat2*(res.Zwi_2) + res.fwat2*(res.Zwn_2)
        res.loc[:,'Z2'] = res.Zi2+res.Zn2
        
        #3 Shoots - Water, lipid and air
        res.loc[:,'Zi3'] = res.fwat3*(res.Zwi_3) + res.Zqi_3 
        res.loc[:,'Zn3'] = res.fwat3*(res.Zwn_3) + res.Zqn_3 + res.fair3*res.Kaw3 
        res.loc[:,'Z3'] = res.Zi3+res.Zn3
        
        #4 Top soil - Water, soil, air
        res.loc[:,'Zi4'] = res.fwat4*(res.Zwi_4) + (1 - res.fwat4 - res.fair4) *\
        res.Zqi_4 
        res.loc[:,'Zn4'] = res.fwat4*(res.Zwn_4) + (1 - res.fwat4 - res.fair4) *\
        res.Zqn_4 + res.fair4*res.Kaw4 
        res.loc[:,'Zw4'] = res.fwat4*(res.Zwi_4) + res.fwat4*(res.Zwn_4)
        res.loc[:,'Zq4'] = (1 - res.fwat4 - res.fair4) * res.Zqi_4  + \
        (1 - res.fwat4 - res.fair4) * res.Zqn_4 
        res.loc[:,'Z4'] = res.Zi4+res.Zn4
        
        #5 Air - water, aerosol, air 
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
        res.loc[:,'fwat5'] = res.fwat5 + res.fpart5*VFQW_a #add cloud water from locsumm
        res.loc[:,'Zi5'] = res.fwat5*(res.Zwi_5) + (res.fpart5) * res.Zqi_5
        res.loc[:,'Zn5'] = res.fwat5*(res.Zwn_5) + (res.fpart5) * res.Zqn_5 + \
        (1- res.fwat5-res.fpart5)*res.Kaw5 
        res.loc[:,'Z5'] = res.Zi5+res.Zn5
        res.loc[:,'Zq5'] = res.fpart5*(res.Zqi_5 + res.Zqn_5)
        res.loc[:,'phi5'] = res.fpart5*(res.Zqi_5 + res.Zqn_5)/res.Z5 #particle bound fraction
        
        #6 Root Body - main portion of the root. Consists of "free space" 
        #(soil pore water), and cytoplasm - could add vaccuol
        res.loc[:,'Zi6'] = res.fwat6*(res.Zwi_6) + res.Zqi_6
        res.loc[:,'Zn6'] = res.fwat6*(res.Zwn_6) + res.Zqn_6 + res.fair6 * res.Kaw6 
        res.loc[:,'Zw6'] = res.fwat6*(res.Zwi_6) + res.fwat6*(res.Zwn_6)
        res.loc[:,'Z6'] = res.Zi3 + res.Zn3
        
        #7 Root xylem
        res.loc[:,'Zi7'] = res.fwat7*(res.Zwi_7) + res.Zqi_7
        res.loc[:,'Zn7'] = res.fwat7*(res.Zwn_7) + res.Zqn_7 + res.fair7 * res.Kaw7 
        res.loc[:,'Zw7'] = res.fwat7*(res.Zwi_7) + res.fwat7*(res.Zwn_7)
        res.loc[:,'Z7'] = res.Zi7+res.Zn7
        
        #8 Root central cylinder
        res.loc[:,'Zi8'] = res.fwat8*(res.Zwi_8) + res.Zqi_8
        res.loc[:,'Zn8'] = res.fwat8*(res.Zwn_8) + res.Zqn_8 + res.fair8 * res.Kaw8
        res.loc[:,'Zw8'] = res.fwat8*(res.Zwi_8) + res.fwat8*(res.Zwn_8)
        res.loc[:,'Z8'] = res.Zi8+res.Zn8
               
        #D values (m³/h), N (mol/h) = a*D (activity based)
        #Loop through compartments to set reactive and out of system advective D values
        for j in range(numc): #Loop through compartments
            Drj, Dadvj, Zj, rrxnj, Vj= 'Dr' + str(j+1),'Dadv' + str(j+1),'Z' + \
            str(j+1),'rrxn' + str(j+1),'V' + str(j+1)
            advj = 'adv' + str(j+1)
            #Assuming that degradation is not species specific and happends on 
            #the bulk medium (unless over-written)
            res.loc[:,Drj] = res.loc[:,Zj] * res.loc[:,Vj] * res.loc[:,rrxnj] 
            res.loc[:,Dadvj] = res.loc[:,Zj] * res.loc[:,Vj] * res.loc[:,advj]
        #For air, different reactive rate for the particle and bulk
        res.loc[:,'Dr5'] = (1-res.phi5) * res.loc[:,'V5'] * res.loc[:,'rrxn5']\
        + res.phi5 * res.airq_rrxn
        
        #1 Water - interacts with subsoil and topsoil. May want to put direct ET to plants too
        #Water - subsoil - transfer to pore water through ET and diffusion. 
        #May need to replace with a calibrated constant
        #From Mackay for water/sediment diffusion. Will be dominated by the ET flow so probably OK
        res.loc[:,'D_d12'] =  1/(1/(params.val.kxw*res.A2*res.Z1)+Y2/(res.A2*res.Deff1*res.Zw2)) 
        res.loc[:,'D_et12'] = res.Qet2*(res.Zwi_1+res.Zwn_1) #ET flow goes through subsoil first - may need to change
        res.loc[:,'D_12'] = res.D_d12 + res.D_et12 #Mobile to immobile phase
        res.loc[:,'D_21'] = res.D_d12 #Immobile to mobile phase
        #Water - topsoil - Diffusion and ET
        res.loc[:,'D_d14'] = 1/(1/(params.val.kxw*res.A4*res.Z1)+Y4/(res.A4*res.Deff4*res.Zw4))
        res.loc[:,'Det14'] = res.Qet4*(res.Zwi_1+res.Zwn_1)
        res.loc[:,'D_14'] = res.D_d12 + res.D_et12 #Transfer to topsoil
        res.loc[:,'D_41'] = res.D_d12 #Transfer from topsoil
        res.loc[:,'DT1'] = res.D_12+res.D_14+res.Dadv1+res.Dr1 #Total D value
        #Water does not go to shoots (3), air (5), roots (6-8). Explicit for error checking.
        res.loc[:,'D_13'] = 0
        res.loc[:,'D_15'] = 0
        res.loc[:,'D_16'] = 0
        res.loc[:,'D_17'] = 0
        res.loc[:,'D_18'] = 0
        
        
        #2 Subsoil - From water, to topsoil, to roots
        #Subsoil-Topsoil - Diffusion in water & particle settling(?). ET goes direct from flowing zone.
        #Bottom is lined, so no settling out of the system
        res.loc[:,'D_d24'] = 1/(Y24/(res.A4*res.Deff1*res.Zw2)+Y2/(res.A4*res.Deff4*res.Zw4)) #Diffusion. A4 is area of 2/4 interface.
        res.loc[:,'D_s42'] = params.val.U42*res.A4*res.Zq4 #Paticle settling
        res.loc[:,'D_24'] = res.D_d24 #sub to topsoil
        res.loc[:,'D_42'] = res.D_d24 + res.D_s42 #top to subsoil
        #Subsoil-Root Body (6)
        #Plant uptake - depends on neutral and ionic processes
        #First, calculate the value of N =zeF/RT
        res.loc[:,'N2'] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.temp2)
        res.loc[:,'N6'] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.temp6)
        #Soil-root body
        res.loc[:,'Dsr_n2'] = res.A6*(1-params.val.froot_top)*(res.kspn*res.Zwn_2)
        res.loc[:,'Dsr_i2'] = res.A6*(1-params.val.froot_top)*(res.kspi*res.Zwi_2*res.N2/(np.exp(res.N2)-1))
        res.loc[:,'D_apo2'] = (1-params.val.froot_top)*(params.val.f_apo)*(res.Zw2) #Apoplast bypass
        res.loc[:,'Drs_n2'] = res.A6*(1-params.val.froot_top)*(res.kspn*res.Zwn_6)
        res.loc[:,'Drs_i2'] = res.A6*(1-params.val.froot_top)*(res.kspi*res.Zwi_6*res.N6/(np.exp(res.N6)-1))
        res.loc[mask,'Dsr_i2'], res.loc[mask,'Drs_i2'] = 0,0 #Set neutral to zero
        res.loc[:,'D_rd62'] = (1-params.val.froot_top)*res.V6 * res.Z6 * params.val.k_rd  #Root death
        res.loc[:,'D_rd72'] = (1-params.val.froot_top)*res.V7 * res.Z7 * params.val.k_rd  #Root death
        res.loc[:,'D_rd82'] = (1-params.val.froot_top)*res.V8 * res.Z8 * params.val.k_rd  #Root death
        res.loc[:,'D_26'] = res.Dsr_n2+res.Dsr_i2+res.D_apo2
        res.loc[:,'D_62'] = res.Drs_n2+res.Drs_i2+res.D_rd62
        res.loc[:,'D_72'] = res.D_rd72
        res.loc[:,'D_82'] = res.D_rd82
        res.loc[:,'DT2'] = res.D_21+res.D_24+res.D_26+res.Dadv2+res.Dr2 #Total D value
        #Subsoil does not go to with shoots (3), air (5), roots (7-8). Explicit for error checking.
        res.loc[:,'D_23'] = 0
        res.loc[:,'D_25'] = 0
        res.loc[:,'D_27'] = 0
        res.loc[:,'D_28'] = 0
        
        #3 Shoots - interacts with central cylinder, air, topsoil
        #Shoots-air (Trapp 2007) see calcs in the calculation of kvv, it includes Qet in stomatal pathway
        res.loc[:,'D_d35'] = res.kvv*res.A3*res.Zn3 #Volatilization to air, only neutral species
        #Shoots-soil - Diamond (2001) - all bulk
        res.loc[:,'D_cd'] = res.A3 * params.val.RainRate\
        *(params.val.Ifw - params.val.Ilw)*params.val.lamb * res.Z5  #Canopy drip 
        res.loc[:,'D_we'] = res.A3 * params.val.kwe * res.Z3   #Wax erosion
        res.loc[:,'D_lf'] = res.V3 * res.Z3 * params.val.Rlf  #litterfall & plant death?
        #Shoots-Root Central Cylinder - Advection through ET
        res.loc[:,'D_et83'] = res.Qet*res.Zw8
        #Shoot growth - Modelled as first order decay
        res.loc[:,'D_sg'] = params.val.k_sg*res.V3*res.Z3
        res.loc[:,'D_35'] = res.D_d35
        res.loc[:,'D_53'] = res.D_d35 #Could add other depostion but doesn't seem worth it =- might eliminate air compartment
        res.loc[:,'D_34'] = res.D_cd + res.D_we + res.D_lf
        res.loc[:,'D_43'] = 0 #Could add rainsplash maybe? 
        res.loc[:,'D_38'] = 0 #Talk to Angela about this direction maybe
        res.loc[:,'D_83'] = res.D_et83
        res.loc[:,'DT3'] = res.D_35+res.D_34+res.D_38+res.D_sg+res.Dadv3+res.Dr3 #Total D value
        #Shoots do not go to water (1), subsoil (2), roots (6-8, see above for 8). Explicit for error checking.
        res.loc[:,'D_31'] = 0
        res.loc[:,'D_32'] = 0
        res.loc[:,'D_36'] = 0
        res.loc[:,'D_37'] = 0
        
        #4 Topsoil - interacts with shoots, water, air, subsoil, roots
        #Topsoil-Air, volatilization
        res.loc[:,'D_d45'] = 1/(1/(params.val.ksa*res.A4*res.Z5)+Y4/\
               (res.A4*res.Bea4*res.Z1+res.A4*res.Deff4*res.Zw4)) #Dry diffusion
        res.loc[:,'D_wd54'] = res.A4*res.Zw4*params.val.RainRate* (1-params.val.Ifw)*(1-res.phi5) #Wet gas deposion
        res.loc[:,'D_qs'] = res.A4 * res.Zq5 * params.val.RainRate * res.fpart5 *\
        params.val.Q * (1-params.val.Ifw)  * res.phi5 #Wet dep of aerosol
        res.loc[:,'D_ds'] = res.A4 * res.Zq5 *  params.val.Up * res.fpart5* (1-Ifd) #dry dep of aerosol
        #Topsoil-roots, same as for subsoil
        res.loc[:,'N4'] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.temp4)
        #Soil-root body
        res.loc[:,'Dsr_n4'] = res.A6*params.val.froot_top*(res.kspn*res.Zwn_4)
        res.loc[:,'Dsr_i4'] = res.A6*params.val.froot_top*(res.kspi*res.Zwi_4*res.N4/(np.exp(res.N4)-1))
        res.loc[:,'D_apo4'] = params.val.froot_top*(params.val.f_apo)*(res.Zw2) #Apoplast bypass
        res.loc[:,'Drs_n4'] = res.A6*params.val.froot_top*(res.kspn*res.Zwn_6)
        res.loc[:,'Drs_i4'] = res.A6*params.val.froot_top*(res.kspi*res.Zwi_6*res.N6/(np.exp(res.N6)-1))
        res.loc[mask,'Dsr_i4'], res.loc[mask,'Drs_i4'] = 0,0 #Set neutral to zero
        res.loc[:,'D_rd64'] = (1-params.val.froot_top)*res.V6 * res.Z6 * params.val.k_rd  #Root death
        res.loc[:,'D_rd74'] = (1-params.val.froot_top)*res.V7 * res.Z7 * params.val.k_rd  #Root death
        res.loc[:,'D_rd84'] = (1-params.val.froot_top)*res.V8 * res.Z8 * params.val.k_rd  #Root death
        res.loc[:,'D_46'] = res.Dsr_n4+res.Dsr_i4+res.D_apo4
        res.loc[:,'D_64'] = res.Drs_n4+res.Drs_i4+res.D_rd64
        res.loc[:,'D_74'] = res.D_rd74
        res.loc[:,'D_84'] = res.D_rd84
        res.loc[:,'D_45'] = res.D_d45
        res.loc[:,'D_54'] = res.D_d45 + res.D_wd54 + res.D_qs + res.D_ds
        res.loc[:,'DT4'] = res.D_41+res.D_42+res.D_43+res.D_45+res.D_46+res.Dadv4+res.Dr4 #Total D val
        #Topsoil does not go to roots (7-8). Explicit for error checking.
        res.loc[:,'D_47'] = 0
        res.loc[:,'D_48'] = 0
        
        #5 Air - shoots, topsoil
        res.loc[:,'DT5'] = res.D_54 + res.D_53 +res.Dadv5+res.Dr5
        #Air does not go to water (1), subsoil (2), roots (6-8). Explicit for error checking.
        res.loc[:,'D_51'] = 0
        res.loc[:,'D_52'] = 0
        res.loc[:,'D_56'] = 0
        res.loc[:,'D_57'] = 0
        res.loc[:,'D_58'] = 0
        
        #6 Root Body - subsoil, xylem, topsoil
        #roots-xylem froot_tip is the fraction of root without secondary epothileum
        res.loc[:,'N7'] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.temp7)
        res.loc[:,'Drx_n'] = (params.val.froot_tip)*res.A7*(res.kspn*res.Zwn_6) #A7 is root/xylem interface
        res.loc[:,'Drx_i'] = (params.val.froot_tip)*res.A7*(res.kspi*res.Zwi_6*res.N2/(np.exp(res.N6)-1))
        res.loc[:,'Dxr_n'] = (params.val.froot_tip)*res.A7*(res.kspn*res.Zwn_7)
        res.loc[:,'Dxr_i'] = (params.val.froot_tip)*res.A7*(res.kspi*res.Zwi_7*res.N7/(np.exp(res.N7)-1))
        res.loc[mask,'Drx_i'], res.loc[mask,'Dxr_i'] = 0,0 #Set neutral to zero
        res.loc[:,'D_et6'] = res.Qet*res.Zw6
        res.loc[:,'D_rg6'] = params.val.k_rg*res.V6*res.Z6 #root growth
        res.loc[:,'D_67'] = res.Drx_n+res.Drx_i+res.D_et6
        res.loc[:,'D_76'] = res.Dxr_n+res.Dxr_i
        res.loc[:,'DT6'] = res.D_62+res.D_64+res.D_67+res.D_rg6+res.Dadv6+res.Dr6 #Total D val
        #Root body does not go to water (1), shoots(3), air (5), root (8). Explicit for error checking.
        res.loc[:,'D_61'] = 0
        res.loc[:,'D_63'] = 0
        res.loc[:,'D_65'] = 0
        res.loc[:,'D_68'] = 0
        
        #7 Xylem - root body, central cylinder
        #xylem-central cylinder - just advection
        res.loc[:,'D_et7'] = res.Qet*res.Zw7
        res.loc[:,'D_rg7'] = params.val.k_rg*res.V7*res.Z7 #root growth
        res.loc[:,'D_78'] = res.D_et7
        res.loc[:,'D_87'] = 0
        res.loc[:,'DT7'] = res.D_72 + res.D_74 + res.D_78 + res.D_rg7 +res.Dadv7+res.Dr7 #Total D val
        #Xylem does not go to water (1), shoots (3), air (5). Explicit for error checking.
        res.loc[:,'D_71'] = 0
        res.loc[:,'D_73'] = 0
        res.loc[:,'D_75'] = 0
        
        #8 Root central cylinder - shoots, xylem
        res.loc[:,'D_rg8'] = params.val.k_rg*res.V8*res.Z8 #root growth
        res.loc[:,'DT8'] = res.D_82 + res.D_83 + res.D_83 + res.D_87 + res.D_rg8 +res.Dadv8+res.Dr8 #Total D val
        #RCC does not go to water (1), air (5). Explicit for error checking.
        res.loc[:,'D_81'] = 0
        res.loc[:,'D_85'] = 0
        res.loc[:,'D_86'] = 0
        
        return res


    def run_it(self,locsumm,chemsumm,params,numc,pp,timeseries):
        """Feed the calculated values into the ADRE equation over time.
        timeseries is the timeseries data of temperatures, rainfall, influent 
        concentrations, influent volumes, redox conditions? What else?
        
        timeseries(df): Contains all of the time-varying parameters. 2d dataframe
        with a "time" column. Must contain Qin, Qout, RainRate, WindSpeed and 
        input concentrations for each compound at each time step, others optional
        """
        
        dt = timeseries.time[0] #Set this so it works
        ntimes = len(timeseries['time'])
        #Set up 4D output dataframe by adding time as the third multi-index
        #Index level 0 = time, level 1 = chems, level 2 = cell number
        times = timeseries.index
        res_t = dict.fromkeys(times,[]) #This dict will contain the outputs
        #pdb.set_trace()
        for t in range(ntimes):
            #First, update params. Updates:
            #Qin, Qout, RainRate, WindSpeed, 
            params.loc['Qin','val'] = timeseries.Qin[t] #m³/s
            params.loc['Qout','val'] = timeseries.Qout[t] #m³/s
            params.loc['RainRate','val'] = timeseries.RainRate[t] #m³/h
            params.loc['WindSpeed','val'] = timeseries.WindSpeed[t] #m/s
            #Next, update locsumm
            #Temperature (Tj), pHj, condj
            comps = locsumm.index
            for j in range(numc):
                Tj, pHj, condj = 'T' + str(j+1), 'pH' + str(j+1), 'cond'+ str(j+1)
                if Tj in timeseries.columns: #Only update if this is a time-varying parameter
                      locsumm.loc[comps[j],'Temp'] = timeseries.loc[t,Tj]
                if pHj in timeseries.columns:
                      locsumm.loc[comps[j],'Temp'] = timeseries.loc[t,pHj]
                if condj in timeseries.columns:
                          locsumm.loc[comps[j],'Temp'] = timeseries.loc[t,pHj]
            #Need to update advective flow in air compartment,   
            res = self.input_calc(locsumm,chemsumm,params,pp,numc)
            
            #Then add the upstream boundary condition
            chems = chemsumm.index
            for i in range(np.size(chems)):
                chem_Cin = chems[i] + '_Cin'
                #Assuming chemical concentration in g/m³ activity [mol/L³] = C/Z,
                #using Z1 in the first cell (x) (0) 
                chemsumm.loc[chems[i],'bc_us'] = timeseries.loc[t,chem_Cin]/\
                chemsumm.MolMass[chems[i]]/res.Z1[chems[i],0] #mol/m³    
            #Put it in res
            res.loc[:,'bc_us'] = res['dummy'].mul(chemsumm.bc_us, level = 0) #mol/m³
            #Initial conditions for each compartment
            if t is 0: #Set initial conditions here. 
                #initial Conditions
                for j in range(0,numc):
                    a_val = 'a'+str(j+1) + '_t'
                    res.loc[:,a_val] = 0 #Can make different for the different compartments
                    dt = timeseries.time[1]-timeseries.time[0]
            else: #Set the previous solution aj_t1 to the inital condition (aj_t)
                for j in range(0,numc):
                    a_val, a_valt1 = 'a'+str(j+1) + '_t', 'a'+str(j+1) + '_t1'
                    res_past = res_t[t-1].copy(deep=True)
                    res.loc[:,a_val] = res_past.loc[:,a_valt1]
                dt = timeseries.time[t] - timeseries.time[t-1] #timestep can vary
            #Now - run it forwards a time step!
            res = self.ADRE_1DUSS(res,params,numc,dt)
            res_t[t] = res.copy(deep=True)
            """
            #Testing - just to have something here
            for j in range(0,numc):
                a_val = 'a'+str(j+1) + '_t1'
                res.loc[:,a_val] = 0 

            """
            #Now comes the twicky part. Put together 
            #for j in range(0,numc):
            #    a_val = 'a'+str(j+1) + '_t'
            #    res.loc[:,a_val] = 0
        #Once we are out of the time loop, put the whole dataframe together
        res_time = pd.concat(res_t)
        """
                chems = chemsumm.index
        numchems = len(chems)
        resi = dict.fromkeys(chems,[])
        #Using the chems as the keys of the dict(resi) then concatenate
        for i in range(numchems):
            resi[chems[i]] = res.copy(deep=True)
        res = pd.concat(resi)
        """
        return res_t, res_time
    