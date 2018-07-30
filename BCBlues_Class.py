# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:32:24 2018

@author: Tim Rodgers
"""

from FugModel import FugModel #Import the parent FugModel class
from HelperFuncs import ppLFER, vant_conv, arr_conv, make_ppLFER #Import helper functions
import numpy as np
import pandas as pd

class BCBlues(FugModel):
    """ Model of contaminant transport in a bioretention cell. BCBlues objects
    have the following properties:
        
    Attributes:
    ----------

            locsumm (df): physical properties of the BC
            chemsumm (df): phyical-chemical properties of modelled compounds
            params (df): Other parameters of the model
            results (df): Results of the BC model
            num_compartments (int): (optional) number of non-equilibirum 
            compartments and size of D value matrix
            name (str): (optional) name of the BC model 
            pplfer_system (df): (optional) input ppLFERs to use in the model
    """
    
    def __init__(self,bcsumm,chemsumm,params,num_compartments = 7,name = None,pplfer_system = None):
        FugModel. __init__(self,bcsumm,chemsumm,params,num_compartments,name)
        self.pp = pplfer_system
        self.ic = self.input_calc(self.bcsumm,self.chemsumm,self.params,self.pp)        
                
    def input_calc(self,locsumm,chemsumm,params,pp):
        """Calculate Z, D and inp values using the compartment parameters from
        bcsumm and the chemical parameters from chemsumm, along with other 
        parameters in params.
        """
        #Declare constants
        R = 8.314 #Ideal gas constant, J/mol/K
        #Initialize results by copying the chemsumm dataframe
        res = pd.DataFrame.copy(chemsumm,deep=True)
        #Calculate chemical-independent parameters
        locsumm.loc[:,'V']= locsumm.Area*locsumm.Depth #Calculate volumes m³
        locsumm.loc[:,'TempK'] = locsumm.Temp +273.15 #°C to K
        locsumm.loc['Air','Density'] = 0.029 * 101325 / (R * locsumm.Temp.Air) #Air density kg/m^3
        delta_blv = 0.004 * ((0.07 / params.Value.WindSpeed) ** 0.5) #leaf boundary layer depth
        Ifd = 1 - np.exp(-2.8 * params.Value.Beta) #Vegetation dry deposition interception fraction
        #Compound-specific transport parameters
        #Fraction soil volume occupied by interstitial air and water
        res.loc[:,'Bea'] = res.AirDiffCoeff*locsumm.VFAir.Soil**(10/3) \
            /(locsumm.VFAir.Soil +locsumm.VFWat.Soil)**2
        res.loc[:,'Bew'] = res.WatDiffCoeff*locsumm.VFWat.Soil**(10/3) \
            /(locsumm.VFAir.Soil +locsumm.VFWat.Soil)**2
        res.loc[:,'k_av'] = res.AirDiffCoeff / delta_blv
        
        #ppLFER system parameters - initialize defaults if not there already
        if pp is None:
            pp = pd.DataFrame(index = ['l','s','a','b','v','c'])
            pp = make_ppLFER(pp)

        #Check if partition coefficients & dU values have been provided, or only solute descriptors
        #add based on ppLFER if not, then adjust partition coefficients for temperature of system
        #Temperatures are going to be a bit of a problem in this system as we can define
        #temperatures for every compartment. In general, use the compartment with the greatest
        #heat capacity as the temperature for the coefficient - for instance assume the boundary layer of 
        #air at the water or soil surface will be at the soil or water temperature not at the air temp.
        #May need to revisit later - in practice might be better to define at use?
        #Possible soln - temp zones of sub/sur surface schmutzdecke at the equilibrium? 
        #Aerosol-Air (Kqa), use octanol-air enthalpy
        if 'LogKqa' not in res.columns:
            res.loc[:,'LogKqa'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKqa.l,pp.logKqa.s,pp.logKqa.a,pp.logKqa.b,pp.logKqa.v,pp.logKqa.c)
        if 'dUoa' not in res.columns: #!!!This might be broken - need to check units & sign!!!
            res.loc[:,'dUoa'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUoa.l,pp.dUoa.s,pp.dUoa.a,pp.dUoa.b,pp.dUoa.v,pp.dUoa.c)
        res.loc[:,'Kqa'] = vant_conv(res.dUoa,locsumm.TempK.Air,10**res.LogKqa,T1 = 288.15) #Using air temp
        res.loc[:,'LogKqa'] = np.log10(res.Kqa)
        #Organic carbon-water (KocW), use octanol-water enthalpy (dUow)
        if 'LogKocW' not in res.columns:
            res.loc[:,'LogKocW'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKocW.l,pp.logKocW.s,pp.logKocW.a,pp.logKocW.b,pp.logKocW.v,pp.logKocW.c)
        if 'dUow' not in res.columns: #!!!This might be broken - need to check units & sign!!!
            res.loc[:,'dUow'] = 1000 * ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUow.l,pp.dUow.s,pp.dUow.a,pp.dUow.b,pp.dUow.v,pp.dUow.c)
        res.loc[:,'KocW'] = vant_conv(res.dUow,locsumm.TempK.Filter,10**res.LogKocW) #Using Filter temp 
        res.loc[:,'LogKocW'] = np.log10(res.KocW)
        #Storage Lipid Water (KslW), use ppLFER for dUslW (kJ/mol) convert to J/mol/K
        if 'LogKslW' not in res.columns:
            res.loc[:,'LogKslW'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKslW.l,pp.logKslW.s,pp.logKslW.a,pp.logKslW.b,pp.logKslW.v,pp.logKslW.c)
        if 'dUslW' not in res.columns:
            res.loc[:,'dUslW'] = 1000 * ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUslW.l,pp.dUslW.s,pp.dUslW.a,pp.dUslW.b,pp.dUslW.v,pp.dUslW.c)
        res.loc[:,'KslW'] = vant_conv(res.dUslW,locsumm.TempK.Shoots,10**res.LogKslW,T1 = 310.15) #Use Shoots temp
        res.loc[:,'LogKslW'] = np.log10(res.KslW)
        #Air-Water (Kaw) use dUaw
        if 'LogKaw' not in res.columns:
            res.loc[:,'LogKaw'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKaw.l,pp.logKaw.s,pp.logKaw.a,pp.logKaw.b,pp.logKaw.v,pp.logKaw.c)
        if 'dUaw' not in res.columns: #!!!This might be broken - need to check units & sign!!!
            res.loc[:,'dUaw'] = 1000 * ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUaw.l,pp.dUaw.s,pp.dUaw.a,pp.dUaw.b,pp.dUaw.v,pp.dUaw.c)
        res.loc[:,'Kaw'] = vant_conv(res.dUaw,locsumm.TempK.Pond,10**res.LogKaw) #Use pond zone temperature
        res.loc[:,'LogKaw'] = np.log10(res.Kaw)
        #Define storage lipid-air (KslA) and organic carbon-air (KocA) using the thermodynamic cycle
        #May have a temperature problem here too.
        res.loc[:,'KslA'] = res.KslW / res.Kaw
        res.loc[:,'KocA'] = res.KocW / res.Kaw
        #Calculate Henry's law constant (H, Pa m³/mol) using Pond temperature
        res.loc[:,'H'] = res.Kaw * R * locsumm.TempK.Pond
        
        #Calculate temperature-corrected media reaction rates
        #Air (air_rrxn /hr), 3600 converts from /s
        res.loc[:,'air_rrxn'] = 3600 * \
        arr_conv(params.Value.EaAir,locsumm.TempK.Air,res.AirOHRateConst * params.Value.OHConc)
        #Air Particles (airq_rrxn) 3600 converts from /s, use 10% of AirOHRateConst if not present
        if 'AirQOHRateConst' not in res.columns:
            res.loc[:,'airq_rrxn'] = 0.1 * res.air_rrxn
        else:
            res.loc[:,'airq_rrxn'] = 3600 * \
            arr_conv(params.Value.EaAir,locsumm.TempK.Air,res.AirQOHRateConst * params.Value.OHConc)
        #Ponding zone (pond_rrxn) and pore water (pore_rrxn) converted from Wat half life (h)
        res.loc[:,'pond_rrxn'] = arr_conv(params.Value.Ea,locsumm.TempK.Pond,np.log(2)/res.WatHL)
        res.loc[:,'pore_rrxn'] = arr_conv(params.Value.Ea,locsumm.TempK.Pore,np.log(2)/res.WatHL)
        #Filter (filt_rrxn) and schmutzdecke (schm_rrxn) converted from soil half life (h)
        #May want to better paramaterize this in the future
        res.loc[:,'filt_rrxn'] = arr_conv(params.Value.Ea,locsumm.TempK.Filter,np.log(2)/res.SoilHL)
        res.loc[:,'schm_rrxn'] = arr_conv(params.Value.Ea,locsumm.TempK.Schmutzdecke,np.log(2)/res.SoilHL)
        #Vegetation is based off of air half life, this can be overridden if chemsumm contains a VegHL column
        #Assume that roots and shoots have same rate of reaction other than temperature, defined by VegHL
        if 'VegHL' in res.columns:
            res.loc[:,'shoots_rrxn'] = arr_conv(params.Value.Ea,locsumm.TempK.Shoots,np.log(2)/res.VegHL)
            res.loc[:,'roots_rrxn'] = arr_conv(params.Value.Ea,locsumm.TempK.Roots,np.log(2)/res.VegHL)
        else:
            res.loc[:,'shoots_rrxn'] = 0.1*arr_conv(params.Value.Ea,locsumm.TempK.Shoots,res.air_rrxn,locsumm.TempK.Air)
            res.loc[:,'roots_rrxn'] = 0.1*arr_conv(params.Value.Ea,locsumm.TempK.Roots,res.air_rrxn,locsumm.TempK.Air)
        #Rhizosphere reaction (rhiz_rrxn) needs to be parameterized, for now assume 10x higher than soil
        if 'RhizHL' in res.columns:
            res.loc[:,'rhiz_rrxn'] = arr_conv(params.Value.Ea,params.Value.TempK,np.log(2)/res.RhizHL)
        else:
            res.loc[:,'rhiz_rrxn'] = 10*arr_conv(params.Value.Ea,locsumm.TempK.Rhizosphere,np.log(2)/res.SoilHL)
        
        #Convert back to half lives (h), good for error checking
        res.loc[:,'AirHL'] = np.log(2)/(res.air_rrxn)
        res.loc[:,'AirQHL'] = np.log(2)/(res.airq_rrxn)
        res.loc[:,'WatHL'] = np.log(2)/(res.wat_rrxn)

            
        
            
            
        
        return res
    
    
    