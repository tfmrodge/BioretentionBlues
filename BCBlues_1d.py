# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:52:42 2018

@author: Tim Rodgers
"""
from FugModel import FugModel #Import the parent FugModel class
from HelperFuncs import ppLFER, vant_conv, arr_conv, make_ppLFER #Import helper functions
import numpy as np
import pandas as pd
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
        #self.ic = self.input_calc(self.locsumm,self.chemsumm,self.params,self.pp)
        
    def make_system(self,locsumm,params,numc,dx = None):
        #This function will build the dimensions of the 1D system based on the "locsumm" input file.
        #If you want to specify more things you can can just skip this and input a dataframe directly
        L = locsumm.Length.Water
        R = 8.3144598
        if dx == None:
            dx = params.val.dx
        #Integer cell number is the index, columns are values, 'x' is the centre of each cell
        res = pd.DataFrame(np.arange(0+dx/2,L,dx),columns = ['x'])
        #Set up the water compartment
        res.loc[:,'Q1'] = params.val.Qin - (params.val.Qin-params.val.Qout)/L*res.x 
        res.loc[:,'q1'] = res.Q1/(locsumm.Depth[0] * locsumm.Width[0])  #darcy flux [L/T] at every x
        res.loc[:,'porosity1'] = locsumm.Porosity[0] #added so that porosity can vary with x
        res.loc[:,'porosity2'] = locsumm.Porosity[1] #added so that porosity can vary with x
        res.loc[:,'porosity4'] = locsumm.Porosity[3]
        res.loc[:,'A1'] = locsumm.Width[0] * locsumm.Depth[0] * res.porosity1
        res.loc[:,'A2'] = locsumm.Width[0] * locsumm.Depth[0] * res.porosity2
        res.loc[:,'v1'] = res.q1/res.porosity1 #velocity [L/T] at every x
        #Now loop through the columns and set the values
        #pdb.set_trace()
        for j in range(numc+1):
            #Area (A), Volume (V), Density (rho), organic fraction (foc)
            #water fraction (fwat), air fraction (fair), temperature (tempi)
            Aj, Vj, rhoj, focj = 'A' + str(j+1), 'V' + str(j+1),'rho' + str(j+1),'foc' + str(j+1)
            fwatj, fairj, tempj = 'fwat' + str(j+1), 'fair' + str(j+1),'temp' + str(j+1)
            if j <= 1: #done above, assuming water and subsoil as 1 and 2
                pass
            else: #Other compartments don't share the same CV
                res.loc[:,Aj] = locsumm.Width[j] * locsumm.Depth[j]
            res.loc[:,Vj] = res.loc[:,Aj] * dx #volume at each x [L³]
            res.loc[:,focj] = locsumm.FrnOC[j] #Fraction organic matter
            res.loc[:,fwatj] = locsumm.FrnWat[j] #Fraction water
            res.loc[:,fairj] = locsumm.FrnAir[j] #Fraction air
            res.loc[:,tempj] = locsumm.Temp[j] + 273.15 #Temperature [K]
            if locsumm.index[j] == 'Air': #Set air density based on temperature
                res.loc[:,rhoj] = 0.029 * 101325 / (R * res.loc[:,tempj])
            else:
                res.loc[:,rhoj] = locsumm.Density[j] #density for every x [M/L³]

        #Longitudinal Dispersivity. Calculate using relationship from Schulze-Makuch (2005) 
        #for unconsolidated sediment unless a value of alpha [L] is given
        if 'alpha' not in params.index:
            params.loc['alpha','val'] = 0.2 * L**0.44 #alpha = c(L)^m, c = 0.2 m = 0.44
        res.loc[:,'ldisp'] = params.val.alpha * res.v1
        return res
                    
    def make_chems(self,chemsumm,pp = None):
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
        #Calculate Henry's law constant (H, Pa m³/mol) using Pond temperature
        res.loc[:,'H'] = 10**res.LogKaw * R * 298.15
        
        return res
    
    def input_calc(self,locsumm,chemsumm,params,pp,numc):
        """Calculate Z, D and inp values using the compartment parameters from
        bcsumm and the chemical parameters from chemsumm, along with other 
        parameters in params.
        """
        #Declare constants
        R = 8.3144598 #Ideal gas constant, J/mol/K
        #delta_blv = 0.004 * ((0.07 / params.val.WindSpeed) ** 0.5) #leaf boundary layer depth
        #Ifd = 1 - np.exp(-2.8 * params.Value.Beta) #Vegetation dry deposition interception fraction
        
        #Set up the output dataframe, res, a multi indexed pandas dataframe with the 
        #index level 0 as the chemical names, 1 as the integer cell number along x
        #First, call make_system if a full system hasn't been given
        if locsumm.index.name == 'Compartment':
            #res = self.make_system(locsumm,params,numc,0.1)
            res = self.make_system(locsumm,params,numc,params.dx)
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
        
        #Set parameters that vary with x and chemicals
        #For the sake of future-proofing, this is most things
        #Deff = 1/tortuosity^2, tortuosity(j)^2 = 1-2.02*ln(porosity) (Shen and Chen, 2007)
        res.loc[:,'tausq1'] = 1/(1-2.02*np.log(res.porosity1))
        res.loc[:,'tausq4'] = 1/(1-2.02*np.log(res.porosity4))
        res.loc[:,'Deff1'] = res['tausq1'].mul(chemsumm.WatDiffCoeff, level = 0)
        res.loc[:,'Deff4'] = res['tausq4'].mul(chemsumm.WatDiffCoeff, level = 0)
        #Dispersivity as the sum of the effective diffusion coefficient (Deff) and ldisp.
        res.loc[:,'disp'] = res.ldisp + res.Deff1
        
        #Calculate temperature-corrected media reaction rates
        #Air (air_rrxn /hr), 3600 converts from /s
        #Add a dummy variable as mul is how I am sorting by levels w/e
        res.loc[:,'dummy'] = 1
        res.loc[:,'air_rrxn'] = res['dummy'].mul(chemsumm.AirOHRateConst, level = 0)  * params.val.OHConc    
        res.loc[:,'air_rrxn'] = 3600 * arr_conv(params.val.EaAir,res.temp5,res.AirOHRateConst * params.Value.OHConc)

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
        res.loc[:,'pore_rrxn'] = arr_conv(params.Value.Ea,locsumm.TempK.Filt_pores,np.log(2)/res.WatHL)
        #Filter (filt_rrxn) and schmutzdecke (schm_rrxn) converted from soil half life (h)
        #May want to better paramaterize this in the future
        res.loc[:,'filt_rrxn'] = arr_conv(params.Value.Ea,locsumm.TempK.Filter,np.log(2)/res.SoilHL)
        res.loc[:,'surf_rrxn'] = arr_conv(params.Value.Ea,locsumm.TempK.Surface,np.log(2)/res.SoilHL)
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
            
        #Calculate Z-values (mol/m³/Pa)
        #Air Za
        res.loc[:,'Za'] = 1/(R*locsumm.TempK.Air)
        #Water in air
        res.loc[:,'Za_w'] = 1/(vant_conv(res.dUaw,locsumm.TempK.Air,10**res.LogKaw) * R * locsumm.TempK.Air)
        #Lower and Upper air Aerosol particles - composed of water and particle, with the water fraction defined
        #by hygroscopic growth of the aerosol. Growth is defined as per the Berlin Spring aerosol from Arp et al. (2008)
        if params.Value.RH > 100: #maximum RH = 100%
            params.Value.RH = 100
        #Hardcoded hygroscopic growth factor (GF) not ideal but ¯\_(ツ)_/¯
        GF = np.interp(params.Value.RH/100,xp = [0.12,0.28,0.77,0.92],fp = [1.0,1.08,1.43,2.2],\
                       left = 1.0,right = params.Value.RH/100*5.13+2.2)
        #Volume fraction of water in aerosol
        VFQW_a = (GF - 1) * locsumm.Density.Pond / ((GF - 1) * \
                  locsumm.Density.Pond + locsumm.loc['Air','PartDensity'])
        #Volume fraction of 'nucleus'
        VFQp_a = 1 - VFQW_a
        #Calculate aerosol Z values
        res.loc[:,'Za_q'] = res.loc[:,'Za']*res.loc[:,'Kqa']*locsumm.loc['Air','PartDensity']\
        *1000*VFQp_a+res.loc[:,'Za_w']*VFQW_a
        #Ponding Zone Water
        #Water Paw (nonequilbrium constant) value is calculated using the 
        #ratio of Zair (Tair) and ZWater (Tpond) as per Scheringer (2000) 10.1021/es991085a
        res.loc[:,'Zp_w'] = locsumm.TempK.Air/(locsumm.TempK.Pond*res.loc[:,'H'])
        #Suspended Particles
        res.loc[:,'Zp_q'] = res.Zp_w*vant_conv(res.dUow,locsumm.TempK.Pond,10**res.LogKocW)\
        *locsumm.PartFrnOC.Pond * locsumm.PartDensity.Pond/1000
        #Schmutzdecke/Surface layer
        #This is a thin layer at the top of the BC consisting of mulch and some soil, where
        #particles will settle. Biofilm? Growth due to particle settling, or definition as top x cm?
        #In that case the filter zone will need to grow over time as well. I need to make sure that I am defining
        #particle fractions based on a mass balance of particles  - this will be done by the bc_dims method
        #Surface Air
        res.loc[:,'Zsurf_a'] = 1/(R*locsumm.TempK.Surf)
        #Captured aerosol particles = Za*Kqa*PartFrnOCAir
        res.loc[:,'Zsurf_aq'] = res.Zsurf_a * vant_conv(res.dUoa,locsumm.TempK.Surface,10**res.LogKqa,T1 = 288.15)\
        *locsumm.PartDensity.Air*1000/(R*locsumm.TempK.Surface) * locsumm.PartFrnOC.Air
        #Captured stormwater particles - check
        res.loc[:,'Zsurf_wq'] = vant_conv(res.dUow,locsumm.TempK.Surface,10**res.LogKocW)\
        *locsumm.PartFrnOC.Pond * locsumm.PartDensity.Pond/1000/ \
        (vant_conv(res.dUaw,locsumm.TempK.Surface,10**res.LogKaw) * R * locsumm.TempK.Surface)
        #Surface film layer -Ksla
        res.loc[:,'Zsurf_f'] = vant_conv(res.dUslW,locsumm.TempK.Surface,10**res.LogKslW,T1 = 310.15)\
        /vant_conv(res.dUaw,locsumm.TempK.Surface,10**res.LogKaw)*params.Value.VFOCFilm
        #Surface Leaf Litter - similar to film, defined from storage lipid
        res.loc[:,'Zsurf_l'] = res.Zsurf_a * vant_conv(res.dUslW,locsumm.TempK.Surface,10**res.LogKslW,T1 = 310.15)\
        /vant_conv(res.dUaw,locsumm.TempK.Surface,10**res.LogKaw)*locsumm.FrnOC.Shoots
        #Surface Soil Particles (mostly mulch?)
        res.loc[:,'Zsurf_s'] = res.Zsurf_a * vant_conv(res.dUow,locsumm.TempK.Surface,10**res.LogKocW)\
        /vant_conv(res.dUaw,locsumm.TempK.Surface,10**res.LogKaw)*locsumm.FrnOC.Surface
        #Filter Zone
        #Filter Zone AIr
        res.loc[:,'Zfilt_a'] = 1/(R*locsumm.TempK.Filter)
        #Filter zone solids - equilibrium 
        res.loc[:,'Zfilt_s'] = res.Zfilt_a * vant_conv(res.dUow,locsumm.TempK.Filter,10**res.LogKocW)\
        /vant_conv(res.dUaw,locsumm.TempK.Filter,10**res.LogKaw)*locsumm.FrnOC.Filter
        #Pore Water
        #Pore Water - what about DOC? Is there a good way to deal with DOC - perhaps by increasing Zpw?
        res.loc[:,'Zpore_w'] = 1/(vant_conv(res.dUaw,locsumm.TempK.Pore_Water,10**res.LogKaw) * R * locsumm.TempK.Air)
        #
        
        
        
        #Bulk Z Values (Zb_j)
        #Air (1) - consists of Zq and Za
        res.loc[:,'Zb_1'] = res.loc[:,'Za'] * (1-locsumm.VFPart.Air) + res.loc[:,'Za_q'] * locsumm.VFPart.Air
        #Pond (2) - Sediment and bulk water
        res.loc[:,'Zb_2'] = res.loc[:,'Zp_w'] * (1-locsumm.VFPart.Pond) + res.loc[:,'Zp_q'] * locsumm.VFPart.Pond
        #Surface (3) - SW particles, aerosol particles, film, leaf litter, air - Water is in pore layer.
        res.loc[:,'Zb_3'] = res.loc[:,'Zsurf_aq'] * (locsumm.VFaq.Surface) + \
        res.loc[:,'Zsurf_wq'] * locsumm.VFwq.Surface + res.loc[:,'Zsurf_f'] * locsumm.VFFilm.Surface\
        +res.loc[:,'Zsurf_a'] * locsumm.VFAir.Surface + res.loc[:,'Zsurf_s'] * locsumm.VFSolids.Surface
        #Filter Zone (4) - just air and solids
        res.loc[:,'Zb_4'] = res.loc[:,'Zfilt_a'] * locsumm.VFAir.Filter + res.loc[:,'Zfilt_s'] * locsumm.VFSolids.Filter
        #Pore Water (5) - Just pore water
        res.loc[:,'Zb_5'] = res.loc[:,'Zpore_w']
        
       
        return res
    