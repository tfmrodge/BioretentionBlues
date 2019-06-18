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
    
    def __init__(self,locsumm,chemsumm,params,num_compartments = 8,name = None,pplfer_system = None):
        FugModel. __init__(self,locsumm,chemsumm,params,num_compartments,name)
        self.pp = pplfer_system
        #self.ic = self.input_calc(self.locsumm,self.chemsumm,self.params,self.pp)        
                
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
        res.loc[:,'Bea'] = res.AirDiffCoeff*locsumm.VFAir.Filter**(10/3) \
            /(locsumm.VFAir.Filter +locsumm.VFWat.Filter)**2
        res.loc[:,'Bew'] = res.WatDiffCoeff*locsumm.VFWat.Filter**(10/3) \
            /(locsumm.VFAir.Filter +locsumm.VFWat.Filter)**2
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
        #Organic carbon-water (KocW), use octanol-water enthalpy (dUow)
        if 'LogKocW' not in res.columns:
            res.loc[:,'LogKocW'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKocW.l,pp.logKocW.s,pp.logKocW.a,pp.logKocW.b,pp.logKocW.v,pp.logKocW.c)
        if 'dUow' not in res.columns: #!!!This might be broken - need to check units & sign!!!
            res.loc[:,'dUow'] = 1000 * ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUow.l,pp.dUow.s,pp.dUow.a,pp.dUow.b,pp.dUow.v,pp.dUow.c)
        res.loc[:,'KocW'] = vant_conv(res.dUow,locsumm.TempK.Filter,10**res.LogKocW) #Using Filter temp 
        #Storage Lipid Water (KslW), use ppLFER for dUslW (kJ/mol) convert to J/mol/K
        if 'LogKslW' not in res.columns:
            res.loc[:,'LogKslW'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKslW.l,pp.logKslW.s,pp.logKslW.a,pp.logKslW.b,pp.logKslW.v,pp.logKslW.c)
        if 'dUslW' not in res.columns:
            res.loc[:,'dUslW'] = 1000 * ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUslW.l,pp.dUslW.s,pp.dUslW.a,pp.dUslW.b,pp.dUslW.v,pp.dUslW.c)
        res.loc[:,'KslW'] = vant_conv(res.dUslW,locsumm.TempK.Shoots,10**res.LogKslW,T1 = 310.15) #Use Shoots temp
        #Air-Water (Kaw) use dUaw
        if 'LogKaw' not in res.columns:
            res.loc[:,'LogKaw'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKaw.l,pp.logKaw.s,pp.logKaw.a,pp.logKaw.b,pp.logKaw.v,pp.logKaw.c)
        if 'dUaw' not in res.columns: #!!!This might be broken - need to check units & sign!!!
            res.loc[:,'dUaw'] = 1000 * ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUaw.l,pp.dUaw.s,pp.dUaw.a,pp.dUaw.b,pp.dUaw.v,pp.dUaw.c)
        res.loc[:,'Kaw'] = vant_conv(res.dUaw,locsumm.TempK.Pond,10**res.LogKaw) #Use pond zone temperature
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
        
    def bc_dims(locsumm,inflow,rainrate,dt,params):
        """
        Calculate BC dimension & compartment information for a given time step.
        Forward calculation of t(n+1) from inputs at t(n)
        The output of this will be a "locsumm" file which can be fed into the rest of the model.
        These calculations do not depend on the contaminant transport calculations.
        
        This module includes particle mass balances, where particles are
        advective transfer media for compounds in the model.
        Particle model based off of 
        Water flow modelling based on Randelovic et al (2016)
        
        locsumm gives the conditions at t(n)
        Inflow in m³/s, rainrate in mm/h, dt in s
        Initial conditions includes height of water (h), saturation(S)
        Could make this just part of locsumm?
        """
        
        res = locsumm.copy(deep=True)
        res.loc[:,'V'] = res.Area * res.Depth #Volume m³
        res.loc[:,'P'] = 2 * (res.Area/res.Depth + res.Depth) #Perimeter, m ## Need to make into hydraulic perimeter##
        #Ponding Zone
        #Convert rain to flow rate (m³/s). Direct to ponding zone
        Qr_in = res.Area.Air*rainrate/3.6E6 #m³/s
        #Infiltration Kf is in mm/hr
        #Potential
        Qinf_poss = params.Value.Kf * (res.Depth.Pond + res.Depth.Filter)\
        /res.Depth.Filter*res.Area.Filter
        #Upstream max volume
        Qinf_us = 1/dt * (res.V.Pond+(Qr_in+inflow)*dt)
        #Downstream capacity
        #Filter saturation in filt_pores depth column
        res.loc['Filt_pores','Depth'] = res.V.Filt_pores /(res.V.Filter * (1-res.VFPart.Filter))
        #Maximum infiltration to native soils, Ks is hydr. cond. of native soil
        Q_max_inf = params.Value.Ks * (res.Area.Filter * res.P.Filter*params.Value.Cs)
        Qinf_ds= 1/dt + ((1-res.Depth.Filt_pores)*(1-res.VFPart.Filter) * res.V.Filter) +Q_max_inf #Why not Qpipe here?
        #FLow from pond to filter zone
        Q26 = min(Qinf_poss,Qinf_us,Qinf_ds)
        #Flow over weir
        if res.Depth.Pond > params.Hw:
            #Physically possible
            Q2_wp = params.Cq * params.Bw * np.sqrt(2*9.81*(res.Depth.Pond - params.Hw)^3)
            #Upstream Control
            Q2_wus = 1/dt * (res.Depth.Pond - params.Hw)*params.Area.Pond + (Qr_in+inflow)*dt - Q26*dt
            Q2_w = min(Q1_wp,Q2_wus)
        else:
            Q2_w = 0
        #Exfiltration to surrounding soil
        #Maximum possible
        if res.Area.Filter > res.Area.Pond:
            Qpexf_poss = params.Value.Ks*((res.Area.Pond-res.Area.Filter) + params.Value.Cs*res.P.Filter)
        else:
            Qpexf_poss = params.Value.Ks*params.Value.Cs*res.P.Filter
        #Upstream availability, no need for downstream as it flows out of the system
        Qpexf_us = 1/dt*(res.V.Pond) + (Qr_in+inflow-Q26-Q1_w)*dt
        Q2_exf = min(Qpexf_poss,Qpexf_us) #Actual exfiltration
        
        #Pond Volume from mass balance
        #Change in pond volume dVp at t
        dVp = (inflow + Qr_in - Q26 - Q2_w - Q2_exf)*dt
        res.loc['Pond','V'] += dVp
        #Ponding height m at t+1
        res.loc['Pond','Depth'] = np.interp(res.V.Pond,params.loc['Vp',:],params.loc['hp',:],\
                       left = 0, right = params.loc['hp','End'])  
        #Area of ponding surface m² at t+1
        res.loc['Pond','Area'] = np.interp(res.V.Pond,params.loc['Vp',:],params.loc['Ap',:],\
                       params.loc['Ap','Value'], right = params.loc['Ap','End'])

        #Pore Water Flow - filter zone
        #Capillary Rise - from drainage/submerged zone to filter zone.
        if res.Depth.Filt_Pores > params.Ss and res.Depth.Filt_Pores < params.Sfc:
            Cr = 4 * params.Emax/(params.Value.Sfc - params.Value.Ss)
            Q10_cp = res.Area.Filter * Cr * (res.Depth.Filt_Pores-params.Ss)*(params.Sfc - res.Depth.Filt_Pores)
            #Upstream volume available (in drainage layer)
            Q10_cus = ((1-res.VFPart.Filter)*res.Depth.Drain_pores*res.Area.Filter)/dt
            #Space available in pore_filt
            Q10_cds = 1/dt * ((1 - res.Depth.Filt_Pores)*(1-res.VFPart.Filter)*res.Depth.Filter*res.Area.Filter - Q26*dt)
            Q106 = min(Q2_cp,Q2_cus,Q10_cds)
        else: 
            Q106 = 0
        #Estimated saturation at time step t+1
        S_est = min(1.0,res.Depth.Filt_Pores+Q26*dt/(res.V.Filter))
        #Infiltration from filter_pore to drainage/submerged_pore layer
        Q6_infp = res.Area.Filter*params.Kf*S_est*(res.Depth.Pond + res.Depth.Filter)/res.Depth.Filter
        Q6_inf_us = 1/dt * ((res.Depth.Filt_Pores-params.Value.Sh)*res.VFPart.Filter*res.Depth.Filter*res.Area.Filter \
                            + Q26 * dt + Q106 * dt)/dt
        Q610 = min(Q6_infp,Q6_inf_us)
        #Flow due to evapotranspiration. Some will go our the air, some will be transferred to the plants for cont. transfer?
        if S_est <= params.Value.Sh:
            Q6_etp = 0
        elif S_est <= params.Value.Sw:
            Q6_etp = res.Area.Filter * params.Value.Ew*(res.Depth.Filt_Pores-params.Value.Sh)\
            /(params.Value.Sw - params.Value.Sh)
        elif S_est <= params.Value.Ss:
            Q6_etp = res.Area.Filter * (params.Value.Ew +(params.Value.Emax - params.Value.Ew)\
            *(res.Depth.Filt_Pores-params.Value.Sw)/(params.Value.Ss - params.Value.Sw))
        else:
            Q6_etp = res.Area.Filter*params.Value.Emax
        #Upstream available - should Q610 be subtracted maybe?
        Q6_etus = 1/dt* ((res.Depth.Filt_Pores-params.Value.Sh)*(1-res.VFPart.Filter)*res.Area.Filter +(Q26+Q106+Q610)*dt)
        Q6_et = min(Q6_etp,Q6_etus)
        
        #Filter Pore Water Volume from mass balance
        #Change in filter pore water volume dVf at t
        dVf = (Q26 + Q106 - Q610 - Q6_et)*dt
        res.loc['Filt_pores','V'] += dVf
        #Filter Saturation (in the filt_pores depth column) at t+1
        res.loc['Filt_pores','Depth'] = res.V.Filt_pores /(res.V.Filter * (1-res.VFPart.Filter))  
        
        #Pore water flow - drainage/submerged zone
        #Exfiltration from filter to native soil
        Q10_exfp = params.Value.Ks * (res.Area.Drainage + params.Value.Cs*\
                   res.P.Drainage*res.Depth.Drain_pores/res.Depth.Drainage)
        Q10_exfus = 1/dt * ((1-res.VFPart.Drainage)*res.Depth.Drain_pores*res.Area.Drainage + (Q610-Q106)*dt)
        Q10_exf = min(Q10_exfp,Q10_exfus)
        #Drain through pipe
        if res.Depth.Drain_pores >= params.Value.hpipe:
            Q10_pipe = 1/dt * ((res.Depth.Drain_pores-params.Value.hpipe)*(1-res.VFPart.Drainage)\
            *res.Area.Drainage + (Q610-Q106-Q10_exf)*dt)

        #Drainage Pore Water Volume from mass balance
        #Change in drainage pore water volume dVd at t
        dVd = (Q610 - Q106 - Q10_exf - Q10_pipe)*dt
        res.loc['Drain_pores','V'] += dVd
        #Height of submerged zone
        res.loc['Drain_pores','Depth'] = res.V.Drain_pores /(res.Area.Drainage * (1-res.VFPart.Drainage))
        
        #Put final flows into the res df. Flow rates are given as flow from a compartment (row)
        #to another compartment (column). Flows out of the system have their own
        #columns (eg exfiltration, ET, outflow), as do flows into the system.
        res.loc['Pond','Qto6'] = Q26 #Infiltration to filter
        res.loc['Pond','QOut'] = Q2_w #Weir overflow
        res.loc['Pond','Qexf'] = Q2_exf #exfiltration from pond
        res.loc['Pond','Qin'] = inflow #From outside system
        res.loc['Filt_pores','Qto10'] = Q610 #Infiltration to drainage layer
        res.loc['Filt_pores','QET'] = Q6_et #Infiltration to drainage layer
        res.loc['Drain_pores','Qto6'] = Q106 #Capillary rise
        res.loc['Drain_pores','Qexf'] = Q10_exf #exfiltration from drainage layer
        res.loc['Drain_pores','QOut'] = Q10_pipe #Weir overflow
        #Calculate VFair for drain and filter zones based on saturation
        res.loc['Filter','VFAir'] = VFPart - res.Filt_pores.Depth
        res.loc['Filter','VFAir'] = VFPart - res.Filt_pores.Depth



        return res
    
    
    