# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:52:42 2018

@author: Tim Rodgers
"""
from FugModel import FugModel #Import the parent FugModel class
from HelperFuncs import ppLFER, vant_conv, arr_conv, make_ppLFER #Import helper functions
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
#import time
import pdb #Turn on for error checking

class SubsurfaceSinks(FugModel):
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
        #self.ic = self.input_calc(self.locsumm,self.chemsumm,self.params,self.pp,self.numc)
                    
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
    
    def sys_chem(self,locsumm,chemsumm,params,pp,numc,timeseries):
        """Put together the system and the chemical parameters into the 3D dataframe
        that will be used to calculate Z and D values. Basically just tidies things
        up a bit, might not be good practice to make this a seperate function
        
        """
        #Set up the output dataframe, res, a multi indexed pandas dataframe with the 
        #index level 0 as the chemical names, 1 as the integer cell number along x
        #First, call make_system if a full system hasn't been given
        #pdb.set_trace()
        #try: #See if there is a compartment index in the timeseries
        #    timeseries.index.levels[1]
        #    res = timeseries.copy(deep=True)
        #except AttributeError: #If there is no compartment index, make the system
        res = self.make_system(locsumm,params,numc,timeseries,params.val.dx)
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
        #the mask dm is used to differentiate compartments that are discretized vs those that are not
        res.loc[res.dm,'tausq_water'] = 1/(1-2.02*np.log(res.porositywater))
        res.loc[res.dm,'Deff_water'] = res['tausq_water'].mul(chemsumm.WatDiffCoeff, level = 0) #Effective water diffusion coefficient 
        res.loc[res.dm,'Deff_subsoil'] = res['dummy'].mul(chemsumm.WatDiffCoeff, level = 0)*\
        res.fwatsubsoil**(10/3)/(res.fairsubsoil +res.fwatsubsoil)**2#Added in case we need it - might not as Deff1 may cover
        if 'pond' in numc: #Add pond for BC model
            res.loc[res.dm==False,'Deff_pond'] = res['dummy'].mul(chemsumm.WatDiffCoeff, level = 0)
            res.loc[res.dm,'Bea_subsoil'] = res['dummy'].mul(chemsumm.AirDiffCoeff, level = 0)*\
            res.fairsubsoil**(10/3)/(res.fairsubsoil +res.fwatsubsoil)**2 #Effective air diffusion coefficient
        if 'topsoil' in numc:
            res.loc[res.dm,'Deff_topsoil'] = res['dummy'].mul(chemsumm.WatDiffCoeff, level = 0)*\
                res.fwattopsoil**(10/3)/(res.fairtopsoil +res.fwattopsoil)**2 #Effective water diffusion coefficient 
            res.loc[res.dm,'Bea_topsoil'] = res['dummy'].mul(chemsumm.AirDiffCoeff, level = 0)*\
                res.fairtopsoil**(10/3)/(res.fairtopsoil +res.fwattopsoil)**2 #Effective air diffusion coefficient 
        #Dispersivity as the sum of the effective diffusion coefficient (Deff) and ldisp.
        res.loc[res.dm,'disp'] = res.ldisp + res.Deff_water #Check units - Ldisp in [m²/T], T is from flow rate
        #Read dU values in for temperature conversions. Probably there is a better way to do this.
        res.loc[:,'dUoa'] = res['dummy'].mul(chemsumm.dUoa,level = 0)
        res.loc[:,'dUow'] = res['dummy'].mul(chemsumm.dUow,level = 0)
        res.loc[:,'dUslw'] = res['dummy'].mul(chemsumm.dUslW,level = 0)
        res.loc[:,'dUaw'] = res['dummy'].mul(chemsumm.dUaw,level = 0)
        #Equilibrium constants  - call by compartment name
        #Calculate temperature-corrected media reaction rates (/h)
        #These are all set so that they can vary in x, even though for now they do not
        for j in numc:
            Kdj, Kdij, focj, tempj = 'Kd' +str(j),'Kdi' +str(j),'foc' +str(j),'temp' +str(j)
            Kawj, rrxnj = 'Kaw' +str(j),'rrxn' +str(j)
            #Kaw is only neutral
            res.loc[:,Kawj] = vant_conv(res.dUaw,res.loc[:,tempj],res['dummy'].mul(10**chemsumm.LogKaw,level = 0))
            #Kd neutral and ionic
            if j in ['water','subsoil','topsoil','pond','drain']: #for water, subsoil and topsoil if not directly input
                if 'Kd' in chemsumm.columns: #If none given leave blank, will estimate from Koc & foc
                    res.loc[:,Kdj] = res['dummy'].mul(chemsumm.Kd,level = 0)
                    maskn = np.isnan(res.loc[:,Kdj])
                else:
                    maskn = res.dummy==1
                if 'Kdi' in chemsumm.columns:#If none given leave blank, will estimate from Koc & foc
                    res.loc[:,Kdij] = res['dummy'].mul(chemsumm.Kdi,level = 0)
                    maski = np.isnan(res.loc[:,Kdij])
                else:
                    maski = res.dummy == 1
                res.loc[maskn,Kdj] = res.loc[:,focj].mul(10**chemsumm.LogKocW, level = 0)
                res.loc[maski,Kdij] = res.loc[maski,Kdj] #Assume same sorption if not given
                res.loc[:,Kdj] = vant_conv(res.dUow,res.loc[:,tempj],res.loc[:,Kdj]) #Convert with dUow
                #The ionic Kd value is based off of pKa and neutral Kow
                res.loc[:,Kdij] = vant_conv(res.dUow,res.loc[:,tempj],res.loc[:,Kdij])
                if j in ['water','pond','drain']:
                    rrxnq_j = 'rrxnq_'+str(j)
                    res.loc[:,rrxnj] = res['dummy'].mul(np.log(2)/chemsumm.WatHL,level = 0)
                    res.loc[:,rrxnj] = arr_conv(params.val.Ea,res.loc[:,tempj],res.loc[:,rrxnj])
                    res.loc[:,rrxnq_j] = res.loc[:,rrxnj] * 0.1 #Can change particle bound rrxn if need be
                if j in ['subsoil','topsoil']:
                    res.loc[:,rrxnj] = res['dummy'].mul(np.log(2)/chemsumm.SoilHL,level = 0)
                    res.loc[:,rrxnj] = arr_conv(params.val.Ea,res.loc[:,tempj],res.loc[:,rrxnj])
            elif j in ['shoots','rootbody','rootxylem','rootcyl']:
                res.loc[maskn,Kdj] = vant_conv(res.dUslw,res.loc[:,tempj],res.loc[:,focj].mul(10**chemsumm.LogKslW, level = 0))
                res.loc[maski,Kdij] = res.loc[maski,Kdj]
                if 'VegHL' in res.columns:
                    res.loc[:,rrxnj] = res['dummy'].mul(np.log(2)/chemsumm.VegHL,level = 0)
                else:#If no HL for vegetation specified, assume 0.1 * wat HL - based on Wan (2017) wheat plants?
                    res.loc[:,rrxnj] = res['dummy'].mul(np.log(2)/(chemsumm.WatHL*0.1),level = 0)
                res.loc[:,rrxnj] = arr_conv(params.val.Ea,res.loc[:,tempj],res.loc[:,rrxnj])
                if j in ['shoots']:
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
                    delta_blv = 0.004 * ((0.07 / params.val.WindSpeed) ** 0.5) #leaf boundary layer depth, windsped in m/s
                    res.loc[:,'AirDiffCoeff'] = res['dummy'].mul(chemsumm.AirDiffCoeff, level = 0)
                    res.loc[:,'kav'] = res.AirDiffCoeff/delta_blv #m/h
                    #Veg side (veg-air) MTC from Trapp (2007). Consists of stomata and cuticles in parallel
                    #Stomata - First need to calculate saturation concentration of water
                    C_h2o = (610.7*10**(7.5*(res.tempshoots-273.15)/(res.tempshoots-36.15)))/(461.9*res.tempshoots)
                    g_h2o = res.Qet/(res.A_shootair*(C_h2o-params.val.RH/100*C_h2o)) #MTC for water
                    g_s = g_h2o*np.sqrt(18)/np.sqrt(res['dummy'].mul(chemsumm.MolMass, level = 0))
                    res.loc[:,'kst'] = g_s * res['dummy'].mul((10**chemsumm.LogKaw), level = 0) #MTC of stomata [L/T] (defined by Qet so m/h)
                    #Cuticle
                    res.loc[:,'kcut'] = 10**(0.704*res['dummy'].mul((chemsumm.LogKow), level = 0)-11.2)*3600 #m/h
                    res.loc[:,'kcuta'] = 1/(1/res.kcut + 1*res['dummy'].mul((10**chemsumm.LogKaw), level = 0)/(res.kav)) #m/h
                    res.loc[:,'kvv'] = res.kcuta+res.kst #m/h
            elif j in ['air']:
                res.loc[maskn,Kdj] = vant_conv(res.dUslw,res.loc[:,tempj],res.loc[:,focj].mul(10**chemsumm.LogKslW, level = 0))
                res.loc[maski,Kdij] = res.loc[maski,Kdj]
                res.loc[:,rrxnj] = res['dummy'].mul(chemsumm.AirOHRateConst, level = 0)  * params.val.OHConc
                res.loc[:,rrxnj] = arr_conv(params.val.EaAir,res.loc[:,tempj],res.loc[:,rrxnj])
                if 'AirQOHRateConst' not in res.columns:
                    res.loc[:,'rrxnq_air'] = 0.1 * res.loc[:,rrxnj]
                else:
                    res.loc[:,'rrxnq_air'] = res['dummy'].mul(chemsumm.AirOHRateConst*0.1, level = 0)*params.val.OHConc    
                    res.loc[:,'rrxnq_air'] = arr_conv(params.val.EaAir,res.loc[:,tempj],res.airq_rrxn)    
    
        return chemsumm, res

    def input_calc(self,locsumm,chemsumm,params,pp,numc,timeseries):
        """Calculate Z, D and inp values using the compartment parameters from
        bcsumm and the chemical parameters from chemsumm, along with other 
        parameters in params.
        
        Must have either timeseries or res dataframe. 
        """               
        #Initialize the results data frame as a pandas multi indexed object with
        #indices of the compound names and cell numbers
        pdb.set_trace()
        #Make the system and add chemical properties
        chemsumm, res = self.sys_chem(self.locsumm,self.chemsumm,self.params,self.pp,self.numc,timeseries)            
        #Declare constants
        #chems = chemsumm.index
        #numchems = len(chems)
        #R = params.val.R #Ideal gas constant, J/mol/K
        #Ifd = 1 - np.exp(-2.8 * params.Value.Beta) #Vegetation dry deposition interception fraction
        Ymob_immob = params.val.Ymob_immob #Diffusion path length from mobile to immobile flow Just guessing here
        Y_subsoil = locsumm.Depth[1]/2 #Half the depth of the mobile phase
        if params.val.index.contains('Ytopsoil'):
            Y_topsoil = params.val.Ytopsoil
        else:
            Y_topsoil = locsumm.Depth[3]/2 #Diffusion path is half the depth. Probably should make vary in X
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
        for j in numc: #Loop through compartments
            dissi_j, dissn_j, pHj, Zwi_j = 'dissi_' + str(j),'dissn_' + str(j),\
            'pH' + str(j),'Zwi_' + str(j)
            gammi_j, gammn_j, Ij, Zwn_j = 'gammi_' + str(j),'gammn_' + str(j),\
            'I' + str(j),'Zwn_' + str(j)
            Zqi_j, Zqn_j, Kdj, Kdij,rhopartj = 'Zqi_' + str(j),'Zqn_' + str(j),\
            'Kd' +str(j),'Kdi' +str(j),'rhopart' +str(j)
            #Dissociation of compounds in environmental media using Henderson-Hasselbalch equation
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
            Zij, Znj, Zwj, Zqj,Zj= 'Zi_' + str(j), 'Zn_' + str(j), 'Zw_' + str(j), 'Zq_' + str(j), 'Z' + str(j)
            fpartj,fwatj,fairj,Kawj='fpart'+str(j),'fwat'+ str(j),'fair'+ str(j),'Kaw'+ str(j)
            #Set the mask for whether the compartment is discretized or not.
            if locsumm.loc[j,'Discrete'] == 'y':
                mask = res.dm.copy(deep=True)
            else: 
                mask = res.dm.copy(deep=True) ==False
            #This mask may not have worked - for now switch to true so that it just doesn't make a difference
            #mask.loc[:]  = True
            #Finally, lets calculate the Z values in the compartments
            if j in 'air': #Air we need to determine hygroscopic growth
                #Aerosol particles - composed of water and particle, with the water fraction defined
                #by hygroscopic growth of the aerosol. Growth is defined as per the Berlin Spring aerosol from Arp et al. (2008)
                if params.val.RH > 100: #maximum RH = 100%
                    params.val.RH = 100
                #Hardcoded hygroscopic growth factor (GF) not ideal but ¯\_(ツ)_/¯
                GF = np.interp(params.val.RH/100,xp = [0.12,0.28,0.77,0.92],fp = \
                        [1.0,1.08,1.43,2.2],left = 1.0,right = params.val.RH/100*5.13+2.2)
                #Volume fraction of water in aerosol 
                VFQW_a = (GF - 1) * locsumm.Density.water / ((GF - 1) * \
                          locsumm.Density.water + locsumm.PartDensity.air)
                res.loc[mask,fwatj] = res.loc[mask,fwatj] + res.loc[mask,fpartj]*VFQW_a #add aerosol water from locsumm
                res.loc[mask,fairj] = 1 - res.loc[mask,fwatj] - res.loc[mask,fpartj]
                #mask = res.dm #Change the mask so that Z values will be calculated across all x, to calculate diffusion
            res.loc[mask,Zij] = res.loc[mask,fwatj]*res.loc[mask,Zwi_j]+res.loc[mask,fpartj]\
            *res.loc[mask,Zqi_j] #No Zair for ionics
            res.loc[mask,Znj] = res.loc[mask,fwatj]*res.loc[mask,Zwn_j]+res.loc[mask,fpartj]\
            *res.loc[mask,Zqn_j]+res.loc[mask,fairj]*res.loc[mask,Kawj]
            res.loc[mask,Zwj] = res.loc[mask,fwatj]*res.loc[mask,Zwi_j] + res.loc[mask,fwatj]\
            *res.loc[mask,Zwn_j] #pure water
            res.loc[mask,Zqj] = res.loc[mask,fpartj]*res.loc[mask,Zqi_j] + res.loc[mask,fpartj]\
            *res.loc[mask,Zqn_j] #Solid/particle phase Z value
            res.loc[mask,Zj] = res.loc[mask,Zij] + res.loc[mask,Znj] #Overall Z value
            if j in ['air','water','pond','drain']: #=Calculate the particle-bound fraction in air and water
                phij = 'phi'+str(j)
                res.loc[mask,phij] = res.loc[mask,fpartj]*(res.loc[mask,Zqi_j] + res.loc[mask,Zqn_j])/res.loc[mask,Zj]

                
        """        
        #1 Water - Consists of suspended solids and pure water
        res.loc[:,'Zi1'] = (1-res.fpart1) * (res.Zwi_1) + res.fpart1 * (res.Zqi_1)
        res.loc[:,'Zn1'] = (1-res.fpart1) * (res.Zwn_1) + res.fpart1 * (res.Zqn_1)
        res.loc[:,'Z1'] = res.Zi1+res.Zn1
        
        #2 Subsoil - Immobile-phase water and soil particles
        res.loc[:,'Zi2'] = res.fwatsubsoil*(res.Zwi_2) + (res.fpart2)*(res.Zqi_2)
        res.loc[:,'Zn2'] = res.fwatsubsoil*(res.Zwn_2) + (res.fpart2)*(res.Zqn_2)
        res.loc[:,'Zw2'] = res.fwatsubsoil*(res.Zwi_2) + res.fwatsubsoil*(res.Zwn_2)
        res.loc[:,'Z2'] = res.Zi2+res.Zn2
        
        #3 Shoots - Water, lipid and air
        res.loc[:,'Zi3'] = res.fwat3*(res.Zwi_3) + res.Zqi_3 
        res.loc[:,'Zn3'] = res.fwat3*(res.Zwn_3) + res.Zqn_3 + res.fair3*res.Kaw3 
        res.loc[:,'Z3'] = res.Zi3+res.Zn3
        
        #4 Top soil - Water, soil, air
        res.loc[:,'Zi4'] = res.fwat4*(res.Zwi_4)+(1-res.fwat4-res.fair4)*res.Zqi_4
        res.loc[:,'Zn4'] = res.fwat4*(res.Zwn_4) + (1 - res.fwat4 - res.fair4)*\
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
        res.loc[:,'Zw5'] = res.fwat5*(res.Zwi_5)+res.fwat5*(res.Zwn_5)
        res.loc[:,'Zq5'] = res.fpart5*(res.Zqi_5 + res.Zqn_5)
        res.loc[:,'phi5'] = res.fpart5*(res.Zqi_5 + res.Zqn_5)/res.Z5 #particle bound fraction
        
        #6 Root Body - main portion of the root. Consists of "free space" 
        #(soil pore water), and cytoplasm - could add vaccuol
        res.loc[:,'Zi6'] = res.fwat6*(res.Zwi_6) + res.Zqi_6
        res.loc[:,'Zn6'] = res.fwat6*(res.Zwn_6) + res.Zqn_6 + res.fair6 * res.Kaw6 
        res.loc[:,'Zw6'] = res.fwat6*(res.Zwi_6) + res.fwat6*(res.Zwn_6)
        res.loc[:,'Z6'] = res.Zi6 + res.Zn6
        
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
        
        #9 Ponding Zone - only for BCs, code was written first without hence why 
        #we put the switch in!
        #Consists of suspended solids & water
        if numc == 9:
            res.loc[:,'Zi9'] = (1-res.fpart9) * (res.Zwi_9) + res.fpart9 * (res.Zqi_9)
            res.loc[:,'Zn9'] = (1-res.fpart9) * (res.Zwn_9) + res.fpart9 * (res.Zqn_9)
            res.loc[:,'Z9'] = res.Zi9+res.Zn9  
            res.loc[:,'Zw9'] = res.fwat9*(res.Zwi_9) + res.fwat9*(res.Zwn_9)
            res.loc[:,'Zq9'] = res.fpart9*(res.Zqi_9)+res.fpart9*(res.Zqn_9) 
        """
        
        #D values (m³/h), N (mol/h) = a*D (activity based)
        #Loop through compartments to set D values
        for j in numc: #Loop through compartments
            Drj, Dadvj, Zj, rrxnj, Vj= 'Dr' + str(j),'Dadv' + str(j),'Z' + \
            str(j),'rrxn' + str(j),'V' + str(j)
            advj, Dtj = 'adv' + str(j),'DT' + str(j)
            if locsumm.loc[j,'Discrete'] == 1:
                mask = res.dm
            else: 
                mask = res.dm ==False
            #Assuming that degradation is not species specific and happens on 
            #the bulk medium (unless over-written)
            if j in ['air','water','pond','drain']:#differentiate particle & bulk portions
                phij , rrxnq_j = 'phi'+str(j),'rrxnq_'+str(j)
                res.loc[mask,Drj] = (1-res.loc[mask,phij])*res.loc[mask,Vj]* res.loc[mask,rrxnj]\
                +res.loc[mask,phij]*res.loc[mask,rrxnq_j]
            res.loc[mask,Drj] = res.loc[mask,Zj] * res.loc[mask,Vj] * res.loc[mask,rrxnj] 
            res.loc[mask,Dadvj] = res.loc[mask,Zj] * res.loc[mask,Vj] * res.loc[mask,advj]
            res.loc[mask,Dtj] = res.loc[mask,Drj] + res.loc[mask,Dadvj] #Initialize total D value
            if j in ['water']: #interacts with subsoil and topsoil.
                for k in numc: #Inner loop for inter-compartmental values
                    D_jk,D_kj = 'D_'+str(j)+str(k),'D_'+str(k)+str(j)
                    try: #First, check if intercompartmental transport between jk already done
                        res.loc[:,D_jk]
                    except KeyError:
                        if k in ['subsoil','topsoil']:
                            if k == 'subsoil':
                                y = Ymob_immob
                                A = res.Asubsoil
                            elif k == 'topsoil':
                                y = Y_topsoil
                                A = res.AsoilV
                            D_djk,Detjk,Zwk,Qetk = 'D_d'+str(j)+str(k),'Det'+str(j)+str(k),\
                            'Zw_'+str(k),'Qet'+str(k)
                            res.loc[mask,D_djk] =  1/(1/(params.val.kxw*A*res.Zwater)+y/(A*res.Deff_water*res.loc[mask,Zwk])) #Diffusion from water to soil
                            res.loc[mask,Detjk] = res.loc[mask,Qetk]*res.loc[mask,Zwk] #ET flow goes through subsoil first - may need to change
                            res.loc[mask,D_jk] = res.loc[mask,D_djk] + res.loc[mask,Detjk]
                            res.loc[mask,D_kj] = res.loc[mask,D_djk]
                        elif k == j: #Processes that come directly out of the compartment go here
                            Zwk = 'Zw_'+str(k)
                            res.loc[mask,'D_waterexf'] = res.loc[mask,'Qwaterexf']*res.loc[mask,'Zw_water']
                            res.loc[mask,D_jk] = res.loc[:,'D_waterexf']
                        else: #Other compartments Djk = Dkj = 0
                            res.loc[mask,D_jk] = 0
                            res.loc[mask,D_kj] = 0
                    res.loc[mask,Dtj] += res.loc[mask,D_jk]
            #Subsoil- water, topsoil, roots, air (if no topsoil or pond),drain, pond(if present)
            elif j in ['subsoil','topsoil']:
                for k in numc: #Inner loop for inter-compartmental values
                    D_jk,D_kj = 'D_'+str(j)+str(k),'D_'+str(k)+str(j)
                    try: #First, check if intercompartmental transport between jk already done
                        res.loc[:,D_jk]
                    except KeyError:
                        if k == j: #Processes that come directly out of the compartment (not to another compartment) go here            
                            res.loc[mask,D_jk] = 0 #Nothing else from subsoil (lined bottom or to drain layer)
                        elif k in ['water']:
                            if j == 'subsoil':
                                y = Ymob_immob
                                A = res.Asubsoil
                            elif j == 'topsoil':
                                y = Y_topsoil
                                A = res.AsoilV
                            D_djk,D_etkj,Zwk,Qetj = 'D_d'+str(j)+str(k),'D_et'+str(k)+str(j),\
                            'Zw_'+str(k),'Qet'+str(j)
                            res.loc[mask,D_djk] =  1/(1/(params.val.kxw*A*res.Zwater)+y/(A*res.Deff_water*res.loc[mask,Zwk])) #Diffusion from water to soil
                            res.loc[mask,D_etkj] = res.loc[mask,Qetj]*res.loc[mask,Zwk] #ET flow goes through subsoil first - may need to change
                            res.loc[mask,D_jk] = res.loc[mask,D_djk] + res.loc[:,Detjk]
                            res.loc[mask,D_kj] = res.loc[mask,D_djk]
                        elif k in ['subsoil','topsoil']:#Subsoil - topsoil and vice-versa
                            D_djk,D_skj,Zwk = 'D_d'+str(j)+str(k),'D_s'+str(k)+str(j),\
                            'Zw_'+str(k)                          
                            res.loc[mask,D_djk] = 1/(Y_subsoil/(res.AsoilV*res.Deff_water*res.Zw_water)\
                                   +Y_topsoil/(res.AsoilV*res.Deff_topsoil*res.Zwtopsoil)) #Diffusion - both ways
                            res.loc[mask,D_skj] = params.val.U42*res.AsoilV*res.Zq4 #Particle settling - only from top to subsoil
                            if j == 'subsoil':
                                res.loc[mask,D_jk] = res.loc[mask,D_djk] #sub- to topsoil
                                res.loc[mask,D_kj] = res.loc[mask,D_djk] + res.loc[mask,D_skj] #top- to subsoil  
                            else: #k = topsoil
                                res.loc[mask,D_jk] = res.loc[mask,D_djk] + res.loc[mask,D_skj] #top- to subsoil 
                                res.loc[mask,D_kj] = res.loc[mask,D_djk]  #sub- to topsoil
                        elif k in ['rootbody','rootxylem','rootcyl']:
                            D_rdkj,Vk,Zk= 'D_rd'+str(k)+str(j),'V'+str(k),'Z'+str(k)
                            if k in ['rootbody']:
                                Nj,Nk,Qetj,tempj,tempk = 'N'+str(j),'N'+str(k),'Qet'+str(j),'temp'+str(j),'temp'+str(k)
                                Dsr_nj,Dsr_ij,Zw_j,Zwn_j,Zwi_j,Arootj = 'Dsr_n'+str(j),'Dsr_i'+str(j),'Zw_'+str(j),'Zwn_'+str(j),'Zwi_'+str(j),'Aroot'+str(j)
                                Drs_nj,Drs_ij,Zwn_k,Zwi_k = 'Drs_n'+str(j),'Drs_i'+str(j),'Zwn_'+str(k),'Zwi_'+str(k)
                                D_apoj,Qetj = 'D_apo'+str(j),'Qet'+str(j)
                                #First, calculate the value of N =zeF/RT
                                res.loc[mask,Nj] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.loc[mask,tempj])
                                res.loc[mask,Nk] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.loc[mask,tempk])       
                                res.loc[mask,Dsr_nj] = res.loc[mask,Arootj]*(res.kspn*res.loc[mask,Zwn_j])
                                res.loc[mask,Dsr_ij] = res.loc[mask,Arootj]*(res.kspi*res.loc[mask,Zwi_j]*res.loc[mask,Nj]/(np.exp(res.loc[mask,Nj])-1))
                                #Root back to soil
                                res.loc[mask,Drs_nj] = res.loc[mask,Arootj]*(res.kspn*res.loc[mask,Zwn_k])
                                res.loc[mask,Drs_ij] = res.loc[mask,Arootj]*(res.kspi*res.loc[mask,Zwi_k]*res.loc[mask,Nk]/(np.exp(res.loc[mask,Nk])-1))
                                res.loc[mask,Drs_nj], res.loc[mask,Drs_ij] = 0,0 #Set neutral to zero
                                #Overall D values
                                res.loc[mask,D_jk] = res.loc[mask,Dsr_nj] + res.loc[mask,Dsr_ij] 
                                res.loc[mask,D_kj] = res.loc[mask,Drs_nj] + res.loc[mask,Drs_ij]  
                            elif k in ['rootxylem']: 
                                res.loc[mask,D_apoj] = res.loc[mask,Qetj]*(params.val.f_apo)*(res.loc[mask,Zw_j]) #Apoplast bypass straight to the xylem
                                res.loc[mask,D_jk] = res.loc[mask,D_apoj] 
                                res.loc[mask,D_kj] = 0
                            else:#Central cylinder just root death to topsoil
                                res.loc[mask,D_jk] = 0
                                res.loc[mask,D_kj] = 0
                            res.loc[mask,D_rdkj] = (1-params.val.froot_top)*res.loc[mask,Vk] * res.loc[mask,Zk] * params.val.k_rd  #Root death
                            res.loc[mask,D_kj] += res.loc[mask,D_rdkj]
                        elif k in ['air']:
                            if 'topsoil' in numc and j =='subsoil':
                                pass
                            else:
                                if j == 'subsoil':
                                    y = res.x #Only the "topsoil" portion has a value for Asoilair
                                    Bea = res.Bea_subsoil
                                else:
                                    y = Y_topsoil
                                    Bea= res.Bea_topsoil
                                Zw_j,D_djk,Deff_j = 'Zw_'+str(j),'D_d'+str(j)+str(k),'Deff_'+str(j)
                                res.loc[mask,D_djk] = 1/(1/(params.val.ksa*res.Asoilair*res.Zair[res.dm==False])+y/\
                                       (res.Asoilair*Bea*res.Zair+res.Asoilair*res.loc[:,Deff_j]*res.loc[:,Zw_j])) #Dry diffusion
                                res.loc[:,'D_wd54'] = res.AtopsoilV*res.Zw5*params.val.RainRate* (1-params.val.Ifw) #Wet gas deposion
                                res.loc[:,'D_qs'] = res.AtopsoilV * res.Zq5 * params.val.RainRate * res.fpart5 *\
                                params.val.Q * (1-params.val.Ifw)  #Wet dep of aerosol
                                res.loc[:,'D_ds'] = res.AtopsoilV * res.Zq5 *  params.val.Up * res.fpart5* (1-Ifd) #dry dep of aerosol                            
                        else: #Other compartments Djk = Dkj = 0
                            res.loc[mask,D_jk] = res.loc[mask,D_kj] = 0
                    res.loc[mask,Dtj] += res.loc[mask,D_jk]
        #2 Subsoil - From water, to topsoil, to roots
        #Subsoil-Topsoil - Diffusion in water & particle settling(?). ET goes direct from flowing zone.
        #Bottom is lined, so no settling out of the system
        res.loc[:,'D_d24'] = 1/(Ymob_immob4/(res.AtopsoilV*res.Deff1*res.Zw2)+Ymob_immob/(res.AtopsoilV*res.Deff4*res.Zw4)) #Diffusion. 
        res.loc[:,'D_s42'] = params.val.U42*res.AtopsoilV*res.Zq4 #Particle settling
        res.loc[:,'D_24'] = res.D_d24 #sub to topsoil
        res.loc[:,'D_42'] = res.D_d24 + res.D_s42 #top to subsoil
        #Subsoil-Root Body (6)
        #Plant uptake - depends on neutral and ionic processes
        #First, calculate the value of N =zeF/RT
        res.loc[:,'N2'] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.temp2)
        res.loc[:,'N6'] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.temp6)
        #Soil-root body
        res.loc[:,'Dsr_n2'] = res.Arootsubsoil*(res.kspn*res.Zwn_2)
        res.loc[:,'Dsr_i2'] = res.Arootsubsoil*(res.kspi*res.Zwi_2*res.N2/(np.exp(res.N2)-1))
        res.loc[:,'D_apo2'] = res.Qet2*(params.val.f_apo)*(res.Zw2) #Apoplast bypass
        res.loc[:,'Drs_n2'] = res.Arootsubsoil*(res.kspn*res.Zwn_6)
        res.loc[:,'Drs_i2'] = res.Arootsubsoil*(res.kspi*res.Zwi_6*res.N6/(np.exp(res.N6)-1))
        res.loc[mask,'Dsr_i2'], res.loc[mask,'Drs_i2'] = 0,0 #Set neutral to zero
        res.loc[:,'D_rd62'] = (1-params.val.froot_top)*res.V6 * res.Z6 * params.val.k_rd  #Root death
        res.loc[:,'D_rd72'] = (1-params.val.froot_top)*res.V7 * res.Z7 * params.val.k_rd  #Root death
        res.loc[:,'D_rd82'] = (1-params.val.froot_top)*res.V8 * res.Z8 * params.val.k_rd  #Root death
        res.loc[:,'D_26'] = res.Dsr_n2+res.Dsr_i2
        res.loc[:,'D_62'] = res.Drs_n2+res.Drs_i2+res.D_rd62
        res.loc[:,'D_27'] = res.D_apo2
        res.loc[:,'D_72'] = res.D_rd72
        res.loc[:,'D_82'] = res.D_rd82
        res.loc[:,'DT2'] = res.D_21+res.D_24+res.D_26+res.D_27+res.Dadv2+res.Dr2 #Total D value
        #Subsoil does not go to with shoots (3), air (5), roots (7-8). Explicit for error checking.
        res.loc[:,'D_23'] = 0
        res.loc[:,'D_25'] = 0
        res.loc[:,'D_28'] = 0
        
        #3 Shoots - interacts with central cylinder, air, topsoil
        #Shoots-air (Trapp 2007, Diamond 2001) see calcs in the calculation of kvv, it includes Qet in stomatal pathway
        res.loc[:,'D_d35'] = res.kvv*res.A35*res.Zn3 #Volatilization to air, only neutral species
        res.loc[:,'D_rv'] = res.A35 * res.Zw5 * params.val.RainRate * params.val.Ifw  #Wet dep of gas to shoots
        res.loc[:,'D_qv'] = res.A35 * res.Zq5 * params.val.RainRate * params.val.Q * params.val.Ifw #Wet dep of aerosol
        res.loc[:,'D_dv'] = res.A35 * res.Zq5 * params.val.Up *Ifd  #dry dep of aerosol
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
        res.loc[:,'D_53'] = res.D_d35+res.D_rv+res.D_qv+res.D_dv #Could add other depostion but doesn't seem worth it =- might eliminate air compartment
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
        res.loc[:,'D_d45'] = 1/(1/(params.val.ksa*res.AtopsoilV*res.Z5)+Ytopsoil/\
               (res.AtopsoilV*res.Bea4*res.Z1+res.AtopsoilV*res.Deff4*res.Zw4)) #Dry diffusion
        res.loc[:,'D_wd54'] = res.AtopsoilV*res.Zw5*params.val.RainRate* (1-params.val.Ifw) #Wet gas deposion
        res.loc[:,'D_qs'] = res.AtopsoilV * res.Zq5 * params.val.RainRate * res.fpart5 *\
        params.val.Q * (1-params.val.Ifw)  #Wet dep of aerosol
        res.loc[:,'D_ds'] = res.AtopsoilV * res.Zq5 *  params.val.Up * res.fpart5* (1-Ifd) #dry dep of aerosol
        #Topsoil-roots, same as for subsoil
        res.loc[:,'N4'] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.temp4)
        #Soil-root body
        res.loc[:,'Dsr_n4'] = res.Aroottopsoil*(res.kspn*res.fwat4*res.Zwn_4)
        res.loc[:,'Dsr_i4'] = res.Aroottopsoil*(res.kspi*res.fwat4*res.Zwi_4*res.N4/(np.exp(res.N4)-1))
        res.loc[:,'D_apo4'] = res.Qet4*(params.val.f_apo)*(res.Zw2) #Apoplast bypass
        res.loc[:,'Drs_n4'] = res.Aroottopsoil*(res.kspn*res.fwat6*res.Zwn_6)
        res.loc[:,'Drs_i4'] = res.Aroottopsoil*(res.kspi*res.fwat6*res.Zwi_6*res.N6/(np.exp(res.N6)-1))
        res.loc[mask,'Dsr_i4'], res.loc[mask,'Drs_i4'] = 0,0 #Set neutral to zero
        res.loc[:,'D_rd64'] = params.val.froot_top*res.V6 * res.Z6 * params.val.k_rd  #Root death
        res.loc[:,'D_rd74'] = params.val.froot_top*res.V7 * res.Z7 * params.val.k_rd  #Root death
        res.loc[:,'D_rd84'] = params.val.froot_top*res.V8 * res.Z8 * params.val.k_rd  #Root death
        res.loc[:,'D_46'] = res.Dsr_n4+res.Dsr_i4
        res.loc[:,'D_64'] = res.Drs_n4+res.Drs_i4+res.D_rd64
        res.loc[:,'D_47'] = res.D_apo4
        res.loc[:,'D_74'] = res.D_rd74
        res.loc[:,'D_84'] = res.D_rd84
        res.loc[:,'D_45'] = res.D_d45
        res.loc[:,'D_54'] = res.D_d45 + res.D_wd54 + res.D_qs + res.D_ds
        res.loc[:,'DT4'] = res.D_41+res.D_42+res.D_43+res.D_45+res.D_46+res.D_47+res.Dadv4+res.Dr4 #Total D val
        #Topsoil does not go to roots (7-8). Explicit for error checking.
        res.loc[:,'D_48'] = 0
        
        #5 Air - shoots, topsoil
        res.loc[:,'DT5'] = res.D_54 + res.D_53 +res.Dadv5 + res.Dr5
        #Air does not go to water (1), subsoil (2), roots (6-8). Explicit for error checking.
        res.loc[:,'D_51'] = 0
        res.loc[:,'D_52'] = 0
        res.loc[:,'D_56'] = 0
        res.loc[:,'D_57'] = 0
        res.loc[:,'D_58'] = 0
        
        #6 Root Body - subsoil, xylem, topsoil
        #roots-xylem froot_tip is the fraction of root without secondary epothileum
        res.loc[:,'N7'] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.temp7)
        res.loc[:,'Drx_n'] = (params.val.froot_tip)*res.A7*(res.kspn*res.fwat6*res.Zwn_6) #A7 is root/xylem interface
        res.loc[:,'Drx_i'] = (params.val.froot_tip)*res.A7*(res.kspi*res.fwat6*res.Zwi_6*res.N2/(np.exp(res.N6)-1))
        res.loc[:,'Dxr_n'] = (params.val.froot_tip)*res.A7*(res.kspn*res.fwat7*res.Zwn_7)
        res.loc[:,'Dxr_i'] = (params.val.froot_tip)*res.A7*(res.kspi*res.fwat7*res.Zwi_7*res.N7/(np.exp(res.N7)-1))
        res.loc[mask,'Drx_i'], res.loc[mask,'Dxr_i'] = 0,0 #Set neutral to zero
        #res.loc[:,'D_et6'] = res.Qet*res.Zw6 #Removed because this should just be diffusion across membrane
        res.loc[:,'D_rg6'] = params.val.k_rg*res.V6*res.Z6 #root growth
        res.loc[:,'D_67'] = res.Drx_n+res.Drx_i#+res.D_et6
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
        res.loc[:,'DT7'] = res.D_72 + res.D_74 + res.D_76 + res.D_78 + res.D_rg7 +res.Dadv7+res.Dr7 #Total D val
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
        #pdb.set_trace()
        #Ponding Zone - mobile water, shoots, topsoil, air
        if numc >= 9: #Need params.val.Qinf_91 - update with timestep
            #NEED res.depth_ts, res.depth_pond
            #Top boundary condition of the ponding zone - Mass flux from ponding zone
            res.loc[:,'D_inf'] = params.val.Qinf_91*res.Zw9 #Infiltration from the ponding zone to mobile water
            res.loc[:,'D_exf9'] = res.Q9_exf*res.Zw9
            res.loc[:,'D_d94'] = 1/(1/(params.val.kxw*res.AtopsoilV\
                  *res.Zw9)+(res.depth_ts/2)/(res.AtopsoilV*res.Deff2*res.Zw2)) #Diffusion between topsoil/pond
            #Particle settling to topsoil - set the value of U94 by mass balance based on flow velocity?
            #If_sw represents interception fraction of stormwater by vegetation
            res.loc[:,'D_s94'] = (1-params.val.If_sw)*params.val.Udx*res.AtopsoilV*res.Zq9 #Particle settling
            res.loc[:,'D_r94'] = params.val.Urx*res.AtopsoilV*res.Zq4 #Sediment resuspension
            res.loc[:,'D_s93'] = (params.val.If_sw)*params.val.Udx*res.AtopsoilV*res.Zq9 #Particle deposition on plants
            res.loc[:,'D_r93'] = params.val.Urx*res.AtopsoilV*res.Zq9 #Sediment resuspension - assume for plants that only deposited sediment is re-suspended
            #Need to figure out plant uptake from water - do plants still work? Maybe just sorption?
            #Assume diffusion only through cuticle (MTC for cuticle permeability), water (MTC water)
            ###
            res.loc[:,'A93'] = params.val.LAI*res.A3 #TEMPORARY ONLY - set in BCBlues module
            ###
            res.loc[:,'D_d93'] = 1/(1/(res.kcut*res.A93*res.Z5)+\
                   1/(params.val.kmw*res.A93*res.Zw9)) #Water-plant diffusion
            #Pond-Air
            res.loc[:,'D_d95'] = 1/(1/(params.val.kma*res.AtopsoilV*res.Z5)+(locsumm.Depth.Pond/2)/\
               (res.AtopsoilV*res.Bea4*res.Z1+res.A9V*res.Deff9*res.Zw9)) #Dry air/water diffusion
            ###
            res.loc[:,'A95'] = res.A9 #TEMPORARY ONLY - set in BCBlues module
            ###
            res.loc[:,'D_wd59'] = res.A95*res.Zw5*params.val.RainRate* (1-params.val.Ifw)#Wet gas deposion
            res.loc[:,'D_qw'] = res.A95 * res.Zq5 * params.val.RainRate * res.fpart5 *\
            params.val.Q * (1-params.val.Ifw) #Wet dep of aerosol
            res.loc[:,'D_dw'] = res.A95 * res.Zq5 *  params.val.Up * res.fpart5* (1-Ifd) #dry dep of aerosol
            #Inter-compartmental D values
            res.loc[:,'D_91'] = res.D_inf
            res.loc[:,'D_92'] = 0
            res.loc[:,'D_93'] = res.D_s93+res.D_d93
            res.loc[:,'D_94'] = res.D_d94+res.D_s94
            res.loc[:,'D_95'] = res.D_d95
            res.loc[:,'D_96'] = 0
            res.loc[:,'D_97'] = 0
            res.loc[:,'D_98'] = 0
            res.loc[:,'D_19'] = 0 
            res.loc[:,'D_29'] = 0
            res.loc[:,'D_39'] = res.D_d93+res.D_r93
            res.loc[:,'D_49'] = res.D_d94+res.D_r94
            res.loc[:,'D_59'] = res.D_d95+res.D_wd59+res.D_qw+res.D_dw
            res.loc[:,'D_69'] = 0
            res.loc[:,'D_79'] = 0
            res.loc[:,'D_89'] = 0
            #Total D values - need to update all
            res.loc[:,'DT9'] = res.D_91+res.D_92+res.D_93+res.D_94+res.D_95+res.D_96\
            +res.D_97+res.D_98+res.Dadv9+res.Dr9 #Total D val D_adv9 is weir overflow
            res.loc[:,'DT3'] += res.D_39
            res.loc[:,'DT4'] += res.D_49
            res.loc[:,'DT5'] += res.D_59
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
                Tj, pHj, condj = 'T' + str(j), 'pH' + str(j), 'cond'+ str(j)
                if Tj in timeseries.columns: #Only update if this is a time-varying parameter
                      locsumm.loc[comps[j],'Temp'] = timeseries.loc[t,Tj]
                if pHj in timeseries.columns:
                      locsumm.loc[comps[j],'pH'] = timeseries.loc[t,pHj]
                if condj in timeseries.columns:
                          locsumm.loc[comps[j],'cond'] = timeseries.loc[t,condj]
            #Need to update advective flow in air compartment, 
            # res = self.input_calc(locsumm,chemsumm,params,pp,numc)
            if numc == 9:
                res_numc = 9
            else:
                res_numc = 8
            res = self.input_calc(locsumm,chemsumm,params,pp,res_numc) #Currently need to have all the compartments
            #Then add the upstream boundary condition
            chems = chemsumm.index
            for i in range(np.size(chems)):
                chem_Cin = chems[i] + '_Cin'
                Z_us = res.Zwn_1[chems[i],0] + res.Zwi_1[chems[i],0] #Inlet activity capacity
                #Assuming chemical concentration in g/m³ activity [mol/m³] = C/Z,
                #using Z1 in the first cell (x) (0) 
                chemsumm.loc[chems[i],'bc_us'] = timeseries.loc[t,chem_Cin]/\
                chemsumm.MolMass[chems[i]]/Z_us#res.Z1[chems[i],0] #mol/m³    
            #Put it in res
            res.loc[:,'bc_us'] = res['dummy'].mul(chemsumm.bc_us, level = 0) #mol/m³
            #Initial conditions for each compartment
            if t is 0: #Set initial conditions here. 
                #initial Conditions
                for j in range(0,numc):
                    a_val = 'a'+str(j+1) + '_t'
                    res.loc[:,a_val] = 0 #1#Can make different for the different compartments
                    dt = timeseries.time[1]-timeseries.time[0]
            else: #Set the previous solution aj_t1 to the inital condition (aj_t)
                for j in range(0,numc):
                    a_val, a_valt1 = 'a'+str(j+1) + '_t', 'a'+str(j+1) + '_t1'
                    res_past = res_t[t-1].copy(deep=True)
                    res.loc[:,a_val] = res_past.loc[:,a_valt1]
                dt = timeseries.time[t] - timeseries.time[t-1] #timestep can vary
            #Now - run it forwards a time step!
            #Feed the time to params
            params.loc['Time','val']=t
            res = self.ADRE_1DUSS(res,params,numc,dt)
            res_t[t] = res.copy(deep=True)
        
        #Outside of loop put it all together
        res_time = pd.concat(res_t)

        return res_t, res_time
    
    def mass_flux(self,res_time,numc):
        """ This function calculates mass fluxes between compartments and
        out of the overall system. Calculations are done at the same discretization 
        level as the system, to get the overall mass fluxes for a compartment use 
        mass_flux.loc[:,'Variable'].groupby(level=[0,1]).sum()
        """
        #First determine the number 
        numx = len(res_time.groupby(level=2))
        dt = res_time.index.levels[0][1]-res_time.index.levels[0][0]
        #Make a dataframe to display mass flux on figure
        mass_flux = pd.DataFrame(index = res_time.index)
        #First, we will add the advective transport out and in to the first and last
        #cell of each compound/time, respectively
        N_effluent = res_time.M_n[slice(None),slice(None),numx-1] - res_time.M_xf[slice(None),slice(None),numx-1]
        mass_flux.loc[:,'N_effluent'] = 0
        mass_flux.loc[(slice(None),slice(None),numx-1),'N_effluent'] = N_effluent
        mass_flux.loc[:,'N_influent'] = res_time.inp_mass1#This assumes inputs are zero
        #Now, lets get to compartment-specific transport
        for j in range(numc):#j is compartment mass is leaving
            Drj,Nrj,a_val, NTj, DTj= 'Dr' + str(j+1),'Nr' + str(j+1),'a'+str(j+1) + '_t1','NT' + str(j+1),'DT' + str(j+1)
            #Transformation (reaction) in each compartment Mr = Dr*a*V
            mass_flux.loc[:,Nrj] = dt*(res_time.loc[:,Drj] * res_time.loc[:,a_val])#Reactive mass loss
            mass_flux.loc[:,NTj] = dt*(res_time.loc[:,DTj] * res_time.loc[:,a_val])#Total mass out
            for k in range(numc):#From compartment j to compartment k
                if j != k:
                    Djk,Njk = 'D_'+str(j+1)+str(k+1),'N' +str(j+1)+str(k+1)
                    mass_flux.loc[:,Njk] = dt*(res_time.loc[:,Djk] * res_time.loc[:,a_val])
        #Growth dilution processes are modelled as first-order decay. For now, add to the reactive M value
        mass_flux.loc[:,'Nr3'] += dt*(res_time.loc[:,'D_sg'] * res_time.loc[:,'a3_t1'])
        mass_flux.loc[:,'Nr6'] += dt*(res_time.loc[:,'D_rg6'] * res_time.loc[:,'a6_t1'])
        mass_flux.loc[:,'Nr7'] += dt*(res_time.loc[:,'D_rg7'] * res_time.loc[:,'a7_t1'])
        mass_flux.loc[:,'Nr8'] += dt*(res_time.loc[:,'D_rg8'] * res_time.loc[:,'a8_t1'])
        
        #Now, let's define the net transfer between compartments that we are interested in.
        #The convention here is that a positive number is a transfer in the direction indicated, and a negative number is the opposite.
        mass_flux.loc[:,'Nwss'] = mass_flux.N12 - mass_flux.N21
        mass_flux.loc[:,'Nssts'] = mass_flux.N14 - mass_flux.N24
        mass_flux.loc[:,'Nssrb'] = mass_flux.N26 - mass_flux.N62
        mass_flux.loc[:,'Ntsrb'] = mass_flux.N46 - mass_flux.N64
        mass_flux.loc[:,'Ntsa'] = mass_flux.N45 - mass_flux.N54
        mass_flux.loc[:,'Nrbx'] = mass_flux.N67 - mass_flux.N76
        mass_flux.loc[:,'Nxc'] = mass_flux.N78 - mass_flux.N87
        mass_flux.loc[:,'Ncs'] = mass_flux.N83 - mass_flux.N38
        mass_flux.loc[:,'Nsts'] = mass_flux.N34 - mass_flux.N43
        mass_flux.loc[:,'Nsa'] = mass_flux.N35 - mass_flux.N53
        if numc == 9:
            mass_flux.loc[:,'Npw'] = mass_flux.N91 - mass_flux.N19
            mass_flux.loc[:,'Npts'] = mass_flux.N94 - mass_flux.N49
            mass_flux.loc[:,'Npa'] = mass_flux.N95 - mass_flux.N59
        
        #Overall 'removal' as 
        mass_flux.loc[:,'removal'] = (mass_flux.N_influent-mass_flux.N_effluent)/mass_flux.N_influent
        
        return mass_flux
    
    #def mass_balance(self,locsumm,chemsumm,params,numc,pp,timeseries):