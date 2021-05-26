# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:52:42 2018

@author: Tim Rodgers
"""
from FugModel import FugModel #Import the parent FugModel class
from HelperFuncs import ppLFER, vant_conv, arr_conv, make_ppLFER, find_nearest #Import helper functions
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
        res.loc[:,'LogKqa'] = np.log10(vant_conv(res.dUoa,298.15,10.**res.LogKqa,T1 = 288.15))
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
        res.loc[:,'LogKslW'] = np.log10(vant_conv(res.dUslW,298.15,10.**res.LogKslW,T1 = 310.15))
        #Air-Water (Kaw) use dUaw
        if 'LogKaw' not in res.columns:
            res.loc[:,'LogKaw'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKaw.l,pp.logKaw.s,pp.logKaw.a,pp.logKaw.b,pp.logKaw.v,pp.logKaw.c)
        if 'dUaw' not in res.columns: #!!!This might be broken - need to check units & sign!!!
            res.loc[:,'dUaw'] = 1000 * ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUaw.l,pp.dUaw.s,pp.dUaw.a,pp.dUaw.b,pp.dUaw.v,pp.dUaw.c)

        #Define storage lipid-air (KslA) and organic carbon-air (KocA) using the thermodynamic cycle
        res.loc[:,'LogKslA'] = np.log10(10.0**res.LogKslW / 10.0**res.LogKaw)
        res.loc[:,'LogKocA'] = np.log10(10.0**res.LogKocW / 10.0**res.LogKaw)
        #Calculate Henry's law constant (H, Pa m³/mol) at 298.15K
        res.loc[:,'H'] = 10.**res.LogKaw * R * 298.15
        #Define Kocw from either an input value or from Kow
        
 
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
        #Read dU values in for temperature conversions. Probably there is a better way to do this.
        res.loc[:,'dUoa'] = res['dummy'].mul(chemsumm.dUoa,level = 0)
        res.loc[:,'dUow'] = res['dummy'].mul(chemsumm.dUow,level = 0)
        res.loc[:,'dUslw'] = res['dummy'].mul(chemsumm.dUslW,level = 0)
        res.loc[:,'dUaw'] = res['dummy'].mul(chemsumm.dUaw,level = 0)
        #Equilibrium constants  - call by compartment name
        #Calculate temperature-corrected media reaction rates (/h)
        #These can vary in x, although this makes the dataframe larger. 
        #pdb.set_trace()
        for j in numc:
            Kdj, Kdij, focj, tempj = 'Kd' +str(j),'Kdi' +str(j),'foc' +str(j),'temp' +str(j)
            Kawj, rrxnj = 'Kaw' +str(j),'rrxn' +str(j)
            #Kaw is only neutral
            
            res.loc[:,Kawj] = vant_conv(res.dUaw,res.loc[:,tempj],res['dummy'].mul(10.**chemsumm.LogKaw,level = 0))
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
                #pdb.set_trace()
                res.loc[maskn,Kdj] = res.loc[:,focj].mul(10.**chemsumm.LogKocW, level = 0)
                res.loc[maski,Kdij] = res.loc[:,focj].mul(10.**(chemsumm.LogKocW-3.5), level = 0) #3.5 log units lower from Franco & Trapp (2010)
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
                res.loc[maskn,Kdj] = vant_conv(res.dUslw,res.loc[:,tempj],res.loc[:,focj].mul(10.**chemsumm.LogKslW, level = 0))
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
                    res.loc[:,'kmvn'] = 10.**(1.2*res['dummy'].mul(chemsumm.LogKow, level = 0) - 7.5) * 3600 #Convert from m/s to m/h
                    res.loc[:,'kmvi'] = 10.**(1.2*(res['dummy'].mul(chemsumm.LogKow, level = 0) -3.5) - 7.5)* 3600 #Convert from m/s to m/h
                    res.loc[:,'kspn'] = 1/(1/params.val.kcw + 1/res.kmvn) #Neutral MTC between soil and plant. Assuming that there is a typo in Trapp (2000)
                    res.loc[:,'kspi'] = 1/(1/params.val.kcw + 1/res.kmvi)
                    #Correct for kmin = 10E-10 m/s for ions
                    kspimin = (10e-10)*3600
                    res.loc[res.kspi<kspimin,'kspi'] = kspimin
                    #Air side MTC for veg (from Diamond 2001)
                    #Back calculate windspeed in m/s. As the delta_blv requires a windspeed to be calculated, then replace with minwindspeed
                    try:
                        windspeed = res.loc[:,'advair']/np.array(np.sqrt(locsumm.Area.air)*locsumm.Depth.air*3600)
                    except AttributeError:
                        windspeed = timeseries.WindSpeed
                    windspeed.loc[windspeed==0] = params.val.MinWindSpeed #This will also replace all other compartments
                    #res.loc[:,'delta_blv'] = 0.004 * ((0.07 / windspeed.reindex(res.index,method = 'bfill')) ** 0.5) 
                    #res.loc[:,'delta_blv'] = 0.004 * ((0.07 / windspeed.reindex(res.index,level = 0)) ** 0.5) 
                    res.loc[:,'delta_blv'] = 0.004 * ((0.07 / windspeed.reindex(res.index,level = 1)) ** 0.5) 
                    #delta_blv = 0.004 * ((0.07 / params.val.WindSpeed) ** 0.5) #leaf boundary layer depth, windsped in m/s
                    res.loc[:,'AirDiffCoeff'] = res['dummy'].mul(chemsumm.AirDiffCoeff, level = 0)
                    res.loc[:,'kav'] = res.AirDiffCoeff/res.delta_blv #m/h
                    #Veg side (veg-air) MTC from Trapp (2007). Consists of stomata and cuticles in parallel
                    #Stomata - First need to calculate saturation concentration of water
                    C_h2o = (610.7*10.**(7.5*(res.tempshoots-273.15)/(res.tempshoots-36.15)))/(461.9*res.tempshoots)
                    g_h2o = res.Qet/(res.A_shootair*(C_h2o-params.val.RH/100*C_h2o)) #MTC for water
                    g_s = g_h2o*np.sqrt(18)/np.sqrt(res['dummy'].mul(chemsumm.MolMass, level = 0))
                    res.loc[:,'kst'] = g_s * res['dummy'].mul((10.**chemsumm.LogKaw), level = 0) #MTC of stomata [L/T] (defined by Qet so m/h)
                    #Cuticle
                    res.loc[:,'kcut'] = 10.**(0.704*res['dummy'].mul((chemsumm.LogKow), level = 0)-11.2)*3600 #m/h
                    res.loc[:,'kcuta'] = 1/(1/res.kcut + 1*res['dummy'].mul((10.**chemsumm.LogKaw), level = 0)/(res.kav)) #m/h
                    res.loc[:,'kvv'] = res.kcuta+res.kst #m/h
            elif j in ['air']:
                res.loc[maskn,Kdj] = vant_conv(res.dUoa,res.loc[:,tempj],res.loc[:,focj].mul(10.**chemsumm.LogKqa, level = 0))
                res.loc[maski,Kdij] = res.loc[maski,Kdj]
                res.loc[:,rrxnj] = res['dummy'].mul(chemsumm.AirOHRateConst, level = 0)  * params.val.OHConc
                res.loc[:,rrxnj] = arr_conv(params.val.EaAir,res.loc[:,tempj],res.loc[:,rrxnj])
                if 'AirQOHRateConst' not in res.columns:
                    res.loc[:,'rrxnq_air'] = 0.1 * res.loc[:,rrxnj]
                else:
                    res.loc[:,'rrxnq_air'] = res['dummy'].mul(chemsumm.AirOHRateConst*0.1, level = 0)*params.val.OHConc    
                    res.loc[:,'rrxnq_air'] = arr_conv(params.val.EaAir,res.loc[:,tempj],res.airq_rrxn)    
        #Then the final transport parameters
        #Deff = 1/tortuosity^2, tortuosity(j)^2 = 1-2.02*ln(porosity) (Shen and Chen, 2007)
        #the mask dm is used to differentiate compartments that are discretized vs those that are not
        #pdb.set_trace()
        res.loc[res.dm,'tausq_water'] = 1/(1-2.02*np.log(res.porositywater))
        res.loc[res.dm,'Deff_water'] = res['tausq_water'].mul(chemsumm.WatDiffCoeff, level = 0) #Effective water diffusion coefficient 
        #Now, we are going to calculate a Kow dependent diffusion coefficient. Taken from the diffusion
        #coefficient for bottom sediment from Wu & Gschwend (1988) as: 
        #Dc,s = Dc,w*porosity**2/((1-porosity)*rho(s)*Kd+porosity)
        #res.loc[res.dm,'Deff_subsoil'] = res['dummy'].mul(chemsumm.WatDiffCoeff, level = 0)*res.porositywater**2\
        #/((1-res.porositywater)*res.rhosubsoil/1000*res.Kdsubsoil+res.porositywater)
        #Or, just depends on water fraction
        res.loc[res.dm,'Deff_subsoil'] = res['dummy'].mul(chemsumm.WatDiffCoeff, level = 0)*res.porositywater**2\
        /((1-res.porositywater)*res.rhosubsoil+res.porositywater)
        if 'pond' in numc: #Add pond for BC model
            res.loc[res.dm==False,'Deff_pond'] = res['dummy'].mul(chemsumm.WatDiffCoeff, level = 0)
            res.loc[res.dm,'Bea_subsoil'] = res['dummy'].mul(chemsumm.AirDiffCoeff, level = 0)*\
            res.fairsubsoil**(10/3)/(res.fairsubsoil +res.fwatsubsoil)**2 #Effective air diffusion coefficient
            res.loc[res.dm==False,'D_air'] = res['dummy'].mul(chemsumm.AirDiffCoeff, level = 0)
        if 'topsoil' in numc:
            res.loc[res.dm,'Deff_topsoil'] = res['dummy'].mul(chemsumm.WatDiffCoeff, level = 0)*\
                res.fwattopsoil**(10/3)/(res.fairtopsoil +res.fwattopsoil)**2 #Effective water diffusion coefficient 
            res.loc[res.dm,'Bea_topsoil'] = res['dummy'].mul(chemsumm.AirDiffCoeff, level = 0)*\
                res.fairtopsoil**(10/3)/(res.fairtopsoil +res.fwattopsoil)**2 #Effective air diffusion coefficient 
        #Dispersivity as the sum of the effective diffusion coefficient (Deff) and ldisp.
        res.loc[res.dm,'disp'] = res.ldisp + res.Deff_water #Check units - Ldisp in [m²/T], T is from flow rate    
        return chemsumm, res

    def input_calc(self,locsumm,chemsumm,params,pp,numc,timeseries):
        """Calculate Z, D and inp values using the compartment parameters from
        bcsumm and the chemical parameters from chemsumm, along with other 
        parameters in params.
        
        Must have either timeseries or res dataframe. 
        """               
        #Initialize the results data frame as a pandas multi indexed object with
        #indices of the compound names and cell numbers
        #pdb.set_trace()
        #Make the system and add chemical properties
        chemsumm, res = self.sys_chem(locsumm,chemsumm,params,pp,numc,timeseries)    
        #Declare constants
        #chems = chemsumm.index
        #numchems = len(chems)
        #R = params.val.R #Ideal gas constant, J/mol/K
        #Ifd = 1 - np.exp(-2.8 * params.Value.Beta) #Vegetation dry deposition interception fraction
        Ymob_immob = params.val.Ymob_immob #Diffusion path length from mobile to immobile flow Just guessing here
        #Y_subsoil = locsumm.Depth[1]/2 #Half the depth of the mobile phase
        try:
            res.loc[:,'Y_subsoil'] = res.depth_ss/2
        except AttributeError:    
            pass

        try:
             res.loc[:,'Y_topsoil'] = params.val.Ytopsoil
        except AttributeError:
            res.loc[:,'Y_topsoil'] = res.depth_ts/2 #Diffusion path is half the depth. 
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
        for jind, j in enumerate(numc): #Loop through compartments
            jind = jind+1 #The compartment number, for overall & intercompartmental D values
            dissi_j, dissn_j, pHj, Zwi_j = 'dissi_' + str(j),'dissn_' + str(j),\
            'pH' + str(j),'Zwi_' + str(j)
            gammi_j, gammn_j, Ij, Zwn_j = 'gammi_' + str(j),'gammn_' + str(j),\
            'I' + str(j),'Zwn_' + str(j)
            Zqi_j, Zqn_j, Kdj, Kdij,rhopartj = 'Zqi_' + str(j),'Zqn_' + str(j),\
            'Kd' +str(j),'Kdi' +str(j),'rhopart' +str(j)
            #Dissociation of compounds in environmental media using Henderson-Hasselbalch equation
            #dissi_j - fraction ionic, dissn_j - fraction neutral. A pka of 999 = neutral
            #Multiplying by chemcharge takes care of the cations and the anions
            res.loc[:,dissi_j] = 1-1/(1+10.**(res.chemcharge*(res.pKa-res.loc[:,pHj])))
            #Deal with the amphoterics
            mask = np.isnan(res.pKb) == False
            if mask.sum() != 0:
                res.loc[mask,dissi_j] = 1/(1+10.**(res.pKa-res.loc[:,pHj])\
                       + 10.**(res.loc[:,pHj]-res.pKb))
            #Deal with the neutrals
            mask = res.chemcharge == 0
            res.loc[mask,dissi_j] = 0
            #Then set the neutral fraction
            res.loc[:,dissn_j] = 1-res.loc[:,dissi_j]
            #Now calculate the activity of water in each compartment or sub compartment
            #From Trapp (2010) gamman = 10^(ks*I), ks = 0.3 /M Setchenov approximation
            res.loc[:,gammn_j] = 10.**(0.3*res.loc[:,Ij])
            #Trapp (2010) Yi = -A*Z^2(sqrt(I)/(1+sqrt(I))-0.3I), A = 0.5 @ 15-20°C Davies approximation
            res.loc[:,gammi_j] = 10.**(-0.5*res.chemcharge**2*(np.sqrt(res.loc[:,Ij])/\
                   (1+np.sqrt(res.loc[:,Ij]))-0.3*res.loc[:,Ij]))
            #Now, we can calculate Zw for every compartment based on the above
            #Z values for neutral and ionic 
            res.loc[:,Zwi_j] =  res.loc[:,dissi_j]/res.loc[:,gammi_j]
            res.loc[:,Zwn_j] =  res.loc[:,dissn_j]/res.loc[:,gammn_j]
            #Now we can calculate the solids based on Kd, water diss and gamma for that compartment
            #pdb.set_trace()
            res.loc[:,Zqi_j] =  res.loc[:,Kdij] * res.loc[:,rhopartj]/1000 * res.loc[:,Zwi_j]
            res.loc[:,Zqn_j] =  res.loc[:,Kdj] * res.loc[:,rhopartj]/1000 * res.loc[:,Zwn_j]
            Zij, Znj, Zwj, Zqj,Zjind,Zj ='Zi_'+str(j),'Zn_'+str(j),'Zw_'+str(j),\
            'Zq_'+str(j),'Z'+str(jind),'Z'+str(j)
            fpartj,fwatj,fairj,Kawj='fpart'+str(j),'fwat'+ str(j),'fair'+ str(j),'Kaw'+ str(j)
            #Set the mask for whether the compartment is discretized or not.
            if locsumm.loc[j,'Discrete'] == 1:
                mask = res.dm.copy(deep=True)
            else: 
                mask = res.dm.copy(deep=True) ==False
            #This mask may not have worked - for now switch to true so that it just doesn't make a difference
            #mask.loc[:]  = True
            #Finally, lets calculate the Z values in the compartments
            if j in 'air': #Air we need to determine hygroscopic growth
                #Aerosol particles - composed of water and particle, with the water fraction defined
                #by hygroscopic growth of the aerosol. Growth is defined as per the Berlin Spring aerosol from Arp et al. (2008)
                #Max RH is 100
                timeseries.loc[timeseries.RH>100.,'RH'] = 100
                #Berlin Spring aerosol from Arp et al. (2008)
                try:
                    GF = np.interp(timeseries.RH.reindex(res.index,level=1)/100,xp = [0.12,0.28,0.77,0.92],fp = \
                            [1.0,1.08,1.43,2.2],left = 1.0,right = 2.5)
                except TypeError:
                    GF = np.interp(timeseries.loc[(slice(None),'air'),'RH']/100,xp = [0.12,0.28,0.77,0.92],fp = \
                            [1.0,1.08,1.43,2.2],left = 1.0,right = 2.5)
                    GF = pd.DataFrame(GF,index = res.index.levels[1])
                    GF = GF.reindex(res.index,level = 1)
                #Volume fraction of water in aerosol 
                VFQW_a = (GF - 1) * locsumm.Density.water / ((GF - 1) * \
                          locsumm.Density.water + locsumm.PartDensity.air)
                res.loc[mask,fwatj] = res.loc[mask,fwatj] + res.loc[mask,fpartj]*VFQW_a[0] #add aerosol water from locsumm
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
            res.loc[mask,Zj] = res.loc[mask,Zij] + res.loc[mask,Znj] #Overall Z value - need to copy for the index version bad code but w/e
            res.loc[mask,Zjind] = res.loc[mask,Zj] 
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
        #Set the rainrate for wet deposition processes
        if 'air' in numc:
            try:
                rainrate = pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'QET']),index = timeseries.index.levels[0]).reindex(res.index,level=1).loc[:,0]
                #Test this.
                #rainrate = rainrate.reindex(res.index,level=1).loc[:,0]
            except KeyError:
                rainrate = timeseries.RainRate.reindex(res.index,method = 'bfill')
        #D values (m³/h), N (mol/h) = a*D (activity based)
        #Loop through compartments to set D values
        #pdb.set_trace()
        for jind, j in enumerate(numc): #Loop through compartments
            jind = jind+1 #The compartment number, for overall & intercompartmental D values
            Drj, Dadvj, Zj, rrxnj, Vj= 'Dr' + str(j),'Dadv' + str(j),'Z' + \
            str(jind),'rrxn' + str(j),'V' + str(j)
            advj, Dtj = 'adv' + str(j),'DT' + str(jind)
            if locsumm.loc[j,'Discrete'] == 1:
                mask = res.dm
            else: 
                mask = res.dm == False
            #Assuming that degradation is not species specific and happens on 
            #the bulk medium (unless over-written)
            if j in ['air','water','pond','drain']:#differentiate particle & bulk portions
                phij , rrxnq_j = 'phi'+str(j),'rrxnq_'+str(j)
                res.loc[mask,Drj] = (1-res.loc[mask,phij])*res.loc[mask,Vj]* res.loc[mask,rrxnj]\
                +res.loc[mask,phij]*res.loc[mask,rrxnq_j]
            res.loc[mask,Drj] = res.loc[mask,Zj] * res.loc[mask,Vj] * res.loc[mask,rrxnj] 
            res.loc[mask,Dadvj] = res.loc[mask,Zj] * res.loc[mask,Vj] * res.loc[mask,advj]
            res.loc[mask,Dtj] = res.loc[mask,Drj] + res.loc[mask,Dadvj] #Initialize total D value
            #Now we will go through compartments. Since this is a model of transport in water, we assume there is always 
            #a water compartment and that the water compartment is always first. This "water" is the mobile subsurface water.
            if j in ['water']: #interacts with subsoil and topsoil and pond (if present)
                for kind, k in enumerate(numc): #Inner loop for inter-compartmental values
                    kind = kind+1 #The compartment number, for overall & intercompartmental D values
                    D_jk,D_kj = 'D_'+str(jind)+str(kind),'D_'+str(kind)+str(jind)
                    try: #First, check if intercompartmental transport between jk already done
                        res.loc[:,D_jk]
                    except KeyError:
                        if k == j: #Processes that come directly out of the compartment go here
                            Zwk = 'Zw_'+str(k)
                            #Water exfiltration
                            res.loc[mask,'D_waterexf'] = res.loc[mask,'Qwaterexf']*res.loc[mask,'Zw_water']
                            res.loc[mask,D_jk] = res.loc[:,'D_waterexf']                        
                        elif k in ['subsoil','topsoil']:
                            if k in ['subsoil']:
                                y = Ymob_immob
                                A = res.Asubsoil
                            elif k in ['topsoil']:
                                y = res.Y_topsoil
                                A = res.AsoilV
                            D_djk,D_mjk,Detjk,Zwk,Zk,Qetk = 'D_d'+str(j)+str(k),'D_m'+str(j)+str(k),'Det'+str(j)+str(k),\
                            'Zw_'+str(k),'Z'+str(k),'Qet'+str(k)
                            fwatk, Vk = 'fwat'+str(k),'V' + str(k)
                            #pdb.set_trace() #Testing for diffusion and Kow
                            #res.loc[mask,D_djk] = 1/(1/(params.val.kxw*A*res.Zwater)+y/(A*res.Deff_water*res.loc[mask,Zk]/res.loc[mask,Zwk]*res.Zw))
                            #res.loc[mask,D_djk] = 1/(1/(params.val.kxw*A*res.Zwater)+y/(A*res.Deff_subsoil*res.loc[mask,Zwk])) #Diffusion from water to soil
                            #Calculate D water subsoil from Paraiba (2002)
                            Lw = res.Zwater*res.Deff_water/y
                            Lss = res.loc[mask,Zk]*res.Deff_subsoil/y
                            #Lss = res.loc[mask,Zk]*res.Deff_water/y
                            res.loc[mask,D_djk] = A*Lw*Lss/(Lw+Lss)
                            res.loc[mask,D_mjk] = params.val.wmim*res.loc[mask,fwatk]*res.loc[mask,Vk]*res.loc[mask,Zwk] #Mixing of mobile & immobile water
                            res.loc[mask,Detjk] = res.loc[mask,Qetk]*res.loc[mask,Zwk] #ET flow goes through subsoil first - may need to change
                            res.loc[mask,D_jk] = res.loc[mask,D_djk] + res.loc[mask,Detjk] + res.loc[mask,D_mjk]
                            res.loc[mask,D_kj] = res.loc[mask,D_djk]+ res.loc[mask,D_mjk]/params.val.immobmobfac
                        #If there is a ponding zone, then the activity at the upstream boundary condition is the activity in the pond.
                        #We will have to make this explicit at the beginning of the ADRE
                        elif k in ['pond']:
                            Zwk = 'Zw_'+str(k)
                            #Flow from pond to first cell. This assumes that all particles are captured in the soil, for now.
                            res.loc[res.dm==False,'D_infps'] = np.array(res.loc[(slice(None),slice(None),0),'Qin'])\
                                *res.loc[res.dm==False,Zwk] #Infiltration from the ponding zone to mobile water
                            #Overall D values - We are going to calculate advection explicitly so won't put it here. 
                            res.loc[mask,D_jk] = 0#res.loc[:,'D_inf']
                            res.loc[mask,D_kj] = 0  
                        else: #Other compartments Djk = 0
                            res.loc[mask,D_jk] = 0
                    #Add Djk to Dt & set nans to zero        
                    res.loc[np.isnan(res.loc[:,D_jk]),D_jk] = 0        
                    res.loc[mask,Dtj] += res.loc[mask,D_jk]
                    
            #Subsoil- water, topsoil, roots, air (if no topsoil or pond),drain, pond(if present)
            elif j in ['subsoil','topsoil']:
                for kind, k in enumerate(numc): #Inner loop for inter-compartmental values
                    kind = kind+1 #The compartment number, for overall & intercompartmental D values
                    D_jk,D_kj = 'D_'+str(jind)+str(kind),'D_'+str(kind)+str(jind)
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
                                y = res.Y_topsoil
                                A = res.AsoilV
                            D_djk,D_mjk,D_etkj,Zwk,Zk,Qetj = 'D_d'+str(j)+str(k),'D_m'+str(j)+str(k),'D_et'+str(k)+str(j),\
                            'Zw_'+str(k),'Z'+str(k),'Qet'+str(j)
                            #pdb.set_trace()
                            #res.loc[mask,D_djk] = 1/(1/(params.val.kxw*A*res.Zwater)+y/(A*res.Deff_water*res.loc[mask,Zk]/res.loc[mask,Zwk])).groupby(level=0).mean()
                            res.loc[mask,D_djk] =  1/(1/(params.val.kxw*A*res.Zwater)+y/(A*res.Deff_water*res.loc[mask,Zwk])) #Diffusion from water to soil
                            res.loc[mask,D_mjk] = params.val.wmim*res.loc[mask,fwatk]*res.loc[mask,Vk]*res.loc[mask,Zwk] #Mixing of mobile & immobile water
                            res.loc[mask,D_etkj] = res.loc[mask,Qetj]*res.loc[mask,Zwk] #ET flow goes through subsoil first - may need to change
                            res.loc[mask,D_jk] = res.loc[mask,D_djk] + res.loc[:,D_etkj] + res.loc[mask,D_mjk]
                            res.loc[mask,D_kj] = res.loc[mask,D_djk] + res.loc[mask,D_mjk]
                        elif k in ['subsoil','topsoil']:#Subsoil - topsoil and vice-versa
                            D_djk,D_skj,Zwk = 'D_d'+str(j)+str(k),'D_s'+str(k)+str(j),\
                            'Zw_'+str(k)                          
                            res.loc[mask,D_djk] = 1/(res.Y_subsoil/(res.AsoilV*res.Deff_water*res.Zw_water)\
                                   +res.Y_topsoil/(res.AsoilV*res.Deff_topsoil*res.Zw_topsoil)) #Diffusion - both ways
                            res.loc[mask,D_skj] = params.val.U42*res.AsoilV*res.Zq_topsoil #Particle settling - only from top to subsoil
                            if j == 'subsoil':
                                res.loc[mask,D_jk] = res.loc[mask,D_djk] #sub- to topsoil
                                res.loc[mask,D_kj] = res.loc[mask,D_djk] + res.loc[mask,D_skj] #top- to subsoil  
                            else: #k = topsoil
                                res.loc[mask,D_jk] = res.loc[mask,D_djk] + res.loc[mask,D_skj] #top- to subsoil 
                                res.loc[mask,D_kj] = res.loc[mask,D_djk]  #sub- to topsoil
                        elif k in ['rootbody','rootxylem','rootcyl']:
                            D_rdkj,Vk,Zk= 'D_rd'+str(k)+str(j),'V'+str(k),'Z'+str(kind)
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
                                res.loc[res.chemcharge == 0,Dsr_ij], res.loc[res.chemcharge == 0,Drs_ij] = 0,0 #Set neutral to zero
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
                        elif k in ['shoots']:
                            if 'topsoil' in numc and j =='subsoil':
                                res.loc[mask,D_jk] = 0
                                res.loc[mask,D_kj] = 0
                            else:
                                #Canopy drip 
                                if locsumm.loc['shoots','Discrete'] == 0:
                                    masksh = res.dm==False
                                else:
                                    masksh = res.dm
                                res.loc[masksh,'D_cd'] = res.A_shootair * rainrate*(params.val.Ifw - params.val.Ilw)*params.val.lamb * res.Zshoots  
                                #Wax erosion
                                res.loc[masksh,'D_we'] = res.A_shootair * params.val.kwe * res.Zshoots   
                                #litterfall & plant death?
                                res.loc[masksh,'D_lf'] = res.Vshoots * res.Zshoots   * params.val.Rlf    
                                #Overall D Values
                                res.loc[mask,D_jk] = 0
                                res.loc[masksh,D_kj] = res.D_cd + res.D_we + res.D_lf
                        elif k in ['air']:
                            if 'topsoil' in numc and j =='subsoil':
                                res.loc[mask,D_jk] = 0
                                res.loc[mask,D_kj] = 0
                            else:
                                if ('topsoil' not in numc) and (j == 'subsoil'):
                                    y = np.array(res.iloc[0].x/2) #Half the top cell.
                                    Bea = np.array(res.loc[(res.x == min(res.x)),'Bea_subsoil'])
                                    #For the BC, only the top of the soil will interact with the air
                                    masks = (res.x == min(res.x)) 
                                    maska = res.dm==False
                                else:
                                    y = res.Y_topsoil
                                    Bea= res.Bea_topsoil
                                    #***CHECK BEFORE USE***
                                    masks = res.dm
                                    maska = res.dm
                                Zw_j,D_djk,Deff_j = 'Zw_'+str(j),'D_d'+str(j)+str(k),'Deff_'+str(j)
                                Zq_k,Zw_k = 'Zq_'+str(k),'Zw_'+str(k)
                                #Getting the values in the correct cells took some creative indexing here, making this formula complicated.
                                res.loc[masks,D_djk] = 1/(1/(params.val.ksa*res[masks].Asoilair*np.array(res.Zair[maska]))\
                                       +y/(res[masks].Asoilair*Bea*np.array(res.Zair[maska])+\
                                    res[masks].Asoilair*np.array(res.loc[masks,Deff_j])*np.array(res.loc[masks,Zw_j]))) #Dry diffusion
                                res.loc[maska,D_djk] = np.array(res.loc[masks,D_djk])
        
                                #From air to top cell of soil. 
                                #res.loc[maska,'D_wdairsoil'] = res[maska].Asoilair*np.array(res.loc[maska,Zw_k])*rainrate.reindex(res.index,level=1).loc[:,0]*(1-params.val.Ifw) #Wet gas deposion
                                res.loc[maska,'D_wdairsoil'] = res[maska].Asoilair*np.array(res.loc[maska,Zw_k])*rainrate*(1-params.val.Ifw) #Wet gas deposion
                                res.loc[masks,'D_wdairsoil'] = 0
                                res.loc[maska,'D_qairsoil'] = res[res.dm==False].Asoilair * res.loc[maska,Zq_k] \
                                    *rainrate*res[maska].fpartair*params.val.Q*(1-params.val.Ifw)  #Wet dep of aerosol
                                res.loc[masks,'D_qairsoil'] = 0
                                res.loc[maska,'D_dairsoil'] = res[maska].Asoilair * res.loc[maska,Zq_k]\
                                    *  params.val.Up * res[maska].fpartair* (1-Ifd) #dry dep of aerosol
                                res.loc[masks,'D_dairsoil'] = 0
                                #Overall D values
                                #pdb.set_trace()
                                #Soil to air - only diffusion
                                res.loc[masks,D_jk] = res.loc[masks,D_djk] 
                                #Air to soil - diffusion, wet & dry gas and particle deposition.
                                res.loc[maska,D_kj] = res.loc[maska,D_djk]+res[maska].D_wdairsoil+res[maska].D_qairsoil+res[maska].D_dairsoil
                        #Soil/Pond. We are going to treat the advective portion explicitly, then the rest will be implicit.
                        elif k in ['pond']:
                            Zq_j,Zq_k,Zw_j,Zw_k = 'Zq_'+str(j),'Zq_'+str(k),'Zw_'+str(j),'Zw_'+str(k),
                            mask0 = (res.x == min(res.x)) 
                            #Define pond/water area - for now, same as pond/air
                            y = np.array(res.iloc[0].x/2) #Half the top cell.
                            pondD = (pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Depth']),index = timeseries.index.levels[0])/2).reindex(res.index,level=1).loc[:,0]
                            Apondsoil = (pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Area']),index = timeseries.index.levels[0])).reindex(res.index,level=1).loc[:,0]
                            Apondsoil[pondD==0] = 0 #The pond area is based on a minimum surveyed value of 5.74m², if there is no depth there is no area.
                            #Particle capture from pond to first cell. This will be advective & explicit, assume 100% capture for now.
                            res.loc[res.dm==False,'D_qps'] = np.array(res.loc[(slice(None),slice(None),0),'Qin'])*res.loc[res.dm==False,Zq_k]
                            #Diffusive transfer
                            res.loc[res.dm==False,'D_dsoilpond'] = 1/(1/(params.val.kxw*Apondsoil[res.dm==False]*res.loc[res.dm==False,Zw_k])\
                                   +(y)/(Apondsoil[res.dm==False]*np.array(res[mask0].Deff_subsoil)*np.array(res.loc[mask0,Zw_j])))
                            res.loc[mask0,'D_dsoilpond'] = np.array(res.loc[res.dm==False,'D_dsoilpond']) #This goes both ways, need in both cells.
                            res.loc[np.isnan(res.D_dsoilpond),'D_dsoilpond'] = 0 #Set nans to zero just in case
                            #Particle Resuspension
                            res.loc[mask0,'D_r94'] = params.val.Urx*Apondsoil*res.loc[:,Zq_j]
                            #Overall D Values - for soil to pond we have diffusion and resuspension
                            res.loc[mask0,D_jk] = res.D_dsoilpond + res.D_r94
                            #For pond to soil we will not put the particle transport in as it is calculated explicitly. So, only diffusion.
                            res.loc[res.dm==False,D_kj] = res.D_dsoilpond
                        else: #Other compartments Djk  = 0
                            res.loc[mask,D_jk] = 0
                    #Add D_jk to DTj for each compartment.
                    #Set nans to zero to prevent problems.
                    res.loc[np.isnan(res.loc[:,D_jk]),D_jk] = 0
                    res.loc[:,Dtj] += res.loc[:,D_jk]
                    
            #Roots interact with subsoil, root body, root central cylinder, xylem, shoots
            elif j in ['rootbody','rootxylem','rootcyl']: 
                for kind, k in enumerate(numc): #Inner loop for inter-compartmental values
                    kind = kind+1 #The compartment number, for overall & intercompartmental D values
                    D_jk,D_kj = 'D_'+str(jind)+str(kind),'D_'+str(kind)+str(jind)
                    try: #First, check if intercompartmental transport between jk already done
                        res.loc[:,D_jk]
                    except KeyError:
                        if k == j: #Processes that come directly out of the compartment (not to another compartment) go here            
                            Vj,Zj,D_gj = 'V'+str(j),'Z'+str(jind),"D_g"+str(j)
                            res.loc[mask,D_gj] = params.val.k_rg*res.loc[mask,Vj]*res.loc[mask,Zj] #Root growth as first order
                            res.loc[mask,D_jk] = res.loc[mask,D_gj]
                        #Root body interacts with xylem.
                        elif (j in ['rootbody']) & (k in ['rootxylem']): 
                            Nj,Nk,Qetj,tempj,tempk = 'N'+str(j),'N'+str(k),'Qet'+str(j),'temp'+str(j),'temp'+str(k)
                            Zw_j,Zwn_j,Zwi_j,Zwn_k,Zwi_k = 'Zw_'+str(j),'Zwn_'+str(j),'Zwi_'+str(j),'Zwn_'+str(k),'Zwi_'+str(k),
                            #These should have been done above but just in case.
                            res.loc[mask,Nj] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.loc[:,tempj])
                            res.loc[mask,Nk] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.loc[mask,tempk])
                            #Root body to xylem - interfacial area is Arootxylem
                            res.loc[mask,'Drx_n'] = res.Arootxylem*(res.kspn*res.fwatrootbody*res.loc[:,Zwn_j]) #A7 is root/xylem interface
                            res.loc[mask,'Drx_i'] = res.Arootxylem*(res.kspi*res.fwatrootbody*res.loc[:,Zwi_j]\
                                   *res.loc[mask,Nj]/(np.exp(res.loc[mask,Nj]) - 1))
                            #xylem to root body - interfacial area is Arootxylem
                            res.loc[mask,'Dxr_n'] = res.Arootxylem*(res.kspn*res.fwatrootbody*res.loc[:,Zwn_k]) #A7 is root/xylem interface
                            res.loc[mask,'Dxr_i'] = res.Arootxylem*(res.kspi*res.fwatrootbody*res.loc[:,Zwi_k]\
                                   *res.loc[mask,Nk]/(np.exp(res.loc[mask,Nk]) - 1))
                            #Set neutral to zero
                            res.loc[res.chemcharge == 0,'Drx_i'], res.loc[res.chemcharge == 0,'Dxr_i'] = 0,0 #Set neutral to zero
                            #Overall D values - note that there isn't any ET flux, all transport across the membrane is diffusive
                            res.loc[mask,D_jk] = res.loc[:,'Drx_n']+res.loc[:,'Drx_i'] 
                            res.loc[mask,D_kj] = res.loc[:,'Dxr_n']+res.loc[:,'Dxr_i']       
                        elif (j in ['rootxylem']) & (k in ['rootcyl']):
                            #Xylem goes to central cylinder, transport to xylem is accounted for above. Only ET flow (no more membranes!)
                            Zw_j = 'Zw_'+str(j)
                            res.loc[mask,'D_xc'] = res.Qet*res.loc[:,Zw_j]
                            #Overall D values - only one way.
                            res.loc[mask,D_jk] = res.loc[:,'D_xc']
                            res.loc[mask,D_kj] = 0
                        elif (j in ['rootcyl']) & (k in ['shoots']):
                            #Cylinder goes to shoots
                            Zw_j = 'Zw_'+str(j)
                            #pdb.set_trace()
                            res.loc[mask,'D_csh'] = res.Qetplant*res.loc[:,Zw_j]
                            #If the system flows vertically - flux goes up discretized units
                            if params.val.vert_flow == 1:
                                #Flux up the central cylinder. Stays in same compartment but goes vertically up discretized units.
                                res.loc[(mask) & (res.x != min(res.x)),'D_'+str(jind)+str(jind)] += res.loc[mask,'D_csh']
                                res.loc[(mask) & (res.x != min(res.x)),Dtj] += res.loc[mask,'D_csh'] 
                                #Overall D values - only one way, only from top cell.
                                res.loc[(res.x == min(res.x)) ,D_jk] = res.loc[:,'D_csh']
                                res.loc[mask,D_kj] = 0               
                            else:
                                res.loc[mask,D_jk] = res.loc[:,'D_csh'] #From central cylinder to shoots
                                res.loc[mask,D_kj] = 0
                                          
                        else: #Other compartments Djk = Dkj = 0
                            res.loc[mask,D_jk] = 0
                    #Add D_jk to DTj for each compartment.
                    #Set nans to zero to prevent problems.
                    res.loc[np.isnan(res.loc[:,D_jk]),D_jk] = 0
                    res.loc[:,Dtj] += res.loc[:,D_jk]                            
                            
            elif j in ['shoots']: #Shoots interact with air, central cylinder & soil. Only air still needs to be done here.
                for kind, k in enumerate(numc): #Inner loop for inter-compartmental values
                    kind = kind+1 #The compartment number, for overall & intercompartmental D values
                    D_jk,D_kj = 'D_'+str(jind)+str(kind),'D_'+str(kind)+str(jind)
                    try: #First, check if intercompartmental transport between jk already done
                        res.loc[:,D_jk]
                    except KeyError:
                        if k == j: #Processes that come directly out of the compartment (not to another compartment) go here            
                            #Shoot growth - Modelled as first order decay
                            Vj,Zj,D_gj = 'V'+str(j),'Z'+str(jind),"D_g"+str(j)
                            #pdb.set_trace()
                            res.loc[mask,D_gj] = params.val.k_sg*res.loc[mask,Vj]*res.loc[mask,Zj] #shoot growth as first order
                            res.loc[mask,D_jk] = res.loc[mask,D_gj]
                        elif k in ['air']:
                            Zn_j,Zw_k,Zq_k = 'Zn_'+str(j),'Zw_'+str(k),'Zq_'+str(k)
                            #Volatilization to air, only neutral species. Ashoots is the interfacial area
                            res.loc[mask,'D_dshootsair'] = res.kvv*res.A_shootair*res.loc[:,Zn_j]
                            #res.loc[mask,'D_rv'] = res.A_shootair * res.loc[:,Zw_k]*rainrate.reindex(res.index,level=1).loc[:,0]* params.val.Ifw  #Wet dep of gas to shoots
                            res.loc[mask,'D_rv'] = res.A_shootair * res.loc[:,Zw_k]*rainrate* params.val.Ifw  #Wet dep of gas to shoots
                            res.loc[mask,'D_qv'] = res.A_shootair * res.loc[:,Zq_k]*rainrate\
                                * params.val.Q * params.val.Ifw #Wet dep of aerosol
                            res.loc[mask,'D_dv'] = res.A_shootair * res.loc[:,Zq_k] * params.val.Up *Ifd  #dry dep of aerosol
                            #Overall D values- only diffusion from shoots to air
                            res.loc[mask,D_jk] = res.loc[:,'D_dshootsair']
                            res.loc[mask,D_kj] = res.loc[:,'D_dshootsair'] +res.loc[:,'D_rv']+res.loc[:,'D_qv']+res.loc[:,'D_dv']       
                        else: #Other compartments Djk = Dkj = 0
                            res.loc[mask,D_jk] = 0
                    #Add D_jk to DTj for each compartment.
                    #Set nans to zero to prevent problems.
                    res.loc[np.isnan(res.loc[:,D_jk]),D_jk] = 0
                    res.loc[:,Dtj] += res.loc[:,D_jk] 

            elif j in ['air']: #Air interacts with shoots, soil, pond. Only need to do pond here. 
                for kind, k in enumerate(numc): #Inner loop for inter-compartmental values
                    kind = kind+1 #The compartment number, for overall & intercompartmental D values
                    D_jk,D_kj = 'D_'+str(jind)+str(kind),'D_'+str(kind)+str(jind)
                    try: #First, check if intercompartmental transport between jk already done
                        res.loc[:,D_jk]
                    except KeyError:
                        if k == j: #Processes that come directly out of the compartment (not to another compartment) go here            
                            #Shoot growth - Modelled as first order decay
                            Vj,Zj = 'V'+str(j),'Z'+str(jind)
                            res.loc[mask,D_jk] = 0 #No additional loss processes from air.
                        elif k in ['pond']: #Volatilization from pond to air, then normal processes from air to pond
                            Zn_j,Zw_j,Zq_j,Zw_k = 'Zn_'+str(j),'Zw_'+str(j),'Zq_'+str(j),'Zw_'+str(k)
                            y = (pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Depth']),index = timeseries.index.levels[0])/2).reindex(res.index,level=1).loc[:,0]
                            Apondair = (pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Area']),index = timeseries.index.levels[0])).reindex(res.index,level=1).loc[:,0]
                            Apondair[y==0] = 0 #The pond area is based on a minimum surveyed value of 5.74m², if there is no depth there is no area.
                            #Volatilization to air, only neutral species. Ashoots is the interfacial area
                            res.loc[mask,'D_dairpond'] = 1/(1/(params.val.kma*Apondair[mask]*res.Zair[mask])\
                                   +y/(Apondair[mask]*res.loc[mask,'D_air']*res[mask].Zair+\
                                Apondair[mask]*res.loc[mask,'Deff_pond']*res.loc[mask,Zw_k])) #Dry diffusion
                            res.loc[np.isnan(res.D_dairpond),'D_dairpond'] = 0 #Set NaNs to zero
                            try: #Code problem - need different indexing for subsurface sinks vs BC blues.
                                #pdb.set_trace()
                                res.loc[mask,'D_rp'] = Apondair * res.loc[:,Zw_j]*rainrate* params.val.Ifw #rainrate.reindex(res.index,level=1).loc[:,0]\
                                    #Wet dep of gas to pond
                                res.loc[mask,'D_qp'] = Apondair * res.loc[:,Zq_j]*rainrate* params.val.Q  #.reindex(res.index,level=1).loc[:,0]\
                                     #Wet dep of aerosol
                            except KeyError:
                                res.loc[mask,'D_rp'] = Apondair * res.loc[:,Zw_j]*rainrate.reindex(res.index,level=1)\
                                    * params.val.Ifw  #Wet dep of gas to pond
                                res.loc[mask,'D_qp'] = Apondair * res.loc[:,Zq_j]*rainrate.reindex(res.index,level=1)\
                                    * params.val.Q * params.val.Ifw #Wet dep of aerosol
                            res.loc[mask,'D_dp'] = Apondair * res.loc[:,Zq_j] * params.val.Up *Ifd  #dry dep of aerosol
                            #Overall D values- only diffusion from shoots to air
                            res.loc[mask,D_jk] = res.loc[:,'D_dairpond']
                            res.loc[mask,D_kj] = res.loc[:,'D_dairpond'] +res.loc[:,'D_rp']+res.loc[:,'D_qp']+res.loc[:,'D_dp']                            
                        else: #Other compartments Djk = 0.
                            res.loc[mask,D_jk] = 0
                    #Add D_jk to DTj for each compartment.
                    #Set nans to zero to prevent problems.
                    res.loc[np.isnan(res.loc[:,D_jk]),D_jk] = 0
                    res.loc[:,Dtj] += res.loc[:,D_jk]                             
                            
            elif j in ['pond']: #Pond interacts with soil, air and water - sometimes through advection. All done previously 
                for kind, k in enumerate(numc): #Inner loop for inter-compartmental values
                    kind = kind+1 #The compartment number, for overall & intercompartmental D values
                    D_jk,D_kj = 'D_'+str(jind)+str(kind),'D_'+str(kind)+str(jind)
                    try: #First, check if intercompartmental transport between jk already done
                        res.loc[:,D_jk]
                    except KeyError:
                        if k == j: #Processes that come directly out of the compartment (not to another compartment) go here            
                            #Exfiltration from pond out of cell & weir overflow
                            Qpondexf = pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Q_exf']),index = timeseries.index.levels[0]).reindex(res.index,level=1).loc[:,0]
                            res.loc[mask,'D_pondexf'] = Qpondexf*res.loc[mask,'Zw_pond']
                            #Weir overflow
                            Qpondover = pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Q_out']),index = timeseries.index.levels[0]).reindex(res.index,level=1).loc[:,0]
                            res.loc[mask,'D_pondover'] = Qpondover*res.loc[mask,'Zpond']
                            res.loc[mask,D_jk] = res.loc[:,'D_pondexf']+res.loc[:,'D_pondover']
                        else: #Other compartments Djk = 0.
                            res.loc[mask,D_jk] = 0
                            
                    #Add D_jk to DTj for each compartment.
                    #Set nans to zero to prevent problems.
                    res.loc[np.isnan(res.loc[:,D_jk]),D_jk] = 0
                    res.loc[:,Dtj] += res.loc[:,D_jk]    
        #'''
                
        """
        Old code before it was cool & modular
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
        
        #6 Root Body - subsoil, xylem
        e fraction of root without secondary epothileum
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
            #Top boundary condition of the mobile zone - Mass flux from ponding zone
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
        """
        return res


    def run_it(self,locsumm,chemsumm,params,pp,numc,timeseries,input_calcs = None,last_step = None):
        """Feed the calculated values into the ADRE equation over time.
        timeseries is the timeseries data of temperatures, rainfall, influent 
        concentrations, influent volumes, redox conditions? What else?
        
        input_calcs(df): 3 options, for 1. and 2. input_calc will be run:
            1. Nothing (None)
            2. Contains the physical parameters of the system for each compartment,
        as in the "BCBlues" submodule. No chemical data, run with input_calc as timeseries
            3. Contains a full "input_calc" output dataframe, with a "time" column,
        and everything else. Index level 0 = chems, level 1 = time, level 2 = cell number 
        """
        #pdb.set_trace()
        try: #See if there is a compartment index in the timeseries
            input_calcs.index.levels[2]
        except AttributeError: #Runs the full flow_time calcs
            input_calcs = self.input_calc(locsumm,chemsumm,params,pp,numc,timeseries)
        except IndexError:  #Runs just the input_calcs part
            input_calcs = self.input_calc(locsumm,chemsumm,params,pp,numc,input_calcs)
        #Now, this will be our outputs dataframe
        #res = input_calcs.copy(deep=True)
        res = input_calcs
        #If we want to run the code in segments, we just need one previous timestep. 
        #Now, we can add the influent concentrations as the upstream boundary condition
        #Assuming chemical concentration in g/m³ activity [mol/m³] = C/Z/molar mass,
        #using Z1 in the first cell (x) (0)
        #pdb.set_trace()
        if params.val.Pulse == True:
            for chem in chemsumm.index:
                chem_Min = str(chem) + '_Min'
                res.loc[(chem,slice(None),0),'Min'] = \
                timeseries.loc[:,chem_Min].reindex(input_calcs.index, level = 1)/chemsumm.MolMass[chem] #mol
                res.loc[(chem,slice(None),slice(None)),'bc_us'] = 0
        else:
            #Initialize mass in as zero - will be calcualated from upstream BC
            res.loc[:,'Min'] = 0
            for chem in chemsumm.index:
                chem_Cin = str(chem) + '_Cin'
                res.loc[(chem,slice(None),slice(None)),'Cin'] = \
                timeseries.loc[:,chem_Cin].reindex(input_calcs.index, level = 1)
                res.loc[(chem,slice(None),slice(None)),'bc_us'] = res.loc[(chem,slice(None),slice(None)),'Cin']/\
                    chemsumm.MolMass[chem]/res.loc[(chem,slice(None),0),'Z1']
        
        #ntimes = len(timeseries.index)
        #Update params on the outside of the timeloop
        if params.val.vert_flow == 1:
            params.loc['L','val'] = locsumm.Depth.subsoil + locsumm.Depth.topsoil
        else:
            params.loc['L','val'] = locsumm.Length.water
        #check = np.zeros([len(timeseries.index),len(chemsumm.index)])
        #Initialize total mass and minp, minq
        res.loc[:,'M_tot'] = 0
        
        #Give number of discretized compartments to params
        params.loc['numc_disc','val'] = locsumm.loc[:,'Discrete'].sum()
        #lexsort outside the timeloop for better performance
        res = res.sort_index()
        for t in res.index.levels[1]: #Just in case index doesn't start at zero
            #Set inlet and outlet flow rates and velocities for each time
            try:
                params.val.Qout = timeseries.Qout[t]
                params.val.Qin = timeseries.Qin[t]
            except AttributeError: #Qout is calculated not given
                params.loc['Qin','val'] = res.loc[(res.index.levels[0][0],t,0),'Qwater'] #Qin to top of the water compartment
                params.loc['Qout','val'] = res.loc[(res.index.levels[0][0],t,0),'Qout'] #Qout from water compartment
            params.val.vin = params.val.Qin/(res.Awater[0])
            params.val.vout = params.val.Qout/(res.Awater[res.dm][-1])

            #Initial conditions for each compartment
            if t == res.index.levels[1][0]: #Set initial conditions here. 
                #initial Conditions
                for j in range(0,len(numc)):
                    a_val = 'a'+str(j+1) + '_t'
                    try: #If there is a last step, use that as the a_t values
                        #res.loc[(slice(None),t,slice(None)),a_val] = last_step.loc[(slice(None),t,slice(None)),a_val]
                        res.loc[(slice(None),t,slice(None)),a_val] = last_step.iloc[:,j].reindex(res.index,level=0)
                    except AttributeError:
                        res.loc[(slice(None),t,slice(None)),a_val] = 0 #1#Can make different for the different compartments
                dt = timeseries.time[1]-timeseries.time[0]
 
                
            else: #Set the previous solution aj_t1 to the inital condition (aj_t)
                for j in range(0,len(numc)):
                    a_val, a_valt1 = 'a'+str(j+1) + '_t', 'a'+str(j+1) + '_t1'
                    M_val, V_val, Z_val = 'M'+str(j+1) + '_t1', 'V' + str(j+1), 'Z' + str(j+1)
                    #Define a_t so as to be mass conservative - since system is explicit for volumes etc. this can cause mass loss
                    res.loc[(slice(None),t,slice(None)),a_val] = np.array(res.loc[(slice(None),(t-1),slice(None)),M_val])\
                    /np.array(res.loc[(slice(None),(t),slice(None)),V_val])/np.array(res.loc[(slice(None),(t),slice(None)),Z_val])
                    #For the pond compartment, need to add Qin - advective step at the beginning. Otherwsie when pond dries up mass disappears.
                    if numc[j] in ['pond']:
                        res.loc[(slice(None),t,slice(None)),a_val] = np.array(res.loc[(slice(None),(t-1),slice(None)),M_val])\
                        /np.array(res.loc[(slice(None),(t),slice(None)),V_val]+dt*res.loc[(slice(None),(t),slice(max(res.index.levels[2]))),'Qin'])\
                        /np.array(res.loc[(slice(None),(t),slice(None)),Z_val])
                    #Set nans to zero - this will happen to compartments with zero volume such as the roots in the drain cell and the pond
                    res.loc[np.isnan(res.loc[:,a_val]),a_val] = 0
                dt = timeseries.time[t] - timeseries.time[t-1] #timestep can vary
            #Now - run it forwards a time step!
            #Feed the time to params
            res_t = res.loc[(slice(None),t,slice(None)),:]
            if t == 216:#260: #216:#412: #630# 260 is location of mass influx from tracer test; stop at spot for error checking
                #pdb.set_trace()
                xxx = 'why'
                yy = 'seriously stop'
            res_t = self.ADRE_1DUSS(res_t,params,numc,dt)
            for j in range(0,len(numc)): #Put sthe results - a value at the next time step and input mass - in the dataframe
                a_valt1,M_val = 'a'+str(j+1) + '_t1','M'+str(j+1) + '_t1'
                res.loc[(slice(None),t,slice(None)),a_valt1] = res_t.loc[(slice(None),t,slice(None)),a_valt1]
                res.loc[(slice(None),t,slice(None)),M_val] = res_t.loc[(slice(None),t,slice(None)),M_val]
                res.loc[(slice(None),t,slice(None)),'M_tot'] += res.loc[(slice(None),t,slice(None)),M_val]
            #Also put the water and soil inputs in.
            if params.val.vert_flow == 1:
                res.loc[(slice(None),t,slice(None)),'Mqin'] = res_t.loc[(slice(None),t,slice(None)),'Mqin']
                res.loc[(slice(None),t,slice(None)),'Min_p'] = res_t.loc[(slice(None),t,slice(None)),'Min_p']
            else: #Set upstream mass in here
                res.loc[(slice(None),t,slice(None)),'Min'] = res_t.loc[(slice(None),t,slice(None)),'Min']
                res.loc[(slice(None),t,slice(None)),'Mqin'] = 0
                res.loc[(slice(None),t,slice(None)),'Min_p'] = res.loc[(slice(None),t,slice(None)),'Min']
            res.loc[(slice(None),t,slice(None)),'M_xf'] = res_t.loc[(slice(None),t,slice(None)),'M_xf']
            res.loc[(slice(None),t,slice(None)),'M_n'] = res_t.loc[(slice(None),t,slice(None)),'M_n']
            #mass = res.loc[(slice(None),t,slice(None)),'a1_t1']*res.loc[(slice(None),t,slice(None)),'Z1']\
            #*res.loc[(slice(None),t,slice(None)),'V1'] + res.loc[(slice(None),t,slice(None)),'a2_t1']\
            #*res.loc[(slice(None),t,slice(None)),'Z2']*res.loc[(slice(None),t,slice(None)),'V2']
            
            #if t >= 260:#260:#260 is injection
            #    pass
                #check[t] = np.array(res.loc[(slice(None),slice(None),0),'Min'].groupby(level=0).sum()) - np.array(res.M_tot.groupby(level = 0).sum())
                #check[t] = np.array(res.loc[(slice(None),260,0),'Min'])\
                #- np.array(mass.groupby(level = 0).sum())
            #else:
            #    check[t] = np.array(res.loc[(slice(None),t,0),'Min']) - np.array(res.M_tot.groupby(level = 0).sum())
                    #res.loc[(slice(None),t,slice(None)),inp_mass] = res_t.loc[(slice(None),t,slice(None)),inp_mass]

            #res.loc[(slice(None),t,slice(None)),'M_tot'] = mass
        return res
    
    def mass_flux(self,res_time,numc):
        """ This function calculates mass fluxes (g/h) between compartments and
        out of the overall system. Calculations are done at the same discretization 
        level as the system, to get the overall mass fluxes for a compartment use 
        mass_flux.loc[:,'Variable'].groupby(level=[0,1]).sum() (result is in mol/h, multiply by dt for total mass)
        """
        #pdb.set_trace()
        #First determine the number
        numx = res_time.loc[(res_time.index.levels[0][0],res_time.index.levels[1][0],slice(None)),'dm'].sum()
        res_time.loc[:,'dt'] =  res_time['time'] - res_time['time'].groupby(level=2).shift(1)
        res_time.loc[(slice(None),min(res_time.index.levels[1]),slice(None)),'dt'] = \
            np.array(res_time.loc[(slice(None),min(res_time.index.levels[1])+1,slice(None)),'dt'])
        #Make a dataframe to display mass flux on figure
        mass_flux = pd.DataFrame(index = res_time.index)
        mass_flux.loc[:,'dt'] = np.array(res_time.loc[:,'dt'])
        #pdb.set_trace()

        #First, we will add the advective transport out and in to the first and last
        #cell of each compound/time, respectively
        #N is mass flux, mol/hr
        #Pipe flow out the back end. 
        #N_effluent = (res_time.M_n[slice(None),slice(None),numx-1] - res_time.M_xf[slice(None),slice(None),numx-1])/dt
        #N_effluent = np.array(res_time.a1_t[slice(None),slice(None),numx-1]*res_time.Z1[slice(None),slice(None),numx-1]\
        #                                    *res_time.Qout[slice(None),slice(None),numx-1])
        N_effluent = np.array(res_time.a1_t[slice(None),slice(None),numx-1]*res_time.Z1[slice(None),slice(None),numx-1]\
                                            *res_time.Qout[slice(None),slice(None),numx-1])
        #N_fd = (res_time.M_n[slice(None),slice(None),numx-2] - res_time.M_xf[slice(None),slice(None),numx-2])/dt
        mass_flux.loc[:,'N_effluent'] = 0
        mass_flux.loc[(slice(None),slice(None),numx-1),'N_effluent'] = N_effluent
        mass_flux.loc[:,'N_influent'] = res_time.Min/res_time.dt #This assumes inputs are zero
        #Now, lets get to compartment-specific transport
        for jind, j in enumerate(numc):#j is compartment mass is leaving
            jind = jind+1 #The compartment number, for overall & intercompartmental D values
            Drj,Nrj,a_val, NTj, DTj= 'Dr' + str(j),'Nr' + str(j),'a'+str(jind) + '_t1','NT' + str(j),'DT' + str(jind)
            Nadvj,Dadvj = 'Nadv' + str(j),'Dadv' + str(j)
            #Transformation (reaction) in each compartment Mr = Dr*a*V
            mass_flux.loc[:,Nrj] = (res_time.loc[:,Drj] * res_time.loc[:,a_val])#Reactive mass loss
            mass_flux.loc[:,NTj] = (res_time.loc[:,DTj] * res_time.loc[:,a_val])#Total mass out
            mass_flux.loc[:,Nadvj] = (res_time.loc[:,Dadvj] * res_time.loc[:,a_val])#Advective mass out.
            if j == 'water': #Water compartment, exfiltration losses
                mass_flux.loc[:,'N_exf'] = (res_time.loc[:,'D_waterexf'] * res_time.loc[:,a_val])#Exfiltration mass loss
            elif j in ['rootbody','rootxylem','rootcyl','shoots']:
                #Growth dilution processes - in DT but nowhere else.
                Nrg_j,D_rgj = "Ng_"+str(j),"D_g"+str(j)
                mass_flux.loc[:,Nrg_j] = (res_time.loc[:,D_rgj] * res_time.loc[:,a_val])            
            for kind, k in enumerate(numc):#From compartment j to compartment k
                if j != k:
                    kind = kind+1
                    Djk,Dkj,Njk,Nkj,Nnet_jk,ak_val = 'D_'+str(jind)+str(kind),'D_'+str(kind)+str(jind),\
                    'N' +str(j)+str(k),'N' +str(k)+str(j),'Nnet_' +str(j)+str(k),'a'+str(kind) + '_t1'
                    mass_flux.loc[:,Njk] = (res_time.loc[:,Djk] * res_time.loc[:,a_val])
                    mass_flux.loc[:,Nkj] = (res_time.loc[:,Dkj] * res_time.loc[:,ak_val])
                    mass_flux.loc[:,Nnet_jk]  = mass_flux.loc[:,Njk] - mass_flux.loc[:,Nkj]
                    
        return mass_flux
    
    def mass_balance(self,res_time,numc,mass_flux = None,normalized = False):
        """ This function calculates a mass balance and the mass transfers (g) between compartments 
        on a whole-compartment basis.
        Attributes:
            res_time (dataframe) - output from self.run_it
            numc (list) - Compartments in the system.
            mass_flux (dataframe, optional) - output from self.mass_flux. Will be calculated from res_time if absent.
            normalized (bool, optional) = Normalize the mass transfers to the total mass that has entered the system.
                Note that this will only normalize certain outputs
        """        
        #pdb.set_trace()
        #For testing
        #Min = res_time.Min.groupby(level=0).sum()
        #stop = 'stop'
        try:
            mass_flux.loc[:,'dt']
        except AttributeError:
            mass_flux = self.mass_flux(res_time,numc)
        #Mass balance at teach time step.
        mbal = pd.DataFrame(index = (mass_flux.N_effluent).groupby(level=[0,1]).sum().index)
        #First, add the things that are always going to be there.
        mbal.loc[:,'time'] = np.array(res_time.loc[(slice(None),slice(None),slice(0)),'time'])
        mbal.loc[:,'Min'] = res_time.Min.groupby(level=[0,1]).sum().groupby(level=0).cumsum()
        mbal.loc[:,'Mtot'] = res_time.M_tot.groupby(level=[0,1]).sum()
        if normalized == True:
            divisor = mbal.Min+mbal.loc[(slice(None),mbal.index.levels[1][0]),'Mtot'].reindex(mbal.index,method ='ffill')
        else:
            divisor = 1
        mbal.loc[:,'Meff'] = (mass_flux.dt*mass_flux.N_effluent).groupby(level=[0,1]).sum()/divisor
        mbal.loc[:,'Mexf'] = (mass_flux.dt*mass_flux.N_exf).groupby(level=[0,1]).sum()/divisor
        mbal.loc[:,'Mout'] = ((mass_flux.dt*mass_flux.N_effluent).groupby(level=[0,1]).sum() +\
                             (mass_flux.dt*mass_flux.N_exf).groupby(level=[0,1]).sum()).groupby(level=0).cumsum()
        #(mbal.loc[:,'Meff'].groupby(level=0).cumsum()+mbal.loc[:,'Mexf'].groupby(level=0).cumsum())
        
        #mbal.loc[:,'Mbal2'] = 0.0
        for jind, j in enumerate(numc):#j is compartment mass is leaving
            jind = jind+1 #The compartment number, for overall & intercompartmental D values
            Nrj,Nadvj,Moutj,Mbalj,Mjind,Mj,Minj,Madvj,Mrj= 'Nr' + str(j),'Nadv' + str(j),'Mout' + str(j),'Mbal' + str(j),\
                'M'+str(jind)+'_t1','M'+str(j),'Min'+str(j),'Madv'+str(j),'Mr'+str(j)
            #Moutj here gives the mass out of the entire system.
            mbal.loc[:,Mrj] = (mass_flux.dt*mass_flux.loc[:,Nrj]).groupby(level=[0,1]).sum()/divisor
            mbal.loc[:,Madvj] = (mass_flux.dt*mass_flux.loc[:,Nadvj]).groupby(level=[0,1]).sum()/divisor                                 
            mbal.loc[:,Moutj] = (mbal.loc[:,Mrj]+mbal.loc[:,Madvj])
            #Total mass out of system - cumulative
            mbal.loc[:,'Mout'] += ((mass_flux.dt*mass_flux.loc[:,Nrj]).groupby(level=[0,1]).sum() +\
                                  (mass_flux.dt*mass_flux.loc[:,Nadvj]).groupby(level=[0,1]).sum()).groupby(level=0).cumsum()
            #(mbal.loc[:,Mrj].groupby(level=0).cumsum()+mbal.loc[:,Madvj].groupby(level=0).cumsum())
            #initialize
            mbal.loc[:,Minj] = 0
            for kind, k in enumerate(numc):
                #We will define net transfer as Mjk - Mkj (positive indicates positive net transfer from j to k)
                if ('Mnet'+str(k)+str(j)) in mbal.columns:
                    pass 
                elif k != j:
                    Njk,Nkj,Mjk,Mkj,Mnetjk = 'N' +str(j)+str(k),'N' +str(k)+str(j),'M'+str(j)+str(k),'M'+str(k)+str(j),'Mnet'+str(j)+str(k)
                    #Mass into each compartment will be recorded as that compartment's Mjk
                    mbal.loc[:,Mkj] = (mass_flux.dt*mass_flux.loc[:,Nkj]).groupby(level=[0,1]).sum()/divisor
                    mbal.loc[:,Minj] += mbal.loc[:,Mkj]
                    #Mass out per time step. 
                    mbal.loc[:,Mjk] = (mass_flux.dt*mass_flux.loc[:,Njk]).groupby(level=[0,1]).sum()/divisor              
                    mbal.loc[:,Moutj] += mbal.loc[:,Mjk]
                    mbal.loc[:,Mnetjk] = mbal.loc[:,Mjk] - mbal.loc[:,Mkj]
            #Mass balance for each compartment            
            #For water and soil, we may also have mass coming in from the pond zone
            if (j in 'water') and ('pond' in numc):
                mbal.loc[:,'Minp']=res_time.Min_p.groupby(level=[0,1]).sum()/divisor
                mbal.loc[:,Minj]+= mbal.loc[:,'Minp']
                mbal.loc[:,'Mnetwaterpond'] += -mbal.loc[:,'Minp'] #Negative as from pond to water.
                #For water also need to account for advection out of the system.
                mbal.loc[:,Moutj]+=mbal.loc[:,'Meff']+mbal.loc[:,'Mexf']
                
            elif j in 'subsoil'and ('pond' in numc):
                mbal.loc[:,'Minq'] = res_time.Mqin.groupby(level=[0,1]).sum()/divisor
                mbal.loc[:,Minj] += mbal.loc[:,'Minq']
                mbal.loc[:,'Mnetsubsoilpond'] += -mbal.loc[:,'Minq'] #Negative as from pond to water

            elif j in ['rootbody','rootxylem','rootcyl','shoots']:
                #Growth dilution processes - in DT but nowhere else.
                Nrg_j,Mgj = "Ng_"+str(j),"Mg"+str(j)
                mbal.loc[:,Mgj] = (mass_flux.dt*mass_flux.loc[:,Nrg_j]).groupby(level=[0,1]).sum()/divisor
                mbal.loc[:,Moutj] += mbal.loc[:,Mgj]
                mbal.loc[:,'Mout'] += (mass_flux.dt*mass_flux.loc[:,Nrg_j]).groupby(level=[0,1]).sum().groupby(level=0).cumsum()
            #If not normalized, Mj = absolute mass in compartment. Otherwise, percentage of mass in compartment at a time step (distribution)
            if normalized == True:
                mbal.loc[:,Mj] = res_time.loc[:,Mjind].groupby(level=[0,1]).sum()/divisor
            else:
                mbal.loc[:,Mj] = res_time.loc[:,Mjind].groupby(level=[0,1]).sum()
            #Positive indicates more mass entered than left in a time step
            mbal.loc[:,Mbalj] = (-res_time.loc[:,Mjind].groupby(level=[0,1]).sum()+res_time.loc[:,Mjind].groupby(level=[0,1]).sum().shift(1))/divisor\
                                +mbal.loc[:,Minj]-mbal.loc[:,Moutj]                                
            mbal.loc[(slice(None),min(mbal.index.levels[1])),Mbalj] = 0.0         
        mbal.loc[:,'Mbal'] = (mbal.loc[:,'Mout']+mbal.loc[:,'Mtot'])/divisor
        if np.sum(divisor) == 1:
            mbal.loc[mbal.Min==0,'Mbal'] = 0
        else: 
            mbal.loc[mbal.Min==0,'Mbal'] = 1
        
        #mbal.loc[:,'Mbal2'] = (mbal.loc[:,'Mbal2'])/mbal.Min           
        return mbal         

    def mass_balance_cumulative(self,numc,res_time=None,mass_flux = None,mass_balance = None,normalized=False):
        """ This function calculates cumulative mass transfers between compartments 
        on a whole-compartment (non spatially discretized) basis. 
        Attributes:
            numc - list of compartments.
            res_time (optional) - output from self.run_it. Needed if mass_balance not given.
            mass_flux (optional) - output from self.mass_flux. Needed if mass_balance not given.
            mass_balance (dataframe, optional) - Non-normalized output from mass_balance
            normalized (bool, optional) = Normalize the mass transfers to the total mass that has entered the system.
                Note that this will only normalize certain outputs
        """  
        #pdb.set_trace()
        #Set up mbal. Need to run as non-normalized in order for normalization here to work. Might fix this later, for now it is good enough
        try:
            mbal = mass_balance
        except AttributeError:                
            try:
                mbal = self.mass_balance(res_time,numc,mass_flux,normalized = False)
            except AttributeError:
                mbal = self.mass_balance(res_time,numc,normalized = False)
        #We need to get the cumulative mass fluxes. The overall values in mbal are cumulative already, so need some caution.
        #Start us off with those that will always be in the mbal dataframe, don't need to be cumulatively summed
        mbal_cum = mbal.loc[:,['time','Min','Mtot']]
        if normalized == True:
            divisor = mbal.Min+mbal.loc[(slice(None),mbal.index.levels[1][0]),'Mtot'].reindex(mbal.index,method ='ffill')
        else:
            divisor = 1    
        try:
            mbal_cum =  pd.concat([mbal_cum,mbal.loc[:,['Meff','Mexf','Minp','Minq']].groupby(level=0).cumsum()],axis=1)
            mbal_cum.loc[:,['Meff','Mexf','Mtot','Minp','Minq']] =  mbal_cum.loc[:,['Meff','Mexf','Mtot','Minp','Minq']].mul(1/divisor,axis="index")
        except KeyError:
            mbal_cum =  pd.concat([mbal_cum,mbal.loc[:,['Meff','Mexf']].groupby(level=0).cumsum()],axis=1)
            mbal_cum.loc[:,['Meff','Mexf','Mtot']] =  mbal_cum.loc[:,['Meff','Mexf','Mtot']].mul(1/divisor,axis="index")
        for j in numc:#j is compartment mass is leaving
            Mj,Madvj,Mrj= 'M'+str(j),'Madv'+str(j),'Mr'+str(j)
            #Mass is not cumulative.
            if normalized == True:
                mbal_cum.loc[:,Mj] = mbal.loc[:,Mj]/divisor
            else:
                mbal_cum.loc[:,Mj] = mbal.loc[:,Mj]
            mbal_cum.loc[:,Madvj] = mbal.loc[:,Madvj].groupby(level=0).cumsum()/divisor
            mbal_cum.loc[:,Mrj] = mbal.loc[:,Mrj].groupby(level=0).cumsum()/divisor
            if j in ['rootbody','rootxylem','rootcyl','shoots']:
                #Growth dilution processes - in DT but nowhere else.
                Mgj = "Mg"+str(j)
                mbal_cum.loc[:,Mgj] = mbal.loc[:,Mgj].groupby(level=0).cumsum()/divisor
            #Intercompartmental transfer. For net transfer, we only need to calculate once - as Mnetjk = Mjk-Mkj
            for k in numc:
                if ('Mnet'+str(k)+str(j)) in mbal_cum.columns:
                    pass
                elif k != j:
                    Mjk, Mkj, Mnetjk = 'M'+str(j)+str(k),'M'+str(k)+str(j),'Mnet'+str(j)+str(k)
                    mbal_cum.loc[:,Mjk] = mbal.loc[:,Mjk].groupby(level=0).cumsum()/divisor
                    mbal_cum.loc[:,Mkj] = mbal.loc[:,Mkj].groupby(level=0).cumsum()/divisor
                    mbal_cum.loc[:,Mnetjk] = mbal.loc[:,Mnetjk].groupby(level=0).cumsum()/divisor
        return mbal_cum
    
    def conc_out(self,numc,timeseries,chemsumm,res_time,mass_flux=None):
        """ This function calculates modeled concentrations at the outlet for all chemicals present. All values g/m³
        """
        #pdb.set_trace()
        try:
            mass_flux.loc[:,'dt']
        except AttributeError:
            mass_flux = self.mass_flux(res_time,numc)
        numx = res_time.loc[(res_time.index.levels[0][0],res_time.index.levels[1][0],slice(None)),'dm'].sum()
        #pdb.set_trace()
        Couts = pd.DataFrame(np.array(res_time.loc[(min(res_time.index.levels[0]),slice(None),numx-1),'time']),
                                          index = res_time.index.levels[1],columns=['time'])
        try: #If there are measured and estimated flows, bring both in
            Couts.loc[:,'Qout_meas'] = timeseries.loc[:,'Qout_meas']
            Couts.loc[:,'Qout'] = np.array(res_time.loc[(min(res_time.index.levels[0]),slice(None),numx-1),'Qout'])
        except KeyError:
            Couts.loc[:,'Qout'] = timeseries.loc[:,'Qout']
        for chem in mass_flux.index.levels[0]:
            try: #If there are measurements
                Couts.loc[:,chem+'_Coutmeas'] = timeseries.loc[:,chem+'_Coutmeas']
            except KeyError:
                pass
            #Concentration = mass flux/Q*MW
            Couts.loc[:,chem+'_Coutest'] = np.array(mass_flux.loc[(chem,slice(None),slice(None)),'N_effluent'].groupby(level=1).sum())\
                                            /np.array(Couts.loc[:,'Qout'])*np.array(chemsumm.loc[chem,'MolMass'])
            Couts.loc[np.isnan(Couts.loc[:,chem+'_Coutest']),chem+'_Coutest'] = 0.                      
        
        return Couts
    
    def concentrations(self,numc,res_time):
        """This method calculates modeled concentrations within the system.
        
        """
        #pdb.set_trace()
        concentrations = pd.DataFrame(res_time.loc[res_time.time>=0,['time','x']])
        for jind,j in enumerate(numc):
            jind = jind+1
            aval,Zval,colname = 'a'+str(jind) + '_t1','Z'+str(jind),j+'_conc'
            concentrations.loc[:,colname] = res_time.loc[:,aval]*res_time.loc[:,Zval]
        return concentrations
    
    def mass_distribution(self,numc,res_time,timeseries,chemsumm,normalized = 'compartment'):
        """ 
        This function calculates modeled mass distributions for the different compartments. Large dataframe, same indices as res_time
        
        Attributes:
            normalized (string, optional) - 'True'/'t' will normalize to total mass in system (inlfuent + M0), 'False'/'f' = no normalization, 
                'compartment'/'c' to mass in each compartment at each time, 'overall'/'o' is overall mass M_tot
        """
        #pdb.set_trace()
        mdist = res_time.loc[:,['x','time']] #If timeseries has spin-up period, skip to actual times
        if normalized[0].lower() == 't':
            res_time.loc[np.isnan(res_time.Min),'Min'] = 0    
            divisor = res_time.Min+res_time.loc[(slice(None),res_time.index.levels[1][0],slice(None)),
                                                    'M_tot'].reindex(res_time.index,method ='ffill')
            divisor = divisor.groupby(level=[0]).cumsum()
        elif normalized[0].lower() == 'o': #overall/total mass 
            divisor = res_time.M_tot
        else:
            divisor = 1.
            
        for jind,j in enumerate(numc):
            jind = jind+1
            M_val= 'M'+str(jind) + '_t1'
            #Divide by the total mass in each compartment
            if normalized[0].lower() == 'c':
                divisor = res_time.loc[:,M_val].groupby(level=[0,1]).sum().reindex(res_time.index,method ='ffill')        
            mdist.loc[:,'M'+str(j)] = res_time.loc[:,M_val]/divisor
        #mdist = mdist.fillna(0)
        return mdist
    
            
    def model_fig(self,numc,mass_balance,dM_locs,M_locs,time=None,compound=None,figname=None,dpi=100,fontsize=8,figheight=6):
        """ 
        Show modeled fluxes and mass distributions on a figure. 
        Attributes:
            mass_balance = Either the cumulative or the instantaneous mass balance, as output by the appropriate functions, normalized or not
            figname (str, optional) = name of figure on which to display outputs. Default figure should be in the same folder
            time (float, optional) = Time to display, in hours. Default will be the last timestep.
            compounds (str, optional) = Compounds to display. Default is all.
        """
        #pdb.set_trace()
        #Set up attributes that weren't given
        mbal = mass_balance
        if time is None:
            time = mbal.index.levels[1][-1]#Default is at the end of the model run.
        else:#Otherwise, find the time index from the given time in hours    
            time = mbal.index.levels[1][np.where(mbal.loc[(mbal.index.levels[0][0],
                                        slice(None)),'time']== find_nearest(mbal.loc[(mbal.index.levels[0][0],
                                        slice(None)),'time'],time))][0]
        if compound != None:
            mbal = mbal.loc[(compound,time),:]
        else:
            mbal = mbal.loc[(mbal.index.levels[0][0],time),:]
        img = plt.imread(figname)
        figsize = (img.shape[1]/img.shape[0]*figheight,figheight)
        fig, ax = plt.subplots(figsize=figsize, dpi = dpi)
        ax.grid(False)
        plt.axis('off')    
        #Define the locations where the mass transfers (g) will be placed.            
        for j in numc:#j is compartment mass is leaving
            Mj,Mrj = 'M'+str(j),'Mr'+str(j) 
            pass
            ax.annotate(f'{mbal.loc[Mj]:.2e}',xy = M_locs[Mj],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction') #Mass distribution if normalized, abs. mass if not. At time t        
            ax.annotate(f'{mbal.loc[Mrj]:.2e}',xy = dM_locs[Mrj],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction')
            if j in ['water']: #Add effluent and exfiltration advection
                ax.annotate(f'{mbal.loc["Meff"]:.2e}',xy = dM_locs['Meff'],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction')
                ax.annotate(f'{mbal.loc["Mexf"]:.2e}',xy = dM_locs['Mexf'],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction')
                if 'pond' in numc: #Need to reverse some to display with appropriate conventions.
                    mbal.loc['Mnetwaterpond'] = -mbal.loc['Mnetwaterpond'] 
            elif j in ['subsoil']:#Mass enters the pond.
                if 'pond' in numc:
                    mbal.loc['Mnetsubsoilpond'] = -mbal.loc['Mnetsubsoilpond'] 
                if 'shoots' in numc:
                    mbal.loc['Mnetsubsoilshoots'] = -mbal.loc['Mnetsubsoilshoots']                 
            elif j in ['pond']:#Add mass in - goes to pond zone, NOT normalized.
                ax.annotate(f'{mbal.loc["Min"]:.2e}',xy = dM_locs['Min'],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction')
                ax.annotate(f'{mbal.loc["Madvpond"]:.2e}',xy = dM_locs['Madvpond'],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction')
            elif j in ['air']:#Air has advection out the back end
                ax.annotate(f'{mbal.loc["Madvair"]:.2e}',xy = dM_locs['Madvair'],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction')
            for k in numc:
                if (('Mnet'+str(j)+str(k)) not in dM_locs.keys()):
                    pass
                else:
                    Mnetjk = 'Mnet'+str(j)+str(k)
                    ax.annotate(f'{mbal[Mnetjk]:.2e}',xy = dM_locs[Mnetjk],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction')
        ax.annotate(compound,xy = (0,1),fontsize = fontsize+2, fontweight = 'bold',xycoords='axes fraction')
#       '''
        #stop = 'stop'               
        ax.imshow(img,aspect='auto')
        return fig,ax
#(((res.dt*mass_flux.N_effluent).groupby(level=[0,1]).sum().cumsum()+(res.dt*mass_flux.Nrwater).groupby(level=[0,1]).sum().cumsum()\
#         +(res.dt*mass_flux.N_exf).groupby(level=[0,1]).sum().cumsum()+(res.dt*mass_flux.Nrsubsoil).groupby(level=[0,1]).sum().cumsum())\
#    +res.loc[:,'M_tot'].groupby(level=[0,1]).sum())/res.Min.groupby(level=[0,1]).sum().cumsum()
        
        
        
        
        
        