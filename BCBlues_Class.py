# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:32:24 2018

@author: Tim Rodgers
"""
import pandas as pd

class BioretentionCellBlues:
    """ Model of contaminant transport in a bioretention cell. BCBlues objects
    have the following properties:
        
    Attributes:
    ----------

            bcsumm (df): physical properties of the BC
            chemsumm (df): phyical-chemical properties of modelled compounds
            results (df): Results of the BC model
            num_compartments (int): (optional) number of non-equilibirum 
            compartments and size of D value matrix
            name (str): (optional) name of the BC model 
    """
    def __init__(self,bcsumm,chemsumm,num_compartments = 7,name = None):
        self.bcsumm = bcsumm
        self.chemsumm = chemsumm
        self.numc = num_compartments
        self.name = name
        
                
    def steady_state(self,bcsumm=bcsumm,chemsumm=chemsumm):
        """ Model the bioretention cell at steady state. A steady state
        bioretention cell is an n compartment fugacity model solved at steady
        state using the compartment parameters from bcsumm and the chemical
        parameters from chemsumm """
        #Declare constants
        R = 8.314 #Ideal gas constant, J/mol/K
        #Initialize results
        res = pd.DataFrame(chemsumm.iloc[:, 0])
        #Calculate chemical-independent parameters
        bcsumm.loc[:,'V']= bcsumm.Area*bcsumm.Depth #Add volumes  m³
        bcsumm.loc[0,'Density'] = 0.029 * 101325 / (R * bcsumm.Temp[0]) #Air density kg/m^3
        #bcsumm.loc[:,'Z']=0 #Add Z values to the bcsumm
        numchems = 0
        for chems in chemsumm.Compound:
            numchems = numchems + 1
        #Calculate Z-Values for chemical chem ZB(j) is the bulk Z value for compartment j
        #0 - Air
        bcsumm.loc[0,'Zbulk']=1/(R*bcsumm.Temp[0])
        return res