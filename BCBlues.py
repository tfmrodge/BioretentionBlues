# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:52:52 2019

@author: Tim Rodgers
"""

from FugModel import FugModel #Import the parent FugModel class
from Subsurface_Sinks import Subsurface_Sinks
from HelperFuncs import vant_conv, arr_conv #Import helper functions
from scipy.integrate import solve_ivp
from ode_helpers import state_plotter
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
#import time
import pdb #Turn on for error checking

class BCBlues(Subsurface_Sinks):
    """Wastewater treatment wetland implementation of the Subsurface_Sinks model.
    Created for the Oro Loma Horizontal Levee, hence the name. This model represents
    a horizontally flowing, planted wetland. It is intended to be solved as a Level V 
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