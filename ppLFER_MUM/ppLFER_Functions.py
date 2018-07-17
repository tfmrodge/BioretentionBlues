# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:09:04 2018
Helper functions for ppLFER-MUM
@author: Tim Rodgers
"""
#
import numpy as np

def arr_conv(Ea,T2,k1,T1 = 298.15,):
    """Arrhenius equation conversion of rate reaction constants (k) from T1 to T2 (K)
    The default value for T1 is 298.15K. Activation energy should be in J. The 
    result (res) will be k2 at T2
    """
    R = 8.314 #J/mol/K
    res =  k1 * np.exp((Ea / R) * (1 / T1 - 1 / T2))
    return res

def vant_conv(dU,T2,k1,T1 = 298.15,):
    """Van't Hoff equation conversion of partition coefficients (Kij) from T1 to T2 (K)
    The default value for T1 is 298.15K. Activation energy should be in J. The 
    result (res) will be K2 at T2
    Yes this is literally the same as the Arrhenius but ¯\_(ツ)_/¯
    """
    R = 8.314 #J/mol/K
    res =  k1 * np.exp((dU / R) * (1 / T1 - 1 / T2))
    return res

def ppLFER(L,S,A,B,V,l,s,a,b,v,c):
    """polyparameter linear free energy relationship (ppLFER) in the 1 equation form from Goss (2005)
    Upper case letters represent Abraham's solute descriptors (compund specific)
    while the lower case letters represent the system parameters.
    """
    res = L*l+S*s+A*a+B*b+V*v+c
    return res