# -*- coding: utf-8 -*-
"""
Load Data for BC Blues Model
Created on Fri Jun 15 14:51:15 2018

@author: Tim Rodgers
"""
#Import packages
import pandas as pd

#For ppLFER-MUM
#chemsumm = pd.read_csv('OPECHEMSUMM.csv') 
chemsumm = pd.read_csv('OPECHEMSUMM_Barebones.csv') 
#Location summary for the modelled area. Descriptors should be in the first column (0)
locsumm = pd.read_csv('locsumm.csv',index_col = 0) 
#parameters must be loaded with the descriptor in the first column (0) and the values in a "Value" column
params = pd.read_csv('params.csv',index_col = 0) 