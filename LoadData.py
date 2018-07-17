# -*- coding: utf-8 -*-
"""
Load Data for BC Blues Model
Created on Fri Jun 15 14:51:15 2018

@author: Tim Rodgers
"""
#Import packages
import numpy as np
import pandas as pd

#cd "D:\Users\Tim Rodgers\Documents\GitHub\BioretentionBlues"
#chemsumm = pd.read_csv('OPECHEMSUMM.csv') 
#For the BCsumm, dimensions are the dimensions of the BC. Since the ponding zone is transient, its depth, area etc. will change.
#bcsumm = pd.read_csv('BCSUMM.csv') 

#For ppLFER-MUM
chemsumm = pd.read_csv('OPECHEMSUMM.csv') 
#For the BCsumm, dimensions are the dimensions of the BC. Since the ponding zone is transient, its depth, area etc. will change.
locsumm = pd.read_csv('locsumm.csv') 
#parameters must be loaded with the descriptor in the first column (0) and the values in a "Value" column
params = pd.read_csv('params.csv',index_col = 0) 