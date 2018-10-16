# -*- coding: utf-8 -*-
"""
Data preperation for the 1D ADRE BC Blues model
Created on Fri Oct 12 18:02:32 2018

@author: Tim Rodgers
"""
import pandas as pd
import numpy as np

def makeOroLoma(locsumm,chemsumm,params):
    #Make the location summary of the Oro Loma system
    #Compartments 1 - mobile water, 2 - soil in contact with mobile water,
    #3 - veg, 4 - topsoil, 5 - air?
    L = 30 #m
    Qin = 2 #m³/s
    Qout = 0.6 #m³/s
    dx = 1 #m
    phi_1 = 0.4 #Average porosity through which the mobile phase moves
    phi_4 = 0.2 #Average porosity of the topsoil
    res = pd.DataFrame(np.arange(0,L+dx,dx),columns = ['x'])


def makeic():
        
    #Testing, so lets set up a simple problem. 
    L = 100 #Length [l]
    A = 1.5 #Area [l²]
    alpha_disp = 0.05 #
    phi = 0.6 #porosity
    dx = 1 #Space step [l]
    Qin = 2 # [l³/t] 
    Qout = 0.6 #[l³/t]  Assuming 70% ET
    #Number of compartments
    numc = 6
    #These would be from an input file for each chemical.
    
    chems = ['c1', 'c2', 'c3', 'c4', 'c5']
    #chems = ['c1']

    #Z values
    Z1 = 0.1 #[M/L³]
    Z2 = 1
    Z3 = 0.05
    Z4 = 0.5
    Z5 = 0.9
    Z6 = 0.5
       
    D12 = 10 #D [M/t] from water to soil
    D13 = 0.1
    D14 = 0.4
    D15 = 0.3
    D16 = 0.1
    D21 = 0.1 #D [M/t] from soil to water
    D23 = 0.1
    D24 = 0.1
    D25 = 0.2
    D26 = 0.1
    D31 = 0.5
    D32 = 0.3
    D34 = 0
    D35 = 0.1
    D36 = 0.9
    D41 = 0.7
    D42 = 0.1
    D43 = 0.1
    D45 = 0.5
    D46 = 0.75
    D51 = 0.7
    D52 = 0.5
    D53 = 0.5
    D54 = 0.52
    D56 = 0.1
    D61 = 0.3
    D62 = 0.4
    D63 = 0.5
    D64 = 0.05
    D65 = 0.2
    

    #inputs
    inp_1 = 0
    inp_2 = 0
    inp_3 = 0
    inp_4 = 0
    inp_5 = 0
    inp_6 = 0
    #Initialize length variable
    numchems = int(len(chems))
    res = pd.DataFrame(np.arange(0,L+dx,dx),columns = ['x'])
    #Define the x term as the centre of each cell
    res.loc[:,'x'] = res.x+dx/2
    res1 = res
    res2 = res
    res3 = res
    res4 = res
    res = pd.concat([res,res1,res2,res3,res4], keys=chems)
    #Calculate the flow for every point in x
    res.loc[:,'V1'] = A*dx #water volume of each x [L³]
    res.loc[:,'A'] = A
    res.loc[:,'dx'] = dx #distance between nodes [L]
    res.loc[:,'Q'] = Qin - (Qin-Qout)/L*res.x #flow at every x
    res.loc[:,'q'] = res.Q/A #darcy flux [L/T] at every x
    res.loc[:,'v'] = res.q/phi #darcy flux [L/T] at every x
    res.loc[:,'disp'] = alpha_disp * res.v # [l²/T] Dispersivity
    #res.loc[:,'c'] = res.q*dt/dx #courant number for each x               
    res.loc[:,'DT1'] = 0 #D leaving the water along x
    res.loc[:,'DT2'] = 0
    res.loc[:,'D_12'] = D12 + (Qin-Qout)/L*dx *Z1
    res.loc[:,'D_13'] = D13  
    res.loc[:,'D_14'] = D14  
    res.loc[:,'D_15'] = D15 
    res.loc[:,'D_16'] = D16
    res.loc[:,'D_21'] = D21  
    res.loc[:,'D_23'] = D23
    res.loc[:,'D_24'] = D24
    res.loc[:,'D_25'] = D25
    res.loc[:,'D_26'] = D26
    res.loc[:,'D_31'] = D31
    res.loc[:,'D_32'] = D32
    res.loc[:,'DT3'] = 0
    res.loc[:,'D_34'] = D34
    res.loc[:,'D_35'] = D35
    res.loc[:,'D_36'] = D36
    res.loc[:,'D_41'] = D41
    res.loc[:,'D_42'] = D42
    res.loc[:,'D_43'] = D43
    res.loc[:,'DT4'] = 0
    res.loc[:,'D_45'] = D45
    res.loc[:,'D_46'] = D46
    res.loc[:,'D_51'] = D51
    res.loc[:,'D_52'] = D52
    res.loc[:,'D_53'] = D53
    res.loc[:,'D_54'] = D54
    res.loc[:,'DT5'] = 0
    res.loc[:,'D_56'] = D56
    res.loc[:,'D_61'] = D61
    res.loc[:,'D_62'] = D62
    res.loc[:,'D_63'] = D63
    res.loc[:,'D_64'] = D64
    res.loc[:,'D_65'] = D65
    res.loc[:,'DT6'] = 0
    #For a conservative compound, DT = sum(Douts).
    for j in range(numc):
        for k in range(numc):
            if j !=k:
                DT_val= 'DT' + str(j+1)
                D_val = 'D_' + str(j+1) + str(k+1)
                res.loc[:,DT_val] += res[D_val]
                
    #Z values
    res.loc[:,'Z1'] = Z1
    res.loc[:,'Z2'] = Z2
    res.loc[:,'Z3'] = Z3
    res.loc[:,'Z4'] = Z4
    res.loc[:,'Z5'] = Z5
    res.loc[:,'Z6'] = Z6
    #Assume equal volumes for now
    res.loc[:,'V2'] = res.V1
    res.loc[:,'V3'] = res.V1
    res.loc[:,'V4'] = res.V1
    res.loc[:,'V5'] = res.V1
    res.loc[:,'V6'] = res.V1
    #Inputs
    res.loc[:,'inp_1'] = inp_1
    res.loc[:,'inp_2'] = inp_2
    res.loc[:,'inp_3'] = inp_3
    res.loc[:,'inp_4'] = inp_4
    res.loc[:,'inp_5'] = inp_5
    res.loc[:,'inp_6'] = inp_6
    
    #Conditions in prior time step (initial conditions for each step)
    res.loc[:,'a1_t'] = 1 #- (res.x - numx)/numx
    res.loc[:,'a2_t'] = 1 #- (res.x - numx)/numx #initial activity in the soil
    res.loc[:,'a3_t'] = 1 #- (res.x - numx)/numx
    res.loc[:,'a4_t'] = 1 #- (res.x - numx)/numx
    res.loc[:,'a5_t'] = 1 #- (res.x - numx)/numx
    res.loc[:,'a6_t'] = 1 #- (res.x - numx)/numx

    return res