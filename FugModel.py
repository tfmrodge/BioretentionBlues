# -*- coding: utf-8 -*-
"""
Fugacity Model class, containing all other fugacity models within
Created on Wed Jul 25 15:52:23 2018

@author: Tim Rodgers
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:48:58 2018

@author: Tim Rodgers
"""
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
#import pdb #Turn on for error checking
#import xarray as xr #cite as https://openresearchsoftware.metajnl.com/articles/10.5334/jors.148/

class FugModel(metaclass=ABCMeta):
    """ Fugacity model object, as described by Mackay (2001). This class will
    contain fugacity models, such as ppLFERMUM (Rodgers et al., 2018), the Multimedia 
    Indoor Model (), and the Bioretention Cell Blues (BCBlues Rodgers et al., unpublished).
    The FugModel object is itself an abstract base class (ABC) and so cannot be
    instantiated by itself. The input_calcs abstractmethod needs to be defined for each model.
    Fugacity models have a number of shared attributes and methods, as defined below.
        
    Attributes:
    ----------

            locsumm (df): Properties of the compartments
            chemsumm (df): phyical-chemical properties of modelled compounds
            params (df): Other parameters of the model
            num_compartments (int): (optional) number of non-equilibirum 
            compartments and size of D value matrix
            name (str): (optional) name of the model run
    
    Methods:
    ----------

            run_model(self,calctype): run the selected model using the calculation type specified:
                fss for forward steady state, bss for backward steady state
            forward_calc_ss(self,ic,num_compartments): Do forwards steady-state calcs
            forward_step_uss(self,ic,num_compartments):
            
    Sub-Classes:
    ----------

            ppLFERMUM - ppLFERMUM of Rodgers et al. (2018) based on MUM of Diamond et al (2001)
            BCBlues - BioretentionCell Blues model of Rodgers et al. (in prep)
            
    """ 
    #(self,locsumm,chemsumm,params,input_calcs,num_compartments,name)
    def __init__(self,locsumm,chemsumm,params,num_compartments,name):
        self.locsumm = locsumm
        self.chemsumm = chemsumm
        self.params = params
        self.numc = num_compartments
        self.name = name
        
    #This method needs to be instantiated for every child class
    @abstractmethod
    def input_calc(self):
        pass
        
    def run_model(self,calctype='fss'):
        if calctype is 'fss': #Peform forward steady-state calcs
            return self.forward_calc_ss(self.ic,self.numc)
        elif calctype is 'bss': #Perform backward calcs with lair concentration as target and emissions location
            return self.backward_calc_ss(self.ic,self.numc)
        
    def forward_calc_ss(self,ic,num_compartments):
        """ Perform forward calculations to determine model steady state fugacities
        based on input emissions. Initial_calcs (ic) are calculated at the initialization
        of the chosen model and include the matrix values DTi, and D_ij for each compartment
        as well as a column named compound
        num_compartments (numc) defines the size of the matrix
        """
        #Determine number of chemicals
        #pdb.set_trace()
        numchems = 0
        for chems in ic.Compound:
            numchems = numchems + 1
            
        #Initialize output - the calculated fugacity of every compartment
        col_name = pd.Series(index = range(num_compartments))
        for i in range(num_compartments):
            col_name[i] = 'f'+str(i+1)
        fw_out = pd.DataFrame(index = ic['Compound'],columns = col_name)
        
        #generate matrix. Names of D values in ic must conform to these labels:
        #DTj for total D val from compartment j and D_jk for transfer between compartments j and k
        #Initialize a blank matrix of D values. We will iterate across this to solve for each compound
        D_mat = pd.DataFrame(index = range(num_compartments),columns = range(num_compartments))
        #initialize a blank dataframe for input vectors, RHS of matrix
        inp_val = pd.DataFrame(index = range(num_compartments),columns = ic.Compound)
        for chem in ic.index: #Index of chemical i
            for j in D_mat.index: #compartment j, index of D_mat
                #Define RHS input for every compartment j
                inp_name = 'inp_' + str(j + 1) #must have an input for every compartment, even if it is zero
                inp_val.iloc[j,chem] = -ic.loc[chem,inp_name]
                for k in D_mat.columns: #compartment k, column of D_mat
                    if j == k:
                        DT = 'DT' + str(j + 1)
                        D_mat.iloc[j,k] = -ic.loc[chem,DT]
                    else:
                        D_val = 'D_' +str(k+1)+str(j+1) #label compartments from 1
                        if D_val in ic.columns: #Check if there is transfer between the two compartments
                            D_mat.iloc[j,k] = ic.loc[chem,D_val]
                        else:
                            D_mat.iloc[j,k] = 0 #If no transfer, set to 0
            #Solve for fugacities f = D_mat\inp_val
            lhs = np.array(D_mat,dtype = float)
            rhs = np.array(inp_val.iloc[:,chem],dtype = float)
            fugs = np.linalg.solve(lhs,rhs)
            fw_out.iloc[chem,:] = fugs
        
        return fw_out

    def backward_calc_ss(self,ic,num_compartments,target_conc = 1,target_emiss = 1):
        """ Inverse modelling to determine emissions from measured concentrations
        as selected by the user through the 'target' attribute at steady state.
        Initial_calcs (ic) are calculated at the initialization of the model and 
        include the matrix values DTi, D_ij and the target fugacity (where given)
        for each compartment. This method needs a target fugacity (NOT concentration)
        to function, but the input in chemsumm is a concentration. num_compartments (numc) defines the 
        size of the matrix, target_conc tells what compartment (numbered from 1 not 0)
        the concentration corresponds with, while target_emiss defines which compartment
        the emissions are to. Default = 1, Lair in ppLFER-MUM. Currently, the output is
        a dataframe with the fugacities of each compartment and the emissions in g/h.
        """
        #Initialize outputs
        #pdb.set_trace()
        col_name = pd.Series(index = range(num_compartments))
        for i in range(num_compartments):
            col_name[i] = 'f'+str(i+1) #Fugacity for every compartment
        #Emissions for the target_emiss compartment
        col_name[num_compartments+1] = 'emiss_'+str(target_emiss)
        bw_out = pd.DataFrame(index = ic['Compound'],columns = col_name)        
        #Define target name and check if there is a value for it in the ic dataframe. If not, abort
        targ_name = 'targ_' + str(target_conc)
        if targ_name not in ic.columns:
            return'Please define a target concentration for the chosen compartment, comp_' + str(target_conc)
        #initialize a matrix of numc x numc compartments.
        D_mat = pd.DataFrame(index = range(num_compartments),columns = range(num_compartments))
        #initialize a blank dataframe for input vectors, RHS of matrix.
        inp_val = pd.DataFrame(index = range(num_compartments),columns = ic.Compound)
        #Loop over the chemicals, solving for each.
        for chem in ic.index: #Index of chemical i starting at 0
            #Put the target fugacity into the output
            bw_out.iloc[chem,target_conc-1] = ic.loc[chem,targ_name]
            #Double loop to set matrix values
            j = 0 #Index to pull values from ic
            while j < num_compartments: #compartment j, index of D_mat
                #Define RHS = -Inp(j) - D(Tj)*f(T) for every compartment j using target T
                D_val = 'D_' +str(target_conc)+str(j+1) #label compartments from 1
                inp_name = 'inp_' + str(j + 1) #must have an input for every compartment, even if it is zero
                if j+1 == target_conc: #Need to use DT value for target concentration
                    DT = 'DT' + str(j + 1)
                    if j+1 == target_emiss: #Set -Inp(j) to zero for the targ_emiss row, we will subtract GCb(target_emiss) later
                        inp_val.iloc[j,chem] = ic.loc[chem,DT] * bw_out.iloc[chem,target_conc-1]
                    else:
                        inp_val.iloc[j,chem] = ic.loc[chem,DT] * bw_out.iloc[chem,target_conc-1]-ic.loc[chem,inp_name]
                elif D_val in ic.columns: #check if there is a D(Tj) value
                    if j+1 == target_emiss: #This is clunky but hopefully functional
                        inp_val.iloc[j,chem] = -ic.loc[chem,D_val] * bw_out.iloc[chem,target_conc-1]
                    else:
                        inp_val.iloc[j,chem] = -ic.loc[chem,inp_name] - ic.loc[chem,D_val]*bw_out.iloc[chem,target_conc-1]
                else: #If there is no D(Tj) then RHS = -Inp(j), unless it is the target_emiss column again
                    if j+1 == target_emiss: 
                        inp_val.iloc[j,chem] = 0
                    else:
                        inp_val.iloc[j,chem] = -ic.loc[chem,inp_name]
          
                #Set D values across each row
                k = 0 #Compartment index
                kk = 0 #Index to fill matrix
                while k < num_compartments: #compartment k, column of D_mat
                    if (k+1) == target_conc:
                        k += 1
                    if j == k:
                        DT = 'DT' + str(j + 1)
                        D_mat.iloc[j,kk] = -ic.loc[chem,DT]
                    else:
                        D_val = 'D_' +str(k+1)+str(j+1) #label compartments from 1
                        if D_val in ic.columns: #Check if there is transfer between the two compartments
                            D_mat.iloc[j,kk] = ic.loc[chem,D_val]
                        else:
                            D_mat.iloc[j,kk] = 0 #If no transfer, set to 0
                    if k+1 == num_compartments: #Final column is the input to the target_emiss compartment
                        if (j+1) == target_emiss: #This is 1 for the target_emiss column and 0 everywhere else
                            D_mat.iloc[j,kk+1] = 1
                        else:
                            D_mat.iloc[j,kk+1] = 0
                    k +=1
                    kk += 1
                j += 1
            #Solve for fugsinp = D_mat\inp_val, the last value in fugs is the total inputs
            lhs = np.array(D_mat,dtype = float)
            rhs = np.array(inp_val.iloc[:,chem],dtype = float)
            fugsinp = np.linalg.solve(lhs,rhs)
            #Subtract out the Gcb to get emissions from total inputs
            gcb_name = 'Gcb_' + str(target_emiss)
            fugsinp[-1] = fugsinp[-1] - ic.loc[chem,gcb_name]
            #Multiply by molar mass to get g/h output
            fugsinp[-1] = fugsinp[-1] * ic.loc[chem,'MolMass']
            #bwout units are mol/m³/pa for fugacities, mol/h for emissions
            bw_out.iloc[chem,0:target_conc-1] = fugsinp[0:target_conc-1]
            bw_out.iloc[chem,target_conc:] = fugsinp[target_conc-1:]
        return bw_out
    
    def forward_step_uss(self,ic,num_compartments):
        """ Perform a forward calculation step to determine model unsteady-state fugacities
        based on input emissions. Input calcs need to include inp(t+1), DTi(t+1),
        and D_ij(t+1) for each compartment, mass M(n), as well as a column named
        compound. num_compartments (numc) defines the size of the matrix. 
        From Csizar, Diamond and Thibodeaux (2012) DOI 10.1016/j.chemosphere.2011.12.044
        Possibly this doesn't belong in the parent class, to use it needs to be called
        in a loop which would be in a child classes method.
        """
    

    
    
    
    
    
    
    
    
    
    
    