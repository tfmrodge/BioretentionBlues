#1DADRE_uss(self,ic,locsumm,num_compartments = 4,dt = 1,dx = 1):
""" Perform a single time step in a 1D ADRE multimedia model.
This solution to the 1D ADRE requires an input velocity to be provided, 
and will calculate across the entire spatial range of the modelled system.
Based on the QUICKEST algorithm or Manson and Wallis (1995), as implemented
in Kilic et al, (2009, DOI 10.1016/j.scitotenv.2009.01.057). I have generalized
the system so that it can be used with a single mobile phase and any number
of stationary compartments, where the mobile phase can be through porous media
or surface flow (if porosity phi = 1) Need to define the number of compartments
and give D values between all compartments.
Inputs should be sorted as a multiindexed dataframe where the index is 
[chems,xindex] and the columns are the values at each x. This code allows
for all inputs (eg Z, D values) to be non-uniform spatially.
"""
import numpy as np
import pandas as pd
import time 
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

start = time.time()
#Testing, so lets set up a simple problem. 
L = 100 #Length [l]
A = 1.5 #Area [l²]
alpha_disp = 0.05 #
phi = 0.6 #porosity
dx = 1 #Space step [l]
#dt = 1 #time step [t]
Qin = 2 # [l³/t] 
Qout = 0.6 #[l³/t]  Assuming 70% ET
#Number of compartments
numc = 6
#These would be from an input file for each chemical.

chems = ['c1', 'c2', 'c3', 'c4', 'c5']

   
D12 = 1 #D [M/t] from water to soil
D13 = 0.1
D14 = 0.4
D15 = 0.3
D16 = 0.1
D21 = 0.1 #D [M/t] from soil to water
DT2 = D21 #[M/t] Total leaving soil
D23 = 0.1
D24 = 0.1
D25 = 0.2
D26 = 0.1
D31 = 0.5
D32 = 0.3
DT3 = 0.9
D34 = 0
D35 = 0.1
D36 = 0.9
D41 = 0.7
D42 = 0.1
D43 = 0.1
DT4 = 0.3
D45 = 0.5
D46 = 0.75
D51 = 0.7
D52 = 0.5
D53 = 0.5
D54 = 0.52
DT5 = 0.2
D56 = 0.1
D61 = 0.3
D62 = 0.4
D63 = 0.5
D64 = 0.05
D65 = 0.2
DT6 = 0.1

Z1 = 0.1 #[M/L³]

#inputs
inp_1 = 0
inp_2 = 0
inp_3 = 0
inp_4 = 0
inp_5 = 0
inp_6 = 0
#Initialize length variable
numchems = int(len(chems))
res = pd.DataFrame(np.arange(0,L+dx-1,dx),columns = ['x'])
#Define the x term as the centre of each cell
res.loc[:,'x'] = res.x+dx/2
res1 = res
res2 = res
res3 = res
res4 = res
res = pd.concat([res,res1,res2,res3,res4], keys=chems)
numx = int(len(res.x)/numchems)
reslen = int(len(res.x))
#Calculate the flow for every point in x
res.loc[:,'V1'] = A*dx #water volume of each x [L³]
res.loc[:,'dx'] = dx #distance between nodes [L]
res.loc[:,'Q'] = Qin - (Qin-Qout)/L*res.x #flow at every x
res.loc[:,'q'] = res.Q/A #darcy flux [L/T] at every x
res.loc[:,'v'] = res.q/phi #darcy flux [L/T] at every x
res.loc[:,'disp'] = alpha_disp * res.v # [l²/T] Dispersivity
#res.loc[:,'c'] = res.q*dt/dx #courant number for each x
res.loc[:,'DT1'] = D12 + (Qin-Qout)/L*dx *Z1 #D leaving the water along x
res.loc[:,'D_12'] = D12 #D [M/t] from water to soil
res.loc[:,'D_13'] = D13  
res.loc[:,'D_14'] = D14  
res.loc[:,'D_15'] = D15 
res.loc[:,'D_16'] = D16
res.loc[:,'D_21'] = D21  
res.loc[:,'DT2'] = DT2 
res.loc[:,'D_23'] = D23
res.loc[:,'D_24'] = D24
res.loc[:,'D_25'] = D25
res.loc[:,'D_26'] = D26
res.loc[:,'D_31'] = D31
res.loc[:,'D_32'] = D32
res.loc[:,'DT3'] = DT3
res.loc[:,'D_34'] = D34
res.loc[:,'D_35'] = D35
res.loc[:,'D_36'] = D36
res.loc[:,'D_41'] = D41
res.loc[:,'D_42'] = D42
res.loc[:,'D_43'] = D43
res.loc[:,'DT4'] = DT4
res.loc[:,'D_45'] = D45
res.loc[:,'D_46'] = D46
res.loc[:,'D_51'] = D51
res.loc[:,'D_52'] = D52
res.loc[:,'D_53'] = D53
res.loc[:,'D_54'] = D54
res.loc[:,'DT5'] = DT5
res.loc[:,'D_56'] = D56
res.loc[:,'D_61'] = D61
res.loc[:,'D_62'] = D62
res.loc[:,'D_63'] = D63
res.loc[:,'D_64'] = D64
res.loc[:,'D_65'] = D65
res.loc[:,'DT6'] = DT6
res.loc[:,'Z1'] = Z1
res.loc[:,'inp_1'] = inp_1
res.loc[:,'inp_2'] = inp_2
res.loc[:,'inp_3'] = inp_3
res.loc[:,'inp_4'] = inp_4
res.loc[:,'inp_5'] = inp_5
res.loc[:,'inp_6'] = inp_6

#Conditions in prior time step (initial conditions for each step)
res.loc[:,'aw_t'] = 1 - (res.x - numx)/numx
res.loc[:,'as_t'] = 0 #initial activity in the soil
#Boundary Conditions - Type 1 upstream
bc_us = 0.05 #Activity at the source
#Type 2 boundary downstream
bc_ds = 0

"""
From here, everything should be general and applicable in the class method
"""
#Calculate forward and backward facial values
#Back and forward facial Volumes (L³)
res.loc[1:reslen,'V1_b'] = (res.V1.shift(1) + res.V1)/2
res.loc[(slice(None), 0),'V1_b'] = res.loc[(slice(None),0),'V1']
res.loc[0:reslen-1,'V1_f'] = (res.V1.shift(-1) + res.V1)/2
res.loc[(slice(None), numx-1),'V1_f'] = res.loc[(slice(None),numx-1),'V1']
#Darcy's flux, q, (L/T)
res.loc[1:reslen,'q_b'] = (res.q.shift(1) + res.q)/2
res.loc[(slice(None), 0),'q_b'] = Qin/A
res.loc[0:reslen-1,'q_f'] = (res.q.shift(-1) + res.q)/2
res.loc[(slice(None), numx-1),'q_f'] = Qout/A
#Dispersivity disp [l²/T]
res.loc[1:reslen,'disp_b'] = (res.disp.shift(1) + res.disp)/2
res.loc[(slice(None), 0),'disp_b'] = res.loc[(slice(None), 0),'disp']
res.loc[0:reslen-1,'disp_f'] = (res.disp.shift(-1) + res.disp)/2
res.loc[(slice(None), numx-1),'disp_f'] = res.loc[(slice(None), numx-1),'disp']
#Activity/Fugacity capacity Z
res.loc[1:reslen,'Z1_b'] = (res.Z1.shift(1) + res.Z1)/2
res.loc[(slice(None), 0),'Z1_b'] = res.loc[(slice(None),0),'Z1']
res.loc[0:reslen-1,'Z1_f'] = (res.Z1.shift(-1) + res.Z1)/2
res.loc[(slice(None), numx-1),'Z1_f'] = res.loc[(slice(None),numx-1),'Z1']

#DISCUS algortithm semi-lagrangian 1D ADRE from Manson & Wallis (2000) DOI: 10.1016/S0043-1354(00)00131-7
#Outside of the time loop, if flow is steady, or inside if flow changes
dt = 3
res.loc[:,'c'] = res.q*dt/dx
#time it takes to pass through each cell
res.loc[:,'del_0'] = res.dx/((res.q_b + res.q_f)/2)
#Set up dummy variables to be used inside the loop
delb_test = pd.Series().reindex_like(res)
delb_test[:] = 0 #Challenger time, accepted if <= dt
#Time taken traversing full cells, not the final partial
delb_test1 = pd.Series().reindex_like(res) 
delrb_test = pd.Series().reindex_like(res)
delrb_test[:] = 0
#"Dumb" distance variable
xb_test = pd.Series().reindex_like(res)
xb_test[:] = 0
#This is a bit clunky, but basically this one will stay zero until del_test>dt
xb_test1 = pd.Series().reindex_like(res)
#Forward variables are the same as the backjward variables
#but they will be shifted one fewer times (dels instead of dels+1)
delf_test = delb_test.copy(deep = True)
delf_test1 = delb_test1.copy(deep = True)
delrf_test = delrb_test.copy(deep = True)
xf_test = xb_test.copy(deep = True)
xf_test1 = xb_test1.copy(deep = True)
#This loop calculates the distance & time backwards that a water packet takes
#in a time step. 
dels = 0
for dels in range(int(max(np.floor(res.c)))): #Max cells any go through (could be wrong if q increases)
    #Time to traverse a full cell
    delb_test += res.groupby(level = 0)['del_0'].shift(dels+1)
    delf_test += res.groupby(level = 0)['del_0'].shift(dels)
    #Calculate del_test1 only where a full spatial step is traversed
    delb_test1[delb_test<=dt] = delb_test[delb_test<=dt]
    delf_test1[delf_test<=dt] = delf_test[delf_test<=dt]
    #Do the same thing in reverse for delrb_test, if delrb_test is zero to prevent overwriting
    #Create a mask showing the cells that are finished
    maskb = (delb_test>dt) & (delrb_test==0)
    maskf = (delf_test>dt) & (delrf_test==0)
    delrb_test[maskb] = dt - delb_test1
    delrf_test[maskf] = dt - delf_test1
    #Using delrb_test and the Darcy flux of the current cell, calculate  Xb_test1
    xb_test1[maskb] = xb_test + delrb_test * res.groupby(level = 0)['q'].shift(dels+1)
    xf_test1[maskf] = xf_test + delrf_test * res.groupby(level = 0)['q'].shift(dels)
    #Then, update the "dumb" distance travelled
    xb_test += res.groupby(level = 0)['dx'].shift(dels+1)
    xf_test += res.groupby(level = 0)['dx'].shift(dels)
#Finally, do the last one last for the remaining NaNs & 0s
delrb_test[delrb_test==0] = dt - delb_test1
delrf_test[delrf_test==0] = dt - delf_test1
xb_test1[np.isnan(xb_test1)] = xb_test + delrb_test * res.groupby(level = 0)['q'].shift(dels+1)
xf_test1[np.isnan(xf_test1)] = xf_test + delrf_test * res.groupby(level = 0)['q'].shift(dels)        
#Bring what we need to res. The above could be made a function to clean things up too.
#Distance from the origin of the forward and back faces
res.loc[:,'xb'] = res.x - res.dx/2 - xb_test1
res.loc[:,'xf'] = res.x + res.dx/2 - xf_test1
#Clean up the US boundary, anything NAN or < 0 comes from before the origin
maskb = (np.isnan(res.xb) | (res.xb < 0))
maskf = (np.isnan(res.xf)) | (res.xf < 0)
res.loc[maskb,'xb'] = 0
res.loc[maskf,'xf'] = 0
#Define the cumulative mass along the entire length of the system as M(x) = sum (Mi)
#This is defined at the right hand face of each cell. M = ai*Zi*Vi, at time n
res.loc[:,'M_i'] = res.aw_t * res.Z1 * res.V1
res.loc[:,'M_n'] = res.groupby(level = 0)['M_i'].cumsum()
#Then, we advect one time step. To advect, just shift everything as calculated above.
#We will use a cubic interpolation. Unfortunately, we have to unpack the data 
#in order to get this to work.
chems = res.index.levels[0]
for ii in range(numchems):
    xx = res.loc[(chems[ii], slice(None)),'x']
    yy = res.loc[(chems[ii], slice(None)),'M_n']
    f = interp1d(xx,yy,kind='cubic',bounds_error = False)
    res.loc[(chems[ii], slice(None)),'M_star'] = f(res.loc[(chems[ii], slice(None)),'xf'])\
    - f(res.loc[(chems[ii], slice(None)),'xb'])
    #US boundary conditions
#Case 1 - both xb and xf are outside the domain
mask = (res.xb == 0) & (res.xf == 0)
res.loc[mask,'M_star'] = Qin*bc_us*dt
#Case 2 - xb is out of the range, but xf is in
#Need to compute the sum of a spatial integral from x = 0 to xf and then the rest is temporal
mask = (res.xb == 0) & (res.xf != 0)
slope = np.array(res.M_n[(slice(None),0)])/np.array((res.dx[(slice(None),0)]))
M_x = slope*np.array(res.xf[mask])
#Time integration
M_t =  (dt-np.array(res.xf[mask]/res.q[mask]))*Qin*bc_us
res.loc[mask,'M_star'] = M_x + M_t
#Case 3 - too close to origin for cubic interpolation, so we will use linear interpolation
mask = np.isnan(res.M_star)
res.loc[mask,'M_star'] = slope * (res.xf[mask] - res.xb[mask])
#Divide out to get back to activity/fugacity entering from advection
res.loc[:,'a_star'] = res.M_star / res.Z1 / res.V1

#Finally, we can now set up & solve our implicit portion. 
#We are using a method based on those of Manson and Wallis (1995) and Kilic & Aral (2009)
#Define the spatial weighting term (P) 
res.loc[:,'P'] =dt/(res.dx)
#Now define the spacial weighting terms as f, m, & b. 
#b for the (i-1) spacial step, m for (i), f for (i+1)
#the back (b) term acting on x(i-1)
res.loc[:,'b'] = 2*res.P*res.V1_b*res.Z1_b*res.disp_b/(res.dx + res.groupby(level = 0)['dx'].shift(1))
#To deal with the upstream boundary condition, we can simply set dx(i-1) = dx so that:
res.loc[(slice(None),0),'b'] = 2*res.P*res.V1_b*res.Z1_b*res.disp_b/(res.dx)
#forward (f) term acting on x(i+1)
res.loc[:,'f'] = 2*res.P*res.V1_f*res.Z1_f*res.disp_f/(res.dx + res.groupby(level = 0)['dx'].shift(-1))
res.loc[(slice(None),numx-1),'f'] = 0
#Middle (m) term acting on x(i) - this will be subracted in the matrix (-m*ai)
#Upstream and downstream BCs have been dealt with in the b and f terms
res.loc[:,'m'] = res.f+res.b+dt*res.V1*res.Z1*res.DT1
#These will make the matrix. For each spacial step, i, there will be
#numc activities that we will track. So, in a system of water, air and sediment
#you would have aw1, as1, aa1, aw2,as2,aa3...awnumc,asnumc,aanumc, leading to a matrix
#of numc * i x numc * i in dimension.
#Initialize transport matrix and RHS vector (inp)
mat = np.zeros([numx*numc,numx*numc,numchems])
inp = np.zeros([numx*numc,numchems])
#FILL THE MATRICES
i,j,k = [0,0,0]
#First, define where the matrix values will go.
m_vals = np.arange(0,numx*numc,numc)
b_vals = np.arange(numc,numx*numc,numc)
start = time.time()
#Then, we can set the ADRE terms. Since there will always be three no need for a loop.
mat[m_vals,m_vals,:] = -np.array(res.m).reshape(numchems,numx).swapaxes(0,1)
mat[b_vals,m_vals[0:numx-1],:] = np.array(res.loc[(slice(None),slice(1,numx)),'b']).reshape(numchems,numx-1).swapaxes(0,1)
mat[m_vals[0:numx-1],b_vals,:] = np.array(res.loc[(slice(None),slice(0,numx-2)),'f']).reshape(numchems,numx-1).swapaxes(0,1)
#Next, set D values and inp values
for j in range(0,numc): #j is the row index
    for k in range(0,numc): #k is the column index
        if (j == k): 
            if j == 1:#Skip DT1 as it is in the m value
                pass
            else: #Otherwise, place the DT values in the matrix
                D_val = 'DT' + str(k+1)
                mat[m_vals+j,m_vals+j,:] = -np.array(res.loc[:,D_val]).reshape(numchems,numx).swapaxes(0,1)
        else: #Place the intercompartmental D values
            D_val = 'D_' +str(k+1)+str(j+1)
            mat[m_vals+j,m_vals+k,:] = -np.array(res.loc[:,D_val]).reshape(numchems,numx).swapaxes(0,1)
#Upstream boundary - need to add the diffusion term. DS boundary is dealt with already
inp[0,:] += -res.b[slice(None),0]*bc_us
end = time.time()
print(end - start)
start = time.time()
#maskb = mat[numx/numc] +
for i in range(0,numx-1):
    if i == 588:
        x = 1
    if (i-1)*numc >= 0: #No b term at US boundary
        mat[i*numc,(i-1)*numc,:] = res.b[slice(None),i]
    mat[i*numc,i*numc,:] = -res.m[slice(None),i] #M BCs already dealt with
    if (i+1)*numc <=(numx-1): #No f term at DS boundary
        mat[i*numc,(i+1)*numc,:] = res.f[slice(None),i]
    inp[i*numc,:] = res.M_i[slice(None),i] - res.M_star[slice(None),i] - res.inp_1[slice(None),i]
    for j in range(2,numc+1):
        D_val = 'D_' +str(j)+str(1)
        mat[i*numc,i*numc+j-1,:] = res.loc[(slice(None),i),D_val]
        inp_val = 'inp_' +str(j)
        inp[i*numc+j-1,:] = -res.loc[(slice(None),i),inp_val] #Inter-compartment transfer just the regular inputs
        for k in range(1,numc+1):
            if j == k:
                D_val = 'DT' +str(k)
                mat[i*numc+j-1,i*numc+k-1,:] = -1*res.loc[(slice(None),i),D_val]
            else:
                D_val = 'D_' +str(k)+str(j)
                mat[i*numc+j-1,i*numc+k-1,:] = res.loc[(slice(None),i),D_val]
#Upstream boundary - need to add the diffusion term. DS boundary is dealt with in the loop
inp[0,:] += -res.b[slice(None),0]*bc_us
end = time.time()
print(end - start)
#Now, we will solve the matrix for each compound in turn (might be possible to do simultaneously)
for ii in range(numchems):
    chem = chems[ii]
    LHS = mat[:,:,ii]
    RHS = inp[:,ii]
    matsol = np.linalg.lstsq(LHS,RHS)
    matsol = np.linalg.solve(mat[:,:,ii],inp[:,ii])
    #Put in to the results array
    res.loc[(chem,slice(None)),'a_n1'] = matsol
#res.loc[:,'matsol'] = matsol
end = time.time()
print(end - start)
        
        
        
        











