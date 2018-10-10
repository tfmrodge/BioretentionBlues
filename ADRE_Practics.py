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
inp_1 = 1
inp_2 = 2
inp_3 = 3
inp_4 = 4
inp_5 = 5
inp_6 = 6
#Initialize length variable
numc = len(chems)
res = pd.DataFrame(np.arange(0,L+dx,dx),columns = ['x'])
#Define the x term as the centre of each cell
res.loc[:,'x'] = res.x+dx/2
res1 = res
res2 = res
res3 = res
res4 = res
res = pd.concat([res,res1,res2,res3,res4], keys=chems)
numx = len(res.x)/numc
reslen = len(res.x)
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
res.loc[:,'inp_1'] = inp_1
res.loc[:,'inp_2'] = inp_2
res.loc[:,'inp_3'] = inp_3
res.loc[:,'inp_4'] = inp_4
res.loc[:,'inp_5'] = inp_5
res.loc[:,'inp_6'] = inp_6

#Conditions in prior time step (initial conditions for each step)
res.loc[:,'aw_t'] = 11 - (res.x - numx)/numx
res.loc[:,'as_t'] = 0 #initial activity in the soil
#Boundary Conditions - Type 1 upstream
bc_us = 1 #Activity at the source
#Type 2 boundary downstream
bc_ds = 0

"""
From here, everything should be general and applicable in the class method
"""
#Calculate forward and backward facial V, q and disp
#Back and forward facial volumes (L³)
res.loc[1:reslen,'V1_b'] = (res.V1.shift(1) + res.V1)/2
res.loc[(slice(None), 0),'V1_b'] = res.loc[(slice(None),0),'V1']
res.loc[0:reslen-1,'V1_f'] = (res.V1.shift(-1) + res.V1)/2
res.loc[(slice(None), numx-1),'V1_f'] = res.loc[(slice(None),numx-1),'V1']
#Darcy's flux (L/T)
res.loc[1:reslen,'q_b'] = (res.q.shift(1) + res.q)/2
res.loc[(slice(None), 0),'q_b'] = res.loc[(slice(None),0),'q']
res.loc[0:reslen-1,'q_f'] = (res.q.shift(-1) + res.q)/2
res.loc[(slice(None), numx-1),'q_f'] = res.loc[(slice(None), numx-1),'q']

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
#And we can deal with those boundary conditions. Everything at the US boundary
#starts before the origin, so we will just define it as zero.
#xb_test1[np.isnan(xb_test1)] = 0
#xf_test1[np.isnan(xf_test1)]  = 0
#Bring what we need to res. The above could be made a function to clean things up too.
#Distance from the origin of the forward and back faces
res.loc[:,'xb'] = res.x - xb_test1
res.loc[:,'xf'] = res.x + res.dx - xf_test1
#Clean up the US boundary, anything NAN or zero comes from before the origin
res.loc[0:dels+1,'xb'] = 0
res.loc[0:dels,'xf'] = 0
#Now we define the cumulative mass along the entire length of the system as M(x) = sum (Mi)



#Distance in x that the CV faces move through in time step dt
res.loc[:,'xb'] = 0
res.loc[:,'xf'] = 0
#Time past last cell back
res.loc[:,'del_rb'] = 0
res.loc[:,'del_rf'] = 0

res.loc[:,'del_rb'] = dt - del_test1
    res.loc[:,'del_tf'] += res.groupby(level = 0)['del_0b'].shift(dels+1)
    delrb_test = dt - del_t
    res.loc[:,'del_rb'] += dt -  res.groupby(level = 0)['del_0b'].shift(dels+1)
    res.loc[:,'xb'] += res.groupby(level = 0)['del_0b'].shift(dels+1)

    
res.loc[:,'c_trunc'] = np.floor(res.c)
#Remnant courant number c_alpha. This is always <1
res.loc[:,'c_alpha'] = res.c - res.c_trunc
#Now, we need to determine the back and forward faces for each CV
#For this, we can calculate the travel time across each previous cell as:
d = {}
for dels in range(int(max(np.floor(res.c)-1))): #Pass fully through floor(c)-1 cells
    d["del_{0}".format(dels+1)] = res.groupby(level = 0)['q'].shift(dels+1)\
    /res.groupby(level = 0)['dx'].shift(dels+1)
res = pd.concat([res,pd.DataFrame(d)],axis = 1, sort = False)
res.loc[:,'del_t']

#Dispersivity [l²/T]
res.loc[1:reslen,'disp_b'] = (res.disp.shift(1) + res.disp)/2
res.loc[(slice(None), 0),'disp_b'] = res.loc[(slice(None), 0),'disp']
res.loc[0:reslen-1,'disp_f'] = (res.disp.shift(-1) + res.disp)/2
res.loc[(slice(None), numx-1),'disp_f'] = res.loc[(slice(None), numx-1),'disp']

#Start solving the QUICKEST algorithm for time t+1, from time t. 
#First, solve for the aw_f (front) for each x
#Internal CVs
res.loc[1:reslen,'aw_f'] = 0.5*(res.aw_t.shift(-1)+res.aw_t) - res.c/2*((res.aw_t.shift(-1)-res.aw_t)\
       - 1/6 *((1-res.c**2)*((res.aw_t.shift(-1)-2*res.aw_t+res.aw_t.shift(1)))))
#Upstream boundary
res.loc[(slice(None), 0),'aw_f'] = np.array(0.5*(res.aw_t[slice(None),1]+res.aw_t[slice(None),0])\
 - res.c[0]/2*((res.aw_t[slice(None),1]-res.aw_t[slice(None),0])-1/6 \
 *((1-res.c[slice(None),0]**2)*((res.aw_t[slice(None),1]-2*res.aw_t[slice(None),0]+bc_us)))))
#Downstream boundary
res.loc[(slice(None),numx-1),'aw_f'] = np.array(res.aw_t[slice(None),numx-1]-res.c[slice(None),numx-1]/2\
 *((1/6)*(1-res.c[slice(None),numx-1]**2)*(res.aw_t[slice(None),numx-1]-2\
    *res.aw_t[slice(None),numx-1]+res.aw_t[slice(None),numx-2])))
#Next, solve fore aw_b (back) for each x
#Internal CVs - no need for DS calculation
res.loc[2:reslen,'aw_b'] = 0.5*(res.aw_t+res.aw_t.shift(1)) - res.c/2*((res.aw_t-res.aw_t.shift(1))\
       - 1/6 *((1-res.c**2)*((res.aw_t-2*res.aw_t.shift(1)+res.aw_t.shift(2)))))
#Upstream boundary
res.loc[(slice(None),0),'aw_b'] = np.array(0.5*(res.aw_t[slice(None),0]+bc_us) - res.c[slice(None),0]\
/2*((res.aw_t[slice(None),0]-bc_us) - 1/6 *((1-res.c[slice(None),0]**2)*\
       ((res.aw_t[slice(None),0]-2*bc_us+bc_us)))))
#Cell 1, need to solve as  this one goes 2 cells back
res.loc[1,'aw_b'] = 0.5*(res.aw_t[1]+res.aw_t[0]) - res.c[1]/2*((res.aw_t[1]-res.aw_t[0])\
       - 1/6 *((1-res.c[0]**2)*((res.aw_t[1]-2*res.aw_t[0]+bc_us))))


#Then we can solve for (aZV)*
#Time and space weighting term, P
res.loc[0,'P'] =dt/(res.dx[0])
res.loc[1:numx-1,'P'] = 2*dt/(res.dx.shift(1) + res.dx)
#Interior cells
res.loc[0:numx-1,'aZV*'] = res.aw_t*Z1*res.V1 + res.P*(res.aw_f*Z1*res.V1_f*res.q_f\
    -res.aw_b*Z1*res.V1_b*res.q_b)
#Finally, we can now set up & solve our implicit portion. 
#Let's define the wpacial weighting terms as f, m, b. b will be
#for the (i-1) spacial step, m for (i), f for (i+1)
#the back (b) term
res.loc[0,'b'] = res.P[0]*res.V1_b[0]*Z1*res.disp_b[0]/res.dx[0]
res.loc[1:numx-1,'b'] = res.P*res.V1_b*Z1*res.disp_b/res.dx.shift(1)
#forward (f) term
res.loc[0:numx-1,'f'] = res.P*res.V1_f*Z1*res.disp_f/res.dx
#Middle (m) term
res.loc[0:numx-1,'m'] = res.P*(-res.f-res.b) - dt*res.DT1
#Upstream boundary condition m_star
m_star = res.m[0]
#downstream boundary condition m_etoile
#Currently not sure if the bc_ds works other than as an on/off switch
m_etoile = res.P[numx-1]*(-res.f[numx-1]*bc_ds-res.b[numx-1]) - dt*res.DT1[numx-1]
#These will make the matrix. For each spacial step, i, there will be
#numc activities that we will track. So, in a system of water, air and sediment
#you would have aw1, as1, aa1, aw2,as2,aa3...awnumc,asnumc,aanumc, leading to a matrix
#of numc * i x numc * i in dimension.
#Initialize transport matrix and RHS vector (inp)
mat = np.zeros([numx*numc,numx*numc])
inp = np.zeros([numx*numc,1])
#mat = np.array(mat)
#FILL THE MATRIX
for i in range(0,numx-1):
    if (i-1)*numc >= 0:
        mat[i*numc,(i-1)*numc] = res.b[i]
    mat[i*numc,i*numc] = res.m[i]
    mat[i*numc,(i+1)*numc] = res.f[i]
    inp[i*numc] = -res.inp_1[i]
    for j in range(2,numc+1):
        D_val = 'D_' +str(j)+str(1)
        mat[i*numc,i*numc+j-1] = res.loc[i,D_val]
        inp_val = 'inp_' +str(j)
        inp[i*numc+j-1] = -res.loc[i,inp_val]
        for k in range(1,numc+1):
            if j == k:
                D_val = 'DT' +str(k)
                mat[i*numc+j-1,i*numc+k-1] = -1*res.loc[i,D_val]
            else:
                D_val = 'D_' +str(k)+str(j)
                mat[i*numc+j-1,i*numc+k-1] = res.loc[i,D_val]
#Upstream boundary
mat[0,0] = m_star
inp[0,0] = -res.loc[0,'inp_1'] - res.f[0]*bc_us
#for i in range(2,numc+1):
#    D_val = 'D_' +str(i)+str(1)
#    mat[0,i-1] = res.loc[i,D_val]
#Downstream Boundary - have to subtract numc as compartments are in line
mat[numx*numc-numc,numx*numc-2*numc] = res.b[i]
mat[numx*numc-numc,numx*numc-numc] = m_etoile
inp[numx*numc-numc] = -res.loc[numx-numc,'inp_1']
for j in range(2,numc+1):
    D_val = 'D_' +str(j)+str(1)
    mat[numx*numc-numc,numx*numc-numc+j-1] = res.loc[i,D_val]
    inp_val = 'inp_' +str(j)
    inp[numx*numc-numc+j-1] = -res.loc[numx-numc+j-1,inp_val]
    for k in range(1,numc+1):
        if j == k:
            D_val = 'DT' +str(k)
            mat[numx*numc-numc+j-1,numx*numc-numc+k-1] = -1*res.loc[i,D_val]
        else:
            D_val = 'D_' +str(k)+str(j)
            #I should probably simplify this but ¯\_(ツ)_/¯
            mat[numx*numc-numc+j-1,numx*numc-numc+k-1] = res.loc[i,D_val]
matsol = np.linalg.solve(mat,inp)
#res.loc[:,'matsol'] = matsol
end = time.time()
print(end - start)
        
        
        
        











