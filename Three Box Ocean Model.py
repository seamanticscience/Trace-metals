# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:19:20 2021

@author: Smah Riki
"""
import numpy as np
import matplotlib.pyplot as plt

## Establishing Dimensions of Three Boxes (all values have units of meters)
# Dimensions of Box 1
dx_1 = 17.0*10**6
dy_1 = 8.0*10**6
dz_1 = 100.0
vol_1 = dx_1*dy_1*dz_1

# Dimensions of Box 2
dx_2 = 17.0*10**6
dy_2 = 8.0*10**6
dz_2 = 100.0
vol_2 = dx_2*dy_2*dz_2

# Dimensions of Box 3 (Deep Ocean Layer)
dx_3 = 17.0*10**6
dy_3 = 16.0*10**6
dz_3 = 5000.0
vol_3 = dx_3*dy_3*dz_3


## Establishing Initial Values for Dynamic Quantities (i.e. flow rates, all
## the following quantities have units of m^3*s^-1)
psi = 20*10**6
k_13 = 1.0*10**6
k_31 = 1.0*10**6
k_23 = 1.0*10**6
k_32 = 1.0*10**6
k_12 = 10.0*10**6
k_21 = 10.0*10**6


## Setting Initial Values for Concentrations in the Three Boxes 
## (units of moles C_i per cubic meter)

C_1 = 0.1
C_2 = 0.1
C_3 = 1.0


## Initiate Time Variables

t = 0
dt = 60*60*24*365  #Total number of seconds representing a year. 
dt_in_years = 1
end_time = 100 #Quantity in years, depicts end-time of simulation and, consequently, the graph.


## The following are functions that will output the values of dC_i/dt, for i in [1, 3].

def dC_1_over_dt():
    """
    Calculates change in concentration of C_1 per cubic meter, in units of 
    moles of C_1 per cubic meter per unit time.
    
    The quanities needed are stored in the variables defined earlier. 
    

    Returns
    -------
    Number quantity reflecting change of C_1 per unit time, governed by the flow
    rates and the concentrations at the given times.

    """
    return (psi*(C_3 - C_1) + k_31*(C_3 - C_1) + k_21*(C_2 - C_1))/vol_1

def dC_2_over_dt():
    """
    Calculates change in concentration of C_2 per cubic meter, in units of 
    moles of C_2 per cubic meter per unit time.
    
    The quanities needed are stored in the variables defined earlier. 
    

    Returns
    -------
    Number quantity reflecting change of C_2 per unit time, governed by the flow
    rates and the concentrations at the given times.

    """
    return (psi*(C_1 - C_2) + k_12*(C_1 - C_2) + k_32*(C_3 - C_2))/vol_2

def dC_3_over_dt():
    """
    Calculates change in concentration of C_3 per cubic meter, in units of 
    moles of C_3 per cubic meter per unit time.
    
    The quanities needed are stored in the variables defined earlier. 
    

    Returns
    -------
    Number quantity reflecting change of C_3 per unit time, governed by the flow
    rates and the concentrations at the given times.

    """
    return (psi*(C_2 - C_3) + k_23*(C_2 - C_3) + k_13*(C_1 - C_3))/vol_3


### Create arrays where the first array depicts the x-axis (time steps) and the three other
### arrays depict concentrations of C_1, C_2, and C_3 over the time steps. 

## Create x-axis array (time)

time_axis_list = [0,]
t_temp = dt_in_years

while t_temp <= end_time:
    time_axis_list.append(t_temp)
    t_temp += dt_in_years
        # At the end of this while loop, we will have a list of all time checkpoints for which
        # we want to calculate the three concentrations. 
time_axis_array = np.array(time_axis_list)

## Create y-axis arrays (C_1, C_2, and C_3)

C_1_list = [C_1,]
C_2_list = [C_2,]
C_3_list = [C_3,]
    # Initiate lists that will store the three concentrations, with initial concentrations already
    # in the lists.

for t_val in time_axis_array:
    if t_val == 0:
        pass
    else:
        C_1 += dC_1_over_dt()*dt
        C_1_list.append(C_1)
            # Use Euler Step Function to change value of C_1 by one time step (i.e. dt). Then 
            # append that value to the C_1_list of concentrations as the concentration for that
            # given time. 
        C_2 += dC_2_over_dt()*dt
        C_2_list.append(C_2)
            # Use Euler Step Function to change value of C_2 by one time step (i.e. dt). Then 
            # append that value to the C_2_list of concentrations as the concentration for that
            # given time. 
        C_3 += dC_3_over_dt()*dt
        C_3_list.append(C_3)
            # Use Euler Step Function to change value of C_3 by one time step (i.e. dt). Then 
            # append that value to the C_3_list of concentrations as the concentration for that
            # given time. 
C_1_array = np.array(C_1_list)
C_2_array = np.array(C_2_list)
C_3_array = np.array(C_3_list)
    # Once the above is complete, we now have three arrays depicting concentrations of C_1, C_2, and C_3
    # over time. 

### Below we plot the arrays.

plt.title('Concentrations of C_1, C_2, and C_3 over time')
plt.xlabel('Time (years)')
plt.ylabel('Concentration (mol C_i per cubic meter)')
plt.plot(time_axis_array, C_1_array, 'r', label = 'C_1')
plt.plot(time_axis_array, C_2_array, 'm', label = 'C_2')
plt.plot(time_axis_array, C_3_array, 'b', label = 'C_3')
plt.legend(loc = 'best')
plt.show()