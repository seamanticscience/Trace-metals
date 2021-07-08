# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:19:20 2021

@author: Smah Riki
"""
import numpy as np
import matplotlib.pyplot as plt

### -----------------------------------------------------------------------------

## Establishing Global Quantities

psi = 20*10**6
k_13 = 1.0*10**6
k_31 = 1.0*10**6
k_23 = 1.0*10**6
k_32 = 1.0*10**6
k_12 = 10.0*10**6
k_21 = 10.0*10**6
rho_0 = 1024.5 #Units of kg per cubic meter, density of water

## The following functions abstract the process of creating the transport model
## (using the ODEs we have generalized to the system).

def create_transport_model(C_1, C_2, C_3, dt_in_years, end_time, title, element_symbol, lambda_1 = 0, lambda_2 = 0):
    """
    Uses first order ODEs to characterize the time dependence of the concentration(s)
    in the three boxes (of fixed dimension).
    
    Parameters:
    -----------
        C_1, C_2, C_3: Initial concentrations in the three boxes (float), 
            quantities in mols per cubic meter.
        dt_in_years: float, length of time step
        end_time: int, number of years for which we want to process the simulation
        title: str, title to be given to graph.
        element_symbol: element to be considered (e.g. N for nitrogen) (str)
        lambda_1, lambda_2: float, rate coefficient relating consumption and export
            (s**-1). Set to 0 by default unless specified. 

    Returns
    -------
        Tuple of the form (array of floats to be used for the x-axis, \
                           (array of the concentrations over time for C_1, \
                            array of the concentrations over time for C_2, \
                                array of the concentrations over time for C_3))
        Plots graph with this information (not returned)
    """

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
    
 
    ## Initiate Time Variables
    
    dt = dt_in_years*60*60*24*365  #Total number of seconds representing a year. 
    
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
        return (psi*(C_3 - C_1) + k_31*(C_3 - C_1) + k_21*(C_2 - C_1))/vol_1 - lambda_1*C_1
    
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
        return (psi*(C_1 - C_2) + k_12*(C_1 - C_2) + k_32*(C_3 - C_2))/vol_2 - lambda_2*C_2
    
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
        return (psi*(C_2 - C_3) + k_23*(C_2 - C_3) + k_13*(C_1 - C_3))/vol_3 + \
            (lambda_1*C_1*vol_1 + lambda_2*C_2*vol_2)/vol_3
    
    
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
                # The initial concentration values (at time t_val = 0) have already been
                # put into the lists, so if the time is 0 you do not have to do anything. 
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
    
    def plot_concentrations(title, element_symbol, time_axis_array, array_of_C_1, array_of_C_2, array_of_C_3):
        """
        Plots the concentrations of material in the three boxes and their
        change over time
    
        Parameters:
        -----------
            title: an str object titling the graph
            time_array: array of ints representing the time scale of the graph.
            array_of_C_1: array of numbers detailing levels of C_1 over time.
            array_of_C_2 and array_of_C_3: as detailed above.
        
        Returns:
        -------
            None. Plots graph with above information.
    
        """
        plt.title(f'{title}')
        plt.xlabel('Time (years)')
        plt.ylabel(f'Concentration (mol {element_symbol}_i per cubic meter)')
        plt.plot(time_axis_array, array_of_C_1, 'r', label = f'{element_symbol}_1')
        plt.plot(time_axis_array, array_of_C_2, 'm', label = f'{element_symbol}_2')
        plt.plot(time_axis_array, array_of_C_3, 'b', label = f'{element_symbol}_3')
        plt.legend(loc = 'best')
        plt.show()
        
        
    plot_concentrations(title, element_symbol, time_axis_array, \
                    C_1_array, C_2_array, C_3_array)
        # Plotting the concentrations and how they change over time. 
    return (time_axis_array, (C_1_array, C_2_array, C_3_array))
    

### --------------------------------------------------------------------------------
### The following section calls the above functions for plotting purposes.


## Part 1: Creating Transport Model for Generic Concentration

# Concentrations of 1.0 for all three boxes:
    
transport_model_info = create_transport_model(1.0, 1.0, 1.0, 1, 100, 'Concentrations of C_1, C_2, and C_3, initially all 1.0,' \
                                              + ' dt = 1', 'C')
    # The tuple above is of the form (time_array, (C1_array, C2_array, C3_array)).

# Concentrations of 0.1 for boxes 1 and 2, and 1.0 for box 3

transport_model_plot = create_transport_model(0.1, 0.1, 1.0, 1, 100, 'Concentrations of C_1 = 0.1, C_2 = 0.1, and C_3 = 1.0 initially,' \
                                              + ' dt = 1', 'C')

# -----------------------------------------------------------------------------------------------------------------------------------------
# Part 2: DIC and air-sea exchange of CO2. 
DIC_1_to_3 = 2000.0*rho_0*10**(-6) 
    # After proper conversions, the above essentially represents C_1 to C_3, with units of 
    # mol per cubic meter. 

### To be continued (maybe).

# -------------------------------------------------------------------------------------------------------------------
# Part 3: Representing Soft Tissue Pump

N_1_to_3 = 30*rho_0*10**(-6) 
    # After proper conversions, the above essentially represents N_1 to N_3, with units of 
    # mol per cubic meter.
    
transport_model_info = create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.001, 10, 'Concentrations of N_1, N_2, and N_3 w/ exports, dt = 0.001', 'N',  3*10**-7, 3*10**-8)
