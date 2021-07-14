# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:19:20 2021

@author: Smah Riki
"""
import numpy as np
import matplotlib.pyplot as plt

### -----------------------------------------------------------------------------

## Establishing Global Quantities

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


## Flow Rates for All Cases (units of m^3*s^-1)
psi = 20*10**6
k_13 = 1.0*10**6
k_31 = 1.0*10**6
k_23 = 1.0*10**6
k_32 = 1.0*10**6
k_12 = 10.0*10**6
k_21 = 10.0*10**6
rho_0 = 1024.5 #Units of kg per cubic meter, density of water

# Productivity Rates (Michaelis-Menten) for Nutrient Growth and Light
V_max = ((16*6*10**-3)/(360*86400)) # Units of mol N m-3 s-1
K_sat_N = 1.6*10**-3 #Units of 1.6e-3, mol N m-3
K_sat_l = 30 #Units of W m-2
Ibox1 = 35 #Units of W m-2
Ibox2 = 60 #Units of W m-2

# Global Values for Iron Cycle and Deposition
alpha = 0.01 #Fe dust solubility
R_Fe = 2.5*10**-5 #mol
K_sat_Fe = 2*10**-7 #units of mole per cubic meter
f_dop = 0.67 # Fraction of particles that makes it into pool of nitrate, unitless.


## The following functions abstract the process of creating the transport model
## (using the ODEs we have generalized to the system).

def create_transport_model(C_1, C_2, C_3, dt_in_years, end_time, title, element_symbol, lambda_1 = 0, lambda_2 = 0, \
                           mic_ment_nolight = 0, mic_ment_light_leibig = 0, mic_ment_light_mult_lim = 0, Ibox1 = 35, \
                               k_scav = 0, mu = 0, Fe_1 = None, Fe_2 = None, Fe_3 = None):
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
        mic_ment_nolight: int (0 or 1) where 1 means we are using the Michaelis–Menten model without considering light.
        mic_ment_light_leibig: int (0 or 1) where 1 means we are using the Michaelis–Menten model with light limitation
            and Leibig's law.
        mic_ment_light_mult_lim: int (0 or 1) where 1 means we are using the Michaelis–Menten model with light limitation
            and multiplicative limitation.
        Ibox1 = Value of intensity of light, given as a float of units W/m2.
            Parameter used solely for diagnostic models, default set to 35 W/m2
            for the average value in the Southern Ocean.
        k_scav: k value associated with scavenging of iron, default set to 0. Input in yr-1
        mu: units of s-1, time constant (where 1/mu is the characteristic transport of 
            nutrients, nitrate in this case.)
        Fe_1, Fe_2, Fe_3: Initial concentrations of iron in the three boxes (float), 
            quantities in moles. Initially set to None. 

    Returns
    -------
        Tuple of the form (array of floats to be used for the x-axis, \
                           (array of the concentrations over time for C_1, \
                            array of the concentrations over time for C_2, \
                                array of the concentrations over time for C_3))
        Plots graph with this information (not returned)
    """
    ## Initiate Time Variables
    
    dt = dt_in_years*60*60*24*365  #Total number of seconds representing a year. 
    
    ## Initiate Variables that will "globally," at least within this function, keep track of 
    ## concentrations of C_1 and C_2 to calculate C_3.
    light_dependent_change_in_C_1 = 0
    nutrient_dependent_change_in_C_1 = 0
    light_dependent_change_in_C_2 = 0
    nutrient_dependent_change_in_C_2 = 0
    
    
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
        # Establishing Michaelis-Menten quantities for light and nutrients.
        global light_dependent_change_in_C_1
        global nutrient_dependent_change_in_C_1
        light_dependent_change_in_C_1 = (Ibox1/(K_sat_l + Ibox1))
        nutrient_dependent_change_in_C_1 = ((C_1)/(K_sat_N + C_1))
            
        return (psi*(C_3 - C_1) + k_31*(C_3 - C_1) + k_21*(C_2 - C_1))/vol_1 \
            - lambda_1*C_1 \
                - mic_ment_nolight*(V_max/100)*((C_1)/(K_sat_N + C_1)) \
                    - mic_ment_light_leibig*V_max*min(light_dependent_change_in_C_1, nutrient_dependent_change_in_C_1) \
                        - mic_ment_light_mult_lim*V_max*light_dependent_change_in_C_1*nutrient_dependent_change_in_C_1 
                        
        # Line 1: General tracer equation, maintains equilibrium among all three boxes with flow rate considered.
        # Line 2: Given fixed export rate lambda_1, considers the box's export rate dependent on nutrient concentration in given box.
        # Line 3: NOT given fixed export, rate of export of organic matter from box 1 is governed by the Michaelis-Menten model.
            # We do not consider light in this model, and as such, we manually cut V_max by 100 to account for no light constraint.
        # Line 4: NOT given fixed export, rate of export of organic matter from box 1 is governed by the Michaelis-Menten model.
            # Light IS considered as a limiting factor, but we take the Liebig approach where we take the minimum value of the potential limiting factors.
        # Line 5: NOT given fixed export, rate of export of organic matter from box 1 is governed by the Michaelis-Menten model.
            # Light IS considered as a limiting factor, but we take the Multiplicative Limit approach where we multiply all limiting factors when considering export.
        
    def dFe_1_over_dt():
        """
        Calculates change in concentration of Fe_1 per cubic meter, in units of 
        moles of Fe_1 per cubic meter per unit time.
        
        The quanities needed are stored in the variables defined earlier. 
        
    
        Returns
        -------
        Number quantity reflecting change of Fe_1 per unit time, governed by the flow
        rates and the concentrations at the given times.
        """
        # Establishing F_in
        F_in1 = 0.071/(55.845*60*60*24*365)
            # The quanitity is initially provided in grams Fe per year. To convert this
            # quantity to mol Fe per second, divide by molar mass as well as the total number
            # of seconds in a year.
            
        # Calculating gamma, which is dependent on nutrient concentrations. 
        gamma_1 = C_1*(Fe_1/(Fe_1 + K_sat_Fe))
        
        return (psi*(Fe_3 - Fe_1) + k_31*(Fe_3 - Fe_1) + k_21*(Fe_2 - Fe_1))/vol_1 + \
            alpha*F_in1/vol_1 - k_scav*Fe_1/(55.845*60*60*24*365*vol_1) - mu*gamma_1*R_Fe
                
                # Line 1: General tracer equation, maintains equilibrium among all three boxes with flow rate considered.
                # Line 2: First term represents source, second term represents sink (in terms of being scavenged)
                    # Third term represents amount being used up ('biological utilization' as in Parekh, 2004)
                    
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
        # Establishing Michaelis-Menten quantities for light and nutrients.
        global light_dependent_change_in_C_2
        global nutrient_dependent_change_in_C_2
        light_dependent_change_in_C_2 = (Ibox2/(K_sat_l + Ibox2))
        nutrient_dependent_change_in_C_2 = ((C_2)/(K_sat_N + C_2))
        
        return (psi*(C_1 - C_2) + k_12*(C_1 - C_2) + k_32*(C_3 - C_2))/vol_2 \
            - lambda_2*C_2 \
                - mic_ment_nolight*V_max*((C_2)/(K_sat_N + C_2)) \
                    - mic_ment_light_leibig*V_max*min(light_dependent_change_in_C_2, nutrient_dependent_change_in_C_2) \
                        - mic_ment_light_mult_lim*V_max*light_dependent_change_in_C_2*nutrient_dependent_change_in_C_2
    
        # Line 1: General tracer equation, maintains equilibrium among all three boxes with flow rate considered.
        # Line 2: Given fixed export rate lambda_2, considers the box's export rate dependent on nutrient concentration in given box.
        # Line 3: NOT given fixed export, rate of export of organic matter from box 2 is governed by the Michaelis-Menten model.
        # Line 4: NOT given fixed export, rate of export of organic matter from box 2 is governed by the Michaelis-Menten model.
            # Light IS considered as a limiting factor, but we take the Liebig approach where we take the minimum value of the potential limiting factors.
        # Line 5: NOT given fixed export, rate of export of organic matter from box 2 is governed by the Michaelis-Menten model.
            # Light IS considered as a limiting factor, but we take the Multiplicative Limit approach where we multiply all limiting factors when considering export.
            
    def dFe_2_over_dt():
        """
        Calculates change in concentration of Fe_2 per cubic meter, in units of 
        moles of Fe_2 per cubic meter per unit time.
        
        The quanities needed are stored in the variables defined earlier. 
        
    
        Returns
        -------
        Number quantity reflecting change of Fe_2 per unit time, governed by the flow
        rates and the concentrations at the given times.
        """
        # Establishing F_in.
        F_in2 = 6.46/(55.845*60*60*24*365)
            # The quanitity is initially provided in grams Fe per year. To convert this
            # quantity to mol Fe per second, divide by molar mass as well as the total number
            # of seconds in a year.
            
        # Calculating gamma, which is dependent on nutrient concentrations. 
        gamma_2 = C_2*(Fe_2/(Fe_2 + K_sat_Fe))
        
        return (psi*(Fe_1 - Fe_2) + k_12*(Fe_1 - Fe_2) + k_32*(Fe_3 - Fe_2))/vol_2 + \
            alpha*F_in2/vol_2 - k_scav*Fe_2/(55.845*60*60*24*365*vol_2) - mu*gamma_2*R_Fe

        
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
                    # Establishing Michaelis-Menten quantities for light and nutrients.
                    # light_dependent_change_in_C_1 = (Ibox1/(K_sat_l + Ibox1))
                    # nutrient_dependent_change_in_C_1 = ((C_1)/(K_sat_N + C_1))
                    # light_dependent_change_in_C_2 = (Ibox2/(K_sat_l + Ibox2))
                    # nutrient_dependent_change_in_C_2 = ((C_2)/(K_sat_N + C_2))
        
        return (psi*(C_2 - C_3) + k_23*(C_2 - C_3) + k_13*(C_1 - C_3))/vol_3 + \
            (lambda_1*C_1*vol_1 + lambda_2*C_2*vol_2)/vol_3 + \
                mic_ment_nolight*(V_max/100*((C_1)/(K_sat_N + C_1))*vol_1 + V_max*((C_2)/(K_sat_N + C_2))*vol_2)/vol_3 + \
                    mic_ment_light_leibig*V_max/vol_3*(vol_1*min(light_dependent_change_in_C_1, nutrient_dependent_change_in_C_1) + vol_2*min(light_dependent_change_in_C_2, nutrient_dependent_change_in_C_2)) + \
                        mic_ment_light_mult_lim*V_max/vol_3*(vol_1*light_dependent_change_in_C_1*nutrient_dependent_change_in_C_1 + vol_2*light_dependent_change_in_C_2*nutrient_dependent_change_in_C_2)

        # Line 1: General tracer equation, maintains equilibrium among all three boxes with flow rate considered.
        # Line 2: Given fixed export rate lambda_1 and lambda_2, considers the box's export rate dependent on nutrient concentration in given box.
            # For this particular box, the export received from the other boxes goes to this box.
        # Line 3: NOT given fixed export, rate of export of organic matter from box 1 and 2 is governed by the Michaelis-Menten model.
            # We do not consider light in this model, and as such, we manually cut V_max by 100 to account for no light constraint in box 1
            # and V_max in itself for box 2.
        # Line 4: NOT given fixed export, rate of export of organic matter from box 1 and 2 is governed by the Michaelis-Menten model.
            # Light IS considered as a limiting factor, but we take the Liebig approach where we take the minimum value of the potential limiting factors.
        # Line 5: NOT given fixed export, rate of export of organic matter from box 1 and 2 is governed by the Michaelis-Menten model.
            # Light IS considered as a limiting factor, but we take the Multiplicative Limit approach where we multiply all limiting factors when considering export.
            
    def dFe_3_over_dt():
        """
        Calculates change in concentration of Fe_3 per cubic meter, in units of 
        moles of Fe_3 per cubic meter per unit time.
        
        The quanities needed are stored in the variables defined earlier. 
        
    
        Returns
        -------
        Number quantity reflecting change of Fe_2 per unit time, governed by the flow
        rates and the concentrations at the given times.
        """
        # Calculating gamma_3
        gamma_3 = C_3*(Fe_3/(Fe_3 + K_sat_Fe))
        
        return (psi*(Fe_2 - Fe_3) + k_23*(Fe_2 - Fe_3) + k_13*(Fe_1 - Fe_3))/vol_3 \
            + gamma_3*(R_Fe**2)*(1 - f_dop) - k_scav*Fe_3/(55.845*60*60*24*365*vol_3)
    
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
    if Fe_1 != None and Fe_2 != None and Fe_3 != None:
        Fe_1_list = [Fe_1,]
        Fe_2_list = [Fe_2,]
        Fe_3_list = [Fe_3,]
    else:
        Fe_1_list = [0,]
        Fe_2_list = [0,]
        Fe_3_list = [0,]
        # Initiate lists that will hold concentration values of iron.
    
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
            if Fe_1 != None and Fe_2 != None and Fe_3 != None:
                Fe_1 += dFe_1_over_dt()*dt
                Fe_1_list.append(Fe_1)
                    # Iron in box 1.
                Fe_2 += dFe_2_over_dt()*dt
                Fe_2_list.append(Fe_2)
                    # Iron in box 2.
                Fe_3 += dFe_3_over_dt()*dt
                Fe_3_list.append(Fe_3)
                    # Iron in box 3.
                
    C_1_array = np.array(C_1_list)
    C_2_array = np.array(C_2_list)
    C_3_array = np.array(C_3_list)
        # Once the above is complete, we now have three arrays depicting concentrations of C_1, C_2, and C_3
        # over time. 
    Fe_1_array = np.array(Fe_1_list)
    Fe_2_array = np.array(Fe_2_list)
    Fe_3_array = np.array(Fe_3_list)
    
    def plot_concentrations(title, element_symbol, time_axis_array, array_of_C_1, array_of_C_2, array_of_C_3, \
                            array_of_Fe_1, array_of_Fe_2, array_of_Fe_3):
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
        if array_of_Fe_1.all() != np.array([0,]) and array_of_Fe_2.all() != np.array([0,]) and array_of_Fe_1.all() != np.array([0,]):
            plt.plot(time_axis_array, array_of_Fe_1, 'r-.', label = 'Fe_1')
            plt.plot(time_axis_array, array_of_Fe_2, 'm-.', label = 'Fe_2')
            plt.plot(time_axis_array, array_of_Fe_3, 'b-.', label = 'Fe_3')
        plt.legend(loc = 'best')
        plt.show()
        
    plot_concentrations(title, element_symbol, time_axis_array, \
                    C_1_array, C_2_array, C_3_array, \
                        Fe_1_array, Fe_2_array, Fe_3_array)
        # Plotting the concentrations and how they change over time. 
        
#    return (time_axis_array, (C_1_array, C_2_array, C_3_array), \
#           (Fe_1_array, Fe_2_array, Fe_3_array))
    

### --------------------------------------------------------------------------------
### The following section calls the above functions for plotting purposes.


## Part 1: Creating Transport Model for Generic Concentration

# Concentrations of 1.0 for all three boxes:
    
transport_model_info = \
    create_transport_model(1.0, 1.0, 1.0, 1, 100, \
                           'Concentrations of C_1, C_2, and C_3, initially all 1.0,' + ' dt = 1', 'C')
        # The tuple above is of the form (time_array, (C1_array, C2_array, C3_array)).

# Concentrations of 0.1 for boxes 1 and 2, and 1.0 for box 3

transport_model_plot = \
    create_transport_model(0.1, 0.1, 1.0, 1, 100, \
                           'Concentrations of C_1 = 0.1, C_2 = 0.1, and C_3 = 1.0 initially,' + ' dt = 1', 'C')

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
    
transport_model_info = \
    create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.001, 10, \
                           'Concentrations of N_1, N_2, and N_3 w/ exports, dt = 0.001', 'N',  \
                               3*10**-8, 3*10**-7)

# -------------------------------------------------------------------------------------------------------------------
# Michaelis-Menten Model, not considering effects of light.

transport_model_graphing = \
    create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.001, 100, \
                           'Concentrations of N_1, N_2, and N_3 w/ exports, dt = 0.001, variable export rate \n (Michaelis-Menten, 100*V_max_1 = V_max_2)', 'N',  \
                               mic_ment_nolight = 1)

# -------------------------------------------------------------------------------------------------------------------
# Michaelis-Menten Model, considering effects of light and Leibig's Law.

transport_model_graphing = \
    create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.001, 8, \
                           'Concentrations of N_1, N_2, and N_3 w/ exports, dt = 0.001, variable export rate \n (Michaelis-Menten, V_max_1 = V_max_2, light-limited, Leibig)', 'N',  \
                               mic_ment_light_leibig = 1)

# -------------------------------------------------------------------------------------------------------------------
# Michaelis-Menten Model, considering effects of light and the Multiplicative Law.

transport_model_graphing = \
    create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.001, 8, \
                           'Concentrations of N_1, N_2, and N_3 w/ exports, dt = 0.001, variable export rate \n (Michaelis-Menten, V_max_1 = V_max_2, light-limited, Multiplicative Law)', 'N',  \
                               mic_ment_light_mult_lim = 1)

# -------------------------------------------------------------------------------------------------------------------
# Michaelis-Menten Model, considering effects of light and the two limit laws, but with Ibox1 = 0.1 W/m2

transport_model_graphing_leibig = \
    create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.001, 100, \
                           'Concentrations of N_1, N_2, and N_3 w/ exports, dt = 0.001, variable export rate \n (Michaelis-Menten, V_max_1 = V_max_2, light-limited, Leibig, \n Ibox1 = 0.1)', 'N',  \
                               mic_ment_light_leibig = 1, Ibox1 = 0.1)
        
transport_model_graphing_mult_law = \
    create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.001, 100, \
                           'Concentrations of N_1, N_2, and N_3 w/ exports, dt = 0.001, variable export rate \n (Michaelis-Menten, V_max_1 = V_max_2, light-limited, Multiplicative Law, \n Ibox1 = 0.1)', 'N',  \
                               mic_ment_light_mult_lim = 1, Ibox1 = 0.1)

# -------------------------------------------------------------------------------------------------------------------
# Michaelis-Menten Model, with Iron and the Net Scavenging Model (Case I in Parekh, 2004)

Fe_1_init = 5*10**-10 #0.5 nanomols of Fe per unit volume
Fe_2_init = 5*10**-10
Fe_3_init = 5*10**-10

transport_model_graphing_mult_law = \
    create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.006849, 10000, \
                           'Concentrations of N_1, N_2, N_3, Fe_1, Fe_2, Fe_3 w/ exports, \n dt = 0.001, variable export rate, Michalis-Menten Model, Leibig Limit Approximation', 'N', \
                               mic_ment_light_leibig = 1, k_scav = 0.004, mu = 3.858*10**-7, \
                                   Fe_1 = Fe_1_init, Fe_2 = Fe_2_init, Fe_3 = Fe_3_init)
                                        # Time step of 2.5 days