# -*- coding: utf-8 -*-

"""
Created on Mon Jul  5 17:19:20 2021
@author: Smah Riki
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import itertools

### -----------------------------------------------------------------------------

## Creating Class to Solve ODEs

class solve_ode_timestep(object):
    pass

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

# Global Values for metal Cycle and Deposition
alpha = 0.01*0.035 #M dust solubility
R_Fe = (2.5*10**-5)*6.625 #unitless, multiplied by 6.625 to convert this M:C ratio to M:N
K_sat_Fe = 2*10**-7 #units of mole per cubic meter
f_dop = 0.67 # Fraction of particles that makes it into pool of nitrate, unitless.

# Global Values for Ligand Cycling and Microbial Production
K_I = 45 #W/m2
gamma_Fe = 5*10**(-5)*(106/16) # Units of mol L/(mol N), converted using Redfield Ratio.
lambda_ligand_Fe = 5*10**(-5)/4398

### -------------------------------------------------------------------------------------------

## The following functions abstract the process of creating the transport model
## (using the ODEs we have generalized to the system).

def create_transport_model(C_1, C_2, C_3, dt_in_years, end_time, title, num_metal_elements, \
                           use_metal = False, metal_type = None, M_1 = None, M_2 = None, M_3 = None, \
                               M_in1 = None, M_in2 = None, alpha = None, R_M = None, K_sat_M = None, \
                                   ligand_use = False, use_ligand_cycling = False, \
                                       gamma = gamma_Fe, lambda_ligand = lambda_ligand_Fe, \
                                           L_1 = None, L_2 = None, L_3 = None, \
                           mic_ment_light_leibig = 0, mic_ment_light_mult_lim = 0, \
                               k_scav = 0, ligand_total_val = 0, beta_val = 0, use_multiple_ligands = False, \
                                   **other_metal_parameters):
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
        num_metal_elements: Number of elements to be studied/ plotted. Will be useful when handling trace metals. 
        K_sat_M = Saturation constant of metal.
        use_metal: boolean, true if we want to measure concentration of metal.
        M_1, M_2, M_3: Initial Concentrations of metal in the three boxes (float)
            in mols per cubic meter, this was iron in the previous model. 
        M_in1, M_in2: Amount of metal inputted to respective box. Value in mols of metal per second for
            consistency.
        alpha: fraction of metal inputted that is usable for purposes. 
        metal_type: string, designates type of metal.
        ligand_use: Boolean parameter, indicates whether we want to use ligands in this model (and whether
            our free metal pool will differ from the total metal pool thanks to complexation).
        use_ligand_cycling: boolean, true if we want to measure ligand concentrations and allow them
            to move between boxes. 
        gamma: float, value associated with ligand source.
        lambda_ligand: float, value associated with ligand sink. 
        L_1, L_2, L_3: Initial Concentrations of Ligand in the three boxes.
        mic_ment_light_leibig: int (0 or 1) where 1 means we are using the Michaelis–Menten model with light limitation
            and Leibig's law.
        mic_ment_light_mult_lim: int (0 or 1) where 1 means we are using the Michaelis–Menten model with light limitation
            and multiplicative limitation.
        k_scav: k value associated with scavenging of metal, default set to 0. Input in yr-1
        ligand_total_val: float, initially set to 0 because by default we do not have any ligands in the model.
        beta_val: float, initially set to 0 because we have no equilibrium between the metal and ligand concentrations.
        use_multiple_ligands: Boolean, if true we will be assigning different ligands to different metals. If false, only
            use ligands generated from the dominant metal (being iron in this model)
        *other_metal_parameters: If we want to include other metals in this model alongside M_1 to M_3, we input them here. 
            The order in which the parameters will be accepted are:
                metal_symbol, metal_1, metal_2, metal_3, metal_in_1, metal_in_2, alpha_metal, k_scav, beta_val R_M_metal, 
                K_sat_M_metal, ligand_use_metal, use_ligand_cycling_metal, L_1_metal, L_2_metal, 
                L_3_metal.
            All of these parameters MUST be passed in to include a new metal. 
    Returns
    -------
        Plots graph with this information (not returned)
    """
    ## Initially check to see if the michaelis-menton parameters passed in are either 0 or 1; any other
    ## value, and raise a value error.
    
    ## Yet to be implemented.    
    
    ## Differential Functions
    
    # Cycling of Matter ---------------------------------------------
    # TBD

    
    # Export Production --------------------------------------------------
    
    def export_1(metal_1_dict_tent, K_sat_M_list_tent, C_1_input = C_1):
        """
        Calculates export of organic matter from box 1, considering the Michaelis-Menten
        approach and the liebig/multiplicative limit approach.
        
        Parameters:
            metal_1_dict_tent: dictionary, with current concentrations of metals in box 1.
            K_sat_M_list_tent: list of half saturation constants.
            C_1_input: Initial concentration of C_1
        Returns
            Float, representing the total export of organic matter from box 1 according to current
            concentration of light, nutrient, and metal.
        """
        global light_dependent_change_in_C_1
        global nutrient_dependent_change_in_C_1
        global metal_dependent_change_in_C_1
        light_dependent_change_in_C_1 = (Ibox1/(K_sat_l + Ibox1))
        nutrient_dependent_change_in_C_1 = ((C_1_input)/(K_sat_N + C_1_input))
        
        # Convert Passed In Dictionary to List, where the list consists of tuples matching
        # Concentration Symbols of Metal to Metal Concentrations. 
        metal_1_list_tent = [element for element in metal_1_dict_tent.items()]
        metal_1_list_tent.sort()
            # Sort list so that the k_sat value positions align with this sorted list. 
        
        # Initiate list of tuples with metal concentration and K_sat_M. Takes symbols from 
        # the list from before and matches them with saturation constant.
        metal_constant_list = []
        for metal_index in range(len(metal_1_list_tent)):
            metal_constant_list.append((metal_1_list_tent[metal_index][1], K_sat_M_list_tent[metal_index][1]))

        # Create list of metal_dependent_changes and save it to global list.
        metal_dependent_change_in_C_1 = \
            [(M_1_input/(K_sat_M_s + M_1_input)) for M_1_input, K_sat_M_s in metal_constant_list]

        # Find minimum in list
        metal_dependent_change_in_C_1_min = min(metal_dependent_change_in_C_1)

        return mic_ment_light_leibig*V_max*min([light_dependent_change_in_C_1, nutrient_dependent_change_in_C_1, metal_dependent_change_in_C_1_min]) \
                        + mic_ment_light_mult_lim*V_max*light_dependent_change_in_C_1*nutrient_dependent_change_in_C_1*float(np.prod(np.array(metal_dependent_change_in_C_1)))

            # Exports governed by the Michaelis-Menton model, considering the Liebig and Multiplicative method of limit.
    
    def export_2(metal_2_dict_tent, K_sat_M_list_tent, C_2_input = C_2):
        """
        Calculates export of organic matter from box 1, considering the Michaelis-Menten
        approach and the liebig/multiplicative limit approach.
        
        Parameters:
            metal_2_dict_tent: dictionary, with current concentrations of metals in box 2.
            K_sat_M_list_tent: list of half saturation constants.
            C_2_input: Initial concentration of C_2
        Returns
            Float, representing the total export of organic matter from box 2 according to current
            concentration of light, nutrient, and metal.
        """
        global light_dependent_change_in_C_2
        global nutrient_dependent_change_in_C_2
        global metal_dependent_change_in_C_2
        light_dependent_change_in_C_2 = (Ibox2/(K_sat_l + Ibox2))
        nutrient_dependent_change_in_C_2 = ((C_2_input)/(K_sat_N + C_2_input))

        # Convert passed-in dict to list
        
        metal_2_list_tent = [element for element in metal_2_dict_tent.items()]
        metal_2_list_tent.sort()

        # Initiate list of tuples with metal concentration and K_sat_M.
        metal_constant_list = []
        for metal_index in range(len(metal_2_list_tent)):
            metal_constant_list.append((metal_2_list_tent[metal_index][1], K_sat_M_list_tent[metal_index][1]))
      
        # Create list of metal_dependent_changes and save it to global list.
        metal_dependent_change_in_C_2 = \
            [(M_2_input/(K_sat_M_s + M_2_input)) for (M_2_input, K_sat_M_s) in metal_constant_list]
        
        # Find minimum in list
        metal_dependent_change_in_C_2_min = min(metal_dependent_change_in_C_2)

        
        return mic_ment_light_leibig*V_max*min([light_dependent_change_in_C_2, nutrient_dependent_change_in_C_2, metal_dependent_change_in_C_2_min]) \
                        + mic_ment_light_mult_lim*V_max*light_dependent_change_in_C_2*nutrient_dependent_change_in_C_2*float(np.prod(np.array(metal_dependent_change_in_C_2)))

            # Exports governed by the Michaelis-Menten model, considering both the Liebig and Multiplicative methods of limitation.    
        
    # Complexation, causes differentiation between total and free metal (or any other metal) ------------------------
    
    
    def dtotal_dt(C_1_input, C_2_input, C_3_input, \
                  L_1_input_dict, L_2_input_dict, L_3_input_dict, \
                  conc_name_list_temp, \
                  gamma_temp_dict, lambda_ligand_temp_dict, \
                  metal_1_dict_temp, metal_2_dict_temp, metal_3_dict_temp, \
                  K_sat_M_list_temp, alpha_dict_temp, M_in1_dict_temp, M_in2_dict_temp, k_scav_dict_temp, R_M_dict_temp, \
                  beta_val_dict_temp, dt_temp):
        """
        Takes all initial concentrations and returns all updated values dependent on said
            initial values, using governing differential equations. 
        Parameters
        ----------
            C_1_input, C_2_input, C_3_input: Current nutrient concentrations in respective boxes.
            L_1_input, L_2_input, L_3_input: Current ligand concentrations in respective boxes. 
            conc_name_list_temp: List of all element symbols (e.g. ['C', 'L', 'Fe', ...])
            gamma_temp: gamma value, currently a fixed value but is subject to change. 
            lambda_ligand_temp: lambda ligand value, currently fixed to that of iron but subject to change. 
            metal_1_dict_temp, metal_2_dict_temp, metal_3_dict_temp: dictionaries storing current concentrations of the metals in the three boxes. 
            K_Sat_M_list_temp, ..., beta_val_dict_temp: Pre-established lists and dictionaries with appropriate values to compute the differential equations. 
            dt_temp: Time step. 

        Returns
        -------
        Dictionary mapping element/metal to new concentration value.

        """
        def dM1dt(M_1_input, M_2_input, M_3_input, L_1_input, L_2_input, L_3_input, alpha_temp, M_in1_temp, k_scav_temp, R_M_temp, beta_val_temp):
            return M_1_input + dt_temp*((psi*(M_3_input - M_1_input) + k_31*(M_3_input - M_1_input) + k_21*(M_2_input - M_1_input))/vol_1 + \
                    alpha_temp*M_in1_temp/dz_1 - k_scav_temp*complexation(M_1_input, L_1_input, beta_val_temp)/(60*60*24*365) - R_M_temp*export_1(metal_1_dict_temp, K_sat_M_list_temp, C_1_input))
            
                # Line 1: General tracer equation, maintains equilibrium among all three boxes with flow rate considered.
                # Line 2: First term represents source, second term represents sink (in terms of being scavenged)
                    # Third term represents amount being used up ('biological utilization' as in Parekh, 2004)

        def dM2dt(M_1_input, M_2_input, M_3_input, L_1_input, L_2_input, L_3_input, alpha_temp, M_in2_temp, k_scav_temp, R_M_temp, beta_val_temp):
            return M_2_input + dt_temp*((psi*(M_1_input - M_2_input) + k_12*(M_1_input - M_2_input) + k_32*(M_3_input - M_2_input))/vol_2 + \
                    alpha_temp*M_in2_temp/dz_2 - k_scav_temp*complexation(M_2_input, L_2_input, beta_val_temp)/(60*60*24*365) - R_M_temp*export_2(metal_2_dict_temp, K_sat_M_list_temp, C_2_input))

        def dM3dt(M_1_input, M_2_input, M_3_input, L_1_input, L_2_input, L_3_input, alpha_temp, k_scav_temp, R_M_temp, beta_val_temp):
            return M_3_input + dt_temp*((psi*(M_2_input - M_3_input) + k_23*(M_2_input - M_3_input) + k_13*(M_1_input - M_3_input))/vol_3 \
                - k_scav_temp*complexation(M_3_input, L_3_input, beta_val_temp)/(60*60*24*365) \
                + R_M_temp*(export_1(metal_1_dict_temp, K_sat_M_list_temp, C_1_input)*vol_1 + export_2(metal_2_dict_temp, K_sat_M_list_temp, C_2_input)*vol_2)/vol_3)
        
        
        def dL1dt(L_1_input, L_2_input, L_3_input, gamma_tempo, lambda_ligand_tempo):
            return L_1_input + dt_temp*((psi*(L_3_input - L_1_input) + k_31*(L_3_input - L_1_input) + k_21*(L_2_input - L_1_input))/vol_1 \
                 + gamma_tempo*export_1(metal_1_dict_temp, K_sat_M_list_temp, C_1_input) \
                - lambda_ligand_tempo*L_1_input)
        
        def dL2dt(L_1_input, L_2_input, L_3_input, gamma_tempo, lambda_ligand_tempo):
            return L_2_input + dt_temp*((psi*(L_1_input - L_2_input) + k_12*(L_1_input - L_2_input) + k_32*(L_3_input - L_2_input))/vol_2 \
                 + gamma_tempo*export_2(metal_2_dict_temp, K_sat_M_list_temp, C_2_input) \
                - lambda_ligand_tempo*L_2_input)
        
        def dL3dt(L_1_input, L_2_input, L_3_input, gamma_tempo, lambda_ligand_tempo):
            return L_3_input + dt_temp*((psi*(L_2_input - L_3_input) + k_23*(L_2_input - L_3_input) + k_13*(L_1_input - L_3_input))/vol_3 \
                 - lambda_ligand_tempo/100*L_3_input \
                + gamma_tempo/vol_3*(export_1(metal_1_dict_temp, K_sat_M_list_temp, C_1_input)*vol_1 + export_2(metal_2_dict_temp, K_sat_M_list_temp, C_2_input)*vol_2))
        
        dC1dt = ('C_1', C_1_input + dt_temp*((psi*(C_3_input - C_1_input) + k_31*(C_3_input - C_1_input) + k_21*(C_2_input - C_1_input))/vol_1 \
                - export_1(metal_1_dict_temp, K_sat_M_list_temp, C_1_input)))
        dC2dt = ('C_2', C_2_input + dt_temp*((psi*(C_1_input - C_2_input) + k_12*(C_1_input - C_2_input) + k_32*(C_3_input - C_2_input))/vol_2 \
                - export_2(metal_2_dict_temp, K_sat_M_list_temp, C_2_input)))
        dC3dt = ('C_3', C_3_input + dt_temp*((psi*(C_2_input - C_3_input) + k_23*(C_2_input - C_3_input) + k_13*(C_1_input - C_3_input))/vol_3 + \
                + (export_1(metal_1_dict_temp, K_sat_M_list_temp, C_1_input)*vol_1 + export_2(metal_2_dict_temp, K_sat_M_list_temp, C_2_input)*vol_2)/vol_3))
            
        return_list = [dC1dt, dC2dt, dC3dt]
            
            # Because, at least for the moment, the above concentrations of nutrients are singular values (and are not associated with newly-added metals), 
            # we can just compute them once and add them to a return-list. 
        
        if not use_multiple_ligands:
            dL1dt = ('L_1_Fe', L_1_input_dict['L_1_Fe'] + dt_temp*((psi*(L_3_input_dict['L_3_Fe'] - L_1_input_dict['L_1_Fe']) + k_31*(L_3_input_dict['L_3_Fe'] - L_1_input_dict['L_1_Fe']) + k_21*(L_2_input_dict['L_2_Fe'] - L_1_input_dict['L_1_Fe']))/vol_1 \
                      + gamma_temp_dict['Fe']*export_1(metal_1_dict_temp, K_sat_M_list_temp, C_1_input) \
                    - lambda_ligand_temp_dict['Fe']*L_1_input_dict['L_1_Fe']))
            dL2dt = ('L_2_Fe', L_2_input_dict['L_2_Fe'] + dt_temp*((psi*(L_1_input_dict['L_1_Fe'] - L_2_input_dict['L_2_Fe']) + k_12*(L_1_input_dict['L_1_Fe'] - L_2_input_dict['L_2_Fe']) + k_32*(L_3_input_dict['L_3_Fe'] - L_2_input_dict['L_2_Fe']))/vol_2 \
                      + gamma_temp_dict['Fe']*export_2(metal_2_dict_temp, K_sat_M_list_temp, C_2_input) \
                    - lambda_ligand_temp_dict['Fe']*L_2_input_dict['L_2_Fe']))
            dL3dt = ('L_3_Fe', L_3_input_dict['L_3_Fe'] + dt_temp*((psi*(L_2_input_dict['L_2_Fe'] - L_3_input_dict['L_3_Fe']) + k_23*(L_2_input_dict['L_2_Fe'] - L_3_input_dict['L_3_Fe']) + k_13*(L_1_input_dict['L_1_Fe'] - L_3_input_dict['L_3_Fe']))/vol_3 \
                      - lambda_ligand_temp_dict['Fe']/100*L_3_input_dict['L_3_Fe'] \
                    + gamma_temp_dict['Fe']/vol_3*(export_1(metal_1_dict_temp, K_sat_M_list_temp, C_1_input)*vol_1 + export_2(metal_2_dict_temp, K_sat_M_list_temp, C_2_input)*vol_2)))
            return_list.extend([dL1dt, dL2dt, dL3dt])
        
        m_1_temp = []
        m_2_temp = []
        m_3_temp = []
        
        if use_multiple_ligands:
            L_1_temp = []
            L_2_temp = []
            L_3_temp = []
        
            # Initiate temporary lists which will store new metal/ligand concentrations designated as either the metal in box 1, box 2, or box 3.
            # The sole purpose of these three lists is to help update the passed-in metal dictionaries.
        
        for metal_conc in conc_name_list_temp:
            if not use_multiple_ligands:
                temp_tuple_1 = (f'{metal_conc}_1', dM1dt(metal_1_dict_temp[f'{metal_conc}_1'], metal_2_dict_temp[f'{metal_conc}_2'], metal_3_dict_temp[f'{metal_conc}_3'], \
                                                         L_1_input_dict['L_1_Fe'], L_2_input_dict['L_2_Fe'], L_3_input_dict['L_3_Fe'], \
                                                             alpha_dict_temp[metal_conc], M_in1_dict_temp[metal_conc], k_scav_dict_temp[metal_conc], \
                                                                 R_M_dict_temp[metal_conc], beta_val_dict_temp[metal_conc]))
                return_list.append(temp_tuple_1)
                m_1_temp.append(temp_tuple_1)
                
                temp_tuple_2 = (f'{metal_conc}_2', dM2dt(metal_1_dict_temp[f'{metal_conc}_1'], metal_2_dict_temp[f'{metal_conc}_2'], metal_3_dict_temp[f'{metal_conc}_3'], \
                                                         L_1_input_dict['L_1_Fe'], L_2_input_dict['L_2_Fe'], L_3_input_dict['L_3_Fe'], \
                                                             alpha_dict_temp[metal_conc], M_in2_dict_temp[metal_conc], k_scav_dict_temp[metal_conc], \
                                                                 R_M_dict_temp[metal_conc], beta_val_dict_temp[metal_conc]))
                return_list.append(temp_tuple_2)
                m_2_temp.append(temp_tuple_2)
                
                temp_tuple_3 = (f'{metal_conc}_3', dM3dt(metal_1_dict_temp[f'{metal_conc}_1'], metal_2_dict_temp[f'{metal_conc}_2'], metal_3_dict_temp[f'{metal_conc}_3'], \
                                                         L_1_input_dict['L_1_Fe'], L_2_input_dict['L_2_Fe'], L_3_input_dict['L_3_Fe'], \
                                                             alpha_dict_temp[metal_conc], k_scav_dict_temp[metal_conc], \
                                                                 R_M_dict_temp[metal_conc], beta_val_dict_temp[metal_conc]))
                return_list.append(temp_tuple_3)
                m_3_temp.append(temp_tuple_3)
                    # Temp tuples store the new metal concentrations corresponding to the metal names, then that tuple is appended to the return list as well as the 
                    # respective metal lists. 
                
            if use_multiple_ligands:   
                temp_tuple_1 = (f'{metal_conc}_1', dM1dt(metal_1_dict_temp[f'{metal_conc}_1'], metal_2_dict_temp[f'{metal_conc}_2'], metal_3_dict_temp[f'{metal_conc}_3'], \
                                                         L_1_input_dict[f'L_1_{metal_conc}'], L_2_input_dict[f'L_2_{metal_conc}'], L_3_input_dict[f'L_3_Fe{metal_conc}'], \
                                                             alpha_dict_temp[metal_conc], M_in1_dict_temp[metal_conc], k_scav_dict_temp[metal_conc], \
                                                                 R_M_dict_temp[metal_conc], beta_val_dict_temp[metal_conc]))
                return_list.append(temp_tuple_1)
                m_1_temp.append(temp_tuple_1)
                
                temp_tuple_2 = (f'{metal_conc}_2', dM2dt(metal_1_dict_temp[f'{metal_conc}_1'], metal_2_dict_temp[f'{metal_conc}_2'], metal_3_dict_temp[f'{metal_conc}_3'], \
                                                         L_1_input_dict[f'L_1_{metal_conc}'], L_2_input_dict[f'L_2_{metal_conc}'], L_3_input_dict[f'L_3_Fe{metal_conc}'], \
                                                             alpha_dict_temp[metal_conc], M_in2_dict_temp[metal_conc], k_scav_dict_temp[metal_conc], \
                                                                 R_M_dict_temp[metal_conc], beta_val_dict_temp[metal_conc]))
                return_list.append(temp_tuple_2)
                m_2_temp.append(temp_tuple_2)
                
                temp_tuple_3 = (f'{metal_conc}_3', dM3dt(metal_1_dict_temp[f'{metal_conc}_1'], metal_2_dict_temp[f'{metal_conc}_2'], metal_3_dict_temp[f'{metal_conc}_3'], \
                                                         L_1_input_dict[f'L_1_{metal_conc}'], L_2_input_dict[f'L_2_{metal_conc}'], L_3_input_dict[f'L_3_Fe{metal_conc}'], \
                                                             alpha_dict_temp[metal_conc], k_scav_dict_temp[metal_conc], \
                                                                 R_M_dict_temp[metal_conc], beta_val_dict_temp[metal_conc]))
                return_list.append(temp_tuple_3)
                m_3_temp.append(temp_tuple_3)
                    # Temp tuples store the new metal concentrations corresponding to the metal names, then that tuple is appended to the return list as well as the 
                    # respective metal lists. 

                
                temp_tuple_L_1 = ('L_1_{metal_conc}', dL1dt(L_1_input_dict[f'L_1_{metal_conc}'], L_2_input_dict[f'L_2_{metal_conc}'], L_3_input_dict[f'L_3_{metal_conc}'], \
                                                            gamma_temp_dict[metal_conc], lambda_ligand_temp_dict[metal_conc]))
                return_list.append(temp_tuple_L_1)
                L_1_temp.append(temp_tuple_L_1)
                
                temp_tuple_L_2 = ('L_2_{metal_conc}', dL2dt(L_1_input_dict[f'L_1_{metal_conc}'], L_2_input_dict[f'L_2_{metal_conc}'], L_3_input_dict[f'L_3_{metal_conc}'], \
                                                            gamma_temp_dict[metal_conc], lambda_ligand_temp_dict[metal_conc]))
                return_list.append(temp_tuple_L_2)
                L_2_temp.append(temp_tuple_L_2)
                
                temp_tuple_L_3 = ('L_3_{metal_conc}', dL3dt(L_1_input_dict[f'L_1_{metal_conc}'], L_2_input_dict[f'L_2_{metal_conc}'], L_3_input_dict[f'L_3_{metal_conc}'], \
                                                            gamma_temp_dict[metal_conc], lambda_ligand_temp_dict[metal_conc]))
                return_list.append(temp_tuple_L_3)
                L_3_temp.append(temp_tuple_L_3)
            
        # Now that we have our return_list, update the dictionaries from which we got the concentration values to be used in the next iteration(s) for the three passed-in metal dictionaries.
        
        for conc_index in range(len(m_1_temp)):
            metal_1_dict_temp[m_1_temp[conc_index][0]] = m_1_temp[conc_index][1]
            metal_2_dict_temp[m_2_temp[conc_index][0]] = m_2_temp[conc_index][1]
            metal_3_dict_temp[m_3_temp[conc_index][0]] = m_3_temp[conc_index][1]
            
            if use_multiple_ligands:
                L_1_input_dict[L_1_temp[conc_index][0]] = L_1_temp[conc_index][1]
                L_2_input_dict[L_2_temp[conc_index][0]] = L_2_temp[conc_index][1]
                L_3_input_dict[L_3_temp[conc_index][0]] = L_3_temp[conc_index][1]

        return dict(return_list)


    def complexation(metal_tot, ligand_tot, beta):
        """
        Given a total metal concentration, ligand concentration, and beta value, this
        function will return the total concentration of free metal that is not bound to
        ligands (i.e. is free and prone to scavanging).
        Parameters
        ----------
        metal_tot : float 
            current total concentration of metal, value in mol per cubic meter 
        ligand_tot : float
            current total concentration of ligand, value in mol per cubic meter
        beta : float
            constant value, defines equilibrium position between free metal + ligand
            and the complexed form. Value in kg/mol
        Returns
        -------
        Float, current concentration of free metal.
        """
        term_1 = (metal_tot - 1/beta - ligand_tot)/2
        term_2 = ((beta*(ligand_tot - metal_tot + 1/beta)**2 + 4*metal_tot)/(4*beta))**(1/2)
    
        return term_1 + term_2
    
    # ...................................................................................
    ### Creation of Variables and Using Above-Defined Functions
    # ...................................................................................
    
    ## Initiate Lists With Additional *args passed in, as well as initial concentrations.
    
    metal_symbol_list = [('symb_Fe', metal_type), ]
    metal_1_list = [('m_conc_Fe_1', M_1), ]
    metal_2_list = [('m_conc_Fe_2', M_2), ]
    metal_3_list = [('m_conc_Fe_3', M_3), ]
    metal_in_1_list = [('in1_Fe', M_in1), ]
    metal_in_2_list = [('in2_Fe', M_in2), ]
    alpha_list = [('alpha_Fe', alpha), ]
    k_scav_list = [('k_scav_Fe', k_scav), ]
    beta_val_list = [('beta_val_Fe', beta_val), ]
    R_M_list = [('R_M_Fe', R_M), ]
    K_sat_M_list = [('K_sat_Fe', K_sat_M), ]
    ligand_use_list = [('ligand_use_Fe', ligand_use), ]
    use_ligand_cycling_list = [('use_ligand_cycling_Fe', use_ligand_cycling), ]
    gamma_list = [('gamma_Fe', gamma), ]
    lambda_ligand_list = [('lambda_ligand_Fe', lambda_ligand), ]
    L_symbol_list = [('L_Fe_symb', 'L_Fe'), ]
    L_1_init_list = [('L_1_Fe', L_1), ]
    L_2_init_list = [('L_2_Fe', L_2), ]
    L_3_init_list = [('L_3_Fe', L_3), ]
        
    
    for parameter_val in other_metal_parameters.items():
        if parameter_val[0].startswith('symb'):
            metal_symbol_list.append(parameter_val)
        elif parameter_val[0].startswith('m_conc_') and parameter_val[0].endswith('1'):
            metal_1_list.append(parameter_val)
        elif parameter_val[0].startswith('m_conc_') and parameter_val[0].endswith('2'):
            metal_2_list.append(parameter_val)
        elif parameter_val[0].startswith('m_conc_') and parameter_val[0].endswith('3'):
            metal_3_list.append(parameter_val)
        elif parameter_val[0].startswith('in1'):
            metal_in_1_list.append(parameter_val)
        elif parameter_val[0].startswith('in2'):
            metal_in_2_list.append(parameter_val)
        elif parameter_val[0].startswith('alpha'):
            alpha_list.append(parameter_val)
        elif parameter_val[0].startswith('k_scav'):
            k_scav_list.append(parameter_val)
        elif parameter_val[0].startswith('beta_val'):
            beta_val_list.append(parameter_val)
        elif parameter_val[0].startswith('R_M'):
            R_M_list.append(parameter_val)
        elif parameter_val[0].startswith('K_sat_'):
            K_sat_M_list.append(parameter_val)
        elif parameter_val[0].startswith('ligand_use_'):
            ligand_use_list.append(parameter_val)
        elif parameter_val[0].startswith('use_ligand_cycling_'):
            use_ligand_cycling_list.append(parameter_val)
        elif parameter_val[0].startswith('gamma'):
            gamma_list.append(parameter_val)
        elif parameter_val[0].startswith('lambda_ligand'):
            lambda_ligand_list.append(parameter_val)
        elif parameter_val[0].startswith('L') and parameter_val[0].endswith('symb'):
            L_symbol_list.append(parameter_val)
        elif parameter_val[0].startswith('L_1_'):
            L_1_init_list.append(parameter_val)
        elif parameter_val[0].startswith('L_2_'):
            L_2_init_list.append(parameter_val)
        elif parameter_val[0].startswith('L_3_'):
            L_3_init_list.append(parameter_val)
            
    # Sort initiated lists in alphabetical order to help with matching values from different lists.
    
    metal_symbol_list.sort()
    metal_1_list.sort()
    metal_2_list.sort()
    metal_3_list.sort()
    metal_in_1_list.sort()
    metal_in_2_list.sort()
    alpha_list.sort()
    k_scav_list.sort()
    beta_val_list.sort()
    R_M_list.sort()
    K_sat_M_list.sort()
    ligand_use_list.sort()
    use_ligand_cycling_list.sort()
    gamma_list.sort()
    lambda_ligand_list.sort()
    L_symbol_list.sort()
    L_1_init_list.sort()
    L_2_init_list.sort()
    L_3_init_list.sort()
    
    ## After collecting the above information into appropriate lists, convert them into dictionaries where
    ## the key is in format 'element + box label' to make the next functions readable and accessible via key rather than order. 
    
    
    # Metal Concentrations
    
    all_concs = [('C_1', C_1), ('C_2', C_2), ('C_3', C_3)]
        # Initiate preliminary list where the initial nutrient and ligand concentrations are stored as
        # tuples. This format helps to easily convert this into a dictionary. 
        
    
        
    # Iterate over all three metal lists such that the first element of the tuple is renamed
    # to the metal symbol, while the second element is a numerical element concentration. Do the same for ligand
    # concentrations. 
    
    metal_conc_symbol_tot = []
    metal_1_conc_symbol = []
    metal_2_conc_symbol = []
    metal_3_conc_symbol = []

    ligand_1_conc_symbol = []
    ligand_2_conc_symbol = []
    ligand_3_conc_symbol = []
    
    for val_index in range(len(metal_symbol_list)):
        # metal_symbol_list was used to calculate range just out of convenience; any of the other lists could have been chosen. 
        metal_conc_symbol_tot.append((f'{metal_symbol_list[val_index][1]}_1', metal_1_list[val_index][1]))
        metal_conc_symbol_tot.append((f'{metal_symbol_list[val_index][1]}_2', metal_2_list[val_index][1]))
        metal_conc_symbol_tot.append((f'{metal_symbol_list[val_index][1]}_3', metal_3_list[val_index][1]))
        if use_multiple_ligands:
            metal_conc_symbol_tot.append((f'L_1_{metal_symbol_list[val_index][1]}', L_1_init_list[val_index][1]))
            metal_conc_symbol_tot.append((f'L_2_{metal_symbol_list[val_index][1]}', L_2_init_list[val_index][1]))
            metal_conc_symbol_tot.append((f'L_3_{metal_symbol_list[val_index][1]}', L_3_init_list[val_index][1]))
            # Above six are appended to main concentration symbol list.
        metal_1_conc_symbol.append((f'{metal_symbol_list[val_index][1]}_1', metal_1_list[val_index][1]))
        metal_2_conc_symbol.append((f'{metal_symbol_list[val_index][1]}_2', metal_2_list[val_index][1]))
        metal_3_conc_symbol.append((f'{metal_symbol_list[val_index][1]}_3', metal_3_list[val_index][1]))
        if use_multiple_ligands:
            ligand_1_conc_symbol.append((f'L_1_{metal_symbol_list[val_index][1]}', L_1_init_list[val_index][1]))
            ligand_2_conc_symbol.append((f'L_2_{metal_symbol_list[val_index][1]}', L_2_init_list[val_index][1]))
            ligand_3_conc_symbol.append((f'L_3_{metal_symbol_list[val_index][1]}', L_3_init_list[val_index][1]))
            # Note that all the lists used here (specifically metal_1, metal_2, and metal_3 lists) are sorted, so numerical indices
            # can be used to sort them. 
            
    if not use_multiple_ligands:
            metal_conc_symbol_tot.append(('L_1_Fe', L_1_init_list[0][1]))
            metal_conc_symbol_tot.append(('L_2_Fe', L_2_init_list[0][1]))
            metal_conc_symbol_tot.append(('L_3_Fe', L_3_init_list[0][1]))
            ligand_1_conc_symbol.append(('L_1_Fe', L_1_init_list[0][1]))
            ligand_2_conc_symbol.append(('L_2_Fe', L_2_init_list[0][1]))
            ligand_3_conc_symbol.append(('L_3_Fe', L_3_init_list[0][1]))
    
    all_concs.extend(metal_conc_symbol_tot)
        # Extend our previously_defined list of concentrations of nutrients + ligands with our concentrations of metals. 
    
    init_concs = dict(all_concs)
    init_concs_metal_1 = dict(metal_1_conc_symbol)
    init_concs_metal_2 = dict(metal_2_conc_symbol)
    init_concs_metal_3 = dict(metal_3_conc_symbol)

    init_concs_ligand_1 = dict(ligand_1_conc_symbol)
    init_concs_ligand_2 = dict(ligand_2_conc_symbol)
    init_concs_ligand_3 = dict(ligand_3_conc_symbol)
    
    print(init_concs)
    print(init_concs_ligand_1)
    print(init_concs_ligand_2)
    print(init_concs_ligand_3)
    
    raise NotImplementedError
    
    # print(init_concs_ligand_1)
    # print(init_concs_ligand_2)
    # print(init_concs_ligand_3)
    
        # Convert all tupled lists from above into dictionaries. This is a pivotal step because now our info can be accessed by the names of the elements/ metals, 
        # thereby avoiding any potential worries with misindexing. 


    ## Converting the rest of the important lists into dictionaries with keys corresponding to elemental symbols/ nutrients/ ligands.
    
    metal_name_list = [element_symbol[1] for element_symbol in metal_symbol_list]
        # metal_symbol_list is stored as tuples where the first element is the name of the passed_in_variable
        # and the second element is the symbol itself; just take the second element. 
    
    metal_in1_dict = {}
    metal_in2_dict = {}
    alpha_dict = {}
    k_scav_dict = {}
    beta_val_dict = {}
    R_M_dict = {}
    gamma_dict = {}
    lambda_ligand_dict = {}
        # Initiate dictionaries where we will add keys associated with element, and value associated with the 
        # respective dictionaries. 
    
    for var_index in range(len(metal_name_list)):
        metal_in1_dict[metal_name_list[var_index]] = metal_in_1_list[var_index][1]
        metal_in2_dict[metal_name_list[var_index]] = metal_in_2_list[var_index][1]
        alpha_dict[metal_name_list[var_index]] = alpha_list[var_index][1]
        k_scav_dict[metal_name_list[var_index]] = k_scav_list[var_index][1]
        beta_val_dict[metal_name_list[var_index]] = beta_val_list[var_index][1]
        R_M_dict[metal_name_list[var_index]] = R_M_list[var_index][1]
        gamma_dict[metal_name_list[var_index]] = gamma_list[var_index][1]
        lambda_ligand_dict[metal_name_list[var_index]] = lambda_ligand_list[var_index][1]
        
    ## Initiate Time Variables
    
    dt = dt_in_years*60*60*24*365  #Total number of seconds representing a year. 
    
    ## Initiate Variables that will "globally," at least within this function, keep track of 
    ## concentrations of C_1 and C_2 so that these values can be reused for C_3.
    light_dependent_change_in_C_1 = 0
    nutrient_dependent_change_in_C_1 = 0
    metal_dependent_change_in_C_1 = [0 for items in metal_1_list]
    light_dependent_change_in_C_2 = 0
    nutrient_dependent_change_in_C_2 = 0
    metal_dependent_change_in_C_2 = [0 for items in metal_2_list]
    
    # ...........................................................................
        
    ### Create arrays where the first array depicts the x-axis (time steps) and the three other
    ### arrays depict concentrations of C_1, C_2, and C_3 over the time steps. 
    
    ## Create x-axis array (time)
    
    time_axis_list = [0,]
    time_axis_list_log10 = [1*10^-10,]
    t_temp = dt_in_years
    
    while t_temp <= end_time:
        t_temp += dt_in_years
        time_axis_list.append(t_temp)
        time_axis_list_log10.append(math.log10(t_temp))
            # At the end of this while loop, we will have a list of all time checkpoints for which
            # we want to calculate the three concentrations. 
    time_axis_array = np.array(time_axis_list)
    time_axis_array_log10 = np.array(time_axis_list_log10)


    ## Create y-axis concentrations
    
    # Initiate symbols of all things to be traced.
        
    # Expandable list of biogeochemistry tracers
    if use_multiple_ligands:
        all_symbols = [('placeholder1', "C")]
    else:
        all_symbols = [('placeholder1', "C"), ('placeholder2', 'L')]
    all_symbols.extend(metal_symbol_list)
    if use_multiple_ligands:
        all_symbols.extend(L_symbol_list)
        # metal_symbol_list consists of metals that were passed in in tuple form; simply extend
        # this list of "C" and "L" with that list.
        
    all_symbols_list = [element_symbol[1] for element_symbol in all_symbols]
        # Now we have a list of all symbols/ items circulating through the boxes. We can use this
        # to iterate through the concentrations as necessary. 
        
        
    # Initiate dictionary where the key is the element, but the value is a mutable list storing
    # concentration values for each time step. 
    
    conc_tracing_dict = {}
    for element_name in init_concs.keys():
        conc_tracing_dict[element_name] = [init_concs[element_name],]
            # The above for-loop initiates this sort of dictionary with mutable lists as the value. 

    for t_val in time_axis_array[1:]:
        temp_dict = dtotal_dt(conc_tracing_dict['C_1'][-1], conc_tracing_dict['C_2'][-1], conc_tracing_dict['C_3'][-1], \
                  init_concs_ligand_1, init_concs_ligand_2, init_concs_ligand_3, \
                  metal_name_list, \
                  gamma_dict, lambda_ligand_dict, \
                  init_concs_metal_1, init_concs_metal_2, init_concs_metal_3, \
                  K_sat_M_list, alpha_dict, metal_in1_dict, metal_in2_dict, k_scav_dict, R_M_dict, \
                  beta_val_dict, dt)
        # print(init_concs)
        # print(temp_dict)
        for element_name in init_concs.keys():
            conc_tracing_dict[element_name].append(temp_dict[element_name])
                # for each key in our init_concs (the same keys present in our conc_tracing_dict),
                # find the value associated with that key (which is a singular list), and to that list append
                # the new concentration value given by the temp_dict with that very same key.

        
    def plot_concentrations(title, metal_symbol_list_temp, time_axis_array_temp, conc_dict_temp):
        """
        Plots the concentrations of material in the three boxes and their
        change over time
    
        Parameters:
        -----------
            title: an str object titling the graph
            metal_symbol_list_temp: List of all symbols involved (e.g. ['C', 'L', 'Fe', ...])
            time_axis_array_temp: array of ints representing the time scale of the graph.
            conc_dict_temp: dictionary where the keys are concentrations of the elements in each of the boxes
                (e.g. C_1, Fe_2, etc.), and the values are lists of concentration values spaced out by the time step.
                In a sense, these lists will be the "y values" for each x value in the time axis array. 
        
        Returns:
        -------
            None. Plots graph with above information.
    
        """

        def plot_sub_conc(ml_name, time_axis_array_tempo, \
                       ml_concentrations_1, ml_concentrations_2, ml_concentrations_3):
            '''
            Parameters
            ----------
            ml_name : str
                Metal name, symbol with which to characterize graphed element. 
            ml_concentrations_1 : list/ numpy array
                Concentrations to be plotted for first box.
            ml_concentrations_2 : list/ numpy array
                Concentrations to be plotted for second box.
            ml_concentrations_3 : list / numpy array
                Concentrations to be plotted for third box.

            Returns
            -------
            None; plots figure. 

            '''
            
            plt.figure()
            plt.title(f'{title} \n {ml_name}: Concentrations Over Time (in mol/m3) \n')
            plt.xlabel('Time [log(years)]')
            plt.ylabel(f'Concentration of {ml_name} (in mol/m3)')
                # Initiate labels of graph
                
            plt.plot(time_axis_array_tempo, ml_concentrations_1, 'r', label = f'{ml_name}_1')
            plt.plot(time_axis_array_tempo, ml_concentrations_2, 'g', label = f'{ml_name}_2')
            plt.plot(time_axis_array_tempo, ml_concentrations_3, 'b', label = f'{ml_name}_3')
                # Plot the three curves/ sets of data.

            plt.legend(loc = 'best')
            plt.show()
            plt.close()
                # Make sure to close so that a separate graph can be plotted. 
            
            
        # Plot Concentrations Using Element Symbols in metal_symbol_list_temp.
        
        for element_symbol in metal_symbol_list_temp:
            plot_sub_conc(element_symbol, time_axis_array_temp, \
                          conc_dict_temp[f'{element_symbol}_1'], \
                              conc_dict_temp[f'{element_symbol}_2'], \
                                  conc_dict_temp[f'{element_symbol}_3'])
        
    plot_concentrations(title, all_symbols_list, time_axis_array_log10, conc_tracing_dict)
        # Plotting the concentrations and how they change over time. 
            

### --------------------------------------------------------------------------------
### --------------------------------------------------------------------------------
### --------------------------------------------------------------------------------
### The following section calls the above functions for plotting purposes.

# -------------------------------------------------------------------
# Plotting Iron with results similar to Lauderdale 2020

N_1_to_3 = 30*rho_0*10**(-6)

alpha_Fe = 0.01*0.035 #Fe dust solubility
R_Fe = (2.5*10**-5)*6.625 #unitless, multiplied by 6.625 to convert this Fe:C ratio to Fe:N
K_sat_Fe = 2*10**-7 #units of mole per cubic meter


ligand_conc = 1*10**-6 # mol/m3
beta_val_1 = 10**8 # kg/mol, as required by the value earlier.
# Establishing F_in1.
F_in1 = 0.071/(55.845*60*60*24*365)
    # The quanitity is initially provided in grams M per year. To convert this
    # quantity to mol M per second, divide by molar mass as well as the total number
    # of seconds in a year.   
# Establishing F_in2.
F_in2 = 6.46/(55.845*60*60*24*365)
    # The quanitity is initially provided in grams M per year. To convert this
    # quantity to mol M per second, divide by molar mass as well as the total number
    # of seconds in a year.



# transport_model_graphing_ligand_approach = \
#         create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.006849, 10000, \
#                             'Concentrations of Nutrients, Iron, and Ligands over time, \n dt = 2.5 days, ligand concentration = 10**-6, beta = 10**8 (kg per mol) \n Michalis-Menten Model, Leibig Limit Approximation', 9, \
#                                 use_metal = True, metal_type = 'Fe', M_1 = 0, M_2 = 0, M_3 = 0, K_sat_M = K_sat_Fe, \
#                                     M_in1 = F_in1, M_in2 = F_in2, alpha = alpha_Fe, R_M = R_Fe, \
#                                         ligand_use = True, use_ligand_cycling = True, \
#                                             L_1 = 0, L_2 = 0, L_3 = 0, \
#                                                 mic_ment_light_leibig = 1, \
#                                                     k_scav = 0.19, ligand_total_val = ligand_conc, beta_val = beta_val_1)

# -------------------------------------------------------------------
# Copper (II): Assume that the surface concentration (i.e. metal and ligands) in the first
# two boxes are uniform. Scale K_sat_Fe to K_sat_Cu(II) according to elemental ratios. 
# Assume maximum concentrations of metal and initial ligand total pool.
# ONLY CONSIDERS COPPER IN THE POOL OF METALS.

N_1_to_3 = 30*rho_0*10**(-6)
Cu_1_2 = (1*10**-9)*(1000) # Converting value from mol/liter to mol/m3

alpha_Cu_II = alpha_Fe
R_Cu_II = R_Fe*(0.38/7.5) # Using elemental ratio.
K_sat_Cu_II = K_sat_Fe*(0.38/7.5) # Using elemental ratios to convert between iron and copper. 

ligand_conc = 2*10**-9 # mol/m3
beta_val_Cu_II = math.exp(8.5)

# transport_model_graphing_ligand_approach = \
#         create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.006849, 10000, \
#                             'Concentrations of Nutrients, Iron, and Ligands over time, \n dt = 2.5 days, ligand concentration = 2*10**-9, beta = e**8.5 (kg per mol) \n Michalis-Menten Model, Leibig Limit Approximation', 9, \
#                                 use_metal = True, metal_type = 'Cu(II)', M_1 = Cu_1_2, M_2 = Cu_1_2, M_3 = 0, K_sat_M = K_sat_Cu_II, \
#                                     M_in1 = F_in1, M_in2 = F_in2, alpha = alpha_Cu_II, R_M = R_Cu_II, \
#                                         ligand_use = True, use_ligand_cycling = True, \
#                                             L_1 = ligand_conc, L_2 = ligand_conc, L_3 = 0, \
#                                                 mic_ment_light_leibig = 1, \
#                                                     k_scav = 0.19, ligand_total_val = ligand_conc, beta_val = beta_val_Cu_II)

# -------------------------------------------------------------------
# Copper (II): Assume that the surface concentration (i.e. metal and ligands) in the first
# two boxes are uniform. Scale K_sat_Fe to K_sat_Cu(II) according to elemental ratios. 
# Assume maximum concentrations of metal and initial ligand total pool.
# Includes Iron in the Copper Pool, no changes to ligand behavior. 


N_1_to_3 = 30*rho_0*10**(-6)
Cu_1_2 = (1*10**-9)*(1000) # Converting value from mol/liter to mol/m3

alpha_Cu_II_val = alpha_Fe
R_Cu_II = R_Fe*(0.38/7.5) # Using elemental ratio.
K_sat_Cu_II_val = K_sat_Fe*(0.38/7.5) # Using elemental ratios to convert between iron and copper. 

ligand_conc = 2*10**-9 # mol/m3
beta_val_Cu_II_val = math.exp(8.5)

gamma_Cu_II_val = 5*10**(-5)*(106/16)*(0.38/7.5) # Units of mol L/(mol N), converted using Redfield Ratio.
lambda_ligand_Cu_II_val = 5*10**(-5)/4398*(0.38/7.5)

transport_model_graphing_ligand_approach = \
        create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.006849, 10000, \
                            'Concentrations of Nutrients, Iron, Copper(II) and Ligands over time, \n dt = 2.5 days, ligand concentration = 2*10**-9, beta = e**8.5 (kg per mol) \n Michalis-Menten Model, Leibig Limit Approximation', 9, \
                                use_metal = True, metal_type = 'Fe', M_1 = 0, M_2 = 0, M_3 = 0, K_sat_M = K_sat_Fe, \
                                    M_in1 = F_in1, M_in2 = F_in2, alpha = alpha_Fe, R_M = R_Fe, \
                                        ligand_use = True, use_ligand_cycling = True, \
                                            L_1 = 0, L_2 = 0, L_3 = 0, \
                                                mic_ment_light_leibig = 1, \
                                                    k_scav = 0.19, ligand_total_val = ligand_conc, beta_val = beta_val_1, \
                                                        symb_Cu = 'Cu(II)', m_conc_Cu_II_1 = 0, m_conc_Cu_II_2 = 0, m_conc_Cu_II_3 = 0, \
                                                            in1_Cu_II = F_in1, in2_Cu_II = F_in2, alpha_Cu_II = alpha_Cu_II_val, k_scav_Cu_II = 0.19, \
                                                                beta_val_Cu_II = beta_val_Cu_II_val, R_M_Cu_II = R_Cu_II, K_sat_Cu_II = K_sat_Cu_II_val, \
                                                                    ligand_use_Cu_II = True, use_ligand_cycling_Cu_II = True, \
                                                                        gamma_Cu_II = gamma_Cu_II_val, lambda_ligand_Cu_II = lambda_ligand_Cu_II_val)
            