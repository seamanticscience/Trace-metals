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
                               k_scav = 0, ligand_total_val = 0, beta_val = 0, \
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
            None
        Returns
            Float, representing the total export of organic matter from box 1 according to current
            concentration of light, nutrient, and metal.
        """
        global light_dependent_change_in_C_1
        global nutrient_dependent_change_in_C_1
        global metal_dependent_change_in_C_1
        light_dependent_change_in_C_1 = (Ibox1/(K_sat_l + Ibox1))
        nutrient_dependent_change_in_C_1 = ((C_1_input)/(K_sat_N + C_1_input))
        
        # Convert Passed In Dictionary to List
        metal_1_list_tent = [element for element in metal_1_dict_tent.items()]
        
        # Initiate list of tuples with metal concentration and K_sat_M.
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
            None
        Returns
            Float, representing the total export of organic matter from box 1 according to current
            concentration of light, nutrient, and metal.
        """
        global light_dependent_change_in_C_2
        global nutrient_dependent_change_in_C_2
        global metal_dependent_change_in_C_2
        light_dependent_change_in_C_2 = (Ibox2/(K_sat_l + Ibox2))
        nutrient_dependent_change_in_C_2 = ((C_2_input)/(K_sat_N + C_2_input))

        # Convert passed-in dict to list
        
        metal_2_list_tent = [element for element in metal_2_dict_tent.items()]

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
          
    # Concentration of Nutrients --------------------------------------------------
    # def dC_1_over_dt(metal_1_list_temp, K_sat_M_list_temp, C_1_input = C_1, C_2_input = C_2, C_3_input = C_3):
    #     """
    #     Calculates change in concentration of C_1 per cubic meter, in units of 
    #     moles of C_1 per cubic meter per unit time.
        
    #     The quanities needed are stored in the variables defined earlier. 
        
    
    #     Returns
    #     -------
    #     Number quantity reflecting change of C_1 per unit time, governed by the flow
    #     rates and the concentrations at the given times.
    
    #     """
            
    #     return (psi*(C_3_input - C_1_input) + k_31*(C_3_input - C_1_input) + k_21*(C_2_input - C_1_input))/vol_1 \
    #             - export_1(metal_1_list_temp, K_sat_M_list_temp, C_1_input) 
                        
    #     # Line 1: General tracer equation, maintains equilibrium among all three boxes with flow rate considered.
    #     # Line 2: Given fixed export rate lambda_1, considers the box's export rate dependent on nutrient concentration in given box.
    #     # Line 3: Amount of matter exported according to the given export function.
                
                    
    # def dC_2_over_dt(metal_2_list_temp, K_sat_M_list_temp, C_1_input = C_1, C_2_input = C_2, C_3_input = C_3):
    #     """
    #     Calculates change in concentration of C_2 per cubic meter, in units of 
    #     moles of C_2 per cubic meter per unit time.
        
    #     The quanities needed are stored in the variables defined earlier. 
        
    
    #     Returns
    #     -------
    #     Number quantity reflecting change of C_2 per unit time, governed by the flow
    #     rates and the concentrations at the given times.
    
    #     """
        
    #     return (psi*(C_1_input - C_2_input) + k_12*(C_1_input - C_2_input) + k_32*(C_3_input - C_2_input))/vol_2 \
    #             - export_2(metal_2_list_temp, K_sat_M_list_temp, C_2_input)

    #     # Line 1: General tracer equation, maintains equilibrium among all three boxes with flow rate considered.
    #     # Line 2: Amount of matter exported according to the given export function.
            
    # def dC_3_over_dt(metal_1_list_temp, metal_2_list_temp, K_sat_M_list_temp, \
    #                  C_1_input = C_1, C_2_input = C_2, C_3_input = C_3):
    #     """
    #     Calculates change in concentration of C_3 per cubic meter, in units of 
    #     moles of C_3 per cubic meter per unit time.
        
    #     The quanities needed are stored in the variables defined earlier. 
        
    
    #     Returns
    #     -------
    #     Number quantity reflecting change of C_3 per unit time, governed by the flow
    #     rates and the concentrations at the given times.
    
    #     """
        
    #     return (psi*(C_2_input - C_3_input) + k_23*(C_2_input - C_3_input) + k_13*(C_1_input - C_3_input))/vol_3 + \
    #             + (export_1(metal_1_list_temp, K_sat_M_list_temp, C_1_input)*vol_1 + export_2(metal_2_list_temp, K_sat_M_list_temp, C_2_input)*vol_2)/vol_3
                
    #     # Line 1: General tracer equation, maintains equilibrium among all three boxes with flow rate considered.
    #     # Line 2: Amount of matter exported according to the given export function.

    # # Concentration of metal (or any trace metal in fact) -----------------------------
    
    # def dM_1_over_dt(metal_1_list_temp, K_sat_M_list_temp, \
    #                  alpha_temp, M_in1_temp, k_scav_temp, beta_val_temp, R_M_temp, \
    #                  M_1_input = M_1, M_2_input = M_2, M_3_input = M_3, \
    #                  C_1_input = C_1, L_1_input = L_1):
    #     """
    #     Calculates change in concentration of M_1_input per cubic meter, in units of 
    #     moles of M_1 per cubic meter per unit time.
        
    #     The quanities needed are stored in the variables defined earlier. 
        
    
    #     Returns
    #     -------
    #     Number quantity reflecting change of M_1 per unit time, governed by the flow
    #     rates and the concentrations at the given times.
    #     """
            
    #     return (psi*(M_3_input - M_1_input) + k_31*(M_3_input - M_1_input) + k_21*(M_2_input - M_1_input))/vol_1 + \
    #         alpha_temp*M_in1_temp/dz_1 - k_scav_temp*complexation(M_1_input, L_1_input, beta_val_temp)/(60*60*24*365) - R_M_temp*export_1(metal_1_list_temp, K_sat_M_list_temp, C_1_input)
            
    #             # Line 1: General tracer equation, maintains equilibrium among all three boxes with flow rate considered.
    #             # Line 2: First term represents source, second term represents sink (in terms of being scavenged)
    #                 # Third term represents amount being used up ('biological utilization' as in Parekh, 2004)

    # def dM_2_over_dt(metal_2_list_temp, K_sat_M_list_temp, \
    #                  alpha_temp, M_in2_temp, k_scav_temp, beta_val_temp, R_M_temp, \
    #                  M_1_input = M_1, M_2_input = M_2, M_3_input = M_3, \
    #                  C_2_input = C_2, L_2_input = L_2):
    #     """
    #     Calculates change in concentration of M_2 per cubic meter, in units of 
    #     moles of M_2 per cubic meter per unit time.
        
    #     The quanities needed are stored in the variables defined earlier. 
        
    
    #     Returns
    #     -------
    #     Number quantity reflecting change of M_2 per unit time, governed by the flow
    #     rates and the concentrations at the given times.
    #     """
            
    #     return (psi*(M_1_input - M_2_input) + k_12*(M_1_input - M_2_input) + k_32*(M_3_input - M_2_input))/vol_2 + \
    #        alpha_temp*M_in2_temp/dz_2 - k_scav_temp*complexation(M_2_input, L_2_input, beta_val_temp)/(60*60*24*365) - R_M_temp*export_2(metal_2_list_temp, K_sat_M_list_temp, C_2_input)
        
    # def dM_3_over_dt(metal_1_list_temp, metal_2_list_temp, K_sat_M_list_temp, \
    #                  alpha_temp, k_scav_temp, beta_val_temp, R_M_temp, \
    #                  M_1_input = M_1, M_2_input = M_2, M_3_input = M_3, \
    #                  C_1_input = C_1, C_2_input = C_2, \
    #                  L_1_input = L_1, L_2_input = L_2, L_3_input = L_3):
    #     """
    #     Calculates change in concentration of M_3 per cubic meter, in units of 
    #     moles of M_3 per cubic meter per unit time.
        
    #     The quanities needed are stored in the variables defined earlier. 
        
    
    #     Returns
    #     -------
    #     Number quantity reflecting change of M_2 per unit time, governed by the flow
    #     rates and the concentrations at the given times.
    #     """

    #     return (psi*(M_2_input - M_3_input) + k_23*(M_2_input - M_3_input) + k_13*(M_1_input - M_3_input))/vol_3 \
    #          - k_scav_temp*complexation(M_3_input, L_3_input, beta_val_temp)/(60*60*24*365) \
    #             + R_M_temp*(export_1(metal_1_list_temp, K_sat_M_list_temp, C_1_input)*vol_1 + export_2(metal_2_list_temp, K_sat_M_list_temp, C_2_input)*vol_2)/vol_3

    # ### To find the free ion concentration at any given moment, the following function
    # ### calculates exactly that given our concentration of ligand, metal, and beta constant.
    # ### (via complexation)
    
    # # Concentration of Ligands -----------------------------------------------------
    
    # def dLt_1_over_dt(gamma_temp, lambda_ligand_temp, \
    #                   metal_1_list_temp, K_sat_M_list_temp, \
    #                   L_1_input = L_1, L_2_input = L_2, L_3_input = L_3, \
    #                   C_1_input = C_1, M_1_input = M_1):
    #     """
    #     Calculates change in total ligand in specified box in units of mols per cubic meter.
    #     Addresses ligands cycling through the three boxes, as well as sources/sinks.
    #     Parameters
    #     ----------
    #         None.
    #     Returns
    #     -------
    #     Value in units of mols per cubic meter per second (changing concentration of ligand)
    #     """
    #     return (psi*(L_3_input - L_1_input) + k_31*(L_3_input - L_1_input) + k_21*(L_2_input - L_1_input))/vol_1 \
    #         + gamma_temp*export_1(metal_1_list_temp, K_sat_M_list_temp, C_1_input) \
    #             - lambda_ligand_temp*L_1_input

    #                 # Line 1: Cycling of ligands in and out of box 1.
    #                 # Line 2: Source (with appropriate gamma)
    #                 # Line 3: Loss of ligands to degredation.
                
    # def dLt_2_over_dt(gamma_temp, lambda_ligand_temp, \
    #                   metal_2_list_temp, K_sat_M_list_temp, \
    #                   L_1_input = L_1, L_2_input = L_2, L_3_input = L_3, \
    #                   C_2_input = C_2, M_2_input = M_2):
    #     """
    #     Calculates change in total ligand in specified box in units of mols per cubic meter.
    #     Addresses ligands cycling through the three boxes, as well as sources/sinks.
    #     Parameters
    #     ----------
    #         None.
    #     Returns
    #     -------
    #     Value in units of mols per cubic meter per second (changing concentration of ligand)
    #     """        
    #     return (psi*(L_1_input - L_2_input) + k_12*(L_1_input - L_2_input) + k_32*(L_3_input - L_2_input))/vol_2 \
    #         + gamma_temp*export_2(metal_2_list_temp, K_sat_M_list_temp, C_2_input) \
    #             - lambda_ligand_temp*L_2_input
                
    #                 # Line 1: Cycling of ligands in and out of box 2.
    #                 # Line 2: Source (with appropriate gamma)
    #                 # Line 3: Loss of ligands to degredation.
                    
    # def dLt_3_over_dt(gamma_temp, lambda_ligand_temp, \
    #                   metal_1_list_temp, metal_2_list_temp, K_sat_M_list_temp, \
    #                   L_1_input = L_1, L_2_input = L_2, L_3_input = L_3, \
    #                   C_1_input = C_1, C_2_input = C_2, \
    #                   M_1_input = M_1, M_2_input = M_2):
    #     """
    #     Calculates change in total ligand in specified box in units of mols per cubic meter.
    #     Addresses ligands cycling through the three boxes, as well as sources/sinks.
    #     Parameters
    #     ----------
    #         None.
    #     Returns
    #     -------
    #     Value in units of mols per cubic meter per second (changing concentration of ligand)
    #     """        
    #     return (psi*(L_2_input - L_3_input) + k_23*(L_2_input - L_3_input) + k_13*(L_1_input - L_3_input))/vol_3 \
    #         - lambda_ligand_temp/100*L_3_input \
    #             + gamma_temp/vol_3*(export_1(metal_1_list_temp, K_sat_M_list_temp, C_1_input)*vol_1 + export_2(metal_2_list_temp, K_sat_M_list_temp, C_2_input)*vol_2)
            
    #         # Line 1: Cycling of ligands
    #         # Line 2: Loss of ligand
    #         # Line 3: Input of ligands based on export 'reception'
        
    # Complexation, causes differentiation between total and free metal (or any other metal) ------------------------
    
    
    def dtotal_dt(C_1_input, C_2_input, C_3_input, \
                  L_1_input, L_2_input, L_3_input, \
                  conc_name_list_temp, \
                  gamma_temp, lambda_ligand_temp, \
                  metal_1_dict_temp, metal_2_dict_temp, metal_3_dict_temp, \
                  K_sat_M_list_temp, alpha_dict_temp, M_in1_dict_temp, M_in2_dict_temp, k_scav_dict_temp, R_M_dict_temp, \
                  beta_val_dict_temp, dt_temp):
        """
        Takes all initial concentrations and returns all updated values dependent on said
            initial values.

        Returns
        -------
        Dictionary mapping element/metal to new concentration value.

        """
        def dM1dt(M_1_input, M_2_input, M_3_input, alpha_temp, M_in1_temp, k_scav_temp, R_M_temp, beta_val_temp):
            return M_1_input + dt_temp*(psi*(M_3_input - M_1_input) + k_31*(M_3_input - M_1_input) + k_21*(M_2_input - M_1_input))/vol_1 + \
                    alpha_temp*M_in1_temp/dz_1 - k_scav_temp*complexation(M_1_input, L_1_input, beta_val_temp)/(60*60*24*365) - R_M_temp*export_1(metal_1_dict_temp, K_sat_M_list_temp, C_1_input)
            
                # Line 1: General tracer equation, maintains equilibrium among all three boxes with flow rate considered.
                # Line 2: First term represents source, second term represents sink (in terms of being scavenged)
                    # Third term represents amount being used up ('biological utilization' as in Parekh, 2004)

        def dM2dt(M_1_input, M_2_input, M_3_input, alpha_temp, M_in2_temp, k_scav_temp, R_M_temp, beta_val_temp):
            return M_2_input + dt_temp*(psi*(M_1_input - M_2_input) + k_12*(M_1_input - M_2_input) + k_32*(M_3_input - M_2_input))/vol_2 + \
                    alpha_temp*M_in2_temp/dz_2 - k_scav_temp*complexation(M_2_input, L_2_input, beta_val_temp)/(60*60*24*365) - R_M_temp*export_2(metal_2_dict_temp, K_sat_M_list_temp, C_2_input)

        def dM3dt(M_1_input, M_2_input, M_3_input, alpha_temp, k_scav_temp, R_M_temp, beta_val_temp):
            return M_3_input + dt_temp*(psi*(M_2_input - M_3_input) + k_23*(M_2_input - M_3_input) + k_13*(M_1_input - M_3_input))/vol_3 \
                - k_scav_temp*complexation(M_3_input, L_3_input, beta_val_temp)/(60*60*24*365) \
                + R_M_temp*(export_1(metal_1_dict_temp, K_sat_M_list_temp, C_1_input)*vol_1 + export_2(metal_2_dict_temp, K_sat_M_list_temp, C_2_input)*vol_2)/vol_3
        
        
        dC1dt = ('C_1', C_1_input + dt_temp*(psi*(C_3_input - C_1_input) + k_31*(C_3_input - C_1_input) + k_21*(C_2_input - C_1_input))/vol_1 \
                - export_1(metal_1_dict_temp, K_sat_M_list_temp, C_1_input))
        dC2dt = ('C_2', C_2_input + dt_temp*(psi*(C_1_input - C_2_input) + k_12*(C_1_input - C_2_input) + k_32*(C_3_input - C_2_input))/vol_2 \
                - export_2(metal_2_dict_temp, K_sat_M_list_temp, C_2_input))
        dC3dt = ('C_3', C_3_input + dt_temp*(psi*(C_2_input - C_3_input) + k_23*(C_2_input - C_3_input) + k_13*(C_1_input - C_3_input))/vol_3 + \
                + (export_1(metal_1_dict_temp, K_sat_M_list_temp, C_1_input)*vol_1 + export_2(metal_2_dict_temp, K_sat_M_list_temp, C_2_input)*vol_2)/vol_3)
        dL1dt = ('L_1', L_1_input + dt_temp*(psi*(L_3_input - L_1_input) + k_31*(L_3_input - L_1_input) + k_21*(L_2_input - L_1_input))/vol_1 \
                 + gamma_temp*export_1(metal_1_dict_temp, K_sat_M_list_temp, C_1_input) \
                - lambda_ligand_temp*L_1_input)
        dL2dt = ('L_2', L_2_input + dt_temp*(psi*(L_1_input - L_2_input) + k_12*(L_1_input - L_2_input) + k_32*(L_3_input - L_2_input))/vol_2 \
                 + gamma_temp*export_2(metal_2_dict_temp, K_sat_M_list_temp, C_2_input) \
                - lambda_ligand_temp*L_2_input)
        dL3dt = ('L_3', L_3_input + dt_temp*(psi*(L_2_input - L_3_input) + k_23*(L_2_input - L_3_input) + k_13*(L_1_input - L_3_input))/vol_3 \
                 - lambda_ligand_temp/100*L_3_input \
                + gamma_temp/vol_3*(export_1(metal_1_dict_temp, K_sat_M_list_temp, C_1_input)*vol_1 + export_2(metal_2_dict_temp, K_sat_M_list_temp, C_2_input)*vol_2))
        
        return_list = [dC1dt, dC2dt, dC3dt, dL1dt, dL2dt, dL3dt]
        
        m_1_temp = []
        m_2_temp = []
        m_3_temp = []
        
        for metal_conc in conc_name_list_temp:
            temp_tuple_1 = (f'{metal_conc}_1', dM1dt(metal_1_dict_temp[f'{metal_conc}_1'], metal_2_dict_temp[f'{metal_conc}_2'], metal_3_dict_temp[f'{metal_conc}_3'], \
                                                         alpha_dict_temp[metal_conc], M_in1_dict_temp[metal_conc], k_scav_dict_temp[metal_conc], \
                                                             R_M_dict_temp[metal_conc], beta_val_dict_temp[metal_conc]))
            return_list.append(temp_tuple_1)
            m_1_temp.append(temp_tuple_1)
            
            temp_tuple_2 = (f'{metal_conc}_2', dM2dt(metal_1_dict_temp[f'{metal_conc}_1'], metal_2_dict_temp[f'{metal_conc}_2'], metal_3_dict_temp[f'{metal_conc}_3'], \
                                                         alpha_dict_temp[metal_conc], M_in2_dict_temp[metal_conc], k_scav_dict_temp[metal_conc], \
                                                             R_M_dict_temp[metal_conc], beta_val_dict_temp[metal_conc]))
            return_list.append(temp_tuple_2)
            m_2_temp.append(temp_tuple_2)
            
            temp_tuple_3 = (f'{metal_conc}_3', dM3dt(metal_1_dict_temp[f'{metal_conc}_1'], metal_2_dict_temp[f'{metal_conc}_2'], metal_3_dict_temp[f'{metal_conc}_3'], \
                                                         alpha_dict_temp[metal_conc], k_scav_dict_temp[metal_conc], \
                                                             R_M_dict_temp[metal_conc], beta_val_dict_temp[metal_conc]))
            return_list.append(temp_tuple_3)
            m_3_temp.append(temp_tuple_3)
                
        # Now that we have our return_list, update the dictionaries from which we got the concentration values to be used in the next iteration(s).
        
        
        for conc_index in range(len(m_1_temp)):
            metal_1_dict_temp[m_1_temp[conc_index][0]] = m_1_temp[conc_index][1]
            metal_2_dict_temp[m_2_temp[conc_index][0]] = m_2_temp[conc_index][1]
            metal_3_dict_temp[m_3_temp[conc_index][0]] = m_3_temp[conc_index][1]

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
    
    metal_symbol_list = [('symb_M', metal_type), ]
    metal_1_list = [('m_conc_M_1', M_1), ]
    metal_2_list = [('m_conc_M_2', M_2), ]
    metal_3_list = [('m_conc_M_3', M_3), ]
    metal_in_1_list = [('in1_M', M_in1), ]
    metal_in_2_list = [('in2_M', M_in2), ]
    alpha_list = [('alpha_M', alpha), ]
    k_scav_list = [('k_scav_M', k_scav), ]
    beta_val_list = [('beta_val_M', beta_val), ]
    R_M_list = [('R_M_M', R_M), ]
    K_sat_M_list = [('K_sat_M', K_sat_M), ]
    ligand_use_list = [('ligand_use_M', ligand_use), ]
    use_ligand_cycling_list = [('use_ligand_cycling_M', use_ligand_cycling), ]
    gamma_list = [('gamma_M', gamma), ]
    lambda_ligand_list = [('lambda_ligand_M', lambda_ligand), ]
    # L_1_init_list = [('L_1_M', L_1), ]
    # L_2_init_list = [('L_2_M', L_2), ]
    # L_3_init_list = [('L_3_M', L_3), ]
    

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
            lambda_ligand.append(parameter_val)
        # elif parameter_val[0].startswith('L_1_'):
        #     L_1_init_list.append(parameter_val)
        # elif parameter_val[0].startswith('L_2_'):
        #     L_2_init_list.append(parameter_val)
        # elif parameter_val[0].startswith('L_3_'):
        #     L_3_init_list.append(parameter_val)
            
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
    # L_1_init_list.sort()
    # L_2_init_list.sort()
    # L_3_init_list.sort()

    ## After collecting the above information into appropriate lists, convert them into dictionaries where
    ## the key is in format 'element + box label' to make the next functions readable and accessible via key rather than order. 
    
    
    # Metal Concentrations
    
    all_concs = [('C_1', C_1), ('C_2', C_2), ('C_3', C_3), \
             ('L_1', L_1), ('L_2', L_2), ('L_3', L_3)]
        
    # Iterate over all three lists so that the first element of the tuple is renamed
    # to the metal symbol, followed by element concentration. 
    
    metal_conc_symbol_tot = []
    metal_1_conc_symbol = []
    metal_2_conc_symbol = []
    metal_3_conc_symbol = []
    for val_index in range(len(metal_symbol_list)):
        metal_conc_symbol_tot.append((f'{metal_symbol_list[val_index][1]}_1', metal_1_list[val_index][1]))
        metal_conc_symbol_tot.append((f'{metal_symbol_list[val_index][1]}_2', metal_2_list[val_index][1]))
        metal_conc_symbol_tot.append((f'{metal_symbol_list[val_index][1]}_3', metal_3_list[val_index][1]))
        metal_1_conc_symbol.append((f'{metal_symbol_list[val_index][1]}_1', metal_1_list[val_index][1]))
        metal_2_conc_symbol.append((f'{metal_symbol_list[val_index][1]}_2', metal_2_list[val_index][1]))
        metal_3_conc_symbol.append((f'{metal_symbol_list[val_index][1]}_3', metal_3_list[val_index][1]))
    
    all_concs.extend(metal_conc_symbol_tot)
    
    init_concs = dict(all_concs)
    init_concs_metal_1 = dict(metal_1_conc_symbol)
    init_concs_metal_2 = dict(metal_2_conc_symbol)
    init_concs_metal_3 = dict(metal_3_conc_symbol)

    ## Converting the rest of the important lists into dictionaries with same keys.
    
    metal_name_list = [element_symbol[1] for element_symbol in metal_symbol_list]
    
    metal_in1_dict = {}
    metal_in2_dict = {}
    alpha_dict = {}
    k_scav_dict = {}
    beta_val_dict = {}
    R_M_dict = {}
    
    for var_index in range(len(metal_name_list)):
        metal_in1_dict[metal_name_list[var_index]] = metal_in_1_list[var_index][1]
        metal_in2_dict[metal_name_list[var_index]] = metal_in_2_list[var_index][1]
        alpha_dict[metal_name_list[var_index]] = alpha_list[var_index][1]
        k_scav_dict[metal_name_list[var_index]] = k_scav_list[var_index][1]
        beta_val_dict[metal_name_list[var_index]] = beta_val_list[var_index][1]
        R_M_dict[metal_name_list[var_index]] = R_M_list[var_index][1]
        
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
    
    # Initiate pandas dataframe with time_axis_array
    conc_pd = pd.DataFrame(data = time_axis_array, \
                 columns = ["Time"]).set_index("Time")

    # Expandable list of biogeochemistry tracers
    all_symbols = [('placeholder1', "C"), ('placeholder2', "L")]
    all_symbols.extend(metal_symbol_list)
        # Creating list with all necessary symbols
    
    conc_list = [element_symbol[1] for element_symbol in all_symbols]
    
    
    # Box number labels
    nbox = list(map(str, np.arange(1,4)))
    
    # Create labels for each tracer in each box
    conc_vars = list(map('_'.join, itertools.product(conc_list, nbox)))
      
    # Create DataFrame with initial concentrations for all variables. 
    
    # First combine lists of all concentrations, with intent to convert to dictionary of
    # initial concentrations.
    
    all_concs = [('C_1', C_1), ('C_2', C_2), ('C_3', C_3), \
                 ('L_1', L_1), ('L_2', L_2), ('L_3', L_3)]
        
    # Iterate over all three lists so that the first element of the tuple is renamed
    # to the metal symbol, followed by element concentration. 
    
    metal_conc_symbol_tot = []
    metal_1_conc_symbol = []
    metal_2_conc_symbol = []
    for val_index in range(len(metal_symbol_list)):
        metal_conc_symbol_tot.append((f'{metal_symbol_list[val_index][1]}_1', metal_1_list[val_index][1]))
        metal_conc_symbol_tot.append((f'{metal_symbol_list[val_index][1]}_2', metal_2_list[val_index][1]))
        metal_conc_symbol_tot.append((f'{metal_symbol_list[val_index][1]}_3', metal_3_list[val_index][1]))
        metal_1_conc_symbol.append((f'{metal_symbol_list[val_index][1]}_1', metal_1_list[val_index][1]))
        metal_2_conc_symbol.append((f'{metal_symbol_list[val_index][1]}_2', metal_2_list[val_index][1]))
    
    all_concs.extend(metal_conc_symbol_tot)
    
    init_concs = dict(all_concs)
    
    # Finally add these initial values to the start of the dataframe
    
    for var in conc_vars:
        conc_pd.loc[0, var] = init_concs[var]

    # Iterate over time axis array.
    
    for t_val in range(len(time_axis_array))[1:]:
        conc_pd.loc[t_val] = dtotal_dt\
            (conc_pd.loc[(t_val - 1), 'C_1'], conc_pd.loc[(t_val - 1), 'C_2'], conc_pd.loc[(t_val - 1), 'C_3'], \
                  conc_pd.loc[(t_val - 1), 'L_1'], conc_pd.loc[(t_val - 1), 'L_2'], conc_pd.loc[(t_val - 1), 'L_3'], \
                  metal_name_list, \
                  gamma, lambda_ligand, \
                  init_concs_metal_1, init_concs_metal_2, init_concs_metal_3, \
                  K_sat_M_list, alpha_dict, metal_in1_dict, metal_in2_dict, k_scav_dict, R_M_dict, \
                  beta_val_dict, dt)
        print(conc_pd)

    print(conc_pd)

    raise NotImplementedError
    
    # init_concs_frame = pd.DataFrame()

        # Initiate lists that will store the three concentrations, with initial concentrations already
        # in the lists.
    
    # Initialize lists that will store values of all metal and ligand concentrations, as well as temporary variables 
    # for the nutrient concentrations. 
    
    
    
    
    
    # C_1_list = [C_1,]
    # C_2_list = [C_2,]
    # C_3_list = [C_3,]
    
                            # metal_1_concentrations = []
                            # metal_2_concentrations = []
                            # metal_3_concentrations = []
                            # ligand_1_concentrations = []
                            # ligand_2_concentrations = []
                            # ligand_3_concentrations = []
                            
                            # for init_conc_index in range(len(metal_symbol_list)):
                            #     metal_1_concentrations.append([metal_1_list[init_conc_index][1],])
                            #     metal_2_concentrations.append([metal_2_list[init_conc_index][1],])
                            #     metal_3_concentrations.append([metal_3_list[init_conc_index][1],])        
                            #     ligand_1_concentrations.append([L_1_init_list[init_conc_index][1],])
                            #     ligand_2_concentrations.append([L_2_init_list[init_conc_index][1],])
                            #     ligand_3_concentrations.append([L_3_init_list[init_conc_index][1],])      
    
    # C_1_temp, C_2_temp, C_3_temp = C_1, C_2, C_3        

    # Finally initiate iteration(s).
    
    for t_val in time_axis_array[1:]:

            C_1_temp += dC_1_over_dt(metal_1_list, K_sat_M_list, C_1, C_2, C_3)*dt
            C_1_list.append(C_1_temp)
                # Use Euler Step Function to change value of C_1 by one time step (i.e. dt). Then 
                # append that value to the C_1_list of concentrations as the concentration for that
                # given time. 
            C_2_temp += dC_2_over_dt(metal_2_list, K_sat_M_list, C_1, C_2, C_3)*dt
            C_2_list.append(C_2_temp)
                # Use Euler Step Function to change value of C_2 by one time step (i.e. dt). Then 
                # append that value to the C_2_list of concentrations as the concentration for that
                # given time. 
            C_3_temp += dC_3_over_dt(metal_1_list, metal_2_list, K_sat_M_list, C_1, C_2, C_3)*dt
            C_3_list.append(C_3_temp)
                # Use Euler Step Function to change value of C_3 by one time step (i.e. dt). Then 
                # append that value to the C_3_list of concentrations as the concentration for that
                # given time. 
                
            ## Run through all metal lists/ ligand lists to update + add concentrations.

            # These lists will store updated init values, later to update the original init list.
            
            metal_1_updated_init_list = []
            metal_2_updated_init_list = []
            metal_3_updated_init_list = []
            ligand_1_updated_init_list = []
            ligand_2_updated_init_list = []
            ligand_3_updated_init_list = []
            
            
            for list_index in range(len(metal_symbol_list)):
                metal_1_concentrations[list_index].append(metal_1_concentrations[list_index][-1] + \
                    dM_1_over_dt(metal_1_list, K_sat_M_list, \
                       alpha_list[list_index][1], metal_in_1_list[list_index][1],\
                       k_scav_list[list_index][1], beta_val_list[list_index][1], R_M_list[list_index][1], \
                       M_1_input = metal_1_list[list_index][1], M_2_input = metal_2_list[list_index][1], \
                       M_3_input = metal_3_list[list_index][1], \
                       C_1_input = C_1, L_1_input = L_1_init_list[list_index][1]))
                metal_1_updated_init_list.append((metal_1_list[list_index][0], metal_1_concentrations[list_index][-1]))
                
                metal_2_concentrations[list_index].append(metal_2_concentrations[list_index][-1] + \
                    dM_2_over_dt(metal_2_list, K_sat_M_list, \
                       alpha_list[list_index][1], metal_in_2_list[list_index][1],\
                       k_scav_list[list_index][1], beta_val_list[list_index][1], R_M_list[list_index][1], \
                       M_1_input = metal_1_list[list_index][1], M_2_input = metal_2_list[list_index][1], \
                       M_3_input = metal_3_list[list_index][1], \
                       C_2_input = C_2, L_2_input = L_2_init_list[list_index][1]))
                metal_2_updated_init_list.append((metal_2_list[list_index][0], metal_2_concentrations[list_index][-1]))
                
                metal_3_concentrations[list_index].append(metal_3_concentrations[list_index][-1] + \
                    dM_3_over_dt(metal_1_list, metal_2_list, K_sat_M_list, \
                       alpha_list[list_index][1], k_scav_list[list_index][1], beta_val_list[list_index][1], R_M_list[list_index][1], \
                       M_1_input = metal_1_list[list_index][1], M_2_input = metal_2_list[list_index][1], \
                       M_3_input = metal_3_list[list_index][1], \
                       C_1_input = C_1, C_2_input = C_2, \
                       L_1_input = L_1_init_list[list_index][1], L_2_input = L_2_init_list[list_index][1], \
                       L_3_input = L_3_init_list[list_index][1]))
                metal_3_updated_init_list.append((metal_3_list[list_index][0], metal_3_concentrations[list_index][-1]))
                                
                ligand_1_concentrations[list_index].append(ligand_1_concentrations[list_index][-1] + \
                    dLt_1_over_dt(gamma_list[list_index][1], lambda_ligand_list[list_index][1], metal_1_list, \
                       K_sat_M_list, L_1_init_list[list_index][1], L_2_init_list[list_index][1], \
                       L_3_init_list[list_index][1], C_1, metal_1_list[list_index][1]))
                ligand_1_updated_init_list.append((L_1_init_list[list_index][0], ligand_1_concentrations[list_index][-1]))
                
                ligand_2_concentrations[list_index].append(ligand_2_concentrations[list_index][-1] + \
                    dLt_2_over_dt(gamma_list[list_index][1], lambda_ligand_list[list_index][1], metal_2_list, \
                       K_sat_M_list, L_1_init_list[list_index][1], L_2_init_list[list_index][1], \
                       L_3_init_list[list_index][1], C_2, metal_2_list[list_index][1]))
                ligand_2_updated_init_list.append((L_2_init_list[list_index][0], ligand_2_concentrations[list_index][-1]))
                
                ligand_3_concentrations[list_index].append(ligand_3_concentrations[list_index][-1] + \
                    dLt_3_over_dt(gamma_list[list_index][1], lambda_ligand_list[list_index][1], metal_1_list, \
                       metal_2_list, K_sat_M_list, L_1_init_list[list_index][1], L_2_init_list[list_index][1], \
                       L_3_init_list[list_index][1], C_1, C_2, metal_1_list[list_index][1], metal_2_list[list_index][1]))
                ligand_3_updated_init_list.append((L_3_init_list[list_index][0], ligand_3_concentrations[list_index][-1]))
                
            ## Now update all values to their updated temp values.
            
            C_1, C_2, C_3 = C_1_temp, C_2_temp, C_3_temp
            
            metal_1_list = metal_1_updated_init_list
            metal_2_list = metal_2_updated_init_list
            metal_3_list = metal_3_updated_init_list
            L_1_init_list = ligand_1_updated_init_list
            L_2_init_list = ligand_2_updated_init_list
            L_3_init_list = ligand_3_updated_init_list

    
    C_array = np.array([C_1_list, C_2_list, C_3_list])

        
    def plot_concentrations(title, metal_symbol_list_temp, time_axis_array_temp, array_of_C, \
                            metal_1_concentrations_temp, metal_2_concentrations_temp, metal_3_concentrations_temp, \
                                ligand_1_concentrations_temp, ligand_2_concentrations_temp, ligand_3_concentrations_temp):
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
        # conc_name_list = ['C_1', 'C_2', 'C_3', f'{metal_type}_1', f'{metal_type}_2', f'{metal_type}_3', 'L_1', 'L_2', 'L_3']

        def plot_metal_or_ligand(ml_name, time_axis_array_tempo, \
                       ml_concentrations_1, ml_concentrations_2, ml_concentrations_3):
            '''
            

            Parameters
            ----------
            ml_name : TYPE
                DESCRIPTION.
            ml_concentrations_1 : TYPE
                DESCRIPTION.
            ml_concentrations_2 : TYPE
                DESCRIPTION.
            ml_concentrations_3 : TYPE
                DESCRIPTION.

            Returns
            -------
            None.

            '''
            
            plt.figure()
            plt.title(f'{title} \n {ml_name} Concentrations Over Time (in mol/m3) \n')
            plt.xlabel('Time [log(years)]')
            plt.ylabel(f'Concentration of {ml_name} (in mol/m3)')
            plt.plot(time_axis_array_tempo, ml_concentrations_1, 'r', label = f'{ml_name}_1')
            plt.plot(time_axis_array_tempo, ml_concentrations_2, 'g', label = f'{ml_name}_2')
            plt.plot(time_axis_array_tempo, ml_concentrations_3, 'b', label = f'{ml_name}_3')

            plt.legend(loc = 'best')
            plt.show()
            plt.close()
            
        # Plot Nutrient Concentration
        plot_metal_or_ligand('C', time_axis_array_temp, \
                             array_of_C[0], array_of_C[1], array_of_C[2])
        
        # Plot Metals and Ligands
        for metal_index in range(len(metal_symbol_list_temp)):
            plot_metal_or_ligand(f'{metal_symbol_list_temp[metal_index][1]}', time_axis_array_temp, \
                                 np.array(metal_1_concentrations_temp[metal_index]), \
                                     np.array(metal_2_concentrations_temp[metal_index]), \
                                         np.array(metal_3_concentrations_temp[metal_index]))
            plot_metal_or_ligand(f'{metal_symbol_list_temp[metal_index][1]} Ligand', time_axis_array_temp, \
                                 np.array(ligand_1_concentrations_temp[metal_index]), \
                                     np.array(ligand_2_concentrations_temp[metal_index]), \
                                         np.array(ligand_3_concentrations_temp[metal_index]))            
            
        
    plot_concentrations(title, metal_symbol_list, time_axis_array_log10, C_array, \
                            metal_1_concentrations, metal_2_concentrations, metal_3_concentrations, \
                                ligand_1_concentrations, ligand_2_concentrations, ligand_3_concentrations)
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



transport_model_graphing_ligand_approach = \
        create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.006849, 10000, \
                            'Concentrations of Nutrients, Iron, and Ligands over time, \n dt = 2.5 days, ligand concentration = 10**-6, beta = 10**8 (kg per mol) \n Michalis-Menten Model, Leibig Limit Approximation', 9, \
                                use_metal = True, metal_type = 'Fe', M_1 = 0, M_2 = 0, M_3 = 0, K_sat_M = K_sat_Fe, \
                                    M_in1 = F_in1, M_in2 = F_in2, alpha = alpha_Fe, R_M = R_Fe, \
                                        ligand_use = True, use_ligand_cycling = True, \
                                            L_1 = 0, L_2 = 0, L_3 = 0, \
                                                mic_ment_light_leibig = 1, \
                                                    k_scav = 0.19, ligand_total_val = ligand_conc, beta_val = beta_val_1)

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




# N_1_to_3 = 30*rho_0*10**(-6)
# Cu_1_2 = (1*10**-9)*(1000) # Converting value from mol/liter to mol/m3

# alpha_Cu_II = alpha_Fe
# R_Cu_II = R_Fe*(0.38/7.5) # Using elemental ratio.
# K_sat_Cu_II = K_sat_Fe*(0.38/7.5) # Using elemental ratios to convert between iron and copper. 

# ligand_conc = 2*10**-9 # mol/m3
# beta_val_Cu_II = math.exp(8.5)

# transport_model_graphing_ligand_approach = \
#         create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.006849, 10000, \
#                             'Concentrations of Nutrients, Iron, and Ligands over time, \n dt = 2.5 days, ligand concentration = 2*10**-9, beta = e**8.5 (kg per mol) \n Michalis-Menten Model, Leibig Limit Approximation', 9, \
#                                 use_metal = True, metal_type = 'Cu(II)', M_1 = Cu_1_2, M_2 = Cu_1_2, M_3 = 0, K_sat_M = K_sat_Cu_II, \
#                                     M_in1 = F_in1, M_in2 = F_in2, alpha = alpha_Cu_II, R_M = R_Cu_II, \
#                                         ligand_use = True, use_ligand_cycling = True, \
#                                             L_1 = ligand_conc, L_2 = ligand_conc, L_3 = 0, \
#                                                 mic_ment_light_leibig = 1, mic_ment_mult_lim = 0, \
#                                                     k_scav = 0.19, ligand_total_val = ligand_conc, beta_val = beta_val_Cu_II,  \
#                                                         symb_Cu = 1, m_conc_Cu_1 = 2, m_conc_Cu_2 = 3, m_conc_Cu_3 = 4, \
#                                                             in1_Cu = 5, in2_Cu = 6, alpha_Cu = 7, R_M_Cu = 8, K_sat_Cu = 9, \
#                                                                 ligand_use_Cu = 10, use_ligand_cycling_Cu = 11, \
#                                                                     L_1_Cu = 12, L_2_Cu = 13, L_3_Cu = 14)