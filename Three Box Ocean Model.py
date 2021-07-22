# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:19:20 2021

@author: Smah Riki
"""
import numpy as np
import matplotlib.pyplot as plt
import math

### -----------------------------------------------------------------------------

## Establishing Special NoneType Class to handle adding two NoneTypes down the road.

class No_obj(object):
    def __init__(self, name = 'def'):
        self.name = name
    def __add__(self, other_object):
        return None
    def __mult__(self, other_object):
        return No_obj()


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
alpha = 0.01*0.035 #Fe dust solubility
R_Fe = (2.5*10**-5)*6.625 #unitless, multiplied by 6.625 to convert this Fe:C ratio to Fe:N
K_sat_Fe = 2*10**-7 #units of mole per cubic meter
f_dop = 0.67 # Fraction of particles that makes it into pool of nitrate, unitless.

# Global Values for Ligand Cycling and Microbial Production
K_I = 45 #W/m2
gamma = 5*10**(-5)*(106/16) # Units of mol L/(mol N), converted using Redfield Ratio.
lambda_ligand = 5*10**(-5)/4398

### -------------------------------------------------------------------------------------------

## The following functions abstract the process of creating the transport model
## (using the ODEs we have generalized to the system).

def create_transport_model(C_1, C_2, C_3, dt_in_years, end_time, title, element_symbol, num_elements, lambda_1 = 0, lambda_2 = 0, \
                           mic_ment_nolight = 0, mic_ment_light_leibig = 0, mic_ment_light_mult_lim = 0, Ibox1 = 35, \
                               k_scav = 0, mu = 0, Fe_1 = None, Fe_2 = None, Fe_3 = None, \
                                   use_iron = False, \
                                   ligand_use = False, ligand_total_val = 0, beta_val = 0, \
                                       L_1 = None, L_2 = None, L_3 = None, \
                                           use_ligand_cycling = False, \
                                               one_graph = True, multi_graph = False):
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
        num_elements: Number of elements to be studied/ plotted. Not too useful 
            here, but will be useful when handling trace metals. 
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
        ligand_use: Boolean parameter, indicates whether we want to use ligands in this model (and whether
            our free iron pool will differ from the total iron pool thanks to complexation).
        ligand_total_val: float, initially set to 0 because by default we do not have any ligands in the model.
        beta_val: float, initially set to 0 because we have no equilibrium between the metal and ligand concentrations.
        L_1, L_2, L_3: Initial Concentrations of Ligand in the three boxes. 

    Returns
    -------
        Tuple of the form (array of floats to be used for the x-axis, \
                           (array of the concentrations over time for C_1, \
                            array of the concentrations over time for C_2, \
                                array of the concentrations over time for C_3))
        Plots graph with this information (not returned)
    """
    ## Initially check to see if the michaelis-menton parameters passed in are either 0 or 1; any other
    ## value, and raise a value error.
    
    ## Yet to be implemented.    
    
    ## Initiate Time Variables
    
    dt = dt_in_years*60*60*24*365  #Total number of seconds representing a year. 
    
    ## Initiate Variables that will "globally," at least within this function, keep track of 
    ## concentrations of C_1 and C_2 so that these values can be reused for C_3.
    light_dependent_change_in_C_1 = 0
    nutrient_dependent_change_in_C_1 = 0
    iron_dependent_change_in_C_1 = 0
    light_dependent_change_in_C_2 = 0
    nutrient_dependent_change_in_C_2 = 0
    iron_dependent_change_in_C_2 = 0
    
    # ...........................................................................
    
    ## Differential Functions
    
    # Cycling of Matter ---------------------------------------------
    # TBD

    
    # Input Concentrations All in a List
    
    # input_concentrations = [C_1_input, C_2_input, C_3_input, \
    #                         Fe_1_input, Fe_2_input, Fe_3_input, \
    #                             L_1_input, L_2_input, L_3_input]
    
    
    # [C_1_input = C_1, C_2_input = C_2, C_3_input = C_3, \
    #  Fe_1_input = Fe_1, Fe_2_input = Fe_2, Fe_3_input = Fe_3, \
    #      L_1_input = L_1, L_2_input = L_2, L_3_input = L_3]
    # Export Production --------------------------------------------------
    
    def export_1(C_1_input = C_1, Fe_1_input = Fe_1):
        """
        Calculates export of organic matter from box 1, considering the Michaelis-Menten
        approach and the liebig/multiplicative limit approach.
        
        Parameters:
            None
        Returns
            Float, representing the total export of organic matter from box 1 according to current
            concentration of light, nutrient, and iron.
        """
        global light_dependent_change_in_C_1
        global nutrient_dependent_change_in_C_1
        global iron_dependent_change_in_C_1
        light_dependent_change_in_C_1 = (Ibox1/(K_sat_l + Ibox1))
        nutrient_dependent_change_in_C_1 = ((C_1_input)/(K_sat_N + C_1_input))
        if Fe_1_input == None:
            iron_dependent_change_in_C_1 = None
        else:
            iron_dependent_change_in_C_1 = (Fe_1_input/(K_sat_Fe + Fe_1_input))
        return mic_ment_light_leibig*V_max*min(conc for conc in [light_dependent_change_in_C_1, nutrient_dependent_change_in_C_1, iron_dependent_change_in_C_1] if conc is not None) \
                        + mic_ment_light_mult_lim*V_max*light_dependent_change_in_C_1*nutrient_dependent_change_in_C_1*float([1 if iron_dependent_change_in_C_1 == None else iron_dependent_change_in_C_1][0]) \
                            + mic_ment_nolight*(V_max/100)*(nutrient_dependent_change_in_C_1)

            # Exports governed by the Michaelis-Menton model, considering the Liebig and Multiplicative method of limit.
    
    def export_2(C_2_input = C_2, Fe_2_input = Fe_2):
        """
        Calculates export of organic matter from box 1, considering the Michaelis-Menten
        approach and the liebig/multiplicative limit approach.
        
        Parameters:
            None
        Returns
            Float, representing the total export of organic matter from box 1 according to current
            concentration of light, nutrient, and iron.
        """
        global light_dependent_change_in_C_2
        global nutrient_dependent_change_in_C_2
        global iron_dependent_change_in_C_2
        light_dependent_change_in_C_2 = (Ibox2/(K_sat_l + Ibox2))
        nutrient_dependent_change_in_C_2 = ((C_2_input)/(K_sat_N + C_2_input))
        if Fe_2_input == None:
            iron_dependent_change_in_C_2 = None
        else:
            iron_dependent_change_in_C_2 = (Fe_2_input/(K_sat_Fe + Fe_2_input))
        
        return mic_ment_light_leibig*V_max*min(conc for conc in [light_dependent_change_in_C_2, nutrient_dependent_change_in_C_2, iron_dependent_change_in_C_2] if conc is not None) \
                        + mic_ment_light_mult_lim*V_max*light_dependent_change_in_C_2*nutrient_dependent_change_in_C_2*float([1 if iron_dependent_change_in_C_2 == None else iron_dependent_change_in_C_2][0]) \
                            + mic_ment_nolight*V_max*(nutrient_dependent_change_in_C_2)

            # Exports governed by the Michaelis-Menten model, considering both the Liebig and Multiplicative methods of limitation.    
          
    # Concentration of Nutrients --------------------------------------------------
    def dC_1_over_dt(C_1_input = C_1, C_2_input = C_2, C_3_input = C_3, \
                      Fe_1_input = Fe_1, Fe_2_input = Fe_2, Fe_3_input = Fe_3, \
                          L_1_input = L_1, L_2_input = L_2, L_3_input = L_3):

        """
        Calculates change in concentration of C_1 per cubic meter, in units of 
        moles of C_1 per cubic meter per unit time.
        
        The quanities needed are stored in the variables defined earlier. 
        
    
        Returns
        -------
        Number quantity reflecting change of C_1 per unit time, governed by the flow
        rates and the concentrations at the given times.
    
        """
            
        return (psi*(C_3_input - C_1_input) + k_31*(C_3_input - C_1_input) + k_21*(C_2_input - C_1_input))/vol_1 \
            - lambda_1*C_1_input \
                   - export_1(C_1_input, Fe_1_input) 
                        
        # Line 1: General tracer equation, maintains equilibrium among all three boxes with flow rate considered.
        # Line 2: Given fixed export rate lambda_1, considers the box's export rate dependent on nutrient concentration in given box.
        # Line 3: Amount of matter exported according to the given export function.
                
                    
    def dC_2_over_dt(C_1_input = C_1, C_2_input = C_2, C_3_input = C_3, \
                      Fe_1_input = Fe_1, Fe_2_input = Fe_2, Fe_3_input = Fe_3, \
                          L_1_input = L_1, L_2_input = L_2, L_3_input = L_3):
        """
        Calculates change in concentration of C_2 per cubic meter, in units of 
        moles of C_2 per cubic meter per unit time.
        
        The quanities needed are stored in the variables defined earlier. 
        
    
        Returns
        -------
        Number quantity reflecting change of C_2 per unit time, governed by the flow
        rates and the concentrations at the given times.
    
        """
        
        return (psi*(C_1_input - C_2_input) + k_12*(C_1_input - C_2_input) + k_32*(C_3_input - C_2_input))/vol_2 \
            - lambda_2*C_2_input \
                    - export_2(C_2_input, Fe_2_input)

        # Line 1: General tracer equation, maintains equilibrium among all three boxes with flow rate considered.
        # Line 2: Given fixed export rate lambda_2, considers the box's export rate dependent on nutrient concentration in given box.
        # Line 3: Amount of matter exported according to the given export function.
            
    def dC_3_over_dt(C_1_input = C_1, C_2_input = C_2, C_3_input = C_3, \
                      Fe_1_input = Fe_1, Fe_2_input = Fe_2, Fe_3_input = Fe_3, \
                          L_1_input = L_1, L_2_input = L_2, L_3_input = L_3):
        """
        Calculates change in concentration of C_3 per cubic meter, in units of 
        moles of C_3 per cubic meter per unit time.
        
        The quanities needed are stored in the variables defined earlier. 
        
    
        Returns
        -------
        Number quantity reflecting change of C_3 per unit time, governed by the flow
        rates and the concentrations at the given times.
    
        """
        
        return (psi*(C_2_input - C_3_input) + k_23*(C_2_input - C_3_input) + k_13*(C_1_input - C_3_input))/vol_3 + \
            (lambda_1*C_1_input*vol_1 + lambda_2*C_2_input*vol_2)/vol_3 \
                    + (export_1(C_1_input, Fe_1_input)*vol_1 + export_2(C_2_input, Fe_2_input)*vol_2)/vol_3
                
        # Line 1: General tracer equation, maintains equilibrium among all three boxes with flow rate considered.
        # Line 2: Given fixed export rate lambda_1 and lambda_2, considers the box's export rate dependent on nutrient concentration in given box.
            # For this particular box, the export received from the other boxes goes to this box.
        # Line 3: Amount of matter exported according to the given export function.

    # Concentration of Iron (or any trace metal in fact) -----------------------------
    
    def dFe_1_over_dt(C_1_input = C_1, C_2_input = C_2, C_3_input = C_3, \
                      Fe_1_input = Fe_1, Fe_2_input = Fe_2, Fe_3_input = Fe_3, \
                          L_1_input = L_1, L_2_input = L_2, L_3_input = L_3):
        """
        Calculates change in concentration of Fe_1_input per cubic meter, in units of 
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
            
        if not use_iron:
            return float('inf')            
        elif not ligand_use:
            return (psi*(Fe_3_input - Fe_1_input) + k_31*(Fe_3_input - Fe_1_input) + k_21*(Fe_2_input - Fe_1_input))/vol_1 + \
                alpha*F_in1/dz_1 - k_scav*Fe_1_input/(60*60*24*365) - R_Fe*export_1(C_1_input, Fe_1_input)
        elif not use_ligand_cycling:
            return (psi*(Fe_3_input - Fe_1_input) + k_31*(Fe_3_input - Fe_1_input) + k_21*(Fe_2_input - Fe_1_input))/vol_1 + \
                alpha*F_in1/dz_1 - k_scav*complexation(Fe_1_input, ligand_total_val, beta_val)/(60*60*24*365) - R_Fe*export_1(C_1_input, Fe_1_input)
        else: 
            return (psi*(Fe_3_input - Fe_1_input) + k_31*(Fe_3_input - Fe_1_input) + k_21*(Fe_2_input - Fe_1_input))/vol_1 + \
                alpha*F_in1/dz_1 - k_scav*complexation(Fe_1_input, L_1_input, beta_val)/(60*60*24*365) - R_Fe*export_1(C_1_input, Fe_1_input)
            
                # Line 1: General tracer equation, maintains equilibrium among all three boxes with flow rate considered.
                # Line 2: First term represents source, second term represents sink (in terms of being scavenged)
                    # Third term represents amount being used up ('biological utilization' as in Parekh, 2004)

    def dFe_2_over_dt(C_1_input = C_1, C_2_input = C_2, C_3_input = C_3, \
                      Fe_1_input = Fe_1, Fe_2_input = Fe_2, Fe_3_input = Fe_3, \
                          L_1_input = L_1, L_2_input = L_2, L_3_input = L_3):
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
            
        if not use_iron:
            return float('inf')
        elif not ligand_use:
            return (psi*(Fe_1_input - Fe_2_input) + k_12*(Fe_1_input - Fe_2_input) + k_32*(Fe_3_input - Fe_2_input))/vol_2 + \
                alpha*F_in2/dz_2 - k_scav*Fe_2_input/(60*60*24*365) - R_Fe*export_2(C_2_input, Fe_2_input)
        elif not use_ligand_cycling:
            return (psi*(Fe_1_input - Fe_2_input) + k_12*(Fe_1_input - Fe_2_input) + k_32*(Fe_3_input - Fe_2_input))/vol_2 + \
                alpha*F_in2/dz_2 - k_scav*complexation(Fe_2_input, ligand_total_val, beta_val)/(60*60*24*365) - R_Fe*export_2(C_2_input, Fe_2_input)
        else:
             return (psi*(Fe_1_input - Fe_2_input) + k_12*(Fe_1_input - Fe_2_input) + k_32*(Fe_3_input - Fe_2_input))/vol_2 + \
                alpha*F_in2/dz_2 - k_scav*complexation(Fe_2_input, L_2_input, beta_val)/(60*60*24*365) - R_Fe*export_2(C_2_input, Fe_2_input)           
        
    def dFe_3_over_dt(C_1_input = C_1, C_2_input = C_2, C_3_input = C_3, \
                      Fe_1_input = Fe_1, Fe_2_input = Fe_2, Fe_3_input = Fe_3, \
                          L_1_input = L_1, L_2_input = L_2, L_3_input = L_3):
        """
        Calculates change in concentration of Fe_3 per cubic meter, in units of 
        moles of Fe_3 per cubic meter per unit time.
        
        The quanities needed are stored in the variables defined earlier. 
        
    
        Returns
        -------
        Number quantity reflecting change of Fe_2 per unit time, governed by the flow
        rates and the concentrations at the given times.
        """
        if not use_iron:
            return float('inf')
        elif not ligand_use:        
            return (psi*(Fe_2_input - Fe_3_input) + k_23*(Fe_2_input - Fe_3_input) + k_13*(Fe_1_input - Fe_3_input))/vol_3 \
                 - k_scav*Fe_3_input/(60*60*24*365) \
                    + R_Fe*(export_1(C_1_input, Fe_1_input)*vol_1 + export_2(C_2_input, Fe_2_input)*vol_2)/vol_3
        elif not use_ligand_cycling:
            return (psi*(Fe_2_input - Fe_3_input) + k_23*(Fe_2_input - Fe_3_input) + k_13*(Fe_1_input - Fe_3_input))/vol_3 \
                 - k_scav*complexation(Fe_3_input, ligand_total_val, beta_val)/(60*60*24*365) \
                    + R_Fe*(export_1(C_1_input, Fe_1_input)*vol_1 + export_2(C_2_input, Fe_2_input)*vol_2)/vol_3
        else: 
            return (psi*(Fe_2_input - Fe_3_input) + k_23*(Fe_2_input - Fe_3_input) + k_13*(Fe_1_input - Fe_3_input))/vol_3 \
                 - k_scav*complexation(Fe_3_input, L_3_input, beta_val)/(60*60*24*365) \
                    + R_Fe*(export_1(C_1_input, Fe_1_input)*vol_1 + export_2(C_2_input, Fe_2_input)*vol_2)/vol_3

    ### To find the free ion concentration at any given moment, the following function
    ### calculates exactly that given our concentration of ligand, metal, and beta constant.
    ### (via complexation)
    
    # Concentration of Ligands -----------------------------------------------------
    
    def dLt_1_over_dt(C_1_input = C_1, C_2_input = C_2, C_3_input = C_3, \
                      Fe_1_input = Fe_1, Fe_2_input = Fe_2, Fe_3_input = Fe_3, \
                          L_1_input = L_1, L_2_input = L_2, L_3_input = L_3):
        """
        Calculates change in total ligand in specified box in units of mols per cubic meter.
        Addresses ligands cycling through the three boxes, as well as sources/sinks.

        Parameters
        ----------
            None.

        Returns
        -------
        Value in units of mols per cubic meter per second (changing concentration of ligand)

        """
        if not ligand_use:
            return float('inf')
        return (psi*(L_3_input - L_1_input) + k_31*(L_3_input - L_1_input) + k_21*(L_2_input - L_1_input))/vol_1 \
            + gamma*export_1(C_1_input, Fe_1_input) \
                - lambda_ligand*L_1_input

                    # Line 1: Cycling of ligands in and out of box 1.
                    # Line 2: Source (with appropriate gamma)
                    # Line 3: Loss of ligands to degredation.
                
    def dLt_2_over_dt(C_1_input = C_1, C_2_input = C_2, C_3_input = C_3, \
                      Fe_1_input = Fe_1, Fe_2_input = Fe_2, Fe_3_input = Fe_3, \
                          L_1_input = L_1, L_2_input = L_2, L_3_input = L_3):
        """
        Calculates change in total ligand in specified box in units of mols per cubic meter.
        Addresses ligands cycling through the three boxes, as well as sources/sinks.

        Parameters
        ----------
            None.

        Returns
        -------
        Value in units of mols per cubic meter per second (changing concentration of ligand)

        """     
        if not ligand_use:
            return float('inf')
        return (psi*(L_1_input - L_2_input) + k_12*(L_1_input - L_2_input) + k_32*(L_3_input - L_2_input))/vol_2 \
            + gamma*export_2(C_2_input, Fe_2_input) \
                - lambda_ligand*L_2_input
                
                    # Line 1: Cycling of ligands in and out of box 2.
                    # Line 2: Source (with appropriate gamma)
                    # Line 3: Loss of ligands to degredation.
                    
    def dLt_3_over_dt(C_1_input = C_1, C_2_input = C_2, C_3_input = C_3, \
                      Fe_1_input = Fe_1, Fe_2_input = Fe_2, Fe_3_input = Fe_3, \
                          L_1_input = L_1, L_2_input = L_2, L_3_input = L_3):
        """
        Calculates change in total ligand in specified box in units of mols per cubic meter.
        Addresses ligands cycling through the three boxes, as well as sources/sinks.

        Parameters
        ----------
            None.

        Returns
        -------
        Value in units of mols per cubic meter per second (changing concentration of ligand)

        """        
        if not ligand_use:
            return float('inf')
        return (psi*(L_2_input - L_3_input) + k_23*(L_2_input - L_3_input) + k_13*(L_1_input - L_3_input))/vol_3 \
            - lambda_ligand/100*L_3_input \
                + gamma/vol_3*(export_1(C_1_input, Fe_1_input)*vol_1 + export_2(C_2_input, Fe_2_input)*vol_2)
            
            # Line 1: Cycling of ligands
            # Line 2: Loss of ligand
            # Line 3: Input of ligands based on export 'reception'
        
    # Complexation, causes differentiation between total and free iron (or any other metal) ------------------------
    
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
    
    ## Create y-axis arrays (C_1, C_2, and C_3)
    
    acfv = np.full((len(time_axis_array), 9), 0)
        # NOTE: 'acfv' stands for 'all concentration feature vector'.
        
    # Initialize initial values of elements to be used (Depending on Passed-in Input Parameter)
    
    list_of_initial_concentrations = [C_1, C_2, C_3, Fe_1, Fe_2, Fe_3, L_1, L_2, L_3]
    for conc_ind in range(len(list_of_initial_concentrations)):
        if list_of_initial_concentrations[conc_ind] == None:
            list_of_initial_concentrations[conc_ind] = float('inf')
    
    feature_vector_height = 0
    initial_concentration_row = np.array(list_of_initial_concentrations)
    # np.array([conc for conc in [C_1, C_2, C_3, Fe_1, Fe_2, Fe_3, L_1, L_2, L_3] if conc is not None])
    acfv[0, :] = initial_concentration_row
        # For given row in the blank all_concentration_feature_vector, fill that row in
        # with the given concentrations. This fills out the first row. 
        
    # Initialize list of all arguments *derived from the numpy array* to be used in following loop.
    
    def crr(t_val_temp):
        """
        Returns current row of concentrations in the acfv. 'crr' represents 'current row return'

        Parameters
        ----------
        t_val_temp : Index value of current 

        Returns
        -------
        List of values in current row.

        """
        
        return [acfv[t_val_temp, 0], acfv[t_val_temp, 1], acfv[t_val_temp, 2], \
                acfv[t_val_temp, 3], acfv[t_val_temp, 4], acfv[t_val_temp, 5], \
                    acfv[t_val_temp, 6], acfv[t_val_temp, 7], acfv[t_val_temp, 8]]
    
    for t_val_index in range(len(time_axis_array[1:])):
        feature_vector_height += 1
        acfv[feature_vector_height, :] = acfv[t_val_index, :] \
                + np.array([dC_1_over_dt(acfv[t_val_index, 0], acfv[t_val_index, 1], acfv[t_val_index, 2], \
                acfv[t_val_index, 3], acfv[t_val_index, 4], acfv[t_val_index, 5], \
                    acfv[t_val_index, 6], acfv[t_val_index, 7], acfv[t_val_index, 8])*dt, \
                            dC_2_over_dt(acfv[t_val_index, 0], acfv[t_val_index, 1], acfv[t_val_index, 2], \
                acfv[t_val_index, 3], acfv[t_val_index, 4], acfv[t_val_index, 5], \
                    acfv[t_val_index, 6], acfv[t_val_index, 7], acfv[t_val_index, 8])*dt, \
                            dC_3_over_dt(acfv[t_val_index, 0], acfv[t_val_index, 1], acfv[t_val_index, 2], \
                acfv[t_val_index, 3], acfv[t_val_index, 4], acfv[t_val_index, 5], \
                    acfv[t_val_index, 6], acfv[t_val_index, 7], acfv[t_val_index, 8])*dt, \
                            dFe_1_over_dt(acfv[t_val_index, 0], acfv[t_val_index, 1], acfv[t_val_index, 2], \
                acfv[t_val_index, 3], acfv[t_val_index, 4], acfv[t_val_index, 5], \
                    acfv[t_val_index, 6], acfv[t_val_index, 7], acfv[t_val_index, 8])*dt, \
                            dFe_2_over_dt(acfv[t_val_index, 0], acfv[t_val_index, 1], acfv[t_val_index, 2], \
                acfv[t_val_index, 3], acfv[t_val_index, 4], acfv[t_val_index, 5], \
                    acfv[t_val_index, 6], acfv[t_val_index, 7], acfv[t_val_index, 8])*dt, \
                            dFe_3_over_dt(acfv[t_val_index, 0], acfv[t_val_index, 1], acfv[t_val_index, 2], \
                acfv[t_val_index, 3], acfv[t_val_index, 4], acfv[t_val_index, 5], \
                    acfv[t_val_index, 6], acfv[t_val_index, 7], acfv[t_val_index, 8])*dt, \
                            dLt_1_over_dt(acfv[t_val_index, 0], acfv[t_val_index, 1], acfv[t_val_index, 2], \
                acfv[t_val_index, 3], acfv[t_val_index, 4], acfv[t_val_index, 5], \
                    acfv[t_val_index, 6], acfv[t_val_index, 7], acfv[t_val_index, 8])*dt, \
                            dLt_2_over_dt(acfv[t_val_index, 0], acfv[t_val_index, 1], acfv[t_val_index, 2], \
                acfv[t_val_index, 3], acfv[t_val_index, 4], acfv[t_val_index, 5], \
                    acfv[t_val_index, 6], acfv[t_val_index, 7], acfv[t_val_index, 8])*dt, \
                            dLt_3_over_dt(acfv[t_val_index, 0], acfv[t_val_index, 1], acfv[t_val_index, 2], \
                acfv[t_val_index, 3], acfv[t_val_index, 4], acfv[t_val_index, 5], \
                    acfv[t_val_index, 6], acfv[t_val_index, 7], acfv[t_val_index, 8])*dt])
    
    # Remove columns with values of infinity.
    
    
        
    def plot_concentrations(title, element_symbol, time_axis_array, acfv_p):
        """
        Plots the concentrations of material in the three boxes and their
        change over time
    
        Parameters:
        -----------
            title: an str object titling the graph
            time_array: array of ints representing the time scale of the graph.
            acfv: all_concentration_feature_vector
        
        Returns:
        -------
            None. Plots graph with above information.
    
        """
        conc_name_list = [C_1, C_2, C_3, Fe_1, Fe_2, Fe_3, L_1, L_2, L_3]
        color_list = ['r', 'g', 'b', 'm', 'k', 'c', 'r-', 'g-', 'b-', 'm-']
        
        acfv_t_p = acfv_p.transpose()
        # Take transpose so that each row now corresponds to each concentration.
        
        if one_graph:
            fig, nutri_conc = plt.subplots()
            nutri_conc.set_title(f'{title}')
            nutri_conc.set_xlabel('Time [log(years)]')
            nutri_conc.set_ylabel('Concentration (mol per cubic meter)')
            if use_iron:                
                iron_axis = nutri_conc.twinx()
                iron_axis.set_ylabel('Concentration of iron per cubic meter \n (in nmol/m3)')
            if ligand_use:
                lig_graph, lig = plt.subplots()
                lig.set_title('Ligand Concentrations Over Time')
                lig.set_xlabel('Time [log(years)]')
                lig.set_ylabel('Concentration (mol per cubic meter)')
            for conc_index in range(len(conc_name_list)):
                if conc_index <= 2:
                    nutri_conc.plot(time_axis_array_log10, acfv_t_p[conc_index, :], color_list[conc_index], label = f'{conc_name_list[conc_index]}')
                if 3 <= conc_index <= 5 and use_iron:
                    iron_axis.plot(time_axis_array_log10, acfv_t_p[conc_index, :], color_list[conc_index], label = f'{conc_name_list[conc_index]}')
                if 6 <= conc_index <= 8 and ligand_use:
                    lig.plot(time_axis_array_log10, acfv_t_p[conc_index, :], color_list[conc_index], label = f'{conc_name_list[conc_index]}')

            nutri_conc.legend(loc = 'best')
            if use_iron:
                iron_axis.legend(loc = 'best')
            fig.tight_layout()
            plt.show()
            
            if ligand_use:
                lig.legend(loc = 'best')
                fig.tight_layout()
                plt.show()
                
    plot_concentrations(title, element_symbol, time_axis_array, acfv)           
                
            # # plt.legend(nutri_conc + iron_axis, [nutri_conc.get_label(), iron_axis.get_label()], loc = 'best')
            # plt.legend(loc = 'best')
            # fig.tight_layout()
            # plt.show()
            
            # if use_ligand_cycling:
            #     lig.plot(time_axis_array_log10, array_of_L_1, 'r', label = 'L_1')
            #     lig.plot(time_axis_array_log10, array_of_L_2, 'm', label = 'L_2')
            #     lig.plot(time_axis_array_log10, array_of_L_3, 'b', label = 'L_3')
            #     plt.legend(loc = 'best')
            #     fig.tight_layout()
            #     plt.show()
            
        # if multi_graph:
        #     nutrient_graph, nutri_conc = plt.subplots()
        #     nutri_conc.set_title('Concentrations of N_1, N_2, and N_3 wrt Time n/ (on log scale)')
        #     nutri_conc.set_xlabel('Time [log(years)]')
        #     nutri_conc.set_ylabel('Concentration (mol N_i per cubic meter)')
    
# ----------------------------------------
    
    # C_1_list = [C_1,]
    # C_2_list = [C_2,]
    # C_3_list = [C_3,]
    #     # Initiate lists that will store the three concentrations, with initial concentrations already
    #     # in the lists.
    
    # if use_iron:
    #     Fe_1_list = [Fe_1,]
    #     Fe_2_list = [Fe_2,]
    #     Fe_3_list = [Fe_3,]
    
    # if use_ligand_cycling:
    #     L_1_list = [L_1,]
    #     L_2_list = [L_2,]
    #     L_3_list = [L_3,]
    
    # ## Create Temporary Variables that will store C_1, C_2, C_3, Fe_1, Fe_2, and Fe_3, as well as L_1 to L_3
    # ## to ensure that the concentrations used for all six time steps happen simultaneously,
    # ## i.e. we don't use the concentrations of the next time step to calculate the changes
    # ## in the current time step.
    
    # C_1_temp, C_2_temp, C_3_temp = C_1, C_2, C_3

    # if use_iron:
    #     Fe_1_temp, Fe_2_temp, Fe_3_temp = Fe_1, Fe_2, Fe_3
        
    # if use_ligand_cycling:
    #     L_1_temp, L_2_temp, L_3_temp = L_1, L_2, L_3
    

    # for t_val in time_axis_array[1:]:
    #         C_1_temp += dC_1_over_dt(C_1, C_2, C_3, Fe_1)*dt
    #         C_1_list.append(C_1_temp)
    #             # Use Euler Step Function to change value of C_1 by one time step (i.e. dt). Then 
    #             # append that value to the C_1_list of concentrations as the concentration for that
    #             # given time. 
    #         C_2_temp += dC_2_over_dt(C_1, C_2, C_3, Fe_2)*dt
    #         C_2_list.append(C_2_temp)
    #             # Use Euler Step Function to change value of C_2 by one time step (i.e. dt). Then 
    #             # append that value to the C_2_list of concentrations as the concentration for that
    #             # given time. 
    #         C_3_temp += dC_3_over_dt(C_1, C_2, C_3, Fe_1, Fe_2)*dt
    #         C_3_list.append(C_3_temp)
    #             # Use Euler Step Function to change value of C_3 by one time step (i.e. dt). Then 
    #             # append that value to the C_3_list of concentrations as the concentration for that
    #             # given time. 
    #         if use_iron:
    #             Fe_1_temp += dFe_1_over_dt(Fe_1, Fe_2, Fe_3, C_1, L_1)*dt
    #             Fe_1_list.append(Fe_1_temp)
    #                 # Iron in box 1.
    #             Fe_2_temp += dFe_2_over_dt(Fe_1, Fe_2, Fe_3, C_2, L_2)*dt
    #             Fe_2_list.append(Fe_2_temp)
    #                 # Iron in box 2.
    #             Fe_3_temp += dFe_3_over_dt(Fe_1, Fe_2, Fe_3, C_1, C_2, L_1, L_2, L_3)*dt
    #             Fe_3_list.append(Fe_3_temp)
    #                 # Iron in box 3.
    #         if use_ligand_cycling:
    #             L_1_temp += dLt_1_over_dt(L_1, L_2, L_3, C_1, Fe_1)*dt
    #             L_1_list.append(L_1_temp)
    #                 # Ligands in box 1.
    #             L_2_temp += dLt_2_over_dt(L_1, L_2, L_3, C_2, Fe_2)*dt
    #             L_2_list.append(L_2_temp)
    #                 # Ligands in box 2.
    #             L_3_temp += dLt_3_over_dt(L_1, L_2, L_3, C_1, C_2, Fe_1, Fe_2)*dt
    #             L_3_list.append(L_3_temp)
    #                 # Ligands in box 3.
    #         ## Now update values of C_1 - C_3, Fe_1 - Fe_3, L_1 - L_3 to the updated temp values.
    #         C_1, C_2, C_3 = C_1_temp, C_2_temp, C_3_temp
    #         if use_iron:
    #             Fe_1, Fe_2, Fe_3 = Fe_1_temp, Fe_2_temp, Fe_3_temp
    #         if use_ligand_cycling:
    #             L_1, L_2, L_3 = L_1_temp, L_2_temp, L_3_temp
            
    
    # C_1_array = np.array(C_1_list)
    # C_2_array = np.array(C_2_list)
    # C_3_array = np.array(C_3_list)
    #     # Once the above is complete, we now have three arrays depicting concentrations of C_1, C_2, and C_3
    #     # over time. 
    # if use_iron:
    #     Fe_1_array = np.array(Fe_1_list)
    #     Fe_2_array = np.array(Fe_2_list)
    #     Fe_3_array = np.array(Fe_3_list)
    # else:
    #     Fe_1_array = None
    #     Fe_2_array = None
    #     Fe_3_array = None
        
    # if use_ligand_cycling:
    #     L_1_array = np.array(L_1_list)
    #     L_2_array = np.array(L_2_list)    
    #     L_3_array = np.array(L_3_list)
    # else:
    #     L_1_array = None
    #     L_2_array = None
    #     L_3_array = None
        
    # def plot_concentrations(title, element_symbol, time_axis_array, array_of_C_1, array_of_C_2, array_of_C_3, \
    #                         array_of_Fe_1, array_of_Fe_2, array_of_Fe_3, \
    #                             array_of_L_1, array_of_L_2, array_of_L_3):
    #     """
    #     Plots the concentrations of material in the three boxes and their
    #     change over time
    
    #     Parameters:
    #     -----------
    #         title: an str object titling the graph
    #         time_array: array of ints representing the time scale of the graph.
    #         array_of_C_1: array of numbers detailing levels of C_1 over time.
    #         array_of_C_2 and array_of_C_3: as detailed above.
        
    #     Returns:
    #     -------
    #         None. Plots graph with above information.
    
    #     """
        
    #     if one_graph:
    #         fig, nutri_conc = plt.subplots()
    #         nutri_conc.set_title(f'{title}')
    #         nutri_conc.set_xlabel('Time [log(years)]')
    #         nutri_conc.set_ylabel(f'Concentration (mol {element_symbol}_i per cubic meter)')
    #         nutri_conc.plot(time_axis_array_log10, array_of_C_1, 'r', label = f'{element_symbol}_1')
    #         nutri_conc.plot(time_axis_array_log10, array_of_C_2, 'm', label = f'{element_symbol}_2')
    #         nutri_conc.plot(time_axis_array_log10, array_of_C_3, 'b', label = f'{element_symbol}_3')
    #         plt.legend(loc = 'best')
    #         if use_iron:
    #             iron_axis = nutri_conc.twinx()
    #             iron_axis.set_ylabel('Concentration of iron per cubic meter \n (in nmol/m3)')
    #             iron_axis.plot(time_axis_array_log10, array_of_Fe_1, 'g-.', label = 'Fe_1')
    #             iron_axis.plot(time_axis_array_log10, array_of_Fe_2, 'k-.', label = 'Fe_2')
    #             iron_axis.plot(time_axis_array_log10, array_of_Fe_3, 'c-.', label = 'Fe_3')
    #         # plt.legend(nutri_conc + iron_axis, [nutri_conc.get_label(), iron_axis.get_label()], loc = 'best')
    #         plt.legend(loc = 'best')
    #         fig.tight_layout()
    #         plt.show()
            
    #         if use_ligand_cycling:
    #             lig_graph, lig = plt.subplots()
    #             lig.set_title('Ligand Concentrations Over Time')
    #             lig.set_xlabel('Time [log(years)]')
    #             lig.set_ylabel(f'Concentration (mol {element_symbol}_i per cubic meter)')
    #             lig.plot(time_axis_array_log10, array_of_L_1, 'r', label = 'L_1')
    #             lig.plot(time_axis_array_log10, array_of_L_2, 'm', label = 'L_2')
    #             lig.plot(time_axis_array_log10, array_of_L_3, 'b', label = 'L_3')
    #             plt.legend(loc = 'best')
    #             fig.tight_layout()
    #             plt.show()
    #     if multi_graph:
    #         nutrient_graph, nutri_conc = plt.subplots()
    #         nutri_conc.set_title('Concentrations of N_1, N_2, and N_3 wrt Time n/ (on log scale)')
    #         nutri_conc.set_xlabel('Time [log(years)]')
    #         nutri_conc.set_ylabel('Concentration (mol N_i per cubic meter)')
            
            
    
    # plot_concentrations(title, element_symbol, time_axis_array, \
    #                 C_1_array, C_2_array, C_3_array, \
    #                     Fe_1_array, Fe_2_array, Fe_3_array, \
    #                         L_1_array, L_2_array, L_3_array)
        # Plotting the concentrations and how they change over time. 
        
#    return (time_axis_array, (C_1_array, C_2_array, C_3_array), \
#           (Fe_1_array, Fe_2_array, Fe_3_array))
    

### --------------------------------------------------------------------------------
### --------------------------------------------------------------------------------
### --------------------------------------------------------------------------------
### The following section calls the above functions for plotting purposes.


## Part 1: Creating Transport Model for Generic Concentration

# Concentrations of 1.0 for all three boxes:
    
transport_model_info = \
    create_transport_model(1.0, 1.0, 1.0, 1, 100, \
                            'Concentrations of C_1, C_2, and C_3, initially all 1.0,' + ' dt = 1', 'C', 3)
        # The tuple above is of the form (time_array, (C1_array, C2_array, C3_array)).

# Concentrations of 0.1 for boxes 1 and 2, and 1.0 for box 3

transport_model_plot = \
    create_transport_model(0.1, 0.1, 1.0, 1, 100, \
                            'Concentrations of C_1 = 0.1, C_2 = 0.1, and C_3 = 1.0 initially,' + ' dt = 1', 'C', 3)

# # -----------------------------------------------------------------------------------------------------------------------------------------
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
                            'Concentrations of N_1, N_2, and N_3 w/ exports, dt = 0.001', 'N', 3,  \
                                3*10**-8, 3*10**-7)

# -------------------------------------------------------------------------------------------------------------------
# Michaelis-Menten Model, not considering effects of light.

transport_model_graphing = \
    create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.001, 100, \
                            'Concentrations of N_1, N_2, and N_3 w/ exports, dt = 0.001, variable export rate \n (Michaelis-Menten, 100*V_max_1 = V_max_2)', 'N', 3,  \
                                mic_ment_nolight = 1)

# -------------------------------------------------------------------------------------------------------------------
# Michaelis-Menten Model, considering effects of light and Leibig's Law.

transport_model_graphing = \
    create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.001, 8, \
                            'Concentrations of N_1, N_2, and N_3 w/ exports, dt = 0.001, variable export rate \n (Michaelis-Menten, V_max_1 = V_max_2, light-limited, Leibig)', 'N', 3,  \
                                mic_ment_light_leibig = 1)

# -------------------------------------------------------------------------------------------------------------------
# Michaelis-Menten Model, considering effects of light and the Multiplicative Law.

transport_model_graphing = \
    create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.001, 8, \
                            'Concentrations of N_1, N_2, and N_3 w/ exports, dt = 0.001, variable export rate \n (Michaelis-Menten, V_max_1 = V_max_2, light-limited, Multiplicative Law)', 'N', 3,  \
                                mic_ment_light_mult_lim = 1)

# -------------------------------------------------------------------------------------------------------------------
# Michaelis-Menten Model, considering effects of light and the two limit laws, but with Ibox1 = 0.1 W/m2

transport_model_graphing_leibig = \
    create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.001, 100, \
                            'Concentrations of N_1, N_2, and N_3 w/ exports, dt = 0.001, variable export rate \n (Michaelis-Menten, V_max_1 = V_max_2, light-limited, Leibig, \n Ibox1 = 0.1)', 'N', 3,  \
                                mic_ment_light_leibig = 1, Ibox1 = 0.1)
        
transport_model_graphing_mult_law = \
    create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.001, 100, \
                            'Concentrations of N_1, N_2, and N_3 w/ exports, dt = 0.001, variable export rate \n (Michaelis-Menten, V_max_1 = V_max_2, light-limited, Multiplicative Law, \n Ibox1 = 0.1)', 'N', 3,  \
                                mic_ment_light_mult_lim = 1, Ibox1 = 0.1)

# -------------------------------------------------------------------------------------------------------------------
# Michaelis-Menten Model, with Iron and the Net Scavenging Model (Case I in Parekh, 2004)

Fe_1_init = 5*10**-10 #0.5 nanomols of Fe per unit volume
Fe_2_init = 5*10**-10
Fe_3_init = 5*10**-10
N_1_to_3 = 30*rho_0*10**(-6)

transport_model_graphing_mult_law = \
    create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.006849, 10000, \
                            'Concentrations of N_1, N_2, N_3, Fe_1, Fe_2, Fe_3 w/ exports, \n dt = 2.5 days, variable export rate, \n Michalis-Menten Model, Leibig Limit Approximation \n  ', 'N', 6, \
                                mic_ment_light_leibig = 1, k_scav = 0.004, mu = 3.858*10**-7, \
                                    Fe_1 = Fe_1_init, Fe_2 = Fe_2_init, Fe_3 = Fe_3_init, \
                                        use_iron = True)
                                        # Time step of 2.5 days
                                        
# Concentration of total iron is about 1 nanomol per liter (mmol/)
# Free iron 0.1 or 0.2 (1% of total iron is free)

# -------------------------------------------------------------------------------------------------------------------
# Michaelis-Menten Model, with Iron and free concentration governed by scavenging of free iron. Fixed ligand concentration and beta value.

ligand_conc = 1*10**-6 # mol/m3
beta_val_1 = 10**8 # kg/mol, as required by the value earlier.

transport_model_graphing_ligand_approach = \
        create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.006849, 10000, \
                            'Concentrations of N_1, N_2, N_3, Fe_1, Fe_2, Fe_3 w/ exports and complexation, \n dt = 2.5 days, variable export rate, \n ligand concentration = 10**-6, beta = 10**8 (kg per mol) \n Michalis-Menten Model, Leibig Limit Approximation \n  ', 'N', 6, \
                                mic_ment_light_leibig = 1, k_scav = 0.19, mu = 3.858*10**-7, \
                                    Fe_1 = Fe_1_init, Fe_2 = Fe_2_init, Fe_3 = Fe_3_init, \
                                        use_iron = True, \
                                            ligand_use = True, ligand_total_val = ligand_conc, beta_val = beta_val_1)
                                        # Time step of 2.5 days
                                        
# -------------------------------------------------------------------------------------------------------------------
# Michaelis-Menten Model, with Iron and free concentration governed by scavenging of free iron. Variable ligand concentration, but initial concentration of 0.

transport_model_graphing_ligand_approach = \
        create_transport_model(N_1_to_3, N_1_to_3, N_1_to_3, 0.006849, 10000, \
                           'Concentrations of N_1, N_2, N_3, Fe_1, Fe_2, Fe_3 w/ exports and complexation, \n dt = 2.5 days, variable export rate, \n ligand concentration = 10**-6, beta = 10**8 (kg per mol) \n Michalis-Menten Model, Leibig Limit Approximation \n  ', 'N', 9, \
                               mic_ment_light_leibig = 1, k_scav = 0.19, mu = 3.858*10**-7, \
                                   Fe_1 = 0, Fe_2 = 0, Fe_3 = 0, \
                                       use_iron = True, \
                                       ligand_use = True, ligand_total_val = ligand_conc, beta_val = beta_val_1, \
                                           L_1 = 0, L_2 = 0, L_3 = 0, \
                                               use_ligand_cycling = True)
                                        # Time step of 2.5 days
                                        
# Geo Tracer Database
# Parallelize with PandaandParallel