# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:53:03 2022

@author: ssnaik
"""


import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
from idaes.core.util import model_statistics
import numpy as np
import matplotlib.pyplot as plt
"We develop a SEE-SVR unit model from Onishi's 2017 paper "

#Create model
m = pyo.ConcreteModel()

"Parameter definitions"
#Number of evaporator effects
N_evap = 1

#Number of compression stages
N_compr = 1

#Flowrate of feed stream
m.flow_feed = pyo.Param(initialize = 10.42,
                        units = pyo.units.kg/pyo.units.s)

#Salt in the feed stream
m.salt_feed = pyo.Param(initialize = 70,
                        units = pyo.units.g/pyo.units.kg)

#Temperature of the feed stream
m.feed_temperature = pyo.Param(initialize = 25,
                               units = pyo.units.C)

#Salt in brine concentrate
m.salt_outlet_spec = pyo.Param(initialize = 300,
                               units = pyo.units.g/pyo.units.kg)

#Temperature constraint parameters
m.DT_min = pyo.Param(initialize = 2, units = pyo.units.C)
m.DT_min_1 = pyo.Param(initialize = 2, units = pyo.units.C)
m.DT_min_2 = pyo.Param(initialize = 2, units = pyo.units.C)

#Antoine coefficients
m.a = pyo.Param(initialize = 12.98437)
m.b = pyo.Param(initialize = -2001.77468)
m.c = pyo.Param(initialize = 139.61335)

#Specific heat capacity of vapor
m.cp_vapor = pyo.Param(initialize = 1.8723)

#Overall heat transfer coefficient (Known parameter)
m.overall_heat_transfer_coef =  pyo.Param(initialize = 0.1)

#Heat capacity ratio
m.gamma = pyo.Param(initialize = 1.33)

#Maximum compression ratio
m.CR_max = pyo.Param(initialize = 10)

#Efficiency of compressors (isentropic efficiency)
m.eta = pyo.Param(initialize = 0.75)

"Set definitions"
#Set of Evaporator effects
m.i = pyo.Set(initialize = range(N_evap))
i_first = m.i.first()
i_last =m.i.last()
#Set of compression stages
m.j = pyo.Set(initialize = range(N_compr))
j_last = m.j.last()
j_first = m.j.first()

"Variable definitions"
#====================================================================
                        #All flow variables
#Flow of brine evaporator
m.flow_brine = pyo.Var(m.i, 
                       domain = pyo.NonNegativeReals, 
                       initialize = [2.43],
                       units = pyo.units.kg/pyo.units.s)
#Flow of vapor evaporator
m.flow_vapor_evaporator = pyo.Var(m.i, 
                       domain = pyo.NonNegativeReals, 
                       initialize = [7.99],
                       units = pyo.units.kg/pyo.units.s)
#Flow of super heated vapor
m.flow_super_heated_vapor = pyo.Var(domain = pyo.NonNegativeReals,
                                    initialize = 7.99)
#======================================================================
                       #All concentration variables
#Flow of salt/TDS (For now we are considering only one component in water)
m.salt = pyo.Var(m.i, 
                 domain = pyo.NonNegativeReals, 
                 initialize = [300],
                 units = pyo.units.g/pyo.units.kg)

#Salt mass fraction (XS in paper)
m.salt_mass_frac= pyo.Var(m.i,
                          domain= pyo.NonNegativeReals,
                          bounds = (0,1),
                          initialize = [0.3])
#Salt mass fraction in the feed 
m.salt_mass_frac_feed= pyo.Var(domain = pyo.NonNegativeReals, 
                               bounds = (0,1),
                               initialize = 0.07)
#======================================================================
                      #All pressure variables
#Vapor pressure in evaporator effects
m.evaporator_vapor_pressure = pyo.Var(m.i,
                                      domain = pyo.NonNegativeReals,
                                      initialize = [19.930])
m.super_heated_vapor_pressure = pyo.Var(m.j,
                                        domain = pyo.NonNegativeReals,
                                        bounds = (1, 200),
                                        initialize = 42.230)
# m.saturated_vapor_pressure= pyo.Var(m.i,
#                                     domain = pyo.NonNegativeReals,
#                                     initialize = 10100)
#=======================================================================
                      #All temperature variables
#Actual temperature of feed entering the evaporator after preheating
m.evaporator_feed_temperature = pyo.Var(domain = pyo.NonNegativeReals,
                                        initialize = 25,
                                        units = pyo.units.C)

m.evaporator_ideal_temperature = pyo.Var(m.i,
                                         domain = pyo.NonNegativeReals,
                                         initialize = 25)
m.evaporator_brine_temperature = pyo.Var(m.i,
                                         domain = pyo.NonNegativeReals,
                                         initialize = 35)

m.super_heated_vapor_temperature = pyo.Var(m.j,
                                           domain = pyo.NonNegativeReals,
                                           initialize = 45)
m.evaporator_condensate_temperature = pyo.Var(m.i,
                                              domain = pyo.NonNegativeReals,
                                              initialize  = 30)

m.LMTD = pyo.Var(m.i,
                 domain = pyo.NonNegativeReals,
                 initialize = 1)
m.theta_1 = pyo.Var(m.i,
                    domain = pyo.NonNegativeReals,
                    initialize = 1)
m.theta_2 = pyo.Var(m.i,
                    domain = pyo.NonNegativeReals,
                    initialize = 1)
m.isentropic_temperature = pyo.Var(m.j,
                                   domain = pyo.NonNegativeReals,
                                   initialize = 45)
#=======================================================================
                    #All enthalpy variables
#Specific Enthalpies of brine and vapor in the evaporator
m.evaporator_brine_enthalpy = pyo.Var(m.i,
                                      domain = pyo.Reals,
                                      initialize = 300)

m.evaporator_vapor_enthalpy = pyo.Var(m.i,
                                      domain = pyo.Reals,
                                      initialize = 400)
m.evaporator_condensate_enthalpy = pyo.Var(m.i,
                                           domain = pyo.Reals,
                                           initialize = 100)

m.evaporator_condensate_vapor_enthalpy = pyo.Var(domain = pyo.Reals,
                                                 initialize = 100)
m.super_heated_vapor_enthalpy = pyo.Var(m.j,
                                  domain = pyo.Reals,
                                  initialize = 100)

#Enthalpy of the feed stream
m.enthalpy_feed = pyo.Var(domain = pyo.Reals,
                          initialize = 200)
#=======================================================================
                        #All area variables
m.evaporator_total_area = pyo.Var(domain = pyo.NonNegativeReals,
                                           initialize = 5)
m.each_evaporator_area = pyo.Var(m.i,
                                 domain = pyo.NonNegativeReals,
                                 initialize = 1)
#=======================================================================
                        #Other Variables
#Evaporator heat flow - Q
m.evaporator_heat_flow = pyo.Var(m.i, 
                                 domain = pyo.NonNegativeReals,
                                 initialize = 1)
#Boiling point elevation
m.bpe = pyo.Var(m.i,
                domain = pyo.NonNegativeReals,
                initialize = [35.054])

#Heat transfer coefficient
m.heat_transfer_coef = pyo.Var(m.i,
                domain = pyo.NonNegativeReals,
                initialize = 1)

#Each compressor work
m.compressor_work = pyo.Var(m.j,
                            domain = pyo.NonNegativeReals,
                            initialize = 1)
m.total_compressor_work = pyo.Var(domain = pyo.NonNegativeReals,
                                  initialize = 1)

#=======================================================================
"Model Constraints"

#=======================================================================
                        #Evaporator Constraints
#Link evaporator feed temp to feed temp
def _evaporator_feed_temp_estimate(m):
    return m.evaporator_feed_temperature == m.feed_temperature
m.evaporator_feed_temp_estimate = pyo.Constraint(rule = _evaporator_feed_temp_estimate)

#Flow balance across the evaporator (Equation 1, 3)
def _evaporator_flow_balance(m, i):
    return m.flow_feed - m.flow_brine[i] -m.flow_vapor_evaporator[i] == 0 
m.evaporator_flow_balance = pyo.Constraint(m.i, rule = _evaporator_flow_balance)

# #Solid balance in the evaporator (Equation 3, 4)
def _evaporator_salt_balance(m, i):
    return m.flow_feed*m.salt_feed - m.flow_brine[i]*m.salt[i] == 0
m.evaporator_salt_balance = pyo.Constraint(m.i, rule = _evaporator_salt_balance)

# #Estimating the ideal temperature in an evaporator (Equation 5)
def _ideal_evaporator_temp_con(m,i):
    return pyo.log(m.evaporator_vapor_pressure[i]) == \
        m.a + m.b/(m.evaporator_ideal_temperature[i] +m.c)
m.evaporator_ideal_temp_con = pyo.Constraint(m.i, rule = _ideal_evaporator_temp_con)

# # #Boiling point elevation (Equation 6)
def _bpe_con(m, i):
    return m.bpe[i] == 0.1581 \
        + 2.769*m.salt_mass_frac[i]\
        - 0.002676*m.evaporator_ideal_temperature[i]\
        + 41.78*m.salt_mass_frac[i]**0.5 \
        + 0.134*m.salt_mass_frac[i]*m.evaporator_ideal_temperature[i]
m.bpe_con = pyo.Constraint(m.i, rule = _bpe_con)

#Relating mass fraction of salt to brine salinity (Equation 7)
def _match_mass_frac_to_salinity(m, i):
    return m.salt_mass_frac[i] - 0.001*m.salt[i] == 0
m.match_mass_frac_to_salinity = pyo.Constraint(m.i, rule = _match_mass_frac_to_salinity)

#Relating salt mass frac of feed to salinity of feed
def _match_mass_frac_to_salinity_feed(m):
    return m.salt_mass_frac_feed -0.001*m.salt_feed== 0
m.match_mass_frac_to_salinity_feed = pyo.Constraint(rule = _match_mass_frac_to_salinity_feed)

#Relate brine temperature to ideal temperature and bpe (Equation 8)
def _brine_temp_con(m, i):
    return m.evaporator_brine_temperature[i] - m.evaporator_ideal_temperature[i]\
        - m.bpe[i] == 0
m.brine_temp_con = pyo.Constraint(m.i, rule = _brine_temp_con)

#Energy balance in the evaporator (Equations 9, 10)
def _evaporator_energy_balance(m, i):
    return m.evaporator_heat_flow[i]\
            + m.flow_feed*m.enthalpy_feed\
            - m.flow_brine[i]*m.evaporator_brine_enthalpy[i]\
            - m.flow_vapor_evaporator[i]*m.evaporator_vapor_enthalpy[i] == 0
m.evaporator_energy_balance = pyo.Constraint(m.i, rule = _evaporator_energy_balance)

#Estimating the enthalpies (Equations 11, 12, 13)
def _enthalpy_vapor_estimate(m, i):
    return m.evaporator_vapor_enthalpy[i] == -13470 + 1.84*m.evaporator_brine_temperature[i]
m.enthalpy_vapor_estimate = pyo.Constraint(m.i, rule = _enthalpy_vapor_estimate)

def _enthalpy_brine_estimate(m, i):
    return m.evaporator_brine_enthalpy[i] == -15940 + 8787*m.salt_mass_frac[i] + 3.557*m.evaporator_brine_temperature[i]
m.enthalpy_brine_estimate = pyo.Constraint(m.i, rule = _enthalpy_brine_estimate)

def _enthalpy_feed_estimate(m):
    return m.enthalpy_feed == -15940 + 8787*m.salt_mass_frac_feed + 3.557*m.evaporator_feed_temperature
m.enthalpy_feed_estimate = pyo.Constraint(rule = _enthalpy_feed_estimate)

def _enthalpy_condensate_vapor_estimate(m):
    return m.evaporator_condensate_vapor_enthalpy == -13470 + 1.84*m.evaporator_condensate_temperature[i_first]
m.enthalpy_condensate_vapor_estimate = pyo.Constraint(rule = _enthalpy_condensate_vapor_estimate)

def _enthalpy_condensate_estimate(m, i):
    return m.evaporator_condensate_enthalpy[i] == -15940 + 3.557*m.evaporator_condensate_temperature[i]
m.enthalpy_condensate_estimate = pyo.Constraint(m.i, rule = _enthalpy_condensate_estimate)

# #Flow balance for super heated vapor (Equation 15)
m.flow_balance_super_heated_vapor = pyo.Constraint(expr = m.flow_super_heated_vapor
                                                    == m.flow_vapor_evaporator[i_first])

#Heat requirements in evaporators
def _evaporator_heat_balance(m,i):
    return m.evaporator_heat_flow[i_first] ==\
            m.flow_super_heated_vapor*m.cp_vapor*\
                (m.super_heated_vapor_temperature[j_last] - m.evaporator_condensate_temperature[i_first])+\
                    m.flow_super_heated_vapor*\
                (m.evaporator_condensate_vapor_enthalpy- m.evaporator_condensate_enthalpy[i_first])
m.evaporator_heat_balance = pyo.Constraint(m.i, rule = _evaporator_heat_balance)

#Calculating the heat transfer coefficient (Equation 20)
def _heat_transfer_coef_calculation(m,i):
    return m.heat_transfer_coef[i] == 0.001*(1939.4 + 1.40562*m.evaporator_brine_temperature[i]
                                              - 0.00207525*m.evaporator_brine_temperature[i]**2
                                              + 0.0023186*m.evaporator_brine_temperature[i]**3)
m.heat_transfer_coef_calculation = pyo.Constraint(m.i, rule = _heat_transfer_coef_calculation)

#Evaporator heat transfer area calculation (Equation 21)
def _total_evaporator_heat_transfer_area(m):
    return m.evaporator_total_area == sum(m.each_evaporator_area[i] for i in m.i)
m.total_evaporator_heat_transfer_area = pyo.Constraint(rule = _total_evaporator_heat_transfer_area)

#Area of the first evaporator
def _first_evaporator_area_calculation(m):
    return m.each_evaporator_area[i_first] == m.flow_super_heated_vapor*m.cp_vapor*\
        (m.super_heated_vapor_temperature[j_last] - m.evaporator_condensate_temperature[i_first])\
        /(m.overall_heat_transfer_coef*m.LMTD[i_first])\
        + m.flow_super_heated_vapor*(m.evaporator_condensate_vapor_enthalpy - m.evaporator_condensate_enthalpy[i_first])\
        /(m.heat_transfer_coef[i_first]*(m.evaporator_condensate_temperature[i_first] - m.evaporator_brine_temperature[i_first]))
m.first_evaporator_area_calculation = pyo.Constraint(rule = _first_evaporator_area_calculation)

#Chen approximation for LMTD (Equation 22-26)
#Temperature of hot in - temp of cold out
def _theta_1_calculation(m, i):
    return m.theta_1[i] == m.super_heated_vapor_temperature[j_last] - m.evaporator_brine_temperature[i]
    
m.theta_1_calculation = pyo.Constraint(m.i, rule = _theta_1_calculation)

#Temp of hot out - Temp of cold in 
def _theta_2_calculation(m, i):
    return m.theta_2[i] == m.evaporator_condensate_temperature[i] -  m.evaporator_feed_temperature     
m.theta_2_calculation = pyo.Constraint(m.i, rule = _theta_2_calculation)
        
def _LMTD_calculation(m, i):
    return m.LMTD[i] == (0.5*m.theta_1[i]*m.theta_2[i]*(m.theta_1[i]+m.theta_2[i]))**(1/3)
m.LMTD_calculation = pyo.Constraint(m.i, rule = _LMTD_calculation)

#Temperature constraints to avoid temperature crossovers in evaporator effects (Equation 29-36)
def _temp_con_1(m):
    return m.super_heated_vapor_temperature[j_last] >= m.evaporator_condensate_temperature[i_first] + m.DT_min_1
m.temp_con_1 = pyo.Constraint(rule = _temp_con_1)

def _temp_con_4(m):
    return m.evaporator_brine_temperature[i_first] >= m.evaporator_feed_temperature + m.DT_min_2
m.temp_con_4 = pyo.Constraint(rule = _temp_con_4)

def _temp_con_5(m, i):
    return m.evaporator_condensate_temperature[i] >= m.evaporator_brine_temperature[i] + m.DT_min
m.temp_con_5 = pyo.Constraint(m.i, rule = _temp_con_5)

#Compressor Equations============================================================
#Isentropic temperature constraints (Equation 51, 52)
def _isentropic_temp_calculation(m, j):
    if j == j_first:
        return m.isentropic_temperature[j] == (m.evaporator_brine_temperature[i_last] + 273.15)*\
            (m.super_heated_vapor_pressure[j]/m.evaporator_vapor_pressure[i_last])**((m.gamma -1)/m.gamma) - 273.15
m.isentropic_temp_calculation = pyo.Constraint(m.j, rule = _isentropic_temp_calculation)

#Maximum possible compression (Equation 53)
def _maximum_compression_calculation(m, j):
    if j == j_first:
        return m.super_heated_vapor_pressure[j] <= m.CR_max*m.evaporator_vapor_pressure[i_first]
m.maximum_compression_calculation = pyo.Constraint(m.j, rule = _maximum_compression_calculation)

#Temperature of superheated vapor
def _temperature_super_heated_vapor_calculation(m, j):
    if j == j_first: 
        return m.super_heated_vapor_temperature[j] == m.evaporator_brine_temperature[i_last] +\
            1/m.eta*(m.isentropic_temperature[j] - m.evaporator_brine_temperature[i_last])
m.temperature_super_heated_vapor_calculation = pyo.Constraint(m.j, rule = _temperature_super_heated_vapor_calculation)

#(Equation 56, 57)
def _compressor_pressure_con(m, j):
    if j ==j_first:
        return m.super_heated_vapor_pressure[j]  >= m.evaporator_vapor_pressure[i_last]
m.compressor_pressure_con = pyo.Constraint(m.j, rule = _compressor_pressure_con)

#Compressor work calculation (Equation 58)
def _compressor_work_calculation(m, j):
    return m.compressor_work[j] == m.flow_super_heated_vapor*(m.super_heated_vapor_enthalpy[j] - m.evaporator_vapor_enthalpy[i_last])
m.compressor_work_calculation = pyo.Constraint(m.j, rule = _compressor_work_calculation)

def _super_heated_vapor_enthalpy_calculation(m, j):
    return m.super_heated_vapor_enthalpy[j]== -13470 + 1.84*m.super_heated_vapor_temperature[j]
m.super_heated_vapor_enthalpy_calculation = pyo.Constraint(m.j, rule = _super_heated_vapor_enthalpy_calculation)

def _total_compressor_work_estimate(m):
    return m.total_compressor_work == sum(m.compressor_work[j] for j in m.j)
m.total_compressor_work_estimate = pyo.Constraint(rule = _total_compressor_work_estimate)

#Salt outlet condition
def _salt_outlet_con(m):
    return m.salt[i_first] >= m.salt_outlet_spec
m.salt_outlet_con = pyo.Constraint(rule = _salt_outlet_con)


#m.evaporator_brine_temperature.fix(67.75)
m.evaporator_vapor_pressure.fix(19.93)

#Objective Function
m.obj = pyo.Objective(expr = m.total_compressor_work + m.each_evaporator_area[0])


ipopt = pyo.SolverFactory('ipopt')
ipopt.options["max_iter"] = 3000
ipopt.solve(m, tee=True)

m.each_evaporator_area.display()
m.evaporator_heat_flow.display()
m.total_compressor_work.display()
m.super_heated_vapor_temperature.display()
m.evaporator_condensate_temperature.display()
m.super_heated_vapor_pressure.display()
m.evaporator_vapor_pressure.display()
m.salt.display()
m.flow_brine.display()
m.flow_vapor_evaporator.display()

#Generating area vs work plots
# weight = np.linspace(0,5,20)
# compressor_work = []
# evap_area = []
# for w in weight:
#     m.obj = pyo.Objective(expr = w*m.total_compressor_work + m.each_evaporator_area[0])
#     ipopt = pyo.SolverFactory('ipopt')
#     ipopt.options["max_iter"] = 3000
#     ipopt.solve(m, tee=True)
#     compressor_work.append(pyo.value(m.total_compressor_work))
#     evap_area.append(pyo.value(m.each_evaporator_area[0]))
   
# plt.figure()
# plt.plot(weight, compressor_work, 'o-')

# plt.figure()
# plt.plot(weight, evap_area, '*-')

# plt.figure()
# plt.plot(evap_area, compressor_work, 'x-')
