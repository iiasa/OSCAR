import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import safe_exp, safe_ratio


#####################################################################
##   WETLANDS
#####################################################################

## initialize
OSCAR_wetland = Model('OSCAR_wetland')


##=====================
## Secondary parameters
##=====================

## preindustrial wetland areal emissions
## note: because of different PI conditions
## note: assumes climate was same between PI and precalibration period
OSCAR_wetland.process(
    Out = 'ewet_pi', 
    Eq = lambda Par: Eq__ewet_pi(Par), 
    units = 'TgC Mha-1 yr-1')

def Eq__ewet_pi(Par):
    f_CO2 = 1 + Par.b_ewet_CO2 * np.log(Par.CO2_pi / Par.CO2_piW)
    f_Tl = np.exp(Par.g_ewet_T * (Par.Tl_pi - Par.Tl_piW))
    f_Pl = np.exp(Par.x_ewet_P * np.log(safe_ratio(Par.Pl_pi / Par.Pl_piW)))
    return Par.k_ewet * Par.ewet_piW * f_CO2 * f_Tl * f_Pl


## preindustrial wetland extent
## note: because of different PI conditions
## note: assumes climate was same between PI and precalibration period
OSCAR_wetland.process(
    Out = 'Awet_pi', 
    Eq = lambda Par: Eq__Awet_pi(Par), 
    units = 'Mha')

def Eq__Awet_pi(Par):
    f_CO2 = (Par.CO2_pi / Par.CO2_piW) ** Par.x_Awet_CO2
    f_Tl = np.exp(Par.x_Awet_T * np.log(safe_ratio(Par.Tl_pi / Par.Tl_piW)))
    f_Pl = np.exp(Par.x_Awet_P * np.log(safe_ratio(Par.Pl_pi / Par.Pl_piW)))
    return Par.Awet_piW * f_CO2 * f_Tl * f_Pl


##=====================
## Diagnostic variables
##=====================

## wetland areal emissions
## note: new formulation combining NPP log-fertilisation and respiration Q10
OSCAR_wetland.process(
    Out = 'D_ewet', 
    In = ('D_CO2', 'D_Tl', 'D_Pl'), 
    Eq = lambda Var, Par: Eq__D_ewet(Var, Par), 
    units = 'TgC Mha-1 yr-1')

def Eq__D_ewet(Var, Par):
    f_CO2 = 1 + Par.b_ewet_CO2 * np.log1p(Var.D_CO2 / Par.CO2_pi)
    f_Tl = safe_exp(Par.g_ewet_T * Var.D_Tl, 5.)
    f_Pl = safe_exp(Par.x_ewet_P * np.log(safe_ratio(1 + Var.D_Pl / Par.Pl_piW)), 5.)
    return Par.ewet_pi * (f_CO2 * f_Tl * f_Pl  - 1)
    

## wetland extent
OSCAR_wetland.process(
    Out = 'D_Awet', 
    In = ('D_CO2', 'D_Tl', 'D_Pl'), 
    Eq = lambda Var, Par: Eq__D_Awet(Var, Par), 
    units = 'Mha')

def Eq__D_Awet(Var, Par):
    f_CO2 = (1 + Var.D_CO2 / Par.CO2_pi) ** Par.x_Awet_CO2
    f_Tl = safe_exp(Par.x_Awet_T * np.log1p(Var.D_Tl / Par.Tl_pi), 5.)
    f_Pl = safe_exp(Par.x_Awet_P * np.log(safe_ratio(1 + Var.D_Pl / Par.Pl_piW)), 5.)
    return Par.Awet_pi * (f_CO2 * f_Tl * f_Pl - 1)


## wetland emissions
OSCAR_wetland.process(
    Out = 'D_Ewet', 
    In = ('D_ewet', 'D_Awet'), 
    Eq = lambda Var, Par: Eq__D_Ewet(Var, Par), 
    units = 'TgC yr-1')
    
def Eq__D_Ewet(Var, Par):
    return Par.ewet_pi * Var.D_Awet + Var.D_ewet * (Par.Awet_pi + Var.D_Awet)


## wetland emissions as CH4
OSCAR_wetland.process(
    Out = 'D_Ewet_CH4', 
    In = ('D_Ewet',), 
    Eq = lambda Var, Par: Eq__D_Ewet_CH4(Var, Par), 
    units = 'TgC yr-1')
    
def Eq__D_Ewet_CH4(Var, Par):
    return Var.D_Ewet.sum('reg_land', min_count=1)


## ADDITIONAL DIAGNOSTICS

## total wetland emissions
OSCAR_wetland.process(
    Out = 'Ewet', 
    In = ('D_Ewet',), 
    Eq = lambda Var, Par: Eq__Ewet(Var, Par), 
    units = 'TgC yr-1')
    
def Eq__Ewet(Var, Par):
    return (Var.D_Ewet + Par.ewet_pi * Par.Awet_pi).sum('reg_land', min_count=1)

