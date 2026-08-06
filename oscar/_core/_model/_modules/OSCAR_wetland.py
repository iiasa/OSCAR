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

## SHIFTED PREINDUSTRIAL STATE

## adjusted preindustrial wetland areal emissions
## note: assumes climate was same between PI and precalibration period
OSCAR_wetland.process(
    Out = 'ewet_pi', 
    Eq = lambda Par: Eq__ewet_pi(Par), 
    units = 'TgC Mha-1 yr-1')

def Eq__ewet_pi(Par):
    fct_CO2 = 1 + Par.b_ewet_CO2 * np.log(Par.CO2_pi / Par.CO2_piW)
    return Par.k_ewet * Par.ewet_piW * fct_CO2


## adjusted preindustrial wetland extent
## note: assumes climate was same between PI and precalibration period
OSCAR_wetland.process(
    Out = 'Awet_pi', 
    Eq = lambda Par: Eq__Awet_pi(Par), 
    units = 'Mha')

def Eq__Awet_pi(Par):
    fct_CO2 = (Par.CO2_pi / Par.CO2_piW) ** Par.x_Awet_CO2
    return Par.Awet_piW * fct_CO2


##=====================
## Diagnostic variables
##=====================

## wetland areal emissions
## note: new formulation combining NPP log-fertilisation and respiration Q10
OSCAR_wetland.process(
    Out = 'D_ewet', 
    In = ('D_CO2', 'D_Tl', 'r_Pl'), 
    Eq = lambda Var, Par: Eq__D_ewet(Var, Par), 
    units = 'TgC Mha-1 yr-1')

def Eq__D_ewet(Var, Par):
    fct_CO2 = 1 + Par.b_ewet_CO2 * np.log1p(Var.D_CO2 / Par.CO2_pi)
    fct_Tl = safe_exp(Par.g_ewet_T * Var.D_Tl, 5.)
    fct_Pl = safe_exp(Par.x_ewet_P * np.log(Var.r_Pl), 5.)
    return Par.ewet_pi * (safe_ratio(fct_CO2 * fct_Tl * fct_Pl)  - 1)
    

## wetland extent
OSCAR_wetland.process(
    Out = 'D_Awet', 
    In = ('D_CO2', 'D_Tl', 'r_Pl'), 
    Eq = lambda Var, Par: Eq__D_Awet(Var, Par), 
    units = 'Mha')

def Eq__D_Awet(Var, Par):
    fct_CO2 = (1 + Var.D_CO2 / Par.CO2_pi) ** Par.x_Awet_CO2
    fct_Tl = safe_exp(Par.x_Awet_T * np.log1p(Var.D_Tl / Par.Tl_pi), 5.)
    fct_Pl = safe_exp(Par.x_Awet_P * np.log(Var.r_Pl), 5.)
    return Par.Awet_pi * (safe_ratio(fct_CO2 * fct_Tl * fct_Pl) - 1)


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

