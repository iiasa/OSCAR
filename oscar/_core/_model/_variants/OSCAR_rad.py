import importlib
import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import safe_exp, safe_ratio


##################################################
##   OSCAR (RAD variant)
##################################################

## import and copy main model
from oscar._core._model.OSCAR import OSCAR
OSCAR_rad = OSCAR.copy(new_name='OSCAR_rad')


## ingoing flux to the ocean
OSCAR_rad.process(
    Out = 'D_Fin', 
    In = ('D_CO2',), 
    Eq = lambda Var, Par: Eq__D_Fin(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Fin(Var, Par):
    return Par.v_fg * Par.a_CO2 * 0.


## net primary productivity factor
OSCAR_rad.process(
    Out = 'f_npp', 
    In = ('D_CO2', 'D_Tl', 'D_Pl'), 
    Eq = lambda Var, Par: Eq__f_npp(Var, Par), 
    units = '1')

def Eq__f_npp(Var, Par):
    f_Tl = np.exp(-Par.g_npp_T2 * (Var.D_Tl**2 - 2 * Par.Topt_npp * Var.D_Tl))
    f_Pl = np.exp(Par.x_npp_P * np.log(safe_ratio(1 + Var.D_Pl / Par.Pl_pi)))
    return f_Tl * f_Pl


## wetland areal emissions
OSCAR_rad.process(
    Out = 'D_ewet', 
    In = ('D_CO2', 'D_Tl', 'D_Pl'), 
    Eq = lambda Var, Par: Eq__D_ewet(Var, Par), 
    units = 'TgC Mha-1 yr-1')

def Eq__D_ewet(Var, Par):
    f_Tl = safe_exp(Par.g_ewet_T * Var.D_Tl)
    f_Pl = safe_exp(Par.x_ewet_P * np.log(safe_ratio(1 + Var.D_Pl / Par.Pl_pi)))
    return Par.ewet_pi * (f_Tl * f_Pl  - 1)
    

## wetland extent
OSCAR_rad.process(
    Out = 'D_Awet', 
    In = ('D_CO2', 'D_Tl', 'D_Pl'), 
    Eq = lambda Var, Par: Eq__D_Awet(Var, Par), 
    units = 'Mha')

def Eq__D_Awet(Var, Par):
    f_Tl = safe_exp(Par.g_Awet_T * Var.D_Tl)
    f_Pl = safe_exp(Par.x_Awet_P * np.log(safe_ratio(1 + Var.D_Pl / Par.Pl_pi)))
    return Par.Awet_pi * (f_Tl * f_Pl - 1)

