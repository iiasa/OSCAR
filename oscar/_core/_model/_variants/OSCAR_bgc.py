import importlib
import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import safe_exp, safe_ratio


##################################################
##   OSCAR (BGC variant)
##################################################

## import and copy main model
from oscar._core._model.OSCAR import OSCAR
OSCAR_bgc = OSCAR.copy(new_name='OSCAR_bgc')


## partial pressure of CO2 at sea surface
OSCAR_bgc.process(
    Out = 'D_pCO2', 
    In = ('D_dic',), 
    Eq = lambda Var, Par: Eq__D_pCO2(Var, Par), 
    units = 'ppm')

def Eq__D_pCO2(Var, Par):
    D_pCO2 = ((1.5568 - 1.3993E-2 * Par.To_0) * Var.D_dic
            + (7.4706 - 0.20207 * Par.To_0) * 1E-3 * Var.D_dic ** 2
            - (1.2748 - 0.12015 * Par.To_0) * 1E-5 * Var.D_dic ** 3
            + (2.4491 - 0.12639 * Par.To_0) * 1E-7 * Var.D_dic ** 4
            - (1.5468 - 0.15326 * Par.To_0) * 1E-10 * Var.D_dic ** 5)
    D_pCO2 = (Par.CO2_pi + Par.k_b_dic * D_pCO2) - Par.CO2_pi
    return D_pCO2


## net primary productivity factor
OSCAR_bgc.process(
    Out = 'f_npp', 
    In = ('D_CO2',), 
    Eq = lambda Var, Par: Eq__f_npp(Var, Par), 
    units = '1')

def Eq__f_npp(Var, Par):
    f_CO2 = (1 + Par.b2_npp_CO2 / Par.x_npp_CO2 * ((1 + Var.D_CO2 / Par.CO2_pi) ** Par.x_npp_CO2 - 1))
    return f_CO2


## wildfire factor
OSCAR_bgc.process(
    Out = 'f_fire', 
    In = ('D_npp',), 
    Eq = lambda Var, Par: Eq__f_fire(Var, Par), 
    units = '1')

def Eq__f_fire(Var, Par):
    f_CO2 = safe_exp(Par.x_fire_npp * np.log(safe_ratio(1 + Var.D_npp / Par.npp_pi)), 1/Par.v_fire ** 0.5)
    return f_CO2


## total mortality factor
OSCAR_bgc.process(
    Out = 'f_mort', 
    In = ('D_npp',), 
    Eq = lambda Var, Par: Eq__f_mort(Var, Par), 
    units = '1')

def Eq__f_mort(Var, Par):
    f_CO2 = safe_exp(Par.x_mort_npp * np.log(safe_ratio(1 + Var.D_npp / Par.npp_pi)), 1/Par.v_mort ** 0.5)
    return f_CO2


## coarse woody debris decay factor
OSCAR_bgc.process(
    Out = 'f_cwd', 
    In = (), 
    Eq = lambda Var, Par: Eq__f_cwd(Var, Par), 
    units = '1')

def Eq__f_cwd(Var, Par):
    return 1


## soil respiration factor
OSCAR_bgc.process(
    Out = 'f_resp', 
    In = ('D_ffall',), 
    Eq = lambda Var, Par: Eq__f_resp(Var, Par), 
    units = '1')

def Eq__f_resp(Var, Par):
    f_CO2 = safe_exp(Par.x_resp_fall * np.log(safe_ratio(1 + Var.D_ffall / Par.ffall_pi)), 1/Par.v_resp ** 0.5)
    return f_CO2


## heterotrophic respiration factor for permafrost
OSCAR_bgc.process(
    Out = 'f_ethaw', 
    In = (), 
    Eq = lambda Var, Par: Eq__f_ethaw(Var, Par), 
    units = '1')

def Eq__f_ethaw(Var, Par):
    return 1


## theoretical thawed fraction
OSCAR_bgc.process(
    Out = 'D_pthaw_bar', 
    In = (), 
    Eq = lambda Var, Par: Eq__D_pthaw_bar(Var, Par), 
    units = '1')

def Eq__D_pthaw_bar(Var, Par):
    return -Par.pthaw_min + (1 + Par.pthaw_min) / (1 + ((1/Par.pthaw_min + 1) ** Par.x_pthaw - 1)) ** (1/Par.x_pthaw)


## wetland areal emissions
OSCAR_bgc.process(
    Out = 'D_ewet', 
    In = ('D_CO2',), 
    Eq = lambda Var, Par: Eq__D_ewet(Var, Par), 
    units = 'TgC Mha-1 yr-1')

def Eq__D_ewet(Var, Par):
    f_CO2 = 1 + Par.b_ewet_CO2 * np.log1p(Var.D_CO2 / Par.CO2_pi)
    return Par.ewet_pi * (f_CO2 - 1)
    

## wetland extent
OSCAR_bgc.process(
    Out = 'D_Awet', 
    In = ('D_CO2',), 
    Eq = lambda Var, Par: Eq__D_Awet(Var, Par), 
    units = 'Mha')

def Eq__D_Awet(Var, Par):
    f_CO2 = (1 + Var.D_CO2 / Par.CO2_pi) ** Par.x_Awet_CO2
    return Par.Awet_pi * (f_CO2 - 1)

