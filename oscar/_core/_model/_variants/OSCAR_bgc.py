import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import safe_exp, f_max, safe_ratio


##################################################
##   OSCAR (BGC variant)
##################################################

## import and copy main model
from oscar._core._model.OSCAR import OSCAR
OSCAR_bgc = OSCAR.copy(new_name='OSCAR_bgc')


## BGC PROCESSES

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
    Out = 'r_npp', 
    In = ('D_CO2',), 
    Eq = lambda Var, Par: Eq__r_npp(Var, Par), 
    units = '1')

def Eq__r_npp(Var, Par):
    fct_CO2 = (1 + Par.b2_npp_CO2 / Par.x_npp_CO2 * ((1 + Var.D_CO2 / Par.CO2_pi) ** Par.x_npp_CO2 - 1))
    return safe_ratio(fct_CO2)


## wildfire factor
OSCAR_bgc.process(
    Out = 'r_vfire', 
    In = ('r_npp',), 
    Eq = lambda Var, Par: Eq__r_vfire(Var, Par), 
    units = '1')

def Eq__r_vfire(Var, Par):
    fct_npp = safe_exp(Par.x_fire_npp * np.log(Var.r_npp), f_max(Par.v_fire))
    fct_npp2 = safe_exp(Par.x_fire_npp2 * np.log(Var.r_npp)**2, f_max(Par.v_fire))
    return fct_npp * fct_npp2


## total mortality factor
OSCAR_bgc.process(
    Out = 'r_vmort', 
    In = ('r_npp',), 
    Eq = lambda Var, Par: Eq__r_vmort(Var, Par), 
    units = '1')

def Eq__r_vmort(Var, Par):
    fct_npp = safe_exp(Par.x_mort_npp * np.log(Var.r_npp), f_max(Par.v_mort))
    return fct_npp


## coarse woody debris decay factor
OSCAR_bgc.process(
    Out = 'r_vcwd', 
    In = (), 
    Eq = lambda Var, Par: Eq__r_vcwd(Var, Par), 
    units = '1')

def Eq__r_vcwd(Var, Par):
    return 1.


## soil respiration factor
OSCAR_bgc.process(
    Out = 'r_vesp', 
    In = ('D_ffall',), 
    Eq = lambda Var, Par: Eq__r_vesp(Var, Par), 
    units = '1')

def Eq__r_vesp(Var, Par):
    fct_in = safe_exp(Par.x_resp_fall * np.log(safe_ratio(1 + Var.D_ffall / Par.ffall_pi)),  f_max(Par.v_resp))
    return fct_in


## heterotrophic respiration factor for permafrost
OSCAR_bgc.process(
    Out = 'r_ethaw', 
    In = (), 
    Eq = lambda Var, Par: Eq__f_ethaw(Var, Par), 
    units = '1')

def Eq__f_ethaw(Var, Par):
    return 1.


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
    fct_CO2 = 1 + Par.b_ewet_CO2 * np.log1p(Var.D_CO2 / Par.CO2_pi)
    return Par.ewet_pi * (safe_ratio(fct_CO2) - 1)
    

## wetland extent
OSCAR_bgc.process(
    Out = 'D_Awet', 
    In = ('D_CO2',), 
    Eq = lambda Var, Par: Eq__D_Awet(Var, Par), 
    units = 'Mha')

def Eq__D_Awet(Var, Par):
    f_CO2 = (1 + Var.D_CO2 / Par.CO2_pi) ** Par.x_Awet_CO2
    return Par.Awet_pi * (safe_ratio(f_CO2) - 1)

