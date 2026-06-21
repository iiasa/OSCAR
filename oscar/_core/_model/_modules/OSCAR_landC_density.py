import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import safe_exp, f_max, safe_ratio


#####################################################################
##   LAND CARBON CYCLE - DENSITIES
#####################################################################

## initialize
OSCAR_landC_density = Model('OSCAR_landC_density')

## module adapted from:
## (Gasser et al., 2020; https://doi.org/10.5194/bg-17-4075-2020)


##=====================
## Secondary parameters
##=====================

## SHIFTED PREINDUSTRIAL STATE

## adjusted preindustrial net primary productivity
OSCAR_landC_density.process(
    Out = 'npp_pi', 
    Eq = lambda Par: Eq__npp_pi(Par), 
    units = 'PgC Mha-1 yr-1')

def Eq__npp_pi(Par):
    f_CO2 = 1 + Par.b_npp_CO2 / Par.x_npp_CO2 * ((Par.CO2_pi / Par.CO2_piL) ** Par.x_npp_CO2 - 1)
    f_Tl = np.exp(Par.g_npp_T2 * 2 * Par.D_Topt_npp * (Par.Tl_pi - Par.Tl_piL)) * np.exp(-Par.g_npp_T2 * (Par.Tl_pi - Par.Tl_piL)**2)
    f_Pl = np.exp(Par.x_npp_P * np.log(safe_ratio(Par.Pl_pi / Par.Pl_piL)))
    return Par.k_npp * Par.npp_piL * f_CO2 * f_Tl * f_Pl


## adjusted CO2 fertilisation parameter
OSCAR_landC_density.process(
    Out = 'b2_npp_CO2', 
    Eq = lambda Par: Eq__b2_npp_CO2(Par), 
    units = '1')

def Eq__b2_npp_CO2(Par):
    return Par.b_npp_CO2 * (Par.CO2_pi / Par.CO2_piL) ** Par.x_npp_CO2 / (1 +  Par.b_npp_CO2 / Par.x_npp_CO2 * ((Par.CO2_pi / Par.CO2_piL) ** Par.x_npp_CO2 - 1))


## adjusted optimal temperature parameter
OSCAR_landC_density.process(
    Out = 'D_Topt2_npp', 
    Eq = lambda Par: Eq__D_Topt2_npp(Par), 
    units = 'K')

def Eq__D_Topt2_npp(Par):
    return Par.D_Topt_npp + Par.Tl_piL - Par.Tl_pi


## adjusted wildfire rate
OSCAR_landC_density.process(
    Out = 'v_fire', 
    Eq = lambda Par: Eq__v_fire(Par), 
    units = 'yr-1')

def Eq__v_fire(Par):
    f_npp = 1.
    f_Tl = np.exp(Par.g_fire_T * (Par.Tl_pi - Par.Tl_piL))
    f_Pl = np.exp(Par.g_fire_P * (Par.Pl_pi - Par.Pl_piL))
    return Par.v_fire_piL * f_npp * f_Tl * f_Pl


## adjusted mortality rate
OSCAR_landC_density.process(
    Out = 'v_mort', 
    Eq = lambda Par: Eq__v_mort(Par), 
    units = 'yr-1')

def Eq__v_mort(Par):
    f_npp = 1.
    f_Tl = np.exp(Par.g_mort_T * (Par.Tl_pi - Par.Tl_piL))
    f_Pl = np.exp(Par.x_mort_P * np.log(safe_ratio(Par.Pl_pi / Par.Pl_piL)))
    return Par.v_mort_piL * f_npp * f_Tl * f_Pl


## adjusted respiration rate
OSCAR_landC_density.process(
    Out = 'v_resp', 
    Eq = lambda Par: Eq__v_resp(Par), 
    units = 'yr-1')

def Eq__v_resp(Par):
    f_in = 1.
    f_Tl = np.exp(Par.g_resp_T * (Par.Tl_pi - Par.Tl_piL))
    f_Pl = np.exp(Par.x_resp_P * np.log(safe_ratio(Par.Pl_pi / Par.Pl_piL)))
    return Par.v_resp_piL * f_in * f_Tl * f_Pl


## COARSE WOODY DEBRIS

## corrected fraction of npp going to woody biomass
## note: set to 0. for anthropogenic biomes to prevent CWD
OSCAR_landC_density.process(
    Out = 'p2_npp_wood', 
    Eq = lambda Par: Eq__p2_npp_wood(Par), 
    units = '1')

def Eq__p2_npp_wood(Par):
    return Par.p_npp_wood.where((Par.bio_land == 'Forest') | (Par.bio_land == 'Non-Forest'), 0.)


## coarse woody debris decay rate
## (Harmon et al., 2020; https://doi.org/10.1186/s13021-019-0136-6)
OSCAR_landC_density.process(
    Out = 'v_cwd', 
    Eq = lambda Par: Eq__v_cwd(Par), 
    units = 'yr-1')

def Eq__v_cwd(Par):
    return Par.v10_cwd * Par.q10_cwd ** ((Par.Tl_pi - Par.degC_to_K - 10) / 10)


## coarse woody debris decay sensitivity to temperature
## (Harmon et al., 2020; https://doi.org/10.1186/s13021-019-0136-6)
OSCAR_landC_density.process(
    Out = 'g_cwd_T', 
    Eq = lambda Par: Eq__g_cwd_T(Par), 
    units = 'K-1')

def Eq__g_cwd_T(Par):
    return np.log(Par.q10_cwd) / 10


## PREINDUSTRIAL STEADY-STATE

## preindustrial litterfall flux
OSCAR_landC_density.process(
    Out = 'ffall_pi', 
    Eq = lambda Par: Eq__ffall_pi(Par), 
    units = 'PgC Mha-1 yr-1')

def Eq__ffall_pi(Par):
    if 'p2_npp_wood' not in Par: return None
    if 'v_mort' not in Par: return None
    if 'cveg_pi' not in Par: return None
    return (1 - Par.p2_npp_wood) * Par.v_mort * Par.cveg_pi


## preindustrial vegetation carbon density
OSCAR_landC_density.process(
    Out = 'cveg_pi', 
    Eq = lambda Par: Eq__cveg_pi(Par), 
    units = 'PgC Mha-1')

def Eq__cveg_pi(Par):
    if 'npp_pi' not in Par: return None
    if 'v_mort' not in Par: return None
    if 'v_fire' not in Par: return None
    return (Par.npp_pi * (1 - Par.p_charv - Par.p_graz) / (Par.v_mort + Par.v_fire)).where((Par.v_mort + Par.v_fire) != 0, 0.)


## preindustrial coarse woody debris density
OSCAR_landC_density.process(
    Out = 'ccwd_pi', 
    Eq = lambda Par: Eq__ccwd_pi(Par), 
    units = 'PgC Mha-1')

def Eq__ccwd_pi(Par):
    if 'p2_npp_wood' not in Par: return None
    if 'v_mort' not in Par: return None
    if 'cveg_pi' not in Par: return None
    if 'v_cwd' not in Par: return None
    return (Par.p2_npp_wood * Par.v_mort * Par.cveg_pi / Par.v_cwd).where(Par.v_cwd != 0, 0.)


## preindustrial soil carbon density
OSCAR_landC_density.process(
    Out = 'csoil_pi', 
    Eq = lambda Par: Eq__csoil_pi(Par), 
    units = 'PgC Mha-1')

def Eq__csoil_pi(Par):
    if 'p2_npp_wood' not in Par: return None
    if 'v_mort' not in Par: return None
    if 'v_resp' not in Par: return None
    if 'cveg_pi' not in Par: return None
    if 'ccwd_pi' not in Par: return None
    if 'v_cwd' not in Par: return None
    return (((1 - Par.p2_npp_wood) * Par.v_mort * Par.cveg_pi + (1 - Par.p_cwd_resp) * Par.v_cwd * Par.ccwd_pi) / Par.v_resp).where(Par.v_resp != 0, 0.)


##=====================
## Diagnostic variables
##=====================

## net primary productivity factor
OSCAR_landC_density.process(
    Out = 'f_npp', 
    In = ('D_CO2', 'D_Tl', 'D_Pl'), 
    Eq = lambda Var, Par: Eq__f_npp(Var, Par), 
    units = '1')

def Eq__f_npp(Var, Par):
    f_CO2 = (1 + Par.b2_npp_CO2 / Par.x_npp_CO2 * ((1 + Var.D_CO2 / Par.CO2_pi) ** Par.x_npp_CO2 - 1))
    f_Tl = safe_exp(Par.g_npp_T2 * 2 * Par.D_Topt2_npp * Var.D_Tl, 100) * np.exp(-Par.g_npp_T2 * Var.D_Tl**2)
    f_Pl = safe_exp(Par.x_npp_P * np.log(safe_ratio(1 + Var.D_Pl / Par.Pl_pi)), 100)
    return f_CO2 * f_Tl * f_Pl


## net primary productivity (areal)
OSCAR_landC_density.process(
    Out = 'D_npp', 
    In = ('f_npp',), 
    Eq = lambda Var, Par: Eq__D_npp(Var, Par), 
    units = 'PgC Mha-1 yr-1')

def Eq__D_npp(Var, Par):
    return Par.npp_pi * (Var.f_npp - 1)


## crop harvesting (areal)
OSCAR_landC_density.process(
    Out = 'D_echarv', 
    In = ('D_npp',), 
    Eq = lambda Var, Par: Eq__D_echarv(Var, Par), 
    units = 'PgC Mha-1 yr-1')

def Eq__D_echarv(Var, Par):
    return Par.p_charv * Var.D_npp


## pasture grazing (areal)
OSCAR_landC_density.process(
    Out = 'D_egraz', 
    In = ('D_npp',), 
    Eq = lambda Var, Par: Eq__D_egraz(Var, Par), 
    units = 'PgC Mha-1 yr-1')

def Eq__D_egraz(Var, Par):
    return Par.p_graz * Var.D_npp


## wildfire factor
OSCAR_landC_density.process(
    Out = 'f_fire', 
    In = ('D_npp', 'D_Tl', 'D_Pl'), 
    Eq = lambda Var, Par: Eq__f_fire(Var, Par), 
    units = '1')

def Eq__f_fire(Var, Par):
    f_npp = safe_exp(Par.x_fire_npp * np.log(safe_ratio(1 + Var.D_npp / Par.npp_pi)), f_max(Par.v_fire))
    f_npp2 = safe_exp(Par.x_fire_npp2 * np.log(safe_ratio(1 + Var.D_npp / Par.npp_pi))**2, f_max(Par.v_fire))
    f_Tl = safe_exp(Par.g_fire_T * Var.D_Tl, f_max(Par.v_fire))
    f_Pl = safe_exp(Par.g_fire_P * Var.D_Pl, f_max(Par.v_fire))
    return f_npp * f_npp2 * f_Tl * f_Pl


## wildfire emissions (areal)
OSCAR_landC_density.process(
    Out = 'D_efire', 
    In = ('f_fire', 'D_cveg'), 
    Eq = lambda Var, Par: Eq__D_efire(Var, Par), 
    units = 'PgC Mha-1 yr-1')

def Eq__D_efire(Var, Par):
    return Par.v_fire * ((Par.cveg_pi + Var.D_cveg) * Var.f_fire - Par.cveg_pi)


## total mortality factor
OSCAR_landC_density.process(
    Out = 'f_mort', 
    In = ('D_npp', 'D_Tl', 'D_Pl'), 
    Eq = lambda Var, Par: Eq__f_mort(Var, Par), 
    units = '1')

def Eq__f_mort(Var, Par):
    f_npp = safe_exp(Par.x_mort_npp * np.log(safe_ratio(1 + Var.D_npp / Par.npp_pi)), f_max(Par.v_mort))
    f_Tl = safe_exp(Par.g_mort_T * Var.D_Tl,  f_max(Par.v_mort))
    f_Pl = safe_exp(Par.x_mort_P * np.log(safe_ratio(1 + Var.D_Pl / Par.Pl_piL)), f_max(Par.v_mort))
    return f_npp * f_Tl * f_Pl


## total mortality flux (areal)
OSCAR_landC_density.process(
    Out = 'D_fmort', 
    In = ('f_mort', 'D_cveg'), 
    Eq = lambda Var, Par: Eq__D_fmort(Var, Par), 
    units='PgC Mha-1 yr-1')

def Eq__D_fmort(Var, Par):
    return Par.v_mort * ((Par.cveg_pi + Var.D_cveg) * Var.f_mort - Par.cveg_pi)


## litterfall flux (areal)
## note: assumes same fraction as during preindustrial
OSCAR_landC_density.process(
    Out = 'D_ffall', 
    In = ('D_fmort',), 
    Eq = lambda Var, Par: Eq__D_ffall(Var, Par), 
    units='PgC Mha-1 yr-1')

def Eq__D_ffall(Var, Par):
    return (1 - Par.p2_npp_wood) * Var.D_fmort


## coarse woody debris decay factor
OSCAR_landC_density.process(
    Out = 'f_cwd', 
    In = ('D_Tl',), 
    Eq = lambda Var, Par: Eq__f_cwd(Var, Par), 
    units = '1')

def Eq__f_cwd(Var, Par):
    return safe_exp(Par.g_cwd_T * Var.D_Tl,  f_max(Par.v_cwd))


## coarse woody debris decay flux (areal)
OSCAR_landC_density.process(
    Out = 'D_fcwd', 
    In = ('f_cwd', 'D_ccwd',), 
    Eq = lambda Var, Par: Eq__D_fcwd(Var, Par), 
    units='PgC Mha-1 yr-1')

def Eq__D_fcwd(Var, Par):
    return Par.v_cwd * ((Par.ccwd_pi + Var.D_ccwd) * Var.f_cwd - Par.ccwd_pi)


## coarse woody debris emissions (areal)
OSCAR_landC_density.process(
    Out = 'D_ecwd', 
    In = ('D_fcwd',), 
    Eq = lambda Var, Par: Eq__D_ecwd(Var, Par), 
    units='PgC Mha-1 yr-1')

def Eq__D_ecwd(Var, Par):
    return Par.p_cwd_resp * Var.D_fcwd


## soil respiration factor
OSCAR_landC_density.process(
    Out = 'f_resp', 
    In = ('D_ffall', 'D_Tl', 'D_Pl'), 
    Eq = lambda Var, Par: Eq__f_resp(Var, Par), 
    units = '1')

def Eq__f_resp(Var, Par):
    f_in = safe_exp(Par.x_resp_fall * np.log(safe_ratio(1 + Var.D_ffall / Par.ffall_pi)),  f_max(Par.v_resp))
    f_Tl = safe_exp(Par.g_resp_T * Var.D_Tl,  f_max(Par.v_resp))
    f_Pl = safe_exp(Par.x_resp_P * np.log(safe_ratio(1 + Var.D_Pl / Par.Pl_piL)),  f_max(Par.v_resp))
    return f_in * f_Tl * f_Pl


## soil respiration (areal)
OSCAR_landC_density.process(
    Out = 'D_esoil', 
    In = ('f_resp', 'D_csoil'), 
    Eq = lambda Var, Par: Eq__D_esoil(Var, Par), 
    units='PgC Mha-1 yr-1')

def Eq__D_esoil(Var, Par):
    return Par.v_resp * ((Par.csoil_pi + Var.D_csoil) * Var.f_resp - Par.csoil_pi)


## net biome productivity (areal)
OSCAR_landC_density.process(
    Out = 'D_nbp', 
    In = ('D_npp', 'D_echarv', 'D_egraz', 'D_efire', 'D_ecwd', 'D_esoil'), 
    Eq = lambda Var, Par: Eq__D_nbp(Var, Par), 
    units='PgC Mha-1 yr-1')

def Eq__D_nbp(Var, Par):
    return Var.D_npp - Var.D_echarv - Var.D_egraz - Var.D_efire - Var.D_ecwd  - Var.D_esoil


##=====================
## Prognostic variables
##=====================

## vegetation carbon stock (areal)
OSCAR_landC_density.process(
    Out = 'D_cveg', 
    In = ('D_cveg', 'D_npp', 'D_echarv', 'D_egraz', 'D_efire', 'D_fmort'), 
    DiffEq = lambda Var, Par: DiffEq__D_cveg(Var, Par), 
    vLin = lambda Par: vLin__D_cveg(Par), 
    units='PgC Mha-1', 
    core_dims=['reg_land', 'bio_land'])

def DiffEq__D_cveg(Var, Par):
    return Var.D_npp - Var.D_echarv - Var.D_egraz - Var.D_efire - Var.D_fmort

def vLin__D_cveg(Par):
    return Par.v_mort + Par.v_fire


## coarse woody debris carbon stock (areal)
OSCAR_landC_density.process(
    Out = 'D_ccwd', 
    In = ('D_ccwd', 'D_fmort', 'D_ffall', 'D_fcwd'), 
    DiffEq = lambda Var, Par: DiffEq__D_ccwd(Var, Par), 
    vLin = lambda Par: vLin__D_ccwd(Par), 
    units='PgC Mha-1', 
    core_dims=['reg_land', 'bio_land'])

def DiffEq__D_ccwd(Var, Par):
    return (Var.D_fmort - Var.D_ffall) - Var.D_fcwd

def vLin__D_ccwd(Par):
    return Par.v_cwd


## soil carbon stock (areal)
OSCAR_landC_density.process(
    Out = 'D_csoil', 
    In = ('D_csoil', 'D_ffall', 'D_fcwd', 'D_ecwd', 'D_esoil'), 
    DiffEq = lambda Var, Par: DiffEq__D_csoil(Var, Par), 
    vLin = lambda Par: vLin__D_csoil(Par),
    units='PgC Mha-1', 
    core_dims=['reg_land', 'bio_land'])

def DiffEq__D_csoil(Var, Par):
    return Var.D_ffall + (Var.D_fcwd - Var.D_ecwd) - Var.D_esoil

def vLin__D_csoil(Par):
    return Par.v_resp

