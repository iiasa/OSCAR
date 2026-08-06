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
## note: assumes climate was same between PI and precalibration period
OSCAR_landC_density.process(
    Out = 'npp_pi', 
    Eq = lambda Par: Eq__npp_pi(Par), 
    units = 'PgC Mha-1 yr-1')

def Eq__npp_pi(Par):
    fct_CO2= 1 + Par.b_npp_CO2 / Par.x_npp_CO2 * ((Par.CO2_pi / Par.CO2_piL) ** Par.x_npp_CO2 - 1)
    return Par.k_npp * Par.npp_piL * fct_CO2


## adjusted CO2 fertilisation parameter
OSCAR_landC_density.process(
    Out = 'b2_npp_CO2', 
    Eq = lambda Par: Eq__b2_npp_CO2(Par), 
    units = '1')

def Eq__b2_npp_CO2(Par):
    return Par.b_npp_CO2 * (Par.CO2_pi / Par.CO2_piL) ** Par.x_npp_CO2 / (1 +  Par.b_npp_CO2 / Par.x_npp_CO2 * ((Par.CO2_pi / Par.CO2_piL) ** Par.x_npp_CO2 - 1))


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

## relative change net primary productivity
OSCAR_landC_density.process(
    Out = 'r_npp', 
    In = ('D_CO2', 'D_Tl', 'r_Pl'), 
    Eq = lambda Var, Par: Eq__r_npp(Var, Par), 
    units = '1')

def Eq__r_npp(Var, Par):
    fct_CO2= (1 + Par.b2_npp_CO2 / Par.x_npp_CO2 * ((1 + Var.D_CO2 / Par.CO2_pi) ** Par.x_npp_CO2 - 1))
    fct_Tl = safe_exp(Par.g_npp_T2 * 2 * Par.D_Topt_npp * Var.D_Tl, 100) * np.exp(-Par.g_npp_T2 * Var.D_Tl**2)
    fct_Pl = safe_exp(Par.x_npp_P * np.log(Var.r_Pl), 100)
    return safe_ratio(fct_CO2* fct_Tl * fct_Pl)


## net primary productivity (areal)
OSCAR_landC_density.process(
    Out = 'D_npp', 
    In = ('r_npp',), 
    Eq = lambda Var, Par: Eq__D_npp(Var, Par), 
    units = 'PgC Mha-1 yr-1')

def Eq__D_npp(Var, Par):
    return Par.npp_pi * (Var.r_npp - 1)


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


## relative change in wildfire rate
OSCAR_landC_density.process(
    Out = 'r_vfire', 
    In = ('r_npp', 'D_Tl', 'r_Pl'), 
    Eq = lambda Var, Par: Eq__r_vfire(Var, Par), 
    units = '1')

def Eq__r_vfire(Var, Par):
    fct_npp = safe_exp(Par.x_fire_npp * np.log(Var.r_npp), f_max(Par.v_fire))
    fct_npp2 = safe_exp(Par.x_fire_npp2 * np.log(Var.r_npp)**2, f_max(Par.v_fire))
    fct_Tl = safe_exp(Par.g_fire_T * Var.D_Tl, f_max(Par.v_fire))
    fct_Pl = safe_exp(Par.g_fire_P * Par.Pl_pi * (Var.r_Pl - 1), f_max(Par.v_fire))
    return fct_npp * fct_npp2 * fct_Tl * fct_Pl


## wildfire emissions (areal)
OSCAR_landC_density.process(
    Out = 'D_efire', 
    In = ('r_vfire', 'D_cveg'), 
    Eq = lambda Var, Par: Eq__D_efire(Var, Par), 
    units = 'PgC Mha-1 yr-1')

def Eq__D_efire(Var, Par):
    return Par.v_fire * ((Par.cveg_pi + Var.D_cveg) * Var.r_vfire - Par.cveg_pi)


## relative change in total mortality rate
OSCAR_landC_density.process(
    Out = 'r_vmort', 
    In = ('r_npp', 'D_Tl', 'r_Pl'), 
    Eq = lambda Var, Par: Eq__r_vmort(Var, Par), 
    units = '1')

def Eq__r_vmort(Var, Par):
    fct_npp = safe_exp(Par.x_mort_npp * np.log(Var.r_npp), f_max(Par.v_mort))
    fct_Tl = safe_exp(Par.g_mort_T * Var.D_Tl,  f_max(Par.v_mort))
    fct_Pl = safe_exp(Par.x_mort_P * np.log(Var.r_Pl), f_max(Par.v_mort))
    return fct_npp * fct_Tl * fct_Pl


## total mortality flux (areal)
OSCAR_landC_density.process(
    Out = 'D_fmort', 
    In = ('r_vmort', 'D_cveg'), 
    Eq = lambda Var, Par: Eq__D_fmort(Var, Par), 
    units='PgC Mha-1 yr-1')

def Eq__D_fmort(Var, Par):
    return Par.v_mort * ((Par.cveg_pi + Var.D_cveg) * Var.r_vmort - Par.cveg_pi)


## litterfall flux (areal)
## note: assumes same fraction as during preindustrial
OSCAR_landC_density.process(
    Out = 'D_ffall', 
    In = ('D_fmort',), 
    Eq = lambda Var, Par: Eq__D_ffall(Var, Par), 
    units='PgC Mha-1 yr-1')

def Eq__D_ffall(Var, Par):
    return (1 - Par.p2_npp_wood) * Var.D_fmort


## relative change in coarse woody debris decay rate
OSCAR_landC_density.process(
    Out = 'r_vcwd', 
    In = ('D_Tl',), 
    Eq = lambda Var, Par: Eq__r_vcwd(Var, Par), 
    units = '1')

def Eq__r_vcwd(Var, Par):
    return safe_exp(Par.g_cwd_T * Var.D_Tl,  f_max(Par.v_cwd))


## coarse woody debris decay flux (areal)
OSCAR_landC_density.process(
    Out = 'D_fcwd', 
    In = ('r_vcwd', 'D_ccwd',), 
    Eq = lambda Var, Par: Eq__D_fcwd(Var, Par), 
    units='PgC Mha-1 yr-1')

def Eq__D_fcwd(Var, Par):
    return Par.v_cwd * ((Par.ccwd_pi + Var.D_ccwd) * Var.r_vcwd - Par.ccwd_pi)


## coarse woody debris emissions (areal)
OSCAR_landC_density.process(
    Out = 'D_ecwd', 
    In = ('D_fcwd',), 
    Eq = lambda Var, Par: Eq__D_ecwd(Var, Par), 
    units='PgC Mha-1 yr-1')

def Eq__D_ecwd(Var, Par):
    return Par.p_cwd_resp * Var.D_fcwd


## relative change in soil respiration rate
OSCAR_landC_density.process(
    Out = 'r_vresp', 
    In = ('D_ffall', 'D_Tl', 'r_Pl'), 
    Eq = lambda Var, Par: Eq__r_vresp(Var, Par), 
    units = '1')

def Eq__r_vresp(Var, Par):
    fct_in = safe_exp(Par.x_resp_fall * np.log(safe_ratio(1 + Var.D_ffall / Par.ffall_pi)),  f_max(Par.v_resp))
    fct_Tl = safe_exp(Par.g_resp_T * Var.D_Tl,  f_max(Par.v_resp))
    fct_Pl = safe_exp(Par.x_resp_P * np.log(Var.r_Pl),  f_max(Par.v_resp))
    return fct_in * fct_Tl * fct_Pl


## soil respiration (areal)
OSCAR_landC_density.process(
    Out = 'D_esoil', 
    In = ('r_vresp', 'D_csoil'), 
    Eq = lambda Var, Par: Eq__D_esoil(Var, Par), 
    units='PgC Mha-1 yr-1')

def Eq__D_esoil(Var, Par):
    return Par.v_resp * ((Par.csoil_pi + Var.D_csoil) * Var.r_vresp - Par.csoil_pi)


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

