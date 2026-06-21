import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model


#####################################################################
##   STRATOSPHERIC CHEMISTRY
#####################################################################

## initialize
OSCAR_strato = Model('OSCAR_strato')

## module based on:
## (Prather et al., 2015; https://doi.org/10.1002/2015JD023267)
## following formulation of:
## (Holmes et al., 2013; https://doi.org/10.5194/acp-13-285-2013)


##=====================
## Secondary parameters
##=====================

## preindustrial equivalent effective stratospheric chlorine
OSCAR_strato.process(
    Out = 'EESC_pi', 
    Eq = lambda Par: Eq__EESC_pi(Par), 
    units = 'ppt')

def Eq__EESC_pi(Par):
    return (Par.p_fracrel * (Par.n_Cl + Par.a_Cl_Br * Par.n_Br) * Par.Xhalo_pi).sum('spc_halo', min_count=1)


## present-day equivalent effective stratospheric chlorine
OSCAR_strato.process(
    Out = 'EESC_pd', 
    Eq = lambda Par: Eq__EESC_pd(Par), 
    units = 'ppt',
    core_dims = ['age_air'])

def Eq__EESC_pd(Par):
    return (Par.p_fracrel * (Par.n_Cl + Par.a_Cl_Br * Par.n_Br) * Par.Xhalo_lag_pd).sum('spc_halo', min_count=1)


## present-day stratospheric mean age of air (relative change)
OSCAR_strato.process(
    Out = 'f_ageair_pd', 
    Eq = lambda Par: Eq__f_ageair_pd(Par), 
    units = '1')

def Eq__f_ageair_pd(Par):
    if 'D_Tg_pd' not in Par: return None
    return 1 / (1 + Par.g_ageair * Par.D_Tg_pd)


## present-day stratospheric sink intensity
OSCAR_strato.process(
    Out = 'f_hv_pd', 
    Eq = lambda Par: Eq__f_hv_pd(Par), 
    units = '1')

def Eq__f_hv_pd(Par):
    if 'EESC_pi' not in Par: return None
    if 'EESC_pd' not in Par: return None
    if 'f_ageair_pd' not in Par: return None
    f_hv_N2O = Par.ch_hv_N2O * np.log(Par.N2O_pd / Par.N2O_pi)
    f_hv_EESC = Par.ch_hv_EESC * np.log(Par.EESC_pd.sel(age_air='mid_lat', drop=True) / Par.EESC_pi.sel(age_air='mid_lat', drop=True))
    f_hv_Tg = Par.ch_hv_ageair * np.log(Par.f_ageair_pd)
    return np.exp(f_hv_N2O + f_hv_EESC + f_hv_Tg)


##=====================
## Diagnostic variables
##=====================

## equivalent effective stratospheric chlorine
## (Newman et al., 2007; https://doi.org/10.5194/acp-7-4537-2007)
OSCAR_strato.process(
    Out = 'D_EESC', 
    In = ('D_Xhalo_lag',), 
    Eq = lambda Var, Par: Eq__D_EESC(Var, Par), 
    units = 'ppt',
    core_dims = ['age_air'])

def Eq__D_EESC(Var, Par):
    return (Par.p_fracrel * (Par.n_Cl + Par.a_Cl_Br * Par.n_Br) * Var.D_Xhalo_lag).sum('spc_halo', min_count=1)


## stratospheric mean age of air (relative change)
## note: functional form determined for OSCARv2 based on CCMVal2 data
## (Morgenstern et al., 2010; https://doi.org/10.1029/2009JD013728)
OSCAR_strato.process(
    Out = 'f_ageair', 
    In = ('D_Tg',), 
    Eq = lambda Var, Par: Eq__f_ageair(Var, Par), 
    units = '1')

def Eq__f_ageair(Var, Par):
    return 1 / (1 + Par.g_ageair * Var.D_Tg)


## stratospheric sink intensity
OSCAR_strato.process(
    Out = 'f_hv', 
    In = ('D_N2O', 'D_EESC', 'f_ageair'), 
    Eq = lambda Var, Par: Eq__f_hv(Var, Par), 
    units = '1')

def Eq__f_hv(Var, Par):
    f_hv_N2O = Par.ch_hv_N2O * np.log1p(Var.D_N2O / Par.N2O_pi)
    f_hv_EESC = Par.ch_hv_EESC * np.log1p(Var.D_EESC.sel(age_air='mid_lat', drop=True) / Par.EESC_pi.sel(age_air='mid_lat', drop=True))
    f_hv_Tg = Par.ch_hv_ageair * np.log(Var.f_ageair)
    return np.exp(f_hv_N2O + f_hv_EESC + f_hv_Tg)

