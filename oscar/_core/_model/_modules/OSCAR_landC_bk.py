import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model


#####################################################################
##   LAND CARBON CYCLE - BOOKKEEPING
#####################################################################

## initialize
OSCAR_landC_bk = Model('OSCAR_landC_bk')

## bookkeeping based on:
## (Gitz & Ciais, 2003; https://doi.org/10.1029/2002GB001963)
## reformulated following definition 3 of:
## (Gasser & Ciais, 2013; https://doi.org/10.5194/esd-4-171-2013)
## with continuous equations adapted from:
## (Gasser et al., 2020; https://doi.org/10.5194/bg-17-4075-2020)


##=====================
## Secondary parameters
##=====================

## STRUCTURAL PARAMETERS

## fraction of reproductive tissues / storage carbon / ephemeral structures (remainder)
OSCAR_landC_bk.process(
    Out = 'p_soft', 
    Eq = lambda Par: Eq__p_soft(Par), 
    units='1')

def Eq__p_soft(Par):
    return 1 - Par.p_wood - Par.p_root - Par.p_leaf


## fraction of woody biomass going to HWP subpools
OSCAR_landC_bk.process(
    Out = 'p_hwp', 
    Eq = lambda Par: Eq__p_hwp(Par), 
    units='1')

def Eq__p_hwp(Par):
    return Par.p_hwp_comm + Par.p_hwp_noncomm * Par.p_fuel_noncomm


## fraction of slash after land cover change
OSCAR_landC_bk.process(
    Out = 'p_slash', 
    Eq = lambda Par: Eq__p_slash(Par), 
    units='1')

def Eq__p_slash(Par):
    if 'p_hwp' not in Par: return None
    return 1 - Par.p_hwp.sum('box_hwp', min_count=1)


## factor for mortality rate assuming regrowth dominated by woody biomass
OSCAR_landC_bk.process(
    Out = 'f_mort_regr', 
    Eq = lambda Par: Eq__f_mort_regr(Par), 
    units='1')

def Eq__f_mort_regr(Par):
    if 'p2_npp_wood' not in Par: return None
    return Par.p2_npp_wood.where(Par.p2_npp_wood != 0, 1.)


## fraction of biomass growth reached under shifting cultivation
OSCAR_landC_bk.process(
    Out = 'f_cveg_shift', 
    Eq = lambda Par: Eq__f_cveg_shift(Par), 
    units='1')

def Eq__f_cveg_shift(Par):
    if 'v_mort' not in Par: return None
    if 'v_fire' not in Par: return None
    if 'f_mort_regr' not in Par: return None
    return 1 - np.exp(-(Par.f_mort_regr * Par.v_mort + Par.v_fire) * Par.t_shift)


## PREINDUSTRIAL STEADY-STATE

## preindustrial bookkeeping imbalance of vegetation pool
OSCAR_landC_bk.process(
    Out = 'Cveg_bk_pi', 
    Eq = lambda Par: Eq__Cveg_bk_pi(Par), 
    units='PgC')

def Eq__Cveg_bk_pi(Par):
    if 'v_mort' not in Par: return None
    if 'v_fire' not in Par: return None
    if 'cveg_pi' not in Par: return None
    if 'f_mort_regr' not in Par: return None
    if 'f_cveg_shift' not in Par: return None
    ## age effect
    f_cveg_age = 1 - np.exp(-(Par.f_mort_regr * Par.v_mort + Par.v_fire) * Par.age_bk_pi)
    ## wood harvest
    dCveg_bk_pi_wharv = -Par.cveg_pi * f_cveg_age * Par.dA_wharv_pi
    ## shifting cultivation
    dCveg_bk_pi_shift = -Par.cveg_pi * Par.f_cveg_shift * Par.dA_shift_pi.sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## all
    dCveg_bk_pi = dCveg_bk_pi_wharv + dCveg_bk_pi_shift
    return (dCveg_bk_pi / (Par.f_mort_regr * Par.v_mort + Par.v_fire)).where((Par.f_mort_regr * Par.v_mort + Par.v_fire) != 0, 0.)


## preindustrial bookkeeping imbalance of coarse woody debris pool
OSCAR_landC_bk.process(
    Out = 'Ccwd_bk_pi', 
    Eq = lambda Par: Eq__Ccwd_bk_pi(Par), 
    units='PgC')

def Eq__Ccwd_bk_pi(Par):
    if 'p2_npp_wood' not in Par: return None
    if 'cveg_pi' not in Par: return None
    if 'v_mort' not in Par: return None
    if 'v_cwd' not in Par: return None
    if 'p_slash' not in Par: return None
    if 'age_bk_pi' not in Par: return None
    if 'f_cveg_shift' not in Par: return None
    if 'Cveg_bk_pi' not in Par: return None
    ## age effect
    f_cveg_age = 1 - np.exp(-(Par.f_mort_regr * Par.v_mort + Par.v_fire) * Par.age_bk_pi)
    ## wood harvest
    dCcwd_bk_pi_wharv = Par.cveg_pi * Par.p_wood * Par.p_slash * f_cveg_age * Par.dA_wharv_pi
    ## shifting cultivation
    dCcwd_bk_pi_shift = Par.cveg_pi * Par.p_wood * Par.p_slash * Par.f_cveg_shift * Par.dA_shift_pi.sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## all
    dCcwd_bk_pi = dCcwd_bk_pi_wharv + dCcwd_bk_pi_shift
    return ((Par.p2_npp_wood * Par.v_mort * Par.Cveg_bk_pi + dCcwd_bk_pi) / Par.v_cwd).where(Par.v_cwd != 0, 0.)


## preindustrial bookkeeping imbalance of soil pool
OSCAR_landC_bk.process(
    Out = 'Csoil_bk_pi', 
    Eq = lambda Par: Eq__Csoil_bk_pi(Par), 
    units='PgC')

def Eq__Csoil_bk_pi(Par):
    if 'p2_npp_wood' not in Par: return None
    if 'cveg_pi' not in Par: return None
    if 'v_mort' not in Par: return None
    if 'v_cwd' not in Par: return None
    if 'v_resp' not in Par: return None
    if 'p_soft' not in Par: return None
    if 'f_cveg_shift' not in Par: return None
    if 'Cveg_bk_pi' not in Par: return None
    if 'Ccwd_bk_pi' not in Par: return None
    ## age effect
    f_cveg_age = 1 - np.exp(-(Par.f_mort_regr * Par.v_mort + Par.v_fire) * Par.age_bk_pi)
    ## wood harvest
    dCsoil_bk_pi_wharv = Par.cveg_pi * (Par.p_leaf + Par.p_soft + Par.p_root) * f_cveg_age * Par.dA_wharv_pi
    ## shifting cultivation
    dCsoil_bk_pi_shift = Par.cveg_pi * (Par.p_leaf + Par.p_soft + Par.p_root) * Par.f_cveg_shift * Par.dA_shift_pi.sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## all
    dCsoil_bk_pi = dCsoil_bk_pi_wharv + dCsoil_bk_pi_shift
    return (((1 - Par.p2_npp_wood) * Par.v_mort * Par.Cveg_bk_pi + (1 - Par.p_cwd_resp) * Par.v_cwd * Par.Ccwd_bk_pi + dCsoil_bk_pi) / Par.v_resp).where(Par.v_resp != 0, 0.)


## preindustrial bookkeeping imbalance of harvested wood products pool
OSCAR_landC_bk.process(
    Out = 'Chwp_bk_pi', 
    Eq = lambda Par: Eq__Chwp_bk_pi(Par), 
    units='PgC')

def Eq__Chwp_bk_pi(Par):
    if 'cveg_pi' not in Par: return None
    if 'p_hwp' not in Par: return None
    if 'age_bk_pi' not in Par: return None
    if 'f_cveg_shift' not in Par: return None
    ## age effect
    f_cveg_age = 1 - np.exp(-(Par.f_mort_regr * Par.v_mort + Par.v_fire) * Par.age_bk_pi)
    ## wood harvest
    dChwp_bk_pi_wharv = Par.cveg_pi * Par.p_wood * Par.p_hwp * f_cveg_age * Par.dA_wharv_pi
    ## shifting cultivation
    dChwp_bk_pi_shift = Par.cveg_pi * Par.p_wood * Par.p_hwp * Par.f_cveg_shift * Par.dA_shift_pi.sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## all
    dChwp_bk_pi = dChwp_bk_pi_wharv + dChwp_bk_pi_shift
    return (dChwp_bk_pi / Par.v_hwp).where(Par.v_hwp != 0, 0.)


##=====================
## Diagnostic variables
##=====================

## BOOKKEEPING INITIALIZATION

## fraction of biomass growth reached for secondary land
OSCAR_landC_bk.process(
    Out = 'f_cveg_age', 
    In = ('D_age_bk',), 
    Eq = lambda Var, Par: Eq__f_cveg_age(Var, Par), 
    units='1')

def Eq__f_cveg_age(Var, Par):
    return 1 - np.exp(-(Par.f_mort_regr * Par.v_mort + Par.v_fire) * (Par.age_bk_pi + Var.D_age_bk))


## bookkeeping initialization of vegetation
OSCAR_landC_bk.process(
    Out = 'D_Fbk_veg2veg', 
    In = ('D_cveg', 'f_cveg_age', 'D_dA_lcc1', 'D_dA_lcc2', 'D_dA_wharv1', 'D_dA_wharv2', 'D_dA_shift'), 
    Eq = lambda Var, Par: Eq__D_Fbk_veg2veg(Var, Par), 
    units='PgC yr-1')

def Eq__D_Fbk_veg2veg(Var, Par):
    ## land cover change
    D_Fbk_lcc1 = (-(Par.cveg_pi + Var.D_cveg).rename({'bio_land':'bio_to'}) * Var.D_dA_lcc1).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    D_Fbk_lcc2 = (-((Par.cveg_pi + Var.D_cveg) * Var.f_cveg_age).rename({'bio_land':'bio_to'}) * Var.D_dA_lcc2).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## wood harvest
    D_Fbk_wharv1 = (-(Par.cveg_pi + Var.D_cveg)) * Var.D_dA_wharv1
    D_Fbk_wharv2 = (-(Par.cveg_pi + Var.D_cveg) * Var.f_cveg_age) * Var.D_dA_wharv2
    ## shifting cultivation
    D_Fbk_shift = (-((Par.cveg_pi + Var.D_cveg) * Par.f_cveg_shift).rename({'bio_land':'bio_to'}) * Var.D_dA_shift).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## all
    return D_Fbk_lcc1 + D_Fbk_lcc2 + D_Fbk_wharv1 + D_Fbk_wharv2 + D_Fbk_shift


## bookkeeping initialization of coarse woody debris
OSCAR_landC_bk.process(
    Out = 'D_Fbk_cwd2cwd', 
    In = ('D_ccwd', 'D_dA_lcc1', 'D_dA_lcc2'), 
    Eq = lambda Var, Par: Eq__D_Fbk_cwd2cwd(Var, Par), 
    units='PgC yr-1')

def Eq__D_Fbk_cwd2cwd(Var, Par):
    ## land cover change
    D_Fbk_lcc1 = (((Par.ccwd_pi + Var.D_ccwd).rename({'bio_land':'bio_from'}) - (Par.ccwd_pi + Var.D_ccwd).rename({'bio_land':'bio_to'})) * Var.D_dA_lcc1).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    D_Fbk_lcc2 = (((Par.ccwd_pi + Var.D_ccwd).rename({'bio_land':'bio_from'}) - (Par.ccwd_pi + Var.D_ccwd).rename({'bio_land':'bio_to'})) * Var.D_dA_lcc2).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## all
    return D_Fbk_lcc1 + D_Fbk_lcc2


## bookkeeping initialization of soil
OSCAR_landC_bk.process(
    Out = 'D_Fbk_soil2soil', 
    In = ('D_csoil', 'D_dA_lcc1', 'D_dA_lcc2'), 
    Eq = lambda Var, Par: Eq__D_Fbk_soil2soil(Var, Par), 
    units='PgC yr-1')

def Eq__D_Fbk_soil2soil(Var, Par):
    ## land cover change
    D_Fbk_lcc1 = (((Par.csoil_pi + Var.D_csoil).rename({'bio_land':'bio_from'}) - (Par.csoil_pi + Var.D_csoil).rename({'bio_land':'bio_to'})) * Var.D_dA_lcc1).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    D_Fbk_lcc2 = (((Par.csoil_pi + Var.D_csoil).rename({'bio_land':'bio_from'}) - (Par.csoil_pi + Var.D_csoil).rename({'bio_land':'bio_to'})) * Var.D_dA_lcc2).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## all
    return D_Fbk_lcc1 + D_Fbk_lcc2


## bookkeeping transfer from vegetation to coarse woody debris
OSCAR_landC_bk.process(
    Out = 'D_Fbk_veg2cwd', 
    In = ('D_cveg', 'f_cveg_age', 'D_dA_lcc1', 'D_dA_lcc2', 'D_dA_wharv1', 'D_dA_wharv2', 'D_dA_shift'), 
    Eq = lambda Var, Par: Eq__D_Fbk_veg2cwd(Var, Par), 
    units='PgC yr-1')

def Eq__D_Fbk_veg2cwd(Var, Par):
    ## land cover change
    D_Fbk_lcc1 = (((Par.cveg_pi + Var.D_cveg) * Par.p_wood * Par.p_slash).rename({'bio_land':'bio_from'}) * Var.D_dA_lcc1).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    D_Fbk_lcc2 = (((Par.cveg_pi + Var.D_cveg) * Par.p_wood * Par.p_slash * Var.f_cveg_age).rename({'bio_land':'bio_from'}) * Var.D_dA_lcc2).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## wood harvest
    D_Fbk_wharv1 = ((Par.cveg_pi + Var.D_cveg) * Par.p_wood * Par.p_slash) * Var.D_dA_wharv1
    D_Fbk_wharv2 = ((Par.cveg_pi + Var.D_cveg) * Par.p_wood * Par.p_slash * Var.f_cveg_age) * Var.D_dA_wharv2
    ## shifting cultivation
    D_Fbk_shift = (((Par.cveg_pi + Var.D_cveg) * Par.p_wood * Par.p_slash * Par.f_cveg_shift).rename({'bio_land':'bio_from'}) * Var.D_dA_shift).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## all
    return D_Fbk_lcc1 + D_Fbk_lcc2 + D_Fbk_wharv1 + D_Fbk_wharv2 + D_Fbk_shift


## bookkeeping transfer from vegetation to litter
OSCAR_landC_bk.process(
    Out = 'D_Fbk_veg2litter', 
    In = ('D_cveg', 'f_cveg_age', 'D_dA_lcc1', 'D_dA_lcc2', 'D_dA_wharv1', 'D_dA_wharv2', 'D_dA_shift'), 
    Eq = lambda Var, Par: Eq__D_Fbk_veg2litter(Var, Par), 
    units='PgC yr-1')

def Eq__D_Fbk_veg2litter(Var, Par):
    ## land cover change
    D_Fbk_lcc1 = (((Par.cveg_pi + Var.D_cveg) * (Par.p_leaf + Par.p_soft)).rename({'bio_land':'bio_from'}) * Var.D_dA_lcc1).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    D_Fbk_lcc2 = (((Par.cveg_pi + Var.D_cveg) * (Par.p_leaf + Par.p_soft) * Var.f_cveg_age).rename({'bio_land':'bio_from'}) * Var.D_dA_lcc2).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## wood harvest
    D_Fbk_wharv1 = ((Par.cveg_pi + Var.D_cveg) * (Par.p_leaf + Par.p_soft)) * Var.D_dA_wharv1
    D_Fbk_wharv2 = ((Par.cveg_pi + Var.D_cveg) * (Par.p_leaf + Par.p_soft) * Var.f_cveg_age) * Var.D_dA_wharv2
    ## shifting cultivation
    D_Fbk_shift = (((Par.cveg_pi + Var.D_cveg) * (Par.p_leaf + Par.p_soft) * Par.f_cveg_shift).rename({'bio_land':'bio_from'}) * Var.D_dA_shift).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## all
    return D_Fbk_lcc1 + D_Fbk_lcc2 + D_Fbk_wharv1 + D_Fbk_wharv2 + D_Fbk_shift


## bookkeeping transfer from vegetation to soil
OSCAR_landC_bk.process(
    Out = 'D_Fbk_veg2soil', 
    In = ('D_cveg', 'f_cveg_age', 'D_dA_lcc1', 'D_dA_lcc2', 'D_dA_wharv1', 'D_dA_wharv2', 'D_dA_shift'), 
    Eq = lambda Var, Par: Eq__D_Fbk_veg2soil(Var, Par), 
    units='PgC yr-1')

def Eq__D_Fbk_veg2soil(Var, Par):
    ## land cover change
    D_Fbk_lcc1 = (((Par.cveg_pi + Var.D_cveg) * Par.p_root).rename({'bio_land':'bio_from'}) * Var.D_dA_lcc1).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    D_Fbk_lcc2 = (((Par.cveg_pi + Var.D_cveg) * Par.p_root * Var.f_cveg_age).rename({'bio_land':'bio_from'}) * Var.D_dA_lcc2).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## wood harvest
    D_Fbk_wharv1 = ((Par.cveg_pi + Var.D_cveg) * Par.p_root) * Var.D_dA_wharv1
    D_Fbk_wharv2 = ((Par.cveg_pi + Var.D_cveg) * Par.p_root * Var.f_cveg_age) * Var.D_dA_wharv2
    ## shifting cultivation
    D_Fbk_shift = (((Par.cveg_pi + Var.D_cveg) * Par.p_root * Par.f_cveg_shift).rename({'bio_land':'bio_from'}) * Var.D_dA_shift).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## all
    return D_Fbk_lcc1 + D_Fbk_lcc2 + D_Fbk_wharv1 + D_Fbk_wharv2 + D_Fbk_shift


## bookkeeping transfer from vegetation to harvested wood products
OSCAR_landC_bk.process(
    Out = 'D_Fbk_veg2hwp', 
    In = ('D_cveg', 'f_cveg_age', 'D_dA_lcc1', 'D_dA_lcc2', 'D_dA_wharv1', 'D_dA_wharv2', 'D_dA_shift'), 
    Eq = lambda Var, Par: Eq__D_Fbk_veg2hwp(Var, Par), 
    units='PgC yr-1')

def Eq__D_Fbk_veg2hwp(Var, Par):
    ## land cover change
    D_Fbk_lcc1 = (((Par.cveg_pi + Var.D_cveg) * Par.p_wood * Par.p_hwp).rename({'bio_land':'bio_from'}) * Var.D_dA_lcc1).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    D_Fbk_lcc2 = (((Par.cveg_pi + Var.D_cveg) * Par.p_wood * Par.p_hwp * Var.f_cveg_age).rename({'bio_land':'bio_from'}) * Var.D_dA_lcc2).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## wood harvest
    D_Fbk_wharv1 = ((Par.cveg_pi + Var.D_cveg) * Par.p_wood * Par.p_hwp) * Var.D_dA_wharv1
    D_Fbk_wharv2 = ((Par.cveg_pi + Var.D_cveg) * Par.p_wood * Par.p_hwp * Var.f_cveg_age) * Var.D_dA_wharv2
    ## shifting cultivation
    D_Fbk_shift = (((Par.cveg_pi + Var.D_cveg) * Par.p_wood * Par.p_hwp * Par.f_cveg_shift).rename({'bio_land':'bio_from'}) * Var.D_dA_shift).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## all
    return D_Fbk_lcc1 + D_Fbk_lcc2 + D_Fbk_wharv1 + D_Fbk_wharv2 + D_Fbk_shift


## bookkeeping transfer from vegetation to atmosphere (instantaneous emission)
OSCAR_landC_bk.process(
    Out = 'D_Fbk_veg2atm', 
    In = ('D_cveg',), 
    Eq = lambda Var, Par: Eq__D_Fbk_veg2atm(Var, Par), 
    units='PgC yr-1')

def Eq__D_Fbk_veg2atm(Var, Par):
    return 0.


## NATURAL CYCLE

## altered NPP (under bookkeeping)
OSCAR_landC_bk.process(
    Out = 'D_NPP_bk', 
    In = (), 
    Eq = lambda Var, Par: Eq__D_NPP_bk(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_NPP_bk(Var, Par):
    return 0.


## crop harvesting (under bookkeeping)
OSCAR_landC_bk.process(
    Out = 'D_Echarv_bk', 
    In = ('D_NPP_bk',), 
    Eq = lambda Var, Par: Eq__D_Echarv_bk(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Echarv_bk(Var, Par):
    return Par.p_charv * Var.D_NPP_bk


## pasture grazing (under bookkeeping)
OSCAR_landC_bk.process(
    Out = 'D_Egraz_bk', 
    In = ('D_NPP_bk',), 
    Eq = lambda Var, Par: Eq__D_Egraz_bk(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Egraz_bk(Var, Par):
    return Par.p_graz * Var.D_NPP_bk


## wildfire emissions (under bookkeeping)
OSCAR_landC_bk.process(
    Out = 'D_Efire_bk', 
    In = ('f_fire', 'D_Cveg_bk'), 
    Eq = lambda Var, Par: Eq__D_Efire_bk(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Efire_bk(Var, Par):
    return Par.v_fire * Var.f_fire * Var.D_Cveg_bk


## total mortality flux (under bookkeeping)
## note: assumes regrowth time dominated by woody biomass
OSCAR_landC_bk.process(
    Out = 'D_Fmort_bk', 
    In = ('f_mort', 'D_Cveg_bk'), 
    Eq = lambda Var, Par: Eq__D_Fmort_bk(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Fmort_bk(Var, Par):
    return Par.f_mort_regr * Par.v_mort * Var.f_mort * Var.D_Cveg_bk


## litterfall flux (under bookkeeping)
OSCAR_landC_bk.process(
    Out = 'D_Ffall_bk', 
    In = ('D_Fmort_bk',), 
    Eq = lambda Var, Par: Eq__D_Ffall_bk(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Ffall_bk(Var, Par):
    return (1 - Par.p2_npp_wood) * Var.D_Fmort_bk


## coarse woody debris decay flux (under bookkeeping)
OSCAR_landC_bk.process(
    Out = 'D_Fcwd_bk', 
    In = ('f_cwd', 'D_Ccwd_bk'), 
    Eq = lambda Var, Par: Eq__D_Fcwd_bk(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Fcwd_bk(Var, Par):
    return Par.v_cwd * Var.f_cwd * Var.D_Ccwd_bk


## coarse woody debris emissions (under bookkeeping)
OSCAR_landC_bk.process(
    Out = 'D_Ecwd_bk', 
    In = ('D_Fcwd_bk',), 
    Eq = lambda Var, Par: Eq__D_Ecwd_bk(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Ecwd_bk(Var, Par):
    return Par.p_cwd_resp * Var.D_Fcwd_bk


## soil respiration (under bookkeeping)
OSCAR_landC_bk.process(
    Out = 'D_Esoil_bk', 
    In = ('f_resp', 'D_Csoil_bk'), 
    Eq = lambda Var, Par: Eq__D_Esoil_bk(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Esoil_bk(Var, Par):
    return Par.v_resp * Var.f_resp * Var.D_Csoil_bk


## harvested wood product decay
OSCAR_landC_bk.process(
    Out = 'D_Ehwp_bk', 
    In = ('D_Chwp_bk',), 
    Eq = lambda Var, Par: Eq__D_Ehwp_bk(Var, Par), 
    units='PgC yr-1')

def Eq__D_Ehwp_bk(Var, Par):
    return Par.v_hwp * Var.D_Chwp_bk


## net biome productivity (under bookkeeping)
OSCAR_landC_bk.process(
    Out = 'D_NBP_bk', 
    In = ('D_Fbk_veg2atm', 'D_NPP_bk', 'D_Echarv_bk', 'D_Egraz_bk', 'D_Efire_bk', 'D_Ecwd_bk', 'D_Esoil_bk', 'D_Ehwp_bk'), 
    Eq = lambda Var, Par: Eq__D_NBP_bk(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_NBP_bk(Var, Par):
    return -Var.D_Fbk_veg2atm + Var.D_NPP_bk - Var.D_Echarv_bk - Var.D_Egraz_bk - Var.D_Efire_bk - Var.D_Ecwd_bk - Var.D_Esoil_bk - Var.D_Ehwp_bk.sum('box_hwp', min_count=1)


##=====================
## Prognostic variables
##=====================

## mean age in bookkeeping
OSCAR_landC_bk.process(
    Out = 'D_age_bk', 
    In = ('D_age_bk', 'D_Aland_bk', 'D_dA_lcc1', 'D_dA_lcc2', 'D_dA_wharv1', 'D_dA_wharv2', 'D_dA_shift'),
    DiffEq = lambda Var, Par: DiffEq__age_bk(Var, Par), 
    units = 'yr', 
    core_dims = ['reg_land', 'bio_land'])

def DiffEq__age_bk(Var, Par):
    ## land cover change
    d_Abk_lcc1 = Var.D_dA_lcc1.sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    d_Abk_lcc2 = Var.D_dA_lcc2.sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## wood harvest
    d_Abk_wharv1 = Var.D_dA_wharv1
    d_Abk_wharv2 = (Par.dA_wharv_pi + Var.D_dA_wharv2)
    ## shifting cultivation
    #d_Abk_shift = (Par.dA_shift_pi + Var.D_dA_shift).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    ## all
    return 1 - (Par.age_bk_pi + Var.D_age_bk) * (d_Abk_lcc1 + d_Abk_lcc2 + d_Abk_wharv1 + d_Abk_wharv2) / (Par.Aland_bk_pi + Var.D_Aland_bk)


## land area under bookkeeping
OSCAR_landC_bk.process(
    Out = 'D_Aland_bk', 
    In = ('D_Aland_bk', 'D_dA_lcc1', 'D_dA_lcc2', 'D_dA_wharv1'),
    DiffEq = lambda Var, Par: DiffEq__D_Aland_bk(Var, Par), 
    units = 'Mha', 
    core_dims = ['reg_land', 'bio_land'])

def DiffEq__D_Aland_bk(Var, Par):
    ## land cover change
    d_Abk_lcc1 = Var.D_dA_lcc1.sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    d_Abk_lcc2 = Var.D_dA_lcc2.sum('bio_from', min_count=1).rename({'bio_to':'bio_land'}) - Var.D_dA_lcc2.sum('bio_to', min_count=1).rename({'bio_from':'bio_land'})
    ## wood harvest
    d_Abk_wharv1 = Var.D_dA_wharv1
    ## all
    return d_Abk_lcc1 + d_Abk_lcc2 + d_Abk_wharv1


## vegetation carbon stock (under bookkeeping)
OSCAR_landC_bk.process(
    Out = 'D_Cveg_bk', 
    In = ('D_Cveg_bk', 'D_Fbk_veg2veg', 'D_NPP_bk', 'D_Echarv_bk', 'D_Egraz_bk', 'D_Efire_bk', 'D_Fmort_bk'), 
    DiffEq = lambda Var, Par: DiffEq__D_Cveg_bk(Var, Par), 
    vLin = lambda Par: vLin__D_Cveg_bk(Par), 
    units = 'PgC', 
    core_dims = ['reg_land', 'bio_land'])

def DiffEq__D_Cveg_bk(Var, Par):
    return Var.D_Fbk_veg2veg + Var.D_NPP_bk - Var.D_Echarv_bk - Var.D_Egraz_bk - Var.D_Efire_bk - Var.D_Fmort_bk

def vLin__D_Cveg_bk(Par):
    return Par.v_mort + Par.v_fire


## coarse woody debris carbon stock (under bookkeeping)
OSCAR_landC_bk.process(
    Out = 'D_Ccwd_bk', 
    In = ('D_Ccwd_bk', 'D_Fbk_cwd2cwd', 'D_Fbk_veg2cwd', 'D_Fmort_bk', 'D_Ffall_bk', 'D_Fcwd_bk'), 
    DiffEq = lambda Var, Par: DiffEq__D_Ccwd_bk(Var, Par), 
    vLin = lambda Par: vLin__D_Ccwd_bk(Par), 
    units = 'PgC', 
    core_dims = ['reg_land', 'bio_land'])
    
def DiffEq__D_Ccwd_bk(Var, Par):
    return Var.D_Fbk_cwd2cwd + Var.D_Fbk_veg2cwd + (Var.D_Fmort_bk - Var.D_Ffall_bk) - Var.D_Fcwd_bk

def vLin__D_Ccwd_bk(Par):
    return Par.v_cwd


## soil carbon stock (under bookkeeping)
OSCAR_landC_bk.process(
    Out = 'D_Csoil_bk', 
    In = ('D_Csoil_bk', 'D_Fbk_soil2soil', 'D_Fbk_veg2litter', 'D_Fbk_veg2soil', 'D_Ffall_bk', 'D_Fcwd_bk', 'D_Ecwd_bk', 'D_Esoil_bk'), 
    DiffEq = lambda Var, Par: DiffEq__D_Csoil_bk(Var, Par), 
    vLin = lambda Par: vLin__D_Csoil_bk(Par), 
    units = 'PgC', 
    core_dims = ['reg_land', 'bio_land'])

def DiffEq__D_Csoil_bk(Var, Par):
    return Var.D_Fbk_soil2soil + Var.D_Fbk_veg2litter + Var.D_Fbk_veg2soil + Var.D_Ffall_bk + (Var.D_Fcwd_bk - Var.D_Ecwd_bk) - Var.D_Esoil_bk

def vLin__D_Csoil_bk(Par):
    return Par.v_resp


## harvested wood products stock
OSCAR_landC_bk.process(
    Out = 'D_Chwp_bk', 
    In = ('D_Chwp_bk', 'D_Fbk_veg2hwp', 'D_Ehwp_bk'), 
    DiffEq = lambda Var, Par: DiffEq__D_Chwp_bk(Var, Par), 
    vLin = lambda Par: vLin__D_Chwp_bk(Par), 
    units = 'PgC', 
    core_dims = ['reg_land', 'bio_land', 'box_hwp'])

def DiffEq__D_Chwp_bk(Var, Par):
    return Var.D_Fbk_veg2hwp - Var.D_Ehwp_bk

def vLin__D_Chwp_bk(Par):
    return Par.v_hwp

