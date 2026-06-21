import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import sum_reg


#####################################################################
##   TROPOSPHERIC CHEMISTRY
#####################################################################

## initialize
OSCAR_tropo = Model('OSCAR_tropo')

## module based on:
## (Holmes et al., 2013; https://doi.org/10.5194/acp-13-285-2013)
## adapted to drivers of CMIP6/AerChemMIP:
## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-853-2021)
## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-1105-2021)


##=====================
## Secondary parameters
##=====================

## present-day tropospheric hydroxyl sink intensity
## note: unknowable present-day BB emissions taken as preindustrial
OSCAR_tropo.process(
    Out = 'f_kOH_pd', 
    Eq = lambda Par: Eq__f_kOH_pd(Par), 
    units = '1')

def Eq__f_kOH_pd(Par):
    if 'EESC_pd' not in Par: return None
    if 'EESC_pi' not in Par: return None
    if 'Ebb_NOx_pi' not in Par: return None
    if 'Ebb_CO_pi' not in Par: return None
    if 'Ebb_VOC_pi' not in Par: return None
    if 'Enat_BVOC_pi' not in Par: return None
    if 'Enat_LNOx_pd' not in Par: return None
    ## environmental factors
    f_kOH_CH4 = Par.ch_OH_CH4 * np.log(Par.CH4_pd / Par.CH4_pi)
    f_kOH_N2O = Par.ch_OH_N2O * np.log(Par.N2O_pd / Par.N2O_pi)
    f_kOH_EESC = Par.ch_OH_EESC * np.log(Par.EESC_pd.sel(age_air='mid_lat', drop=True) / Par.EESC_pi.sel(age_air='mid_lat', drop=True))
    f_kOH_H2 = Par.ch_OH_H2 * np.log(Par.H2_pd / Par.H2_pi)
    f_kOH_Tg = Par.ch_OH_Tg * np.log1p(Par.D_Tg_pd / Par.Tg_pi)
    ## natural emissions
    f_kOH_BVOC = Par.ch_OH_BVOC * np.log(sum_reg(Par.Enat_BVOC_pd) / sum_reg(Par.Enat_BVOC_pi))
    f_kOH_LNOx = Par.ch_OH_LNOx * np.log(Par.Enat_LNOx_pd / Par.Enat_LNOx_pi)
    ## anthropogenic emissions
    f_kOH_NOx = Par.ch_OH_NOx * np.log((sum_reg(Par.Eant_NOx_pd) + sum_reg(Par.Ebb_NOx_pd)) / (sum_reg(Par.Eant_NOx_pi) + sum_reg(Par.Ebb_NOx_pi)))
    f_kOH_CO = Par.ch_OH_CO * np.log((sum_reg(Par.Eant_CO_pd) + sum_reg(Par.Ebb_CO_pd)) / (sum_reg(Par.Eant_CO_pi) + sum_reg(Par.Ebb_CO_pi)))
    f_kOH_VOC = Par.ch_OH_VOC * np.log((sum_reg(Par.Eant_VOC_pd) + sum_reg(Par.Ebb_VOC_pd)) / (sum_reg(Par.Eant_VOC_pi) + sum_reg(Par.Ebb_VOC_pi)))
    ## return
    return np.exp(f_kOH_CH4 + f_kOH_N2O + f_kOH_EESC + f_kOH_H2 + f_kOH_Tg + f_kOH_BVOC + f_kOH_LNOx + f_kOH_NOx + f_kOH_CO + f_kOH_VOC)


##=====================
## Diagnostic variables
##=====================

## tropospheric hydroxyl sink intensity
OSCAR_tropo.process(
    Out = 'f_kOH', 
    In = ('D_CH4', 'D_N2O', 'D_EESC', 'D_H2', 'D_Tg', 'D_Enat_BVOC', 'D_Enat_LNOx', 'D_Eant_NOx', 'D_Ebb_NOx', 'D_Eant_CO', 'D_Ebb_CO', 'D_Eant_VOC', 'D_Ebb_VOC'), 
    Eq = lambda Var, Par: Eq__f_kOH(Var, Par), 
    units = '1')

def Eq__f_kOH(Var, Par):
    ## environmental factors
    f_kOH_CH4 = Par.ch_OH_CH4 * np.log1p(Var.D_CH4 / Par.CH4_pi)
    f_kOH_N2O = Par.ch_OH_N2O * np.log1p(Var.D_N2O / Par.N2O_pi)
    f_kOH_EESC = Par.ch_OH_EESC * np.log1p(Var.D_EESC.sel(age_air='mid_lat', drop=True) / Par.EESC_pi.sel(age_air='mid_lat', drop=True))
    f_kOH_H2 = Par.ch_OH_H2 * np.log1p(Var.D_H2 / Par.H2_pi)
    f_kOH_Tg = Par.ch_OH_Tg * np.log1p(Var.D_Tg / Par.Tg_pi)
    ## natural emissions
    f_kOH_BVOC = Par.ch_OH_BVOC * np.log1p(sum_reg(Var.D_Enat_BVOC) / sum_reg(Par.Enat_BVOC_pi))
    f_kOH_LNOx = Par.ch_OH_LNOx * np.log1p(Var.D_Enat_LNOx / Par.Enat_LNOx_pi)
    ## anthropogenic emissions
    f_kOH_NOx = Par.ch_OH_NOx * np.log1p((sum_reg(Var.D_Eant_NOx) + sum_reg(Var.D_Ebb_NOx)) / (sum_reg(Par.Eant_NOx_pi) + sum_reg(Par.Ebb_NOx_pi)))
    f_kOH_CO = Par.ch_OH_CO * np.log1p((sum_reg(Var.D_Eant_CO) + sum_reg(Var.D_Ebb_CO)) / (sum_reg(Par.Eant_CO_pi) + sum_reg(Par.Ebb_CO_pi)))
    f_kOH_VOC = Par.ch_OH_VOC * np.log1p((sum_reg(Var.D_Eant_VOC) + sum_reg(Var.D_Ebb_VOC)) / (sum_reg(Par.Eant_VOC_pi) + sum_reg(Par.Ebb_VOC_pi)))
    ## return
    return np.exp(f_kOH_CH4 + f_kOH_N2O + f_kOH_EESC + f_kOH_H2 + f_kOH_Tg + f_kOH_BVOC + f_kOH_LNOx + f_kOH_NOx + f_kOH_CO + f_kOH_VOC)

