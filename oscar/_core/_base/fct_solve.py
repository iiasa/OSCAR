import numpy as np
import xarray as xr


##################################################
##   1. SAFETY FUNCTIONS
##################################################

## bounded exponential (soft)
## note: caps at f_max for x >> 1, with nf the sharpness of transition
def safe_exp(x, f_max=5., nf=5.):
    return np.exp(x - 1/nf * np.log1p(np.exp(nf * (x - np.log(f_max)))) + 1/nf * np.log1p(np.exp(-nf * np.log(f_max))))


## moving cap as a function of flux rate
## note: close to 1/v**nl for v<<1, anchors at 2 for v=1, quickly goes to 1 for v>>1
def f_max(v, nl=0.5, nh=2.):
    return 1 + 1 / (v**nl + v**nh)


## bounded ratio (soft)
## note: caps at 0 for x << 1, with nf the sharpness of transition
## note: must add safety terms for x < 1 and x >= 1 to avoid NaNs
def safe_ratio_(x, nf=500.):
    #return np.log1p(np.exp(nf * x)) / np.log1p(np.exp(nf))
    return np.nan_to_num(np.log1p(np.exp(nf * np.maximum(x, -1))) / np.log1p(np.exp(nf))) * (x < 1) + x * (x >= 1)


## bounded ratio (hard)
def safe_ratio(x, x_lim=1E-99):
    return x * (x >= x_lim) + x_lim * (x < x_lim)



##################################################
##   2. NUMERICAL SCHEMES
##################################################

## explicit Eulerian scheme
def scheme_ex(dX_dt, dt, v):
    return dt * dX_dt


## implicit-explicit scheme
## if v is constant, only the linear part is implicit
## if v is the loss rate, Patankar trick
def scheme_imex(dX_dt, dt, v):
    return  dt * dX_dt / (1 + v * dt)


##################################################
##   3. HEURISTIC FOR TIME STEPPING
##################################################

## heuristic function to estimate number of substeps
def adapt_nt(D_CO2, CO2_nt=300., nt_min=2, nt_max=12):
    return int(min(max(2 + D_CO2 // CO2_nt, nt_min), nt_max))
adapt_nt.var = 'D_CO2'


##################################################
##   4. REGIONAL SUMMING
##################################################

## convenience function for flexible summing of regions
def sum_reg(da, reg_axis='reg_land'):
    return da.sum(reg_axis, min_count=1) if reg_axis in da.dims else da

