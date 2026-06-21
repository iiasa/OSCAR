import warnings
import numpy as np
import xarray as xr
import scipy.stats as st

from scipy.integrate import quad
from scipy.optimize import fsolve


##################################################
##   1. ANCILLARY FUNCTIONS
##################################################

## function to get lognorm distrib parameters
def lognorm_distrib_param(mean, std):
    mu = np.log(mean / np.sqrt(1. + std**2. / mean**2.))
    sigma = np.sqrt(np.log(1. + std**2. / mean**2.))
    return mu, sigma


## function to infer logitnorm distrib parameters
def logitnorm_distrib_param(mean, std):
    ## error function
    def err(par):
        exp, _ = quad(lambda x, mu, sigma: 1/(1.-x) * 1./np.sqrt(2*np.pi*sigma**2.) * np.exp(-0.5*(np.log(x/(1.-x))-mu)**2./sigma**2.), 0, 1, args=tuple(par), limit=100)
        var, _ = quad(lambda x, mu, sigma: x/(1.-x) * 1./np.sqrt(2*np.pi*sigma**2.) * np.exp(-0.5*(np.log(x/(1.-x))-mu)**2./sigma**2.), 0, 1, args=tuple(par), limit=100)
        return np.array([exp-mean, np.sqrt(var-exp**2)-std])**2
    ## minimize error function
    try:
        par, _, fsolve_flag, _ = fsolve(err, [np.log(mean/(1.-mean)), np.sqrt(std/mean)], full_output=True)
        mu, sigma = par[0], np.abs(par[1])
    except ZeroDivisionError:
        fsolve_flag = 0
    ## return
    if fsolve_flag == 1: return mu, sigma
    else: return np.nan, np.nan


##################################################
## 2. GENERATE MONTE CARLO PARAMETERS
##################################################

## generate all Monte Carlo configurations 
def generate_config(Par0, nMC, kde_to_mod=False, dir_to_mod=False, mod_to_unc=False, mod_noise=0.1, kde_bw=None, seed=None):
    '''
    Function to generate Monte Carlo configuration (= parameters) for OSCAR.
    
    Input:
    ------
    Par0 (xr.Dataset)       dataset containing initial parameters
    nMC (int)               number of MC elements
    
    Output:
    -------
    Par_mc (xr.Dataset)     dataset containing MC parameters

    Options:
    --------
    kde_to_mod (bool)       turn all kde_ options to mod_ options;
                            default = False
    dir_to_mod (bool)       turn all dir_ options to mod_ options;
                            default = False
    mod_to_unc (bool)       turn all mod_ options to unc_ options;
                            default = False
    mod_noise (float)       equivalent s.d. of relative noise added on top of mod_ options;
                            default = 0.1
    kde_bw                  bandwith option for kde_ options forwarded to scipy.stats.gaussian_kde;
                            default = None
    seed (int)              seed for random number generation forwarded to numpy.random.default_rng;
                            default = None
    '''

    print('generating MC configurations')

    ## copy as precaution
    Par = Par0.copy(deep=True)

    ## list mod_ and kde_ dimensions
    mod_list = [coord for coord in Par.coords if coord[:4] == 'mod_']
    kde_list = [coord for coord in Par.coords if coord[:4] == 'kde_']
    dir_list = [coord for coord in Par.coords if coord[:4] == 'dir_']

    ## list uncertainty options, parameters and check no mixing
    par_unc_list, par_mod_list, par_kde_list, par_dir_list = [], [], [], []
    for par in Par:
        is_unc = any(['unc_' in dim for dim in Par[par].dims])
        is_mod = any(['mod_' in dim for dim in Par[par].dims])
        is_kde = any(['kde_' in dim for dim in Par[par].dims])
        is_dir = any(['dir_' in dim for dim in Par[par].dims])
        if is_unc + is_mod + is_kde + is_dir > 1:
            raise RuntimeError("Cannot mix unc_, mod_, kde_ and/or dir_ approaches; change parameter '{}'".format(par))     
        elif is_unc: par_unc_list.append(par)
        elif is_mod: par_mod_list.append(par)
        elif is_kde: par_kde_list.append(par)   
        elif is_dir: par_dir_list.append(par)   

    ## turn kde_ into mod_ (if requested)
    if kde_to_mod:
        Par = Par.rename({kde_: kde_.replace('kde_', 'mod_', 1) for kde_ in kde_list})
        mod_list, kde_list = mod_list + kde_list, []
        par_mod_list, par_kde_list = par_mod_list + par_kde_list, []

    ## turn dir_ into mod_ (if requested)
    if dir_to_mod:
        Par = Par.rename({dir_: dir_.replace('dir_', 'mod_', 1) for dir_ in dir_list})
        mod_list, dir_list = mod_list + dir_list, []
        par_mod_list, par_dir_list = par_mod_list + par_dir_list, []

    ## check kde has only one dim
    for par in par_kde_list: 
        if len(Par[par].dims) > 1:
            raise RuntimeError("Cannot have kde_ parameters with extra dims; change parameter '{}'".format(par))  

    ## turn mod_ to unc_ (if requested)
    ## assumes functional form based on provided values
    ## TODO: adapt to multi dims
    if mod_to_unc:
        for par in par_mod_list:
            if (Par[par] == Par[par].mean()).all():
                Par[par] = xr.DataArray(Par[par].mean(), attrs=Par[par].attrs)
                par_mod_list.remove(par)
            elif (Par[par] == Par[par]**2).all(): # switch
                Par[par] = xr.DataArray([0, 1], coords=['mini', 'maxi'], dims='unc_Choice', attrs=Par[par].attrs)
            elif (Par[par] >= 0).all() and (Par[par] <= 1).all():
                Par[par] = xr.DataArray([Par[par].mean(), Par[par].std()], coords=['mean', 'std'], dims='unc_LogitNorm', attrs=Par[par].attrs)
            elif (Par[par] >= 0).all() or (Par[par] <= 0).all():
                Par[par] = xr.DataArray([Par[par].mean(), Par[par].std()], coords=['mean', 'std'], dims='unc_LogNorm', attrs=Par[par].attrs)
            else:
                Par[par] = xr.DataArray([Par[par].mean(), Par[par].std()], coords=['mean', 'std'], dims='unc_Norm', attrs=Par[par].attrs)
        mod_list = []
        par_unc_list, par_mod_list = par_unc_list + par_mod_list, []

    ## list of fixed parameters
    par_fix_list = [par for par in Par if par not in par_unc_list + par_mod_list + par_kde_list + par_dir_list]

    ## initialize MC dataset
    Par_mc = xr.Dataset()
    Par_mc.coords['config'] = np.arange(nMC)

    ## set random state
    n_rng = np.random.default_rng(seed)


    ## STEP 1
    ## draw unc_ configurations
    for par in par_unc_list:

        ## get distrib and extra dims
        distrib = [dim.split('unc_')[-1] for dim in Par[par].dims if 'unc_' in dim]
        if len(distrib) == 1:
            distrib = distrib[0]
            dims_shape = Par[par].isel({'unc_'+distrib: 0}).shape
            dims_names = Par[par].isel({'unc_'+distrib: 0}).dims
        else:
            raise RuntimeError("Parameter '{}' has none or multiple unc_ dims: {}".format(par, Par[par].dims))

        ## Normal distrib
        if distrib == 'Norm':
            mean = Par[par].sel(unc_Norm='mean', drop=True)
            std = Par[par].sel(unc_Norm='std', drop=True)
            mu, sigma = mean, np.abs(std)
            Norm = st.norm.rvs(size=(nMC, *dims_shape), random_state=n_rng)
            Norm = xr.DataArray(Norm, coords={**{'config': Par_mc.config}, **{dim: Par[dim].values for dim in dims_names}})
            Par_mc[par] = mu + sigma * Norm

        ## LogNormal distrib
        elif distrib == 'LogNorm':
            mean = Par[par].sel(unc_LogNorm='mean', drop=True)
            std = Par[par].sel(unc_LogNorm='std', drop=True)
            mu, sigma = lognorm_distrib_param(np.abs(mean), np.abs(std))
            Norm = st.norm.rvs(size=(nMC, *dims_shape), random_state=n_rng)
            Norm = xr.DataArray(Norm, coords={**{'config': Par_mc.config}, **{dim: Par[dim].values for dim in dims_names}})
            Par_mc[par] = np.sign(mean) * np.exp(mu + sigma * Norm)

        ## LogitNormal distrib
        elif distrib == 'LogitNorm':
            mean = Par[par].sel(unc_LogitNorm='mean', drop=True)
            std = Par[par].sel(unc_LogitNorm='std', drop=True)
            mu, sigma = xr.apply_ufunc(logitnorm_distrib_param, np.abs(mean), np.abs(std),
                input_core_dims=[[], []], output_core_dims=[[], []], vectorize=True)
            if np.isnan([mu, sigma]).any(): raise RuntimeError('Could not infer LogitNorm distribution for parameter {}'.format(par)) 
            Norm = st.norm.rvs(size=(nMC, *dims_shape), random_state=n_rng)
            Norm = xr.DataArray(Norm, coords={**{'config': Par_mc.config}, **{dim: Par[dim].values for dim in dims_names}})
            Par_mc[par] = (1 + np.exp(mu + sigma * Norm) ** -1) ** -1

        ## two HalfNormal distribs
        elif distrib == '2HalfNorm':
            mean = Par[par].sel(unc_2HalfNorm='mean', drop=True)
            std_neg = Par[par].sel(unc_2HalfNorm='std_neg', drop=True)
            std_pos = Par[par].sel(unc_2HalfNorm='std_pos', drop=True)
            mu, sigma_neg, sigma_pos = mean, np.abs(std_neg), np.abs(std_pos)
            Bool = st.randint(0, 2).rvs(size=(nMC, *dims_shape), random_state=n_rng)
            Bool = xr.DataArray(Bool, coords={**{'config': Par_mc.config}, **{dim: Par[dim].values for dim in dims_names}})
            HalfNorm = st.halfnorm.rvs(size=(nMC, *dims_shape), random_state=n_rng)
            HalfNorm = xr.DataArray(HalfNorm, coords={**{'config': Par_mc.config}, **{dim: Par[dim].values for dim in dims_names}})
            Par_mc[par] = mu + (Bool * sigma_pos - (1 - Bool) * sigma_neg) * HalfNorm

        ## Uniform distrib
        elif distrib == 'Uniform':
            mini = Par[par].sel(unc_Uniform='mini', drop=True)
            maxi = Par[par].sel(unc_Uniform='maxi', drop=True)
            Uniform = st.uniform.rvs(size=(nMC, *dims_shape), random_state=n_rng)
            Uniform = xr.DataArray(Uniform, coords={**{'config': Par_mc.config}, **{dim: Par[dim].values for dim in dims_names}})
            Par_mc[par] = mini + (maxi - mini) * Uniform

        ## Triangle distrib
        elif distrib == 'Triangle':
            mode = Par[par].sel(unc_Triangle='mode', drop=True)
            mini = Par[par].sel(unc_Triangle='mini', drop=True)
            maxi = Par[par].sel(unc_Triangle='maxi', drop=True)
            Triang = st.triang(c=(mode-mini)/(maxi-mini)).rvs(size=(nMC, *dims_shape), random_state=n_rng)
            Triang = xr.DataArray(Triang, coords={**{'config': Par_mc.config}, **{dim: Par[dim].values for dim in dims_names}})
            Par_mc[par] = mini + (maxi - mini) * Triang

        ## Discrete Uniform distrib
        elif distrib == 'Choice':
            mini = Par[par].sel(unc_Choice='mini', drop=True)
            maxi = Par[par].sel(unc_Choice='maxi', drop=True)
            mini, maxi = np.minimum(mini, maxi), np.maximum(mini, maxi)
            Choice = st.randint(mini, maxi + 1).rvs(size=(nMC, *dims_shape), random_state=n_rng)
            Choice = xr.DataArray(Choice, coords={**{'config': Par_mc.config}, **{dim: Par[dim].values for dim in dims_names}})
            Par_mc[par] = Choice

        ## error otherwise
        else:
            raise RuntimeError("Distribution {} not implemented for parameter '{}'".format(distrib, par))


    ## STEP 2
    ## draw mod_ configurations
    ## discrete draw of each mod
    Mod = xr.Dataset()
    for mod in mod_list:
        Mod[mod] = xr.DataArray(st.randint(0, len(Par[mod])).rvs(size=nMC, random_state=n_rng), coords={'config': Par_mc.config})
    
    ## applying selection (this keeps mod_ as secondary coordinate)
    Par_mod = xr.merge([Par[par] for par in par_mod_list], join='outer', compat='no_conflicts')
    Par_mc = xr.merge([Par_mc, Par_mod.isel({mod: Mod[mod] for mod in mod_list})], join='outer', compat='no_conflicts')
    
    ## adding noise based on Von Mises (if requested)
    if mod_noise > 0:
        for mod in mod_list: del Par_mc[mod]
        for par in [par for par in par_mod_list]:
            noise_par = float(Par[par].attrs['mod_noise_override']) if 'mod_noise_override' in Par[par].attrs else mod_noise
            if noise_par > 0:
                Noise = st.vonmises_line(kappa=1/noise_par**2).rvs(size=Par_mc[par].shape, random_state=n_rng)
                Noise = xr.DataArray(Noise, coords=Par_mc[par].coords) / np.pi
                Par_mc[par] *= 1 + Noise


    ## STEP 3
    ## draw kde_ configurations
    for kde_ in kde_list:
        par_list = [par for par in par_kde_list if kde_ in Par[par].dims]
        kde_draw = st.gaussian_kde(np.array([Par[par].values for par in par_list]), bw_method=kde_bw).resample(nMC)
        for n, par in enumerate(par_list):
            Par_mc[par] = ('config', kde_draw[n, :])


    ## STEP 4
    ## draw dir_ configurations
    for dir_ in dir_list:
        par_list = [par for par in par_dir_list if dir_ in Par[par].dims]

        ## get extra dimensions
        dims_names = sum([Par[par] for par in par_list]).mean(dir_).dims
        dims_shape = sum([Par[par] for par in par_list]).mean(dir_).shape

        ## method of moments
        means = [Par[par].mean(dir_).values for par in par_list] + [(1-sum([Par[par] for par in par_list])).mean(dir_).values]
        variances = [Par[par].var(dir_).values for par in par_list] + [(1-sum([Par[par] for par in par_list])).var(dir_).values]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            a0 = sum([mean * (1 - mean) / variance - 1 for mean, variance in zip(means, variances)]) / (len(par_list) + 1)
        ai = np.array([mean * a0 for mean in means])

        ## draw from Dirichlet distribution
        flat_alpha = np.moveaxis(ai, 0, -1).reshape(-1, ai.shape[0])
        flat_mean = np.moveaxis(np.array(means), 0, -1).reshape(-1, ai.shape[0])
        dir_draw = [st.dirichlet(alpha).rvs(size=nMC, random_state=n_rng) if np.isfinite(alpha).all() and (alpha > 0).all() else np.tile(mean, (nMC, 1)) for alpha, mean in zip(flat_alpha, flat_mean)]
        dir_draw = np.array(dir_draw).reshape(*dims_shape, nMC, (len(par_list) + 1))
        for n, par in enumerate(par_list):
            Par_mc[par] = ((*dims_names, 'config'), dir_draw[..., n])


    ## add parameters without uncertainty
    for par in par_fix_list:
        Par_mc[par] = Par[par]

    ## copy attributes
    for par in Par:
        Par_mc[par].attrs = Par[par].attrs

    ## return
    return Par_mc.astype(np.float32)

