####################################
# Constrain function for OSCAR v3
####################################

# import libraries
import os
import yaml
import numpy as np
import xarray as xr
from scipy.stats import norm, lognorm, skewnorm, qmc
from scipy.optimize import minimize

import matplotlib.pyplot as plt


#region constrain
#region define functions
#region 1. load and parse constraints
def load_and_parse_constraints(yaml_path):
    '''
    Load constraint specifications from a YAML file and evaluate mathematical expressions

    Input:
    ------
    yaml_path (str)             path to the constraints YAML file

    Output:
    -------
    parsed_specs (list)         list of dicts with evaluated mean and std values
    '''
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    parsed_specs = []
    for spec in config['constraints']:
        # Evaluate math expressions if provided as strings
        if 'mean_eqn' in spec:
            spec['mean'] = eval(str(spec['mean_eqn']))
            
        if 'std_eqn' in spec:
            spec['std'] = eval(str(spec['std_eqn']))
            
        parsed_specs.append(spec)
    return parsed_specs
#endregion

#region 2. format var
def format_var(ds_var, var_specs):
    '''
    format variable according to variable specifications from a dataset

    Input:
    ------
    ds (xr.Dataset)             dataset containing the variables
    var_specs (dict)            variable specifications (from YAML)

    Output:
    -------
    var_new (xr.DataArray)      formatted variable values
    '''

    name = var_specs['name']

    # 1. Get time-frames from the specs (now expected as lists/tuples from YAML)
    baseline = var_specs.get('base', None)
    period = var_specs.get('period', None)
    
    print(f'Formatting {name}: period={period}, baseline={baseline}')

    # 2. Calculate baseline mean
    # We use index [0] and [1] instead of * to prevent argument count errors
    if baseline:
        var_baseline = ds_var.sel(year=slice(baseline[0], baseline[1])).mean(dim='year')
    else:
        var_baseline = 0
    
    # 3. Calculate target period mean
    if period:
        var_target = ds_var.sel(year=slice(period[0], period[1])).mean(dim='year')
    else:
        # If no period is specified, use the full available timeseries
        var_target = ds_var
    
    # 4. Calculate the anomaly/delta
    var_new = var_target - var_baseline
    
    # Ensure the DataArray carries the name defined in the constraint specs
    var_new.name = name
    
    return var_new
#endregion

#region 3. mahalanobis distance
def mahalanobis_distance(x, mean, inv_cov):
    '''
    Compute the Mahalanobis distance of each row in x from the mean

    Input:
    ------
    x (np.ndarray)          array of shape (n_samples, n_features)
    mean (np.ndarray)       mean vector of shape (n_features,)
    inv_cov (np.ndarray)    inverse covariance matrix of shape (n_features, n_features)

    Output:
    -------
    (np.ndarray)            array of shape (n_samples,) with Mahalanobis distances
    '''
    diff = x - mean
    return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
#endregion

#region 4. apply constraints and select configurations
def dist_spec(spec, values, method='cdf'):
    '''
    Evaluate distribution specified by spec at given values
    Input:
    ------
    spec (dict)         distribution specification
    values (np.ndarray) values at which to evaluate

    Output:
    -------
    vals (np.ndarray)   evaluated values

    Options:
    --------
    method (str)       method to use: 'cdf', 'pdf', or 'ppf'
                       default = 'cdf'
    '''
    
    if spec['type'] == 'normal':
        # normal distribution
        if method == 'cdf': vals = norm.cdf(values, loc=spec['mean'], scale=spec['std'])
        if method == 'pdf': vals = norm.pdf(values, loc=spec['mean'], scale=spec['std'])
        if method == 'ppf': vals = norm.ppf(values, loc=spec['mean'], scale=spec['std'])
        
    elif spec['type'] == 'lognormal':
        # lognormal distribution - convert mean/std to lognormal parameters
        mean, std = spec['mean'], spec['std']
        mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
        sigma = np.sqrt(np.log(1 + (std/mean)**2))
        if method == 'cdf': vals = lognorm.cdf(values, s=sigma, scale=np.exp(mu))
        if method == 'pdf': vals = lognorm.pdf(values, s=sigma, scale=np.exp(mu))
        if method == 'ppf': vals = lognorm.ppf(values, s=sigma, scale=np.exp(mu))
    
    elif spec['type'] == 'percentile':
        # percentile-based distribution (e.g., 5th, median, 95th)
        p5, median, p95 = spec['p5'], spec['median'], spec['p95']
        
        if abs((median - p5) - (p95 - median)) < 1e-10:
            # symmetric percentiles - use normal approximation
            mean = median
            std = (p95 - p5) / (norm.ppf(0.95) - norm.ppf(0.05))
            if method == 'cdf': vals = norm.cdf(values, loc=mean, scale=std)
            if method == 'pdf': vals = norm.pdf(values, loc=mean, scale=std)
            if method == 'ppf': vals = norm.ppf(values, loc=mean, scale=std)
        else:
            # non-symmetric - use properly fitted skew-normal distribution
            loc, scale, shape = fit_skewnorm_from_percentiles(p5, median, p95)
            if method == 'cdf': vals = skewnorm.cdf(values, shape, loc=loc, scale=scale)
            if method == 'pdf': vals = skewnorm.pdf(values, shape, loc=loc, scale=scale)
            if method == 'ppf': vals = skewnorm.ppf(values, shape, loc=loc, scale=scale)

    elif spec['type'] == 'skewnormal':
        # direct skew-normal parameters
        loc, scale, shape = spec['loc'], spec['scale'], spec['shape']
        if method == 'cdf': vals = skewnorm.cdf(values, shape, loc=loc, scale=scale)
        if method == 'pdf': vals = skewnorm.pdf(values, shape, loc=loc, scale=scale)
        if method == 'ppf': vals = skewnorm.ppf(values, shape, loc=loc, scale=scale)

    return vals
#endregion

#region 5. fit skew-normal distribution to percentiles
def fit_skewnorm_from_percentiles(p5, median, p95, max_iter=100):
    '''
    Properly fit skew-normal distribution to percentiles using optimization

    Input:
    ------
    p5 (float)          5th percentile
    median (float)      50th percentile (median)
    p95 (float)         95th percentile

    Output:
    -------
    loc (float)         location parameter of fitted skew-normal
    scale (float)       scale parameter of fitted skew-normal
    shape (float)       shape parameter of fitted skew-normal

    Options:
    --------
    max_iter (int)      maximum iterations for optimization
                        default = 100
    '''
    def objective(params):
        loc, scale, shape = params
        try:
            dist = skewnorm(shape, loc=loc, scale=scale)
            p5_est = dist.ppf(0.05)
            median_est = dist.ppf(0.5)
            p95_est = dist.ppf(0.95)
            
            # weight errors by importance (focus on matching percentiles)
            error = (abs(p5_est - p5) + 
                    2 * abs(median_est - median) +  # Emphasize median
                    abs(p95_est - p95))
            return error
        except:
            return np.inf
    
    # better initial guesses
    initial_guesses = [
        [median, (p95 - p5)/3.0, 0.0],      # near-normal
        [median, (p95 - p5)/2.5, 2.0],      # positive skew
        [median, (p95 - p5)/2.5, -2.0],     # negative skew
        [(p5 + median + p95)/3, (p95 - p5)/2.0, 1.0],  # balanced
    ]
    
    best_params = None
    best_error = np.inf
    
    for init_guess in initial_guesses:
        try:
            result = minimize(objective, init_guess, 
                            bounds=[(None, None), (1e-6, None), (None, None)],
                            method='L-BFGS-B',
                            options={'maxiter': max_iter})
            
            if result.success and result.fun < best_error:
                best_error = result.fun
                best_params = result.x
        except:
            continue
    
    if best_params is None:
        # fallback: use empirical CDF
        print(f'Warning: Skew-normal fit failed for percentiles p5={p5}, median={median}, p95={p95}')
        print('Falling back to empirical distribution')
        return None
    
    loc, scale, shape = best_params
    
    # verify the fit
    dist = skewnorm(shape, loc=loc, scale=scale)
    p5_fit = dist.ppf(0.05)
    median_fit = dist.ppf(0.5)
    p95_fit = dist.ppf(0.95)
    
    print(f'Skew-normal fit: p5={p5_fit:.3f} (target {p5:.3f}), '
          f'median={median_fit:.3f} (target {median:.3f}), '
          f'p95={p95_fit:.3f} (target {p95:.3f})')
    
    return loc, scale, shape
#endregion

#region 6. main function to apply constraints and select configurations
def LHS_configs(simulated_results, constraint_specs, N_post, frac_ma=1, use_scipy=True):
    '''
    Select configurations using Latin Hypercube sampling in constraint space
    Returns the selected CONFIG INDICES
    
    Input:
    ------
    simulated_results (np.ndarray)          simulated results for each configuration
    constraint_specs (list of dicts)        specification for temperature and CO2 distributions
    N_post (int)                            number of configurations to select
    
    Output:
    --------
    selected_indices (np.ndarray of int)    config indices that were selected

    Options:
    --------
    frac_ma (float)                         fraction of Mahalanobis distance in combined distance metric
                                            default = 1 (only Mahalanobis distance)
    use_scipy (bool)                        whether to use scipy's LHS sampler
                                            default = True
    '''
    
    n_constraints = simulated_results.shape[1]
    
    # create LHS space in constraint space
    if use_scipy:
        sampler = qmc.LatinHypercube(d=n_constraints)
        lhs_design = sampler.random(n=N_post)
        print('Using scipy\'s LatinHypercube sampler')
    else:
        # Generate optimal Latin Hypercube design
        lhs_design = np.zeros((N_post, n_constraints), dtype=float)
        for j in range(n_constraints):
            ## ? whether to use fixed or random position within each bin
            lhs_design[:, j] = (np.random.permutation(N_post) + np.random.uniform(0.1, 0.9)) / N_post
        print('Using custom LatinHypercube sampler')

    uniform_space = np.zeros((N_post, n_constraints), dtype=float)
    
    for j, spec in enumerate(constraint_specs):
        values = lhs_design[:, j]
        uniform_space[:, j] = dist_spec(spec, values, method='ppf')

    # calculate covariance for Mahalanobis distance
    cov_matrix = np.cov(uniform_space.T)
    try:
        inv_cov_ma = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov_ma = np.eye(n_constraints)

    inv_cov_eu = np.eye(n_constraints)

    # for each LHS point drawn from the observational space, find the closest prior sample
    selected_indices = []
    used_indices = set()
    
    for lhs_point in uniform_space:
        available_mask = ~np.isin(np.arange(len(simulated_results)), list(used_indices))
        available_indices = np.where(available_mask)[0]
        
        if len(available_indices) == 0: break
            
        available_points = simulated_results[available_indices]
        
        # calculate both distances
        dist_mahalanobis = mahalanobis_distance(available_points, lhs_point, inv_cov_ma)
        dist_euclidean = mahalanobis_distance(available_points, lhs_point, inv_cov_eu)

        # combine distances
        combined_distances = frac_ma * dist_mahalanobis + (1 - frac_ma) * dist_euclidean
        
        best_idx = available_indices[np.argmin(combined_distances)]
        selected_indices.append(best_idx)
        used_indices.add(best_idx)
    
    return selected_indices
#endregion
#region 7. constraining pipeline
def run_constraining_pipeline(ds, specs, n_post, vars_to_constrain):
    """
    A generic pipeline: 
    1. Filters specs based on vars_to_constrain
    2. Formats variables based on specs
    3. Runs LHS
    4. Returns selected indices

    Input:
    ------
    ds (xr.Dataset)             dataset containing the variables
    specs (list of dicts)       list of constraint specifications
    n_post (int)                number of configurations to select
    vars_to_constrain (list)    list of variable names to constrain (must match 'name' in specs)
    
    Output:
    -------
    indices (np.ndarray)        selected configuration indices
    valid_specs (list)          specs that were successfully used
    sim_results (np.ndarray)    the formatted data matrix used for selection
    """
    formatted_list = []
    valid_specs = []
    
    for spec in specs:
        name = spec['name']
        
        # 1. Filter: only proceed if this spec name is in target list
        if name not in vars_to_constrain:
            continue
            
        try:    
            # 1. Handle variable selection and basic math (e.g., "D_Fland - D_Eluc")
            if " - " in name:
                parts = name.split(" - ")
                # Subtract the DataArrays directly from the dataset
                ds_var = ds[parts[0]] - ds[parts[1]]
            else:
                ds_var = ds[name]

            # 2. Format the time-series into a scalar value (e.g. mean of 2014-2023)
            v_eff = format_var(ds_var, spec)
            
            formatted_list.append(v_eff.values)
            valid_specs.append(spec)
            
        except KeyError:
            print(f"Skipping {name}: Variable not found in dataset.")
        except Exception as e:
            print(f"Skipping {name}: Error during formatting: {e}")

    if not formatted_list:
        raise ValueError("No variables from vars_to_constrain were found or successfully formatted.")

    # 3. Stack results into (n_samples, n_constraints) matrix
    sim_results = np.column_stack(formatted_list)
    
    # 4. Run Latin Hypercube Selection
    print(f"Running LHS selection for {n_post} samples using {len(valid_specs)} constraints...")
    indices = LHS_configs(sim_results, valid_specs, N_post=n_post)
    
    return indices, valid_specs, sim_results
#endregion

#region validation & visualization
def analyze_selection(selected_indices, valid_specs, sim_results):
    print('\n=== LATIN HYPERCUBE SELECTION RESULTS ===')
    print(f'Selected {len(selected_indices)} configurations')
    print(f'Selected config indices: {selected_indices[:10]}...')  # Show first 10

    for j, spec in enumerate(valid_specs):
        selected_vals = sim_results[:, j][selected_indices]
        all_vals = sim_results[:, j]
        
        print(f'\n--- {spec['name']} ({spec['type']}) ---')
        print(f'Observed constraint: {spec}')
        print(f'All configs:    mean={all_vals.mean():.2f}, std={all_vals.std():.2f}')
        print(f'Selected configs: mean={selected_vals.mean():.2f}, std={selected_vals.std():.2f}')
        print(f'Selected range: [{selected_vals.min():.2f}, {selected_vals.max():.2f}]')
        
        if spec['type'] == 'lognormal':
            # log-space analysis
            log_selected = np.log(selected_vals)
            log_all = np.log(all_vals)
            print(f'Log-space - All: μ={log_all.mean():.3f}, σ={log_all.std():.3f}')
            print(f'Log-space - Selected: μ={log_selected.mean():.3f}, σ={log_selected.std():.3f}')
        if spec['type'] == 'percentile':
            # percentile analysis
            p5, median, p95 = np.percentile(selected_vals, [5, 50, 95])
            print(f'Percentiles of selected: 5th={p5:.2f}, 50th={median:.2f}, 95th={p95:.2f}')

def plot_selection_results(selected_indices, valid_specs, sim_results):

    if len(sim_results[1]) > 2:
        fig, axes = plt.subplots(2, (len(sim_results[1]) + 1) // 2, figsize=((len(sim_results[1]) + 1) // 2 * 4, 8))
    else:
        fig, axes = plt.subplots(1, len(sim_results[1]), figsize=(len(sim_results[1]) * 4, 4))

    for i in range(len(sim_results[1])):
        if len(sim_results[1]) != 1:
            ax = axes[i % 2, i // 2] if len(sim_results[1]) > 2 else axes[i]
        else:
            ax = axes
        var = sim_results[:, i]
        selected_var = sim_results[:, i][selected_indices]

        ax.hist(var, bins=30, alpha=0.5, density=True, label='All', color='gray')
        ax.hist(selected_var, bins=20, alpha=0.5, density=True, label='Sel', color='blue')
        x = np.linspace(var.min(), var.max(), 100)
        ax.plot(x, dist_spec(valid_specs[i], x, method='pdf'), color='red', linestyle='--', lw=2, label='Obs')
        ax.set_ylabel('Density')
        ax.set_xlabel(f'{valid_specs[i]['units']}')
        ax.set_title(f'{valid_specs[i]['name']}')
        ax.legend()
    
    plt.tight_layout()
    return fig, axes
#endregion
