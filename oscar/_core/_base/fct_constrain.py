import os
import ast
import yaml
import numpy as np
import xarray as xr
import operator as op
import matplotlib.pyplot as plt

from scipy.stats import norm, lognorm, skewnorm, qmc, spearmanr
from scipy.optimize import differential_evolution, least_squares


##################################################
##   1. READ CONSTRAINTS AS YAML
##################################################

## load constraints from yaml file
def load_constraints_yaml(path):
    """
    Example structure
    ------------------
    constraints:
      - variable: D_Tg              # must be an OSCAR variable
        base: [1850, 1900]
        period: [2014, 2023]
        mean: 1.19
        std: 0.10
      - var: D_CO2
        distrib: lognormal
        period: [2023, 2023]
        pcts: {5: 130, 50: 141, 95: 152}
      - variable: D_Tg               # same variable, another period -> needs a 'period_name'
        period_name: 1990s           # else auto-named 'period1', 'period2', ...
        base: [1850, 1900]
        period: [1990, 1999]
        mean: 0.55
        std_eqn: 0.10/1.19*0.55
    """
    
    ## load file as list
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    constraints = [_format_constraint(c) for c in config['constraints']]

    ## sort out names
    ## default naming scheme
    def _key(c):
        return f"{c['name']}_{c['period_name']}" if c.get('period_name') else c['name']

    ## count repeats
    counts = {}
    for c in constraints:
        counts[_key(c)] = counts.get(_key(c), 0) + 1

    ## make final dict
    constraints_final, i_per_var = {}, {}
    for c in constraints:
        if counts[_key(c)] > 1 and 'period_name' not in c:
            i_per_var[c['variable']] = i_per_var.get(c['variable'], 0) + 1
            c['period_name'] = f"period{i_per_var[c['variable']]}"
        name = _key(c)
        if name in constraints_final:
            raise ValueError(f"Duplicate constraint '{name}': give distinct 'period_name' values")
        constraints_final[name] = c
    
    ## return
    return constraints_final


## save a (possibly hand-crafted) set of constraints to a yaml file
## note: doesn't quite work in all cases / needs improvement
'''
def save_constraints_yaml(constraints, path):
    if isinstance(constraints, dict):
        constraints = list(constraints.values())
    with open(path, 'w') as f:
        yaml.safe_dump({'constraints': constraints}, f, sort_keys=False)
'''


## format constraint input, allowing for variations in writing
def _format_constraint(constraint):
    
    ## initialize
    c = dict(constraint)

    ## accept "name", "var" or "variable" as synonyms for the default "name"
    name = c.pop("name", None) or c.get("var") or c.get("variable")
    c.pop("var", None)
    c.pop("variable", None)
    c["name"] = name

    ## mean and std (can be equations)
    if 'mean' in c:
        c['mean'] = _safe_eval_numeric(c['mean'])
    if 'std' in c:
        c['std'] = _safe_eval_numeric(c['std'])

    ## separate percentiles (can be equations)
    flat_pcts = {}
    for key in list(c.keys()):
        if key == 'median':
            flat_pcts[50.] = _safe_eval_numeric(c.pop('median'))
        elif key[:1] == 'p' and key[1:].replace('.', '', 1).isdigit():
            flat_pcts[float(key[1:])] = _safe_eval_numeric(c.pop(key))
    if flat_pcts and 'pcts' not in c:
        c['pcts'] = flat_pcts
    elif 'pcts' in c:
        c['pcts'] = {float(k): _safe_eval_numeric(v) for k, v in c['pcts'].items()}

    ## return
    return c


## safe operators for safe eval
_SAFE_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
            ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos}


## function to safely evaluate provided expression
def _safe_eval_numeric(expr):
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression in constraint YAML: {expr!r}")
    return _eval(ast.parse(str(expr), mode='eval'))


##################################################
##   2. BUILD DISTRIBUTIONS
##################################################

## small wrapper to get multiple distribs
def build_distribs(constraints, only=None):
    distribs = {}
    for name, c in [(name, c) for name, c in constraints.items() if only is None or name in only]:
        distribs[name] = build_distrib(mean=c.get('mean'), std=c.get('std'), pcts=c.get('pcts'), distrib=c.get('distrib'))
    return distribs


## make one frozen scipy distrib
def build_distrib(mean=None, std=None, pcts=None, distrib=None):

    ## if percentiles provided
    if pcts is not None:
        pcts = {float(k): float(v) for k, v in pcts.items()}
        if len(pcts) not in (3, 5):
            raise ValueError('percentiles must have exactly 3 or 5 points')
        if 50 not in pcts:
            raise ValueError('percentiles must include the 50th percentile (median)')

        ## take distrib if provived otherwise test symmetry
        distrib = distrib or ('normal' if _pcts_are_symmetric(pcts) else 'skewnormal')

        ## fit distribution
        if distrib == 'normal':
            loc, scale = _fit_normal_to_pcts(pcts)
            return norm(loc=loc, scale=scale)
        elif distrib == 'lognormal':
            log_pcts = {k: np.log(v) for k, v in pcts.items()}
            mu, sigma = _fit_normal_to_pcts(log_pcts)
            return lognorm(s=sigma, scale=np.exp(mu))
        elif distrib == 'skewnormal':
            return _fit_skewnormal_to_pcts(pcts)
        else:
            raise ValueError(f"Unsupported distrib '{distrib}' for percentiles input")

    ## if mean & std provided
    elif mean is not None and std is not None:
        distrib = distrib or 'normal'
        if distrib == 'normal':
            return norm(loc=mean, scale=std)
        elif distrib == 'lognormal':
            mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
            sigma = np.sqrt(np.log(1 + (std / mean)**2))
            return lognorm(s=sigma, scale=np.exp(mu))
        else:
            raise ValueError(f"Unsupported dist '{distrib}' for mean & std input")

    else:
        raise ValueError('Provide either mean & std, or percentiles')


## function to check if symmetric
def _pcts_are_symmetric(pcts, tol=0.05):
    median = pcts[50]
    spread = max(pcts.values()) - min(pcts.values())
    lows = sorted(p for p in pcts if p < 50)
    highs = sorted((p for p in pcts if p > 50), reverse=True)
    if len(lows) != len(highs):
        return False
    for lo, hi in zip(lows, highs):
        if abs((100 - hi) - lo) > 1e-6:  # not a matched pair, e.g. 5 & 95
            return False
        if abs((median - pcts[lo]) - (pcts[hi] - median)) > tol * spread:
            return False
    return True


## function to fit normal distrib to pcts
def _fit_normal_to_pcts(pcts):
    ## 
    q = np.array(sorted(pcts.keys())) / 100.0
    v = np.array([pcts[100 * qi] for qi in q])
    z = norm.ppf(q)
    A = np.column_stack([np.ones_like(z), z])
    (loc, scale), *_ = np.linalg.lstsq(A, v, rcond=None)
    return loc, max(scale, 1e-9)


## function to fit skewed normal distrib to pcts
def _fit_skewnormal_to_pcts(pcts, max_nfev=300, warn_tol=0.02, bounds=([-5, 1e-6, -30], [5, 5, 30])):

    ## read pcts
    q = np.array(sorted(pcts.keys())) / 100.0
    v_raw = np.array([pcts[100 * qi] for qi in q])
    
    ## normalize
    median = pcts[50]
    spread = v_raw.max() - v_raw.min()
    if spread <= 0:
        raise ValueError('percentile values must be strictly increasing')
    v = (v_raw - median) / spread

    ## define distance
    def residuals(params):
        loc, scale, shape = params
        if scale <= 0:
            return np.full_like(v, 1e6)
        return skewnorm.ppf(q, shape, loc=loc, scale=scale) - v

    ## try solving with least_squares and different shape parameters
    best_params, best_cost = None, np.inf
    for shape0 in (-8, -4, -2, -1, 0, 1, 2, 4, 8):
        try:
            result = least_squares(residuals, [0.0, 1.0, shape0], bounds=bounds, max_nfev=max_nfev)
        except Exception:
            continue
        cost = np.sum(result.fun**2)
        if cost < best_cost:
            best_cost, best_params = cost, result.x

    ## bounded global search if large residuals
    if best_params is None or best_cost > warn_tol**2 * len(v):
        result = differential_evolution(
            lambda p: np.sum(residuals(p)**2),
            bounds=list(zip(*bounds)), seed=0, maxiter=max_nfev, polish=True,
        )
        if result.fun < best_cost:
            best_cost, best_params = result.fun, result.x

    ## raise if failed fit
    if best_params is None:
        raise RuntimeError(f'skew-normal fit failed for pcts {pcts}; change constraint!')

    ## de-normalize
    loc_n, scale_n, shape = best_params
    loc, scale = loc_n * spread + median, scale_n * spread

    ## check fit error
    fit_error = np.sqrt(best_cost / len(v)) * spread
    if fit_error > warn_tol * spread:
        print(f"Warning: skew-normal fit residual is {fit_error:.3g} "
              f"({100 * fit_error / spread:.1f}% of percentile spread) for {pcts}")

    ## return
    return skewnorm(shape, loc=loc, scale=scale)


##################################################
##   3. APPLY CONSTRAINING
##################################################

##==========
## Utilities
##==========

## convert constraints to xr.ds.attrs
def _constraint_to_attrs(c):
    attrs = {}
    for k, v in c.items():
        if v is None:
            continue
        if k == 'pcts':
            attrs['pcts_q'] = np.array(list(v.keys()), dtype=float)
            attrs['pcts_v'] = np.array(list(v.values()), dtype=float)
        elif isinstance(v, bool):
            attrs[k] = int(v)
        elif isinstance(v, (list, tuple)):
            attrs[k] = np.asarray(v)
        else:
            attrs[k] = v
    return attrs


## convert xr.ds.attrs to constraints
def _constraint_from_attrs(attrs):
    c = {}
    for k, v in attrs.items():
        if k in ('pcts_q', 'pcts_v'):
            continue
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, np.generic):
            v = v.item()
        c[k] = bool(v) if k in ['skip'] else v
    if 'pcts_q' in attrs:
        c['pcts'] = dict(zip(np.atleast_1d(attrs['pcts_q']).tolist(), np.atleast_1d(attrs['pcts_v']).tolist()))
    return c


## get constraints from targets
def _constraints_from_targets(targets):
    return {name: _constraint_from_attrs(targets[name].attrs) for name in targets.data_vars}


## get constraints that were actually used for the draws (skipped ones are kept as NaN)
def _target_vars(targets):
    return [v for v in targets.data_vars if not targets[v].attrs.get('skip', 0)]


## resolve where constraints are provided
def _resolve_constraints(constraints, targets):
    if isinstance(constraints, (str, os.PathLike)):
        return load_constraints_yaml(constraints)
    if constraints is None:
        if targets is None:
            raise ValueError("Provide 'constraints', or a 'targets' dataset that carries them")
        return _constraints_from_targets(targets)
    return constraints


## get variable value over specified period and w.r.t. specified baseline
## note: period=None (or missing) means time-independent variable, taken as is
def _get_var_eval(da, base=None, period=None, time_axis='year',
    sum_dims=['reg_land', 'bio_land', 'bio_from', 'bio_to']):
    for dim in [dim for dim in da.dims if dim in sum_dims]: da = da.sum(dim, min_count=1)
    target = da.sel({time_axis: slice(*period)}).mean(time_axis) if period else da
    baseline = da.sel({time_axis: slice(*base)}).mean(time_axis) if base else 0
    return target - baseline


## Mahalanobis distance
def _mahalanobis_dist(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))


## Spearman rank-correlation matrix
def _spearman_corr(X):
    d = X.shape[1]
    if d == 1: return np.array([[1.]])
    corr, _ = spearmanr(X)
    if d == 2: corr = np.array([[1., corr], [corr, 1.]])
    return np.atleast_2d(corr)


## Iman-Conover method
## note: reorders columns so the joint sample approximately matches the target correlation
## note: does not alter any column, so marginals are preserved
def _iman_conover(sample, target_corr, seed=None):
    n, d = sample.shape
    rng = np.random.default_rng(seed)
    ref = rng.standard_normal((n, d)) @ np.linalg.cholesky(target_corr).T
    out = np.empty_like(sample)
    for j in range(d):
        rank = np.argsort(np.argsort(ref[:, j]))
        out[:, j] = np.sort(sample[:, j])[rank]
    return out


##==========
## Main algo
##==========

## main function to apply constraining and select ensemble members
def apply_constraining(Out_prior, constraints=None, n_select=None, distribs=None, 
    targets=None, return_targets=False, 
    config_axis='config', time_axis='year', draw_axis='draw', 
    frac_mahalanobis=1.0, corr_shrinkage=1.0, seed=None):
    print(26*'=')
    print('---', 'APPLY CONSTRAINING', '---')
    print(26*'=')

    ## get constraints (from arg, yaml path or targets dataset)
    constraints = _resolve_constraints(constraints, targets)

    ## list constraints
    if targets is None:
        skip_vars = [name for name, c in constraints.items() if c.get('skip', False)]
        used_vars = [name for name in constraints if name not in skip_vars]
        ## draw one seed if not provided
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])  # 32-bit, so it is netcdf-safe
        print('***', f'random seed: {seed}', '***')
        if n_select is None:
            raise ValueError("Provide 'n_select' (or a 'targets' dataset)")
    else:
        used_vars = _target_vars(targets)
        skip_vars = [name for name in constraints if name not in used_vars]
        n_select = targets.sizes[draw_axis]
        seed = targets.attrs.get('seed')
        missing = [v for v in used_vars if v not in constraints]
        if missing:
            raise ValueError(f'targets contain variables not in constraints: {missing}')
    n_constraints = len(used_vars)

    ## build distributions
    print('***', 'building distributions', '***')
    if distribs is None:
        distribs = build_distribs(constraints, only=used_vars)
    else:
        distribs = {name: distrib for name, distrib in distribs.items() if name in used_vars}

    ## get model results and format
    print('***', 'getting model outputs', '***')
    Out_eval = xr.Dataset({name: _get_var_eval(Out_prior[c['name']], base=c.get('base'), period=c.get('period'), time_axis=time_axis) for name, c in constraints.items()})
    X = Out_eval[used_vars].to_array(dim='_constraint').transpose(config_axis, '_constraint').values

    ## check and track members with any NaN (failed computations)
    ## will be excluded from pool of available configs
    valid = ~np.isnan(X).any(axis=1)
    Out_eval = Out_eval.assign_coords(valid=(config_axis, valid))

    ## draw targets if not provided
    if targets is None:

        ## LHS
        print('***', 'Latin Hypercube sampling', '***')
        sampler = qmc.LatinHypercube(d=n_constraints, seed=seed)
        lhs_design = sampler.random(n=n_select)
        real_space = np.column_stack([distribs[c].ppf(lhs_design[:, n]) for n, c in enumerate(used_vars)])

        ## use Iman-Conover method to reoder LHS
        if n_constraints > 1:
            print('***', 'Iman-Conover trick', '***')
            if 0. <= corr_shrinkage <= 1.:
                target_corr = _spearman_corr(X[valid])
                target_corr = corr_shrinkage * target_corr + (1 - corr_shrinkage) * np.eye(n_constraints)
            real_space = _iman_conover(real_space, target_corr, seed=seed)
            achieved_corr = _spearman_corr(real_space)
            ## keep diagnostics
            Out_eval.attrs['corr_vars'] = used_vars
            Out_eval.attrs['target_corr'] = target_corr
            Out_eval.attrs['achieved_corr'] = achieved_corr

        ## package the drawn targets
        ## note: order matters because configs are matched sequentially
        n_draw = real_space.shape[0]
        draws = {c: real_space[:, n] for n, c in enumerate(used_vars)}
        targets = xr.Dataset(
            {name: (draw_axis, draws.get(name, np.full(n_draw, np.nan))) for name in constraints},
            coords={draw_axis: np.arange(n_draw)})
        for name, c in constraints.items():
            targets[name].attrs = _constraint_to_attrs(c)
        targets.attrs.update(seed=seed, n_select=int(n_select), 
            frac_mahalanobis=float(frac_mahalanobis), corr_shrinkage=float(corr_shrinkage))
    
    ## otherwise use provided targets
    else:
        print('***', 'using provided targets (no sampling)', '***')
        real_space = targets[used_vars].to_array(dim='_constraint').transpose(draw_axis, '_constraint').values
    
    ## save seed
    if seed is not None: Out_eval.attrs['seed'] = seed

    ## covariations
    cov = np.atleast_2d(np.cov(real_space.T))
    inv_cov_ma = np.linalg.inv(cov)
    inv_cov_eu = np.eye(n_constraints)

    ## select configs
    print('***', 'selecting configurations', '***')
    selected, used, match_dist = [], [], []
    for real in real_space:
        available = np.array([i for i in range(X.shape[0]) if valid[i] and i not in used])
        if available.size == 0:
            break
        points = X[available]
        dist_ma = _mahalanobis_dist(points, real, inv_cov_ma)
        dist_eu = _mahalanobis_dist(points, real, inv_cov_eu)
        dist_combined = frac_mahalanobis * dist_ma + (1 - frac_mahalanobis) * dist_eu
        best_config = available[np.argmin(dist_combined)]
        selected.append(best_config)
        used.append(best_config)
        match_dist.append(dist_combined.min())

    ## track draw/config pairing as coordinates of targets
    ## note: unmatched draws, if the pool ran out, get config -1 and NaN distance
    n_draw = real_space.shape[0]
    n_unmatched = n_draw - len(selected)
    if n_unmatched:
        print(f'Warning: pool exhausted, {n_unmatched} draws left unmatched')
    matched_configs = Out_eval[config_axis].isel({config_axis: selected}).values
    matched_configs = np.concatenate([matched_configs, np.full(n_unmatched, -1, dtype=matched_configs.dtype)])
    matched_dist = np.concatenate([match_dist, np.full(n_unmatched, np.nan)])
    targets = targets.drop_vars([v for v in (f'selected_{config_axis}', 'match_dist') if v in targets.coords])
    targets = targets.assign_coords({f'selected_{config_axis}': (draw_axis, matched_configs), 'match_dist': (draw_axis, matched_dist)})

    ## return
    selected = np.sort(Out_eval[config_axis].isel({config_axis: selected}).values)
    if return_targets: return selected, Out_eval, targets
    return selected, Out_eval


##################################################
##   4. DIAGNOSTICS & PLOTTING
##################################################

## model values of the matched configs as xr.dataset
def get_matched(Out_eval, targets, config_axis='config', draw_axis='draw'):
    sel = targets[f'selected_{config_axis}']
    sel = sel.where(sel >= 0, drop=True)  # drop unmatched draws
    names = _target_vars(targets)
    model = Out_eval[names].sel({config_axis: sel})
    out = model.drop_vars('valid', errors='ignore')
    for v in names:
        out[f'{v}_target'] = targets[v].sel(draw=sel[draw_axis])
    return out.assign_coords(match_dist=targets['match_dist'].sel(draw=sel[draw_axis]))


## quick summary print
def print_constraining(Out_eval, constraints, selected, distribs=None, config_axis='config'):
    
    ## get distribs
    if distribs is None:
        distribs = build_distribs(constraints)
    
    ## formating function
    def _fmt(x):
        if not np.isfinite(x): return f'{x}'
        n_int_digits = len(str(int(abs(x)))) if abs(x) >= 1 else 0
        if n_int_digits >= 4: return f'{x:.0f}'
        elif n_int_digits == 3: return f'{x:.0f}.'
        elif n_int_digits == 2: return f'{x:.1f}'
        else: return f'{x:.2f}'

    ## print
    for name, distrib in distribs.items():
        prior, post = Out_eval[name].values, Out_eval[name].sel({config_axis: selected}).values
        units = constraints[name].get('units', '')
        print(f"\n--- {name}{f' ({units})' if units else ''} ---")
        print(f"target:      mean={_fmt(distrib.mean())}, std={_fmt(distrib.std())}" + "  (SKIPPED)" * constraints[name].get('skip', False))
        print(f"prior:       mean={_fmt(np.nanmean(prior))}, std={_fmt(np.nanstd(prior))}  (n={int(Out_eval['valid'].sum())})")
        print(f"posterior:   mean={_fmt(post.mean())}, std={_fmt(post.std())}  (n={post.size})")


## control plot
def plot_constraining(Out_eval, constraints, selected, distribs=None, config_axis='config', plot_ncols=4):

    ## get distribs
    if distribs is None:
        distribs = build_distribs(constraints)

    ## names & skip status
    names = [c for c in distribs if c in Out_eval]
    skip_vars = {c for c in names if constraints[c].get('skip', False)}

    ## layout
    have_corr = 'target_corr' in Out_eval.attrs
    n_panels = len(names) + (1 if have_corr else 0)
    panel_size = (3, 2.5)
    nrows = int(np.ceil(n_panels / plot_ncols))
    fig, axes = plt.subplots(nrows, plot_ncols, squeeze=False, figsize=(panel_size[0] * plot_ncols, panel_size[1] * nrows))
    axes_flat = axes.flatten()

    ## loop over constraints
    for ax, name in zip(axes_flat, names):

        ## get info
        is_skipped = name in skip_vars
        prior = Out_eval[name].values[Out_eval['valid'].values] # take only valid configs
        post = Out_eval[name].sel({config_axis: selected}).values
        prior, post = prior[np.isfinite(prior)], post[np.isfinite(post)] # drop NaNs (e.g. failed fits)
        units = constraints[name].get('units', '')

        ## histograms
        ax.hist(prior, bins=30, density=True, alpha=0.4, color='gray', label=f'prior (n={prior.size})')
        if is_skipped:
            ax.hist(post, bins=20, density=True, histtype='step', color='steelblue', linewidth=1.5, alpha=0.6)
        else:
            ax.hist(post, bins=20, density=True, alpha=0.5, color='steelblue', label=f'posterior (n={post.size})')

        ## target distribution curve
        distrib = distribs[name]
        lo = min(prior.min(), post.min(), distrib.ppf(0.01))
        hi = max(prior.max(), post.max(), distrib.ppf(0.99))
        x = np.linspace(lo, hi, 200)
        ax.plot(x, distrib.pdf(x), color='crimson', ls='--', lw=2, label='target')

        ## restrict visible x-range
        #view_lo = min(distrib.mean() - 5 * distrib.std(), post.min())
        #view_hi = max(distrib.mean() + 5 * distrib.std(), post.max())
        #ax.set_xlim(view_lo, view_hi)

        ## highlight skipped constraints
        if is_skipped:
            ax.hist(post, bins=20, density=True, alpha=0.4, color='lightgray', label='posterior')
            ax.set_title(f"{name} ({units}) · skipped" if units else f'{name} · skipped', color='gray', fontsize='medium')
            for spine in ax.spines.values(): spine.set_edgecolor('lightgray')
            ax.tick_params(colors='gray')
        else:
            ax.set_title(f'{name} ({units})' if units else name, fontsize='medium')

    ## correlation diagnostic panel
    if have_corr:
        ax = axes_flat[len(names)]
        corr_vars = Out_eval.attrs['corr_vars']
        target_corr = np.array(Out_eval.attrs['target_corr'])
        achieved_corr = np.array(Out_eval.attrs['achieved_corr'])
        n = len(corr_vars)
        combined = np.triu(target_corr, 1) + np.tril(achieved_corr, -1) + np.eye(n)
        im = ax.imshow(combined, vmin=-1, vmax=1, cmap='coolwarm')
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax.text(j, i, f'{combined[i, j]:.2f}'.replace("-0.", "-.").replace("0.", "."), ha='center', va='center', fontsize=5)
        ax.set_xticks(range(n)); ax.set_xticklabels(corr_vars, rotation=70, fontsize='xx-small')
        ax.set_yticks(range(n)); ax.set_yticklabels(corr_vars, fontsize='xx-small')
        ax.set_title('target corr (upper) vs achieved corr (lower)', fontsize='small')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize='small')

    ## hide unused axes
    for ax in axes_flat[n_panels:]:
        ax.axis('off')

    ## shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    ## return
    return fig, axes


##################################################
##   5. WRAPPER
##################################################

## wrapper for constraining pipeline
def constraining_pipeline(Out_prior, constraints=None, n_select=None, 
    targets=None, return_targets=False, 
    config_axis='config', time_axis='year', 
    frac_mahalanobis=1.0, corr_shrinkage=1.0, seed=None, 
    plot_ncols=4, save_figure=None):

    ## get constraints (from arg, yaml path or targets dataset)
    constraints = _resolve_constraints(constraints, targets)

    ## build distribs once
    distribs = build_distribs(constraints)

    ## apply
    selected, Out_eval, targets = apply_constraining(Out_prior, constraints, n_select, distribs=distribs, 
        targets=targets, return_targets=True, 
        config_axis=config_axis, time_axis=time_axis, 
        frac_mahalanobis=frac_mahalanobis, corr_shrinkage=corr_shrinkage, seed=seed)

    ## print
    print_constraining(Out_eval, constraints, selected, distribs=distribs, 
        config_axis=config_axis)

    ## plot
    plot_constraining(Out_eval, constraints, selected, distribs=distribs, 
        config_axis=config_axis, plot_ncols=plot_ncols)
    if save_figure:
        plt.savefig(save_figure, dpi=200)

    ## return
    if return_targets: return selected, Out_eval, targets
    return selected, Out_eval

