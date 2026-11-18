"""Customized runs: a research project's inputs applied to a CMIP baseline scenario.

A project folder (<project-id>/) holds one settings file per experiment
(settings_<experiment-id>.yaml), the user's input files (inputs/), and per
experiment the compiled forcing (forcing/) and results (results/<experiment-id>_results.nc).

Inputs are continuous annual time series covering the projection period; ensuring
their coverage and continuity is the user's responsibility. Variables the user does
not supply come from the baseline scenario. OSCAR does not interpolate or rescale.
"""
from pathlib import Path
import json

import pandas as pd
import xarray as xr
import yaml

from oscar._io import handlers
from oscar._io._download import ensure_configured_library, open_library, file_digest, load_dataset
from oscar._io.paths import get_paths, validate_name
from oscar._utils.load_config import load_config
from oscar._utils.metadata import load_var_registry
from oscar._utils.results import combine_history, finish, check_overwrite
from oscar._workflows.configured_runs import resolve_selection

SETTINGS = {'scientific_setup', 'user_files', 'theme', 'custom_vars',
            'preprocess_forcing', 'plot_user_forcing', 'run_model', 'plot_outputs'}
SCIENTIFIC = {'hist_type', 'model_region', 'baseline_forcing', 'projection_end_year'}
PROJECTION_END = 2100  # default last projection year


# --- Project settings -------------------------------------------------------------

def project_path(project):
    path = Path(project).expanduser()
    if path.is_absolute():
        return path.resolve()
    validate_name(str(project), 'project name')
    return get_paths()['projects'] / str(project)


def input_path(project, name):
    path = (project / name).resolve()
    if project.resolve() not in path.parents:
        raise ValueError('Input files must be inside the project directory.')
    if not path.is_file():
        raise FileNotFoundError(f'Project input does not exist: {path}')
    return path


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a YAML mapping.')
    return value


def _input_files(settings):
    """(compiled NetCDF, atmospheric CSV settings, land-use CSV settings) of a project."""
    files = _mapping(settings.get('user_files') or {}, 'user_files')
    csv_inputs = _mapping(files.get('csv_inputs') or {}, 'csv_inputs')
    atmospheric = _mapping(csv_inputs.get('atmospheric') or {}, 'atmospheric')
    land_use = _mapping(csv_inputs.get('lulcc') or {}, 'lulcc')
    return files.get('compiled_nc'), atmospheric, land_use


def validate_project(project, experiment, check_inputs=True):
    """Check a project's settings (and, with check_inputs, read its input files) offline."""
    validate_name(experiment, 'experiment name')
    path = project_path(project)
    settings_file = path / f'settings_{experiment}.yaml'
    if not settings_file.exists():
        raise FileNotFoundError(f'Missing {settings_file}. Create the project with oscar.create_project().')
    u = _mapping(yaml.safe_load(settings_file.read_text()), 'Project settings')
    if 'output_identifier' in u:
        raise ValueError('output_identifier is no longer used: forcing and results are named after the '
                         'experiment id (settings_<experiment-id>.yaml). Remove it.')
    if set(u) - SETTINGS:
        raise ValueError(f'Unknown project settings: {sorted(set(u) - SETTINGS)}')
    scientific = _mapping(u.get('scientific_setup') or {}, 'scientific_setup')
    if 'connect_method' in scientific:
        raise ValueError('connect_method is no longer used: OSCAR takes your values as supplied. '
                         'Remove it and provide continuous annual time series.')
    if set(scientific) - SCIENTIFIC:
        raise ValueError(f'Unknown scientific settings: {sorted(set(scientific) - SCIENTIFIC)}')
    defaults = load_config()['configured_mode']['defaults']
    hist = scientific.get('hist_type') or defaults['hist_type']
    sel = resolve_selection(scientific.get('baseline_forcing'), scientific.get('model_region'), hist)
    end = scientific.get('projection_end_year')
    if end is not None and (isinstance(end, bool) or not isinstance(end, int) or not 1750 <= end <= 2100):
        raise ValueError('projection_end_year must be an integer between 1750 and 2100, or null for 2100.')
    for key in ('preprocess_forcing', 'plot_user_forcing', 'run_model', 'plot_outputs'):
        if key in u and not isinstance(u[key], bool):
            raise ValueError(f'{key} must be true or false.')
    themes = load_config()['customized_mode']['output_themes']
    theme = u.get('theme', 'climate')
    if theme not in themes:
        raise ValueError(f'Unknown theme {theme!r}. Available: {", ".join(themes)}')
    variables = u.get('custom_vars') if theme == 'custom' else themes[theme]
    if not isinstance(variables, list) or not variables or len(set(variables)) != len(variables):
        raise ValueError('Output variables must be a nonempty list of unique variable IDs.')
    for v in variables:
        if v not in load_var_registry():
            raise ValueError(f'Unknown output variable {v!r}.')
    compiled, atmospheric, land_use = _input_files(u)
    if compiled and (atmospheric.get('file') or land_use.get('file')):
        raise ValueError('Choose compiled_nc or CSV inputs, not both.')
    forcing = None
    if check_inputs:
        parts = []
        if compiled:
            parts.append(handlers.read_netcdf(input_path(path, compiled)))
        if atmospheric.get('file'):
            parts.append(handlers.read_atmospheric(input_path(path, atmospheric['file']),
                                                   atmospheric.get('format', 'oscar-csv'), sel.region))
        if land_use.get('file'):
            parts.append(handlers.read_land_use(input_path(path, land_use['file']), sel.region))
        if parts:
            forcing = xr.merge(parts, join='outer', combine_attrs='drop_conflicts')
    return {'project_path': str(path), 'settings_file': str(settings_file), 'experiment': experiment,
            'hist_type': sel.hist_type, 'region': sel.region,
            'baseline': sel.scenarios[0], 'projection_end_year': end, 'variables': variables,
            'custom_variables': sorted(forcing.data_vars) if forcing is not None else [],
            'settings': u, 'forcing': forcing}


# --- Forcing ----------------------------------------------------------------------

def _apply(base, user, history, first, last):
    """Baseline forcing (restart year onwards) with the user's variables replaced over first..last."""
    merged = base.copy()
    years = list(range(first, last + 1))
    for var in user.data_vars:
        if not handlers.supplied(user[var]):
            continue  # Not supplied for this scenario: baseline values.
        values = user[var].sel(year=years)
        if var not in base:
            # Not a library forcing (e.g. D_Eluc, computed in CMIP): the restart year comes from the history.
            anchor = history[var].sel(year=[first - 1])
            values = values.expand_dims(reg_land=anchor.reg_land).where(anchor.reg_land == anchor.reg_land[0], 0.)
            merged[var] = xr.concat([anchor, values.broadcast_like(anchor.isel(year=0, drop=True))], dim='year')
            merged[var].attrs['units'] = handlers.UNITS[var]
            continue
        target = base[var]
        if var == 'D_Eant_Xhalo':
            unknown = set(map(str, values.spc_halo.values)) - set(map(str, target.spc_halo.values))
            if unknown:
                raise ValueError(f'Unknown halogen species {sorted(unknown)}.')
            supplied = values.notnull().any('year')
            values = values.reindex(spc_halo=target.spc_halo)
            new = xr.where(supplied.reindex(spc_halo=target.spc_halo, fill_value=False),
                           values, target.sel(year=years))
        elif var in handlers.LAND_USE:
            new = values.reindex_like(target.sel(year=years), fill_value=0.).fillna(0.)
        elif 'reg_land' in target.dims:
            # Emission totals are placed in the first model region (ROWO).
            new = values.expand_dims(reg_land=target.reg_land).where(target.reg_land == target.reg_land[0], 0.)
        else:
            new = values
        projection = xr.where(target.year >= first, new.reindex(year=target.year), target)
        merged[var] = projection.transpose(*target.dims).assign_attrs(target.attrs)
    return merged


def _user_forcing(spec, lib, data, end):
    """Forcing of the user's scenarios (each applied to the baseline), after the checks."""
    user, h_end = handlers.copy_last_year(spec['forcing'], end), lib.hist_end
    reserved = {'Historical'} | {s + '-ref' for s in lib.scenarios}
    for name in map(str, user.scen.values):
        if not name.strip() or name in reserved:
            raise ValueError(f'Invalid or reserved scenario name: {name!r}')
    handlers.check_full_coverage(user, h_end + 1, end)
    history = lib.load('history', ['D_Eluc']) if 'D_Eluc' in user else None
    if history is not None:
        history = history.sel(config=data['params'].config)
    scenarios = data['forcing_scen']
    handlers.continuity_warnings(user, data['forcing_hist'], scenarios, history, h_end)
    base = scenarios.sel(scen=spec['baseline'], drop=True)
    runs = [_apply(base, user.sel(scen=name, drop=True), history, h_end + 1, end).expand_dims(scen=[str(name)])
            for name in user.scen.values]
    return xr.concat(runs, dim='scen', data_vars='different', coords='minimal', compat='equals', join='exact')


def _reference_forcing(spec, scenarios):
    """Library forcing of the baseline (baseline-only run) or of the high/low reference scenarios."""
    if spec['forcing'] is None:
        return scenarios.sel(scen=[spec['baseline']])
    markers = [s for s in load_config()['customized_mode']['marker_scenarios'][spec['hist_type']]
               if s in scenarios.scen.values]
    return scenarios.sel(scen=markers).assign_coords(scen=[s + '-ref' for s in markers]) if markers else None


def _concat_scenarios(datasets, **options):
    """Join batches along scen; library data carry extra coordinates (e.g. scen_group), so keep shared ones."""
    shared = set.intersection(*(set(ds.coords) for ds in datasets))
    parts = [ds.drop_vars([c for c in ds.coords if c not in shared and c not in ds.dims]) for ds in datasets]
    return xr.concat(parts, dim='scen', coords='minimal', **options)


# --- Run --------------------------------------------------------------------------

def run_customized(project=None, experiment=None, library_dir=None, output_dir=None,
                   save=True, plot=None, show_plot=False, overwrite=False,
                   run_model=None, preprocess_forcing=None, **solver_options):
    if project is None or experiment is None:
        raise ValueError('Customized mode needs project="<project-id>" and experiment="<experiment-id>".')
    spec = validate_project(project, experiment, check_inputs=False)
    u, p_path = spec['settings'], Path(spec['project_path'])
    run_model = u.get('run_model', True) if run_model is None else run_model
    preprocess = u.get('preprocess_forcing', True) if preprocess_forcing is None else preprocess_forcing
    plot = u.get('plot_outputs', False) if plot is None else plot
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else p_path / 'results'
    output_id = experiment  # forcing and results are named after the experiment id
    result_file = out_dir / f'{output_id}_results.nc'
    if not run_model:
        if not result_file.exists():
            raise FileNotFoundError(f'No saved results at {result_file}; run with run_model=True first.')
        out = load_dataset(result_file)
        return finish(out, 'customized', json.loads(out.attrs.get('run_settings', '{}')),
                      out.attrs['historical_end'], result_file, False, plot, show_plot,
                      plot_prefix=experiment, plot_dir=out_dir / 'plots')
    spec = validate_project(project, experiment)
    if save:
        check_overwrite(result_file, overwrite)
    lib = open_library(spec['hist_type'], spec['region'],
                       ensure_configured_library(spec['hist_type'], spec['region'], library_dir))
    if spec['baseline'] not in lib.scenarios:
        raise ValueError(f"Baseline {spec['baseline']} is not in this library: {', '.join(lib.scenarios)}")
    h_end = lib.hist_end
    # Projections run to 2100 by default; a library ending in 2099 has its last year copied to 2100.
    latest = PROJECTION_END if lib.end_year >= PROJECTION_END - 1 else lib.end_year
    end = spec['projection_end_year'] or latest
    if not h_end < end <= latest:
        raise ValueError(f'projection_end_year must be between {h_end + 1} and {latest}, or null for {latest}.')
    data = lib.prepare(variables=spec['variables'])
    data['forcing_scen'] = handlers.copy_last_year(data['forcing_scen'], end).sel(year=slice(h_end, end))
    forcing_file = p_path / 'forcing' / f'forcing_processed_{output_id}.nc'
    compiled, atmospheric, land_use = _input_files(u)
    names = [compiled, atmospheric.get('file'), land_use.get('file')]
    processing = {'history': spec['hist_type'], 'region': spec['region'], 'baseline': spec['baseline'],
                  'hist_end': h_end, 'end_year': end, 'library_dir': str(lib.path),
                  'data_version': lib.data_version,
                  'inputs': {name: file_digest(p_path / name) for name in names if name}}
    signature = json.dumps(processing, sort_keys=True)
    user_forcing = None
    if spec['forcing'] is not None:
        if preprocess:
            check_overwrite(forcing_file, overwrite)
            user_forcing = _user_forcing(spec, lib, data, end)
            forcing_file.parent.mkdir(parents=True, exist_ok=True)
            user_forcing.attrs['processing_settings'] = signature
            user_forcing.to_netcdf(forcing_file, engine='h5netcdf')
        else:
            if not forcing_file.exists():
                raise FileNotFoundError(f'No processed forcing at {forcing_file}; set preprocess_forcing: true.')
            user_forcing = load_dataset(forcing_file)
            if user_forcing.attrs.get('processing_settings') != signature:
                raise ValueError('Saved forcing was built from other inputs or settings; '
                                 'set preprocess_forcing: true to rebuild it.')
    reference_forcing = _reference_forcing(spec, data['forcing_scen'])
    if u.get('plot_user_forcing', False) and user_forcing is not None:
        _plot_inputs(user_forcing, reference_forcing, data['forcing_hist'], spec, forcing_file.parent, output_id)
    from oscar._core._model.OSCAR import OSCAR
    # The user's scenarios and the library scenarios run as separate batches, so that
    # inputs only the user supplies (such as D_Eluc) never leave gaps in the others.
    projections = [OSCAR(Ini=data['initial_state'], Par=data['params'], For=forcing,
                         var_keep=spec['variables'], **solver_options)
                   for forcing in (user_forcing, reference_forcing) if forcing is not None]
    projection = _concat_scenarios(projections) if len(projections) > 1 else projections[0]
    out = combine_history(data['history'], projection, h_end, spec['variables'])
    resolved = {k: v for k, v in spec.items() if k != 'forcing'}
    resolved.update(projection_end_year=end, library_dir=str(lib.path), solver_options=solver_options,
                    data_version=lib.data_version, settings_sha256=file_digest(spec['settings_file']),
                    inputs_sha256=processing['inputs'])
    # Plots: results/plots/<experiment-id>_<variable>.png.
    return finish(out, 'customized', resolved, h_end, result_file, save, plot, show_plot, overwrite,
                  plot_prefix=experiment, plot_dir=out_dir / 'plots')


def _plot_inputs(user_forcing, reference_forcing, forcing_hist, spec, folder, output_id):
    """Audit plots of the user's inputs next to the library forcing."""
    from oscar._viz import plot_forcing_audit
    audit_dir = folder / f'audit_plots_{output_id}'
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scen in user_forcing.scen.values:
        for var in spec['custom_variables']:
            species = spec['forcing'][var].spc_halo.values if 'spc_halo' in spec['forcing'][var].dims else ['None']
            rows += [{'scenario': str(scen), 'variable': var, 'species': str(s)} for s in species]
    registry = audit_dir / 'source_registry.csv'
    pd.DataFrame(rows, columns=['scenario', 'variable', 'species']).to_csv(registry, index=False)
    forcing = user_forcing if reference_forcing is None else _concat_scenarios(
        [user_forcing, reference_forcing], join='outer', data_vars='all', compat='override')
    plot_forcing_audit(forcing, forcing_hist, registry, audit_dir)
