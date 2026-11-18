"""Configured runs: available CMIP scenarios re-run from a library.

Users choose the history type, region, scenarios and output variables offered in
config.yaml (configured_mode). The 500-member ensemble and solver are fixed.
"""
from dataclasses import dataclass
import json
from oscar._io._download import ensure_configured_library, open_library, check_choice, canonical_scenario, load_dataset
from oscar._io import handlers
from oscar._io.paths import get_paths, notice_default_data_dir
from oscar._utils.load_config import load_config
from oscar._utils.results import (combine_history, finish, output_path, check_overwrite,
                                     result_prefix)

ALL_VARIABLES = 'all'  # variables='all' (-v all): every output variable of the library


def options():
    """The configured_mode block of config.yaml: the choices offered to users."""
    return load_config()['configured_mode']


def _as_list(values):
    return [values] if isinstance(values, str) else list(values)


@dataclass
class Selection:
    hist_type: str
    region: str
    scenarios: list
    variables: list  # None: every library variable


def resolve_selection(scenario=None, region=None, hist_type=None, variables=None):
    """Check user choices against config.yaml before any data access."""
    opt = options()
    defaults = opt['defaults']
    hist_type = hist_type or defaults['hist_type']
    check_choice(hist_type, opt['allowed_hist'], 'history')
    region = region or defaults['region']
    check_choice(region, opt['allowed_regions'], 'region')
    scenarios = [canonical_scenario(s) for s in _as_list(scenario or defaults['scenario'][hist_type])]
    for s in scenarios:
        check_choice(s, opt['scenarios'][hist_type], f'{hist_type} scenario')
    if len(set(scenarios)) != len(scenarios):
        raise ValueError('Scenario selections must be unique.')
    variables = _as_list(variables or defaults['var_select'])
    if variables == [ALL_VARIABLES]:
        return Selection(hist_type, region, scenarios, None)
    from oscar._utils.metadata import load_var_registry
    registry = load_var_registry()
    for var in variables:
        check_choice(var, registry, 'output variable')
    if len(set(variables)) != len(variables):
        raise ValueError('Variable selections must be unique.')
    return Selection(hist_type, region, scenarios, variables)


def resolve_selections(scenario=None, region=None, hist_type=None, variables=None):
    """(hist_type, region, scenarios, variables) tuple, used by customized projects."""
    s = resolve_selection(scenario, region, hist_type, variables)
    return s.hist_type, s.region, s.scenarios, s.variables


def library_details(path, hist_type, region):
    """Coverage summary of a local library (used by customized runs)."""
    lib = open_library(hist_type, region, path)
    return {'path': lib.path, 'manifest': lib.attrs, 'n_mc': lib.n_config, 'hist_end': lib.hist_end,
            'end_year': lib.end_year, 'scenarios': lib.scenarios}


def results_file(sel, output_dir=None, run_name=None, save=True, plot=True):
    """<data>/results/configured/configured_<hist>_<region>_results.nc, or <run_name>_results.nc."""
    return output_path('configured', output_dir, run_name, save, plot,
                       default_dir=get_paths()['results'] / 'configured',
                       default_name=f'configured_{sel.hist_type}_{sel.region}')


def run_configured(scenario=None, region=None, hist_type=None, variables=None,
                   show_plot=False, run_model=True, save=True, plot=True,
                   output_dir=None, run_name=None, overwrite=False, library_dir=None):
    sel = resolve_selection(scenario, region, hist_type, variables)
    notice_default_data_dir()
    path = results_file(sel, output_dir, run_name, save, plot or show_plot)
    plot_variables = sel.variables or options()['defaults']['var_select']
    if not run_model:
        if path is None or not path.exists():
            raise FileNotFoundError(f'No saved configured results at {path}. Run the same selection first '
                                    '(with the same run name, if any).')
        out = load_dataset(path)
        settings = json.loads(out.attrs.get('run_settings', '{}'))
        return finish(out, 'configured', settings, out.attrs['historical_end'], path, False, plot, show_plot,
                      plot_variables=plot_variables, plot_prefix=result_prefix(path))
    if save:
        check_overwrite(path, overwrite)
    lib = open_library(sel.hist_type, sel.region,
                       ensure_configured_library(sel.hist_type, sel.region, library_dir))
    # variables=all keeps every library output; plots then show the default variables only.
    variables = sel.variables or lib.history_variables
    data = lib.prepare(sel.scenarios, variables)
    # Projections end in run_end (2100); a library ending one year earlier gets its last year copied.
    run_end = options()['run_end']
    end = run_end if lib.end_year >= run_end - 1 else lib.end_year
    data['forcing_scen'] = handlers.copy_last_year(data['forcing_scen'], end).sel(year=slice(lib.hist_end, end))
    print(f"[OSCAR] {sel.hist_type} history, {sel.region}: {lib.n_config} Monte Carlo members, "
          f"{lib.hist_end + 1}–{end} projections, {', '.join(sel.scenarios)}")
    from oscar._core._model.OSCAR import OSCAR
    projection = OSCAR(Ini=data['initial_state'], Par=data['params'], For=data['forcing_scen'],
                       var_keep=variables)
    out = combine_history(data['history'], projection, lib.hist_end, variables)
    settings = {**lib.provenance(), 'scenarios': sel.scenarios, 'variables': variables}
    return finish(out, 'configured', settings, lib.hist_end, path, save, plot, show_plot, overwrite,
                  plot_variables=plot_variables, plot_prefix=result_prefix(path))
