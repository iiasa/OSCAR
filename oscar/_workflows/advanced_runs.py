"""Advanced runs: your emissions on a run option with a fixed boundary, for close collaborators.

No settings file and no project folder: the call names a run option (option='SCI-lite'),
the output variables and, optionally, inputs (a NetCDF file or an in-memory xr.Dataset),
an output NetCDF path, the baseline, the last projection year and reference scenarios.
Everything about a run option (members, years, scenarios, input units) comes from its
library, downloaded to <advanced_library>/<option>/<region>/ with its
<advanced_library>/INFO_<option>.md.

Inputs follow the customized rules: continuous annual time series covering the
projection period, global totals for emissions, and variables not supplied taken
from the baseline. Reference scenarios and baseline-only runs are read from the
library's precomputed results instead of being re-run.
"""
from pathlib import Path

import xarray as xr

from oscar._io import handlers
from oscar._io._download import ensure_advanced_option, open_advanced_library
from oscar._io.paths import notice_default_data_dir
from oscar._utils.metadata import load_var_registry
from oscar._utils.results import combine_history, finish, check_overwrite
from oscar._workflows.customized_runs import _apply, _concat_scenarios

# Each run option has a fixed boundary (historical forcing, model regions to choose from) and a
# default baseline; only the inputs vary. land_cover: land-cover inputs (D_dA_...) are accepted
# (SCI-luc, D_Eluc computed); otherwise land use enters through D_Eluc and ERF_luc (SCI-lite).
OPTIONS = {
    'SCI-lite': {'hist_type': 'CMIP7', 'regions': ('IAMC_R10',), 'baseline_forcing': 'scen7-VL',
                 'land_cover': False},
    'SCI-luc': {'hist_type': 'CMIP7', 'regions': ('IAMC_R5', 'IAMC_R9', 'IAMC_R10', 'AIM_R17', 'COFFEE_R18',
                                                  'GCAM_R32', 'IMAGE_R26', 'MESSAGE_R12', 'REMIND_R12', 'WITCH_R17'),
                'baseline_forcing': 'scen7-VL', 'land_cover': True},
}
REFERENCE_RESULTS = 'reference_scen_results_nMC{n}.nc'


def _choice(option, region):
    """(option settings, region): region defaults to the option's only region."""
    if option not in OPTIONS:
        raise ValueError(f'Advanced mode needs option=, one of: {", ".join(OPTIONS)}.')
    fixed = OPTIONS[option]
    if region is None and len(fixed['regions']) == 1:
        region = fixed['regions'][0]
    if region not in fixed['regions']:
        raise ValueError(f'{option} needs region=, one of: {", ".join(fixed["regions"])}.')
    return fixed, region


def download_option(option, region=None):
    """Download a run option region once to <advanced_library>/<option>/<region>/, with
    <advanced_library>/INFO_<option>.md; returns the region folder."""
    _, region = _choice(option, region)
    notice_default_data_dir()
    folder = ensure_advanced_option(option, region)
    print(f'[OSCAR] {option} / {region} ready at {folder}; see {folder.parent.parent / f"INFO_{option}.md"} to run it.')
    return folder


def _variables(variables):
    """Output variables: a nonempty list of unique IDs of the variable registry."""
    variables = [variables] if isinstance(variables, str) else list(variables or [])
    if not variables or len(set(variables)) != len(variables):
        raise ValueError('variables must be a nonempty list of unique output variable IDs, '
                         'e.g. variables=["D_Tg", "D_CO2"] (oscar info --variables).')
    registry = load_var_registry()
    for var in variables:
        if var not in registry:
            raise ValueError(f'Unknown output variable {var!r} (oscar info --variables).')
    return variables


def input_units(forcing, land_cover=False):
    """Inputs a run option accepts and their units, from its library forcing: time-dependent
    anthropogenic emissions (D_E..., not biomass burning), radiative forcing (ERF_...) and, with
    land_cover, the land-cover variables (D_dA_..., per region and biome, as OSCAR uses them)."""
    def accepted(var):
        if var in handlers.LAND_USE:
            return land_cover
        return var.startswith('ERF_') or (var.startswith('D_E') and not var.startswith(('D_Ebb', 'D_Emiss')))
    return {var: forcing[var].attrs.get('units') for var in forcing.data_vars
            if 'year' in forcing[var].dims and accepted(var)}


def _read_reference(lib, scenarios, variables, first, last, params):
    """Precomputed scenario results of the library, years first..last, aligned on parameter IDs."""
    path = lib.path / REFERENCE_RESULTS.format(n=lib.n_config)
    if not path.is_file():
        raise FileNotFoundError(f'{path.name} is missing from {lib.path}.')
    with xr.open_dataset(path) as ds:
        missing = [v for v in variables if v not in ds]
        if missing:
            raise ValueError(f'Not in the precomputed results: {missing}.')
        unknown = [s for s in scenarios if s not in ds.scen.values]
        if unknown:
            raise ValueError(f'Not in the precomputed results: {unknown}.')
        out = ds[variables].sel(scen=list(scenarios), year=slice(first, last)).load()
    return out.sel(config=params.config)


def compile_scenarios(user, baseline, first, last):
    """Scenario forcing of your runs: your inputs, glued onto the baseline scenario, which
    provides land use (D_Eluc, ERF_luc) and every other variable you do not supply."""
    runs = [_apply(baseline, user.sel(scen=name, drop=True), None, first, last).expand_dims(scen=[str(name)])
            for name in user.scen.values]
    return xr.concat(runs, dim='scen', data_vars='different', coords='minimal', compat='equals', join='exact')


def run_advanced(option=None, variables=None, forcing=None, output=None, region=None, baseline_forcing=None,
                 projection_end_year=None, reference_scenarios=None, overwrite=False, **solver_options):
    """An advanced run, in four steps:
    1. the run option's library gives the matching history (to 2023) and initial state;
    2. your inputs (forcing) are read as scenarios;
    3. they are glued onto the baseline scenario (land use and everything not supplied);
    4. OSCAR runs them from the initial state; precomputed reference scenarios are added.

    option: run option (e.g. 'SCI-lite'); variables: output variables (list);
    region: model region of the option (default: its only region; SCI-luc has several);
    forcing: NetCDF path, xr.Dataset or None (baseline only, read from the precomputed results);
    output: NetCDF path, or None to keep the results in memory only;
    baseline_forcing: library scenario that fills what you do not supply (default: the option's);
    projection_end_year: last year (default: the library's, 2100);
    reference_scenarios: precomputed library scenarios to add as <name>-ref (default: none).
    The library is <advanced_library>/<option>/<region>/ (.oscar_settings.json, oscar init --data-dir PATH).
    Returns the results: history and scenarios along scen, all members along config."""
    fixed, region = _choice(option, region)
    variables = _variables(variables)
    references = [reference_scenarios] if isinstance(reference_scenarios, str) else list(reference_scenarios or [])
    output = Path(output).expanduser().resolve() if output else None
    if output is not None:
        check_overwrite(output, overwrite)
    notice_default_data_dir()

    # 1. Matching history and initial state: the library of the run option.
    hist = fixed['hist_type']
    base_name = baseline_forcing or fixed['baseline_forcing']
    lib = open_advanced_library(option, hist, region, ensure_advanced_option(option, region))
    for name in [base_name, *references]:
        if name not in lib.scenarios:
            raise ValueError(f'Scenario {name!r} is not in this library: {", ".join(lib.scenarios)}')
    h_end, end = lib.hist_end, projection_end_year or lib.end_year
    if not h_end < end <= lib.end_year:
        raise ValueError(f'projection_end_year must be between {h_end + 1} and {lib.end_year}, or None.')
    missing = [v for v in variables if v not in lib.history_variables]
    if missing:
        raise ValueError(f'Not saved by this run option: {missing}.')
    data = lib.prepare(variables=variables)
    baseline = data['forcing_scen'].sel(scen=base_name, drop=True).sel(year=slice(h_end, end))

    parts = []
    if forcing is None:
        # Baseline only: read from the precomputed results, not run.
        parts.append(_read_reference(lib, [base_name], variables, h_end + 1, end, data['params']))
    else:
        # 2. Your inputs, as scenarios: names, units and full coverage are checked; values are yours.
        user = handlers.read_netcdf(forcing, units=input_units(data['forcing_scen'], fixed['land_cover']))
        reserved = {'Historical'} | {s + '-ref' for s in lib.scenarios}
        for name in map(str, user.scen.values):
            if not name.strip() or name in reserved:
                raise ValueError(f'Invalid or reserved scenario name: {name!r}')
        handlers.check_full_coverage(user, h_end + 1, end)
        # 3. Glued onto the baseline (land use and everything not supplied).
        scenarios = compile_scenarios(user, baseline, h_end + 1, end)
        # 4. Run from the initial state at the end of history.
        from oscar._core._model.OSCAR import OSCAR
        projection = OSCAR(Ini=data['initial_state'], Par=data['params'], For=scenarios,
                           var_keep=variables, **solver_options)
        parts.append(projection.sel(year=slice(h_end + 1, end)))
    if references:
        ref = _read_reference(lib, references, variables, h_end + 1, end, data['params'])
        parts.append(ref.assign_coords(scen=[s + '-ref' for s in references]))

    # Results: history, then your scenarios and the references, in memory and optionally saved.
    projection = _concat_scenarios(parts, join='exact') if len(parts) > 1 else parts[0]
    out = combine_history(data['history'], projection, h_end, variables)
    record = {'option': option, 'hist_type': hist, 'region': region, 'library_dir': str(lib.path),
              'baseline_forcing': base_name, 'projection_end_year': end, 'variables': variables,
              'reference_scenarios': references, 'data_version': lib.data_version, 'solver_options': solver_options,
              'input': 'in-memory Dataset' if isinstance(forcing, xr.Dataset) else (str(forcing) if forcing else None)}
    return finish(out, 'advanced', record, h_end, output, output is not None, False, False, overwrite)
