"""Bundled CMIP demonstration; no network or initialization required."""
from pathlib import Path
import json
import numpy as np
from oscar._io._download import Library, BOOTSTRAP_FILES, load_dataset
from oscar._io.paths import INTERNAL_BOOTSTRAP_DIR, get_paths
from oscar._utils.load_config import load_config
from oscar._utils.results import (combine_history, finish, output_path, check_overwrite,
                                     result_prefix)


def _repair_legacy_labels(data):
    """Known beta2 bootstrap format: parameters/history were ranked but scenario
    forcing kept sorted prior IDs. Never apply this repair to external data."""
    par, forcing = data['params'], data['forcing_scen']
    if not np.array_equal(forcing.config.values, par.config.values):
        if not (np.array_equal(par.config, np.arange(par.sizes['config']))
                and forcing.sizes['config'] == par.sizes['config']
                and np.all(np.diff(forcing.config.values) > 0)):
            raise ValueError('Bundled demonstration has incompatible ensemble labels; reinstall OSCAR.')
        data['forcing_scen'] = forcing.assign_coords(config=par.config)
    return data


def run_standard(show_plot=False, run_model=True, save=True, plot=True,
                 output_dir=None, run_name=None, overwrite=False):
    path = output_path('standard', output_dir, run_name, save, plot or show_plot,
                       default_dir=get_paths()['results'] / 'standard')
    if not run_model:
        if path is None or not path.exists():
            raise FileNotFoundError(f'No saved demonstration results at {path}. Run the demonstration first, '
                                    'using the same output directory and run name, if any.')
        out = load_dataset(path)
        return finish(out, 'standard', json.loads(out.attrs.get('run_settings', '{}')), out.attrs['historical_end'],
                      path, False, plot, show_plot, **_plot_options(path))
    if save:
        check_overwrite(path, overwrite)
    cfg = load_config()['standard_mode']
    # The bundle has no manifest; its identity comes from the package configuration.
    lib = Library(INTERNAL_BOOTSTRAP_DIR, cfg['hist_type'], cfg['region'], files=BOOTSTRAP_FILES)
    data = lib.prepare(variables=cfg['var_select'], repair=_repair_legacy_labels)
    par, forcing = data['params'], data['forcing_scen']
    print(f"[OSCAR] Demonstration: {par.sizes['config']} Monte Carlo members, "
          f"{lib.hist_end + 1}–{int(forcing.year[-1])} projections.")
    from oscar._core._model.OSCAR import OSCAR
    projection = OSCAR(Ini=data['initial_state'], Par=par, For=forcing, nt=4, var_keep=cfg['var_select'])
    out = combine_history(data['history'], projection, lib.hist_end, cfg['var_select'])
    settings = {**lib.provenance(), 'scenarios': forcing.scen.values.tolist(), 'nt': 4,
                'data_version': par.attrs.get('release_version', 'bundled-beta2-bootstrap')}
    return finish(out, 'standard', settings, lib.hist_end, path, save, plot, show_plot, overwrite,
                  **_plot_options(path))


def _plot_options(path):
    """The demonstration plots only its configured variables, named after the results file."""
    return {'plot_variables': load_config()['standard_mode']['plot_vars'], 'plot_prefix': result_prefix(path)}
