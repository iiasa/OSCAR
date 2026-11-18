"""Combining history with projections and saving results."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import uuid
import numpy as np
import xarray as xr
from oscar._io.paths import get_paths, validate_name
from oscar._utils.metadata import apply_variable_metadata


def combine_history(history, projection, hist_end, variables):
    history = history[variables].sel(year=slice(None, hist_end))
    if 'scen' in history.dims:
        if history.sizes['scen'] != 1:
            raise ValueError('Expected one historical scenario.')
        history = history.isel(scen=0, drop=True)
    history = history.drop_vars('scen', errors='ignore').expand_dims(scen=['Historical'])
    projection = projection[variables].sel(year=slice(hist_end + 1, None))
    out = xr.concat([history, projection], dim='year', join='outer', combine_attrs='drop_conflicts')
    if 'scen' in projection.dims:
        out = out.sel(scen=['Historical', *projection.scen.values])  # run order, not alphabetical
    if len(np.unique(out.year)) != out.sizes['year']:
        raise ValueError('Results contain duplicate years at the historical transition.')
    return out


def output_path(mode, output_dir=None, run_name=None, save=True, plot=False, default_dir=None, default_name=None):
    """Result file of a run. With default_dir, results go to one fixed, visible file
    (<dir>/<run_name or default_name or mode>_results.nc) instead of a new timestamped folder."""
    if not (save or plot or output_dir or run_name):
        return None
    if run_name:
        validate_name(run_name, 'run name')
    if default_dir is not None:
        directory = Path(output_dir or default_dir).expanduser().resolve()
        return directory / f'{run_name or default_name or mode}_results.nc'
    name = run_name or (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:6])
    directory = Path(output_dir).expanduser().resolve() if output_dir else get_paths()['results'] / mode / name
    return directory / f"{run_name or 'oscar_' + mode}_results.nc"


def result_prefix(path):
    """Plot files are named after the results file: <prefix>_results.nc -> <prefix>_<var>.png."""
    stem = path.stem if path is not None else 'oscar'
    return stem[:-len('_results')] if stem.endswith('_results') else stem


def check_overwrite(path, overwrite):
    if path and path.exists() and not overwrite:
        raise FileExistsError(f'Results already exist: {path}. Replace them with --overwrite (overwrite=True), '
                              'or choose another --output-dir (output_dir=).')


def finish(out, mode, settings, hist_end, path=None, save=True, plot=False, show_plot=False, overwrite=False,
           plot_variables=None, plot_prefix='plot', plot_dir=None):
    """plot_variables: variables to plot (default: all year/config/scen series);
    plot_prefix: plot file names, <prefix>_<var>.png; plot_dir: their folder (default: next to the results)."""
    out = apply_variable_metadata(out)
    # Identity of the scientific selection and library, readable without parsing run_settings.
    identity = {'hist_type': settings.get('hist_type'),
                'model_region': settings.get('region'), 'data_version': settings.get('data_version')}
    out.attrs.update({k: v for k, v in identity.items() if v is not None})
    if 'config' in out.dims:
        out.attrs['ensemble_size'] = int(out.sizes['config'])
    out.attrs.update(run_mode=mode, historical_end=int(hist_end),
                     run_settings=json.dumps(settings, sort_keys=True, default=str))
    if path:
        out.attrs['output_file'] = str(path)
    if save:
        check_overwrite(path, overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
        try:
            out.to_netcdf(temp, engine='h5netcdf')
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()
        print(f'[OSCAR] Results saved to {path}')
    if plot or show_plot:
        from oscar._viz import plot_timeseries_summary
        folder = Path(plot_dir) if plot_dir else path.parent
        folder.mkdir(parents=True, exist_ok=True)
        variables = [v for v in (plot_variables or out.data_vars) if v in out and set(out[v].dims) <= {'year', 'config', 'scen'}]
        plot_timeseries_summary(out, hist_end, variables, folder, show_plot=show_plot, prefix=plot_prefix)
        print(f"[OSCAR] Plots saved to {folder}: {', '.join(f'{plot_prefix}_{v}.png' for v in variables)}")
    return out
