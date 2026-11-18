"""Configured-mode libraries: where they live, how they are downloaded and read.

A library holds one history type (CMIP7 or CMIP6) for one model region: the
500-member parameters, the historical forcing and outputs, the final historical
state and the scenario forcing. Runs restart from the final historical state.
"""
from pathlib import Path
import hashlib
import os
import shutil
import stat
import tempfile
import zipfile

import numpy as np
import requests
import xarray as xr

from oscar._io.paths import get_paths, PACKAGE_ROOT, SETTINGS_FILE
from oscar._utils.load_config import load_config
from oscar._utils.metadata import get_oscar_version

# Library files ({n}: ensemble size).
LIBRARY_FILES = {
    'params': 'params_nMC{n}.nc',
    'initial_state': 'ini_state_nMC{n}.nc',
    'history': 'hist_results_nMC{n}.nc',
    'forcing_hist': 'forcing_hist.nc',
    'forcing_scen': 'forcing_scen.nc',
}

# Bundled standard-mode demonstration (a subset of a CMIP7 / IAMC_R5 library).
BOOTSTRAP_FILES = {
    'params': 'parameters_mc_standard.nc',
    'initial_state': 'scen_initial_state_standard.nc',
    'history': 'output_hist_standard.nc',
    'forcing_hist': 'forcing_hist_standard.nc',
    'forcing_scen': 'forcing_scen_standard.nc',
}

# Raw scenario labels in the library files -> names shown to users.
SCENARIO_NAMES = {
    'h': 'scen7-H', 'hl': 'scen7-HL', 'l': 'scen7-L', 'ln': 'scen7-LN',
    'm': 'scen7-M', 'ml': 'scen7-ML', 'vl': 'scen7-VL',
    'ssp119': 'SSP1-1.9', 'ssp126': 'SSP1-2.6', 'ssp245': 'SSP2-4.5',
    'ssp370': 'SSP3-7.0', 'ssp370-lowNTCF': 'SSP3-7.0-LowNTCF',
    'ssp434': 'SSP4-3.4', 'ssp460': 'SSP4-6.0', 'ssp534-over': 'SSP5-3.4-OS', 'ssp585': 'SSP5-8.5',
}


def check_choice(value, allowed, label):
    if value not in allowed:
        raise ValueError(f"Invalid {label} {value!r}. Available: {', '.join(map(str, allowed))}")


def canonical_scenario(value):
    return SCENARIO_NAMES.get(str(value), str(value))


def scenario_names(ds):
    if 'scen' in ds.dims:
        names = [canonical_scenario(s) for s in ds.scen.values]
        if len(names) != len(set(names)):
            raise ValueError('Scenario aliases resolve to duplicate names in this library.')
        ds = ds.assign_coords(scen=names)
    return ds


def file_digest(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_members(reference, *datasets):
    expected = reference.config.values
    if len(np.unique(expected)) != len(expected):
        raise ValueError('Library parameter ensemble contains duplicate IDs.')
    for ds in datasets:
        if 'config' in ds.dims:
            ids = ds.config.values
            if len(np.unique(ids)) != len(ids) or set(ids.tolist()) != set(expected.tolist()):
                raise ValueError('Library ensemble IDs do not match across parameters, forcing, history and initial state.')


def restart_forcing(forcing, history, hist_end):
    """OSCAR advances from the second forcing year, so scenario forcing starts at hist_end."""
    if int(forcing.year.values[0]) == hist_end:
        return forcing
    if int(forcing.year.values[0]) != hist_end + 1:
        raise ValueError(f'Scenario forcing must begin at {hist_end} (restart year) or {hist_end + 1}.')
    # Beta2 libraries omitted the restart year: take it from the historical forcing.
    anchor = forcing.isel(year=0, drop=True).copy(deep=True)
    for name in forcing.data_vars:
        if 'year' in forcing[name].dims and name in history and 'year' in history[name].dims:
            anchor[name] = history[name].sel(year=hist_end, drop=True).drop_vars('scen', errors='ignore')
    anchor = anchor.drop_vars('year', errors='ignore').expand_dims(year=[hist_end])
    return xr.concat([anchor, forcing], dim='year', data_vars='minimal', coords='minimal', compat='equals')


def load_dataset(path, variables=None):
    with xr.open_dataset(path) as ds:
        return (ds[variables] if variables else ds).load()


class Library:
    """One library folder. hist_type, region and n_config, when given, are checked
    against the identity attributes of the NetCDF files (if the files carry them)."""

    def __init__(self, path, hist_type=None, region=None, n_config=None, files=LIBRARY_FILES):
        self.path = Path(path).expanduser().resolve()
        self.hist_type, self.region, self.n_config = hist_type, region, n_config
        self.files = {kind: self.path / name.format(n=n_config) for kind, name in files.items()}
        missing = [p.name for p in self.files.values() if not p.is_file() or not p.stat().st_size]
        if missing:
            raise FileNotFoundError(f'Incomplete library at {self.path}. Missing: {", ".join(missing)}')
        with xr.open_dataset(self.files['params']) as ds:
            self.attrs = dict(ds.attrs)
            members = ds.sizes.get('config')
        for key, expected in [('history', hist_type), ('region', region)]:
            found = self.attrs.get(key)
            if expected is not None and found is not None and found != expected:
                raise ValueError(f'Library at {self.path} is for {key} {found!r}; expected {expected!r}.')
        if n_config is not None and members != n_config:
            raise ValueError(f'Library at {self.path} has {members} members; expected {n_config}.')
        with xr.open_dataset(self.files['initial_state']) as ds:
            ini_year = int(ds.year.item()) if 'year' in ds.coords else None
        with xr.open_dataset(self.files['forcing_hist']) as ds:
            forcing_end = int(ds.year[-1])
        # The initial state is the end-of-history state; beta2 libraries saved it without a year.
        self.hist_end = ini_year if ini_year is not None else forcing_end
        with xr.open_dataset(self.files['history']) as ds:
            self.hist_start = int(ds.year[0])
            self.history_variables = list(ds.data_vars)
        with xr.open_dataset(self.files['forcing_scen']) as ds:
            self.scenarios = [canonical_scenario(s) for s in np.atleast_1d(ds.scen.values)]
            self.end_year = int(ds.year[-1])

    @property
    def data_version(self):
        return self.attrs.get('release_version', 'legacy-unversioned')

    def provenance(self):
        """Library identity recorded with every result."""
        return {'hist_type': self.hist_type, 'region': self.region, 'library_dir': str(self.path),
                'data_version': self.data_version}

    def load(self, kind, variables=None):
        return load_dataset(self.files[kind], variables)

    def load_scenarios(self, scenarios=None):
        with xr.open_dataset(self.files['forcing_scen']) as ds:
            ds = scenario_names(ds)
            if scenarios is not None:
                for s in scenarios:
                    check_choice(s, self.scenarios, 'scenario in this library')
                ds = ds.sel(scen=list(scenarios))
            return ds.load()

    def prepare(self, scenarios=None, variables=None, repair=None):
        """Scenario-run inputs aligned on parameter IDs, with forcing starting at the restart year.

        Returns params, initial_state, history (selected variables), forcing_hist
        and forcing_scen. repair(data) may fix known legacy formats before checks.
        """
        for var in variables or []:
            check_choice(var, self.history_variables, 'historical output variable')
        data = {'params': self.load('params'), 'initial_state': self.load('initial_state'),
                'history': self.load('history', variables), 'forcing_hist': self.load('forcing_hist'),
                'forcing_scen': self.load_scenarios(scenarios)}
        if repair:
            data = repair(data)
        par = data['params']
        check_members(par, *(data[k] for k in ('initial_state', 'history', 'forcing_hist', 'forcing_scen')))
        # Parameter order is authoritative; align by IDs, never position.
        for kind in ('initial_state', 'history', 'forcing_scen'):
            if 'config' in data[kind].dims:
                data[kind] = data[kind].sel(config=par.config)
        data['forcing_scen'] = restart_forcing(data['forcing_scen'], data['forcing_hist'], self.hist_end)
        return data


def n_members():
    return load_config()['configured_mode']['official_nMC']


def required_assets():
    return [name.format(n=n_members()) for name in LIBRARY_FILES.values()]


def open_library(hist_type, region, path):
    return Library(path, hist_type, region, n_config=n_members())


def validate_library(path, hist_type, region):
    """Check a library folder; returns its resolved path."""
    return open_library(hist_type, region, path).path


def library_location(hist_type, region):
    """Local library folder: <data>/library/run_mode/configured/<hist>/<region>."""
    return get_paths()['configured_library'] / hist_type / region


def _same_version(a, b):
    return a.lstrip('v').replace('-beta', 'b') == b.lstrip('v').replace('-beta', 'b')


def ensure_configured_library(hist_type, region, library_dir=None):
    """Path of a checked library: library_dir (for testing), the fixed location, otherwise a download."""
    if library_dir:
        return validate_library(library_dir, hist_type, region)
    target = library_location(hist_type, region)
    if target.exists():
        return validate_library(target, hist_type, region)
    cfg = load_config()
    meta = cfg['metadata']['run_mode']
    # Never download data published for another version of the code.
    if not _same_version(meta['version'], get_oscar_version()):
        raise RuntimeError(f'No {hist_type} / {region} library at {target}. Place the library there '
                           f'({", ".join(required_assets())}). Automatic download: data for this '
                           'version are not published yet.')
    archive = _fetch(meta.get('bundles', {}).get(f'{hist_type}/{region}'),
                     f'OSCAR_lib_configured_{hist_type}_{region}.zip', target.parent)
    with tempfile.TemporaryDirectory(prefix='.oscar-', dir=target.parent) as temp:
        temp = Path(temp)
        _extract(archive, temp)
        source = next((p for p in [temp / region, temp] if all((p / n).is_file() for n in required_assets())), None)
        if source is None:
            raise RuntimeError(f'Downloaded archive does not contain the {hist_type} / {region} library files.')
        validate_library(source, hist_type, region)
        if target.exists():
            return validate_library(target, hist_type, region)
        os.rename(source, target)
    archive.unlink(missing_ok=True)
    return target


def advanced_location(option, region):
    """Library of a run option region: <advanced_library>/<option>/<region>/ (beside <advanced_library>/INFO_<option>.md)."""
    return get_paths()['advanced_library'] / option / region


def open_advanced_library(option, hist_type, region, path):
    """The library of a run option region; everything about it (members, years, scenarios, units)
    comes from its files. Its NetCDF attribute scientific_setup must name the run option (when present)."""
    path = Path(path).expanduser()
    params = sorted(path.glob('params_nMC*.nc'))
    if len(params) != 1:
        raise FileNotFoundError(f'Expected one params_nMC<members>.nc in {path}.')
    n_config = int(params[0].stem.split('nMC')[-1])
    lib = Library(path, hist_type, region, n_config=n_config)
    found = lib.attrs.get('scientific_setup')
    if found is not None and found != option:
        raise ValueError(f'Library at {lib.path} is for run option {found!r}; expected {option!r}.')
    return lib


def ensure_advanced_option(option, region):
    """Library folder of a run option region, downloaded once (OSCAR_lib_advanced_<option>_<region>.zip:
    INFO_<option>.md and <option>/<region>/, unpacked into <advanced_library>/)."""
    target = advanced_location(option, region)
    info = target.parent.parent / f'INFO_{option}.md'
    if any(target.glob('params_nMC*.nc')) and info.is_file():
        return target
    meta = load_config()['metadata']['run_mode']
    if not _same_version(meta['version'], get_oscar_version()):
        raise RuntimeError(f'No {option} / {region} library at {target}; set advanced_library in {SETTINGS_FILE} '
                           'to a folder that has it.')
    root = target.parent.parent
    root.mkdir(parents=True, exist_ok=True)
    archive = _fetch(meta.get('bundles', {}).get(f'{option}/{region}'),
                     f'OSCAR_lib_advanced_{option}_{region}.zip', root)
    with tempfile.TemporaryDirectory(prefix='.oscar-', dir=root) as temp:
        temp = Path(temp)
        _extract(archive, temp)
        source = temp / option / region
        if not any(source.glob('params_nMC*.nc')) or not (temp / info.name).is_file():
            raise RuntimeError(f'{archive.name} does not contain {info.name} and {option}/{region}/.')
        if target.exists():
            shutil.rmtree(target)  # an incomplete earlier installation
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
        os.replace(temp / info.name, info)
    archive.unlink(missing_ok=True)
    return target


def _fetch(entry, filename, folder):
    """Download a published archive into folder (checked against the publisher's checksum).

    entry: a release-registry entry (filename, urls, checksum, size) from config.yaml,
    or None to look the file up in the published Zenodo record.
    """
    cfg = load_config()
    if entry:
        filename, urls = entry['filename'], entry['urls']
    else:
        record_url = f"{cfg['metadata']['zenodo_base_url']}{cfg['metadata']['run_mode']['zenodo_id']}"
        with requests.get(record_url, timeout=(10, 60)) as response:
            response.raise_for_status()
            record = response.json()
        entry = next((f for f in record.get('files', []) if f.get('key') == filename), None)
        urls = [entry.get('links', {}).get('self')] if entry else []
    if not entry or not entry.get('checksum') or not urls or not all(urls):
        raise RuntimeError(f'No checksummed download of {filename} in the published record.')
    if Path(filename).name != filename:
        raise ValueError('Release archive filename must not contain directory components.')
    # As in beta2, the archive sits next to its library folder and is removed once installed.
    archive = folder / filename
    folder.mkdir(parents=True, exist_ok=True)
    print(f"[OSCAR] Downloading {filename} ({entry.get('size', 'unknown')} bytes) to {folder}")
    for index, url in enumerate(urls):
        try:
            _download(url, archive, entry['checksum'])
            break
        except (requests.RequestException, RuntimeError):
            if index == len(urls) - 1:
                raise
            print('[OSCAR] Trying the next release mirror...')
    return archive


def _download(url, archive, checksum):
    algorithm, expected = checksum.split(':', 1)
    if algorithm not in ('sha256', 'md5'):
        raise ValueError(f'Unsupported release checksum: {algorithm}')
    if archive.exists() and file_digest(archive, algorithm) == expected:
        return
    part = archive.with_suffix(archive.suffix + '.part')
    if part.exists() and file_digest(part, algorithm) == expected:
        os.replace(part, archive)
        return
    for attempt in range(3):
        offset = part.stat().st_size if part.exists() else 0
        try:
            with requests.get(url, stream=True, headers={'Range': f'bytes={offset}-'} if offset else {},
                              timeout=(10, 120)) as response:
                if response.status_code == 416 and offset:
                    part.unlink()
                    raise RuntimeError('Restarting an invalid partial download.')
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Server returned an inconsistent download range.')
                with part.open('ab' if append else 'wb') as stream:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            stream.write(block)
            if file_digest(part, algorithm) != expected:
                part.unlink()
                raise RuntimeError(f'Checksum mismatch downloading {archive.name}')
            os.replace(part, archive)
            return
        except (requests.RequestException, RuntimeError):
            if attempt == 2:
                raise


def _extract(archive, destination):
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            target = (destination / item.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError('Archive contains a path outside the extraction directory.')
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Archive contains a symbolic link.')
        bundle.extractall(destination)


CUSTOMIZED_ARCHIVE = 'OSCAR_lib_customized.zip'


def ensure_customized_library():
    """Templates and examples for customized projects, downloaded once to
    <data>/library/run_mode/customized/ (templates/settings_template.yaml, examples/)."""
    target = get_paths()['customized_library']
    template = target / 'templates' / 'settings_template.yaml'
    if template.is_file():
        return target
    archive = _fetch(load_config()['metadata']['run_mode'].get('bundles', {}).get('customized'),
                     CUSTOMIZED_ARCHIVE, target.parent)
    with tempfile.TemporaryDirectory(prefix='.oscar-', dir=target.parent) as temp:
        temp = Path(temp)
        _extract(archive, temp)
        if not (temp / 'templates' / 'settings_template.yaml').is_file():
            raise RuntimeError(f'{archive.name} does not contain templates/settings_template.yaml.')
        if target.exists():
            shutil.rmtree(target)  # an incomplete earlier installation (no template)
        os.rename(temp, target)
        temp.mkdir()  # recreated for the temporary-directory cleanup
    archive.unlink(missing_ok=True)
    return target
