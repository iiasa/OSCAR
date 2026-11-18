"""Read customized-mode input files: emissions, radiative forcing and land use.

OSCAR takes continuous annual time series. A variable the user supplies must
cover every projection year; OSCAR does not interpolate, extrapolate or rescale
user values. Variables the user does not supply come from the baseline scenario.
"""
import csv

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from oscar._io.paths import PACKAGE_ROOT, REGIONS_INFO_PATH

# Supported inputs, their units and dimensions: oscar/_utils/_resources/input_species.yaml.
_REGISTRY = yaml.safe_load((PACKAGE_ROOT / 'oscar' / '_utils' / '_resources' / 'input_species.yaml').read_text())
EMISSIONS = {var: spec['unit'] for var, spec in _REGISTRY['anthropogenic_emissions'].items()}
RADIATIVE_FORCING = {var: spec['unit'] for var, spec in _REGISTRY['rf'].items()}
LAND_USE = {var: spec['unit'] for var, spec in _REGISTRY['lulcc'].items()}
UNITS = {**EMISSIONS, **RADIATIVE_FORCING, **LAND_USE}
FORMATS = ('oscar-csv', 'iamc-ar6-public', 'ceds')
GLOBAL_NAMES = {'World', 'Global', 'Globe'}


def load_var_mapping(format_type):
    """Translation registry of an external format (vars_map_<format>.yaml)."""
    path = PACKAGE_ROOT / 'oscar' / '_utils' / '_resources' / f'vars_map_{format_type}.yaml'
    if not path.exists():
        raise FileNotFoundError(f'[OSCAR] Missing mapping registry: {path.name}')
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def model_regions(region):
    """Region codes of a model region set, in model order, and the names they may appear under."""
    with (REGIONS_INFO_PATH / 'OSCAR_reg_names.csv').open() as stream:
        row = next(row for row in csv.reader(stream) if row[0] == region)
    codes = [item.split(':', 1)[0] for item in row[1:] if item]
    aliases = {item.split(':', 1)[-1].strip(): item.split(':', 1)[0] for item in row[1:] if item}
    aliases.update({code: code for code in codes})
    return codes, aliases


# --- Reading files -------------------------------------------------------------

def _long_table(path):
    """CSV as long rows; wide files (one column per year) are melted."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    years = [c for c in df if c.isdigit()]
    if years:
        df = df.melt(id_vars=[c for c in df if c not in years], value_vars=years,
                     var_name='year', value_name='value')
    return df


def _translate(df, format_type):
    """IAMC / CEDS rows -> oscar-csv rows (scen, year, reg_land, oscar_var, oscar_id, value, unit)."""
    df = df.rename(columns={c: c.lower() for c in df if c.lower() in
                            ('scenario', 'variable', 'region', 'unit', 'year', 'value')})
    required = {'scenario', 'variable', 'region', 'unit', 'year', 'value'}
    if not required <= set(df):
        raise ValueError(f'CSV missing columns: {sorted(required - set(df))}')
    mapping = load_var_mapping(format_type)
    emissions, forcing = mapping.get('anthropogenic_emissions', {}), mapping.get('radiative_forcing', {})
    rows = []
    for _, row in df.iterrows():
        name, unit = str(row.variable).strip(), str(row.unit).strip()
        if name in emissions:
            var, species, factor, *expected = emissions[name]
            # Older mapping metadata spells NOx as NOX; the model input is NOx.
            var = 'D_Eant_NOx' if var == 'D_Eant_NOX' else var
            species = var if species == 'D_Eant_NOX' else species
        elif name in forcing:
            var, factor, *expected = forcing[name]
            species = var
        else:
            raise ValueError(f'Unsupported variable {name!r}; remove unrelated rows from this input.')
        if var not in UNITS:
            raise ValueError(f'{name!r} maps to {var}, which OSCAR does not take as an input.')
        if not expected or unit != expected[0]:
            raise ValueError(f'Unexpected units for {name}: {unit!r}. Expected {expected[0] if expected else "?"}.')
        rows.append({'scen': row.scenario, 'year': row.year, 'reg_land': row.region, 'oscar_var': var,
                     'oscar_id': species, 'value': float(row.value) * float(pd.eval(factor)), 'unit': UNITS[var]})
    return pd.DataFrame(rows)


def _clean(df, required):
    if not required <= set(df):
        raise ValueError(f'CSV missing columns: {sorted(required - set(df))}')
    for col in required - {'year', 'value'}:
        if df[col].isna().any():
            raise ValueError(f'CSV has missing {col} values.')
        df[col] = df[col].astype(str).str.strip()
    for col in ('year', 'value'):
        df[col] = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(df[col]).all():
            raise ValueError(f'CSV {col} must contain finite numeric values.')
    if (df.year != df.year.astype(int)).any():
        raise ValueError('CSV years must be integers.')
    df['year'] = df.year.astype(int)
    return df


def read_atmospheric(path, format_type, region):
    """Emissions and radiative forcing as global totals (no region dimension).

    For each scenario, year, variable and species: a World (Global) row is used
    as the total and regional rows are ignored, so nothing is counted twice;
    without a World row, the regional rows are added up.
    """
    if format_type not in FORMATS:
        raise ValueError(f'Atmospheric format must be one of {", ".join(FORMATS)}.')
    df = _long_table(path)
    if format_type != 'oscar-csv':
        df = _translate(df, format_type)
    df = _clean(df, {'scen', 'year', 'reg_land', 'oscar_var', 'oscar_id', 'value', 'unit'})
    if df.empty:
        raise ValueError('Input CSV contains no rows.')
    keys = ['scen', 'year', 'reg_land', 'oscar_var', 'oscar_id']
    if df.duplicated(keys).any():
        raise ValueError('Duplicate rows found; aggregate them explicitly before running.')
    _, aliases = model_regions(region)
    unknown = sorted(set(df.reg_land) - GLOBAL_NAMES - set(aliases))
    if unknown:
        raise ValueError(f'Unknown regions {unknown}; use World or {region} region names.')
    for var, rows in df.groupby('oscar_var'):
        if var not in EMISSIONS and var not in RADIATIVE_FORCING:
            raise ValueError(f'{var} is not an emissions or radiative-forcing input. '
                             f'Supported: {", ".join({**EMISSIONS, **RADIATIVE_FORCING})}.')
        if set(rows.unit) != {UNITS[var]}:
            raise ValueError(f'{var} units must be {UNITS[var]!r}; found {sorted(set(rows.unit))}.')
        if var != 'D_Eant_Xhalo' and set(rows.oscar_id) != {var}:
            raise ValueError(f'oscar_id must equal {var} except for halogens (D_Eant_Xhalo).')
    is_global = df.reg_land.isin(GLOBAL_NAMES)
    group = ['scen', 'year', 'oscar_var', 'oscar_id']
    has_global = df[is_global].set_index(group).index
    ignored = df[~is_global & df.set_index(group).index.isin(has_global)]
    if not ignored.empty:
        print(f"[OSCAR] Using World totals; regional rows ignored for: {', '.join(sorted(set(ignored.oscar_var)))}")
    totals = pd.concat([df[is_global], df[~is_global & ~df.set_index(group).index.isin(has_global)]])
    totals = totals.groupby(group, as_index=False)['value'].sum()
    pieces = []
    for var, rows in totals.groupby('oscar_var'):
        if var == 'D_Eant_Xhalo':
            da = rows.rename(columns={'oscar_id': 'spc_halo'}).set_index(['scen', 'year', 'spc_halo'])['value']
        else:
            da = rows.set_index(['scen', 'year'])['value']
        da = da.to_xarray().rename(var)
        da.attrs['units'] = UNITS[var]
        pieces.append(da)
    return xr.merge(pieces, join='outer')


def read_land_use(path, region):
    """Land-use inputs, kept per model region (names must match the region set).

    Columns: scen, year, reg_land, oscar_var, bio_from, bio_to, value and an optional
    unit; harvest variables (D_dA_wharv1/2) use bio_land instead of bio_from/bio_to.
    """
    df = _long_table(path)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={'scenario': 'scen', 'region': 'reg_land', 'variable': 'oscar_var'})
    df = _clean(df, {'scen', 'year', 'reg_land', 'oscar_var', 'value'})
    codes, aliases = model_regions(region)
    unknown = sorted(set(df.reg_land) - set(aliases))
    if unknown:
        raise ValueError(f'Unknown land-use regions {unknown}; use {region} region names (no World totals).')
    df['reg_land'] = df.reg_land.map(aliases)
    pieces = []
    for var, rows in df.groupby('oscar_var'):
        if var not in LAND_USE:
            raise ValueError(f'{var} is not a land-use input. Supported: {", ".join(LAND_USE)}.')
        if 'unit' in rows and set(rows.unit.astype(str).str.strip()) != {LAND_USE[var]}:
            raise ValueError(f'{var} units must be {LAND_USE[var]!r}.')
        biomes = ['bio_land'] if var.startswith('D_dA_wharv') else ['bio_from', 'bio_to']
        if not set(biomes) <= set(rows):
            raise ValueError(f'{var} rows need columns {", ".join(biomes)}.')
        index = ['scen', 'year', 'reg_land'] + biomes
        rows = rows.assign(**{b: rows[b].astype(str).str.strip() for b in biomes})
        if rows.duplicated(index).any():
            raise ValueError(f'Duplicate {var} rows found.')
        da = rows.set_index(index)['value'].to_xarray().rename(var)
        da.attrs['units'] = LAND_USE[var]
        pieces.append(da)
    if not pieces:
        raise ValueError('Land-use CSV contains no rows.')
    return xr.merge(pieces, join='outer')


def read_netcdf(source, units=None):
    """Inputs from a NetCDF file or an in-memory xr.Dataset; each variable needs year and units.

    units: {variable: unit} accepted (default: the customized inputs of input_species.yaml).
    """
    units = units or UNITS
    if isinstance(source, xr.Dataset):
        ds = source.copy()
    else:
        with xr.open_dataset(source) as ds:
            ds = ds.load()
    if not ds.data_vars or 'year' not in ds.dims:
        raise ValueError('Custom forcing must contain variables and a year dimension.')
    if 'config' in ds.dims:
        raise ValueError('Custom forcing must be shared across the ensemble; omit the config dimension.')
    for var in ds.data_vars:
        if var not in units:
            raise ValueError(f'Unsupported input {var!r}. Supported: {", ".join(units)}.')
        if ds[var].attrs.get('units') != units[var]:
            raise ValueError(f'{var} units must be {units[var]!r}.')
        if var == 'D_Eant_Xhalo' and 'spc_halo' not in ds[var].dims:
            raise ValueError('D_Eant_Xhalo needs a spc_halo dimension.')
    years = ds.year.values
    if np.any(years != years.astype(int)) or np.any(np.diff(years) <= 0):
        raise ValueError('Custom forcing years must be unique, increasing integers.')
    if 'scen' in ds.dims and len(set(map(str, ds.scen.values))) != ds.sizes['scen']:
        raise ValueError('Custom scenario names must be unique.')
    if 'scen' not in ds.dims:
        name = str(ds.scen.item()) if 'scen' in ds.coords else 'my-scenario'
        ds = ds.drop_vars('scen', errors='ignore').expand_dims(scen=[name])
    # Emissions and radiative forcing are used as global totals.
    for var in ds.data_vars:
        if var not in LAND_USE and 'reg_land' in ds[var].dims:
            ds[var] = ds[var].sum('reg_land', min_count=1).assign_attrs(ds[var].attrs)
    return ds


# --- Checks against the projection period ---------------------------------------

def _ranges(years):
    """[2024, 2025, 2026, 2030] -> '2024–2026, 2030'."""
    years, out = sorted(int(y) for y in years), []
    for year in years:
        if out and year == out[-1][1] + 1:
            out[-1][1] = year
        else:
            out.append([year, year])
    return ', '.join(f'{a}–{b}' if a != b else f'{a}' for a, b in out)


def _series(da):
    """Supplied series of a variable: one per halogen species, otherwise one per variable."""
    if 'spc_halo' in da.dims:
        return [(f'{da.name} ({s})', da.sel(spc_halo=s)) for s in da.spc_halo.values
                if bool(da.sel(spc_halo=s).notnull().any())]
    return [(da.name, da)]


def supplied(da):
    """True where a scenario supplies this variable at all (any value in any year)."""
    return bool(da.notnull().any())


def copy_last_year(ds, end):
    """Copy the values of end-1 to end where end is missing (libraries and many inputs stop in 2099).

    Only the final year is filled; any other gap is left to the coverage check.
    """
    if 'year' not in ds.dims or end - 1 not in ds.year.values:
        return ds
    if end not in ds.year.values:
        ds = xr.concat([ds, ds.sel(year=[end - 1]).assign_coords(year=[end])], dim='year',
                       data_vars='minimal', coords='minimal', compat='override')
    ds = ds.copy()
    for var in ds.data_vars:
        if 'year' in ds[var].dims:
            da = ds[var]
            filled = da.sel(year=end).fillna(da.sel(year=end - 1))
            ds[var] = xr.where(da.year == end, filled, da).transpose(*da.dims).assign_attrs(da.attrs)
    return ds


def check_full_coverage(user, first, last):
    """Each supplied variable (each halogen species) needs a value in every year first..last.

    A variable a scenario does not supply at all is filled from the baseline. Land use
    may cover only some regions and biomes (the others mean no change).
    """
    needed = set(range(first, last + 1))
    if 'D_Eluc' in user and len({supplied(user.D_Eluc.sel(scen=s)) for s in user.scen.values}) > 1:
        raise ValueError('D_Eluc must be supplied for every scenario in your inputs, or for none.')
    for scen in user.scen.values:
        for var in user.data_vars:
            if not supplied(user[var].sel(scen=scen)):
                continue
            for label, da in _series(user[var].sel(scen=scen)):
                present = da.notnull()
                for dim in set(da.dims) - {'year'}:
                    present = present.any(dim) if var in LAND_USE else present.all(dim)
                covered = {int(y) for y in da.year.values[present.values]}
                missing = needed - covered
                if missing:
                    raise ValueError(
                        f'{label} in scenario "{scen}" is missing {_ranges(missing)}. OSCAR needs a continuous '
                        f'annual time series covering {first}–{last}; please provide the full time series.')


def _global(da):
    """Global total per year (and scenario, halogen species); ensemble mean where members differ."""
    for dim in set(da.dims) - {'year', 'scen', 'spc_halo'}:
        da = da.mean(dim) if dim == 'config' else da.sum(dim, min_count=1)
    return da


def continuity_warnings(user, history_forcing, scenario_forcing, history_output, hist_end):
    """Warn (values unchanged) when a series jumps at the history/projection boundary.

    Allowed jump: twice the largest jump at that boundary among the library scenarios,
    and never less than 10% of the historical value. For a variable the model computes
    (not in the library forcing, e.g. D_Eluc in CMIP), the reference is the largest
    year-to-year change over the last 10 historical years.
    """
    first = hist_end + 1
    for var in user.data_vars:
        if var not in history_forcing:
            # Computed by the model (e.g. D_Eluc in CMIP): compare with the computed history.
            hist = _global(history_output[var])
            past = hist.sel(year=slice(hist_end - 10, hist_end))
            last = hist.sel(year=hist_end)
            reference = abs(past.diff('year')).max('year')
        else:
            last = _global(history_forcing[var].sel(year=hist_end))
            reference = abs(_global(scenario_forcing[var].sel(year=first)) - last)
            if 'scen' in reference.dims:
                reference = reference.max('scen')
        for scen in user.scen.values:
            if not supplied(user[var].sel(scen=scen)):
                continue
            value = _global(user[var].sel(scen=scen, year=first))
            for label, da in _series(value.rename(var)):
                # da is one series (one halogen species already selected); pick the same in the references.
                select = {'spc_halo': da.spc_halo.item()} if 'spc_halo' in da.coords else {}
                h, ref, u = float(last.sel(select)), float(reference.sel(select)), float(da)
                allowed = max(2 * ref, 0.1 * abs(h))
                if abs(u - h) > allowed:
                    print(f'[OSCAR] Warning: {label} in "{scen}" jumps from {h:.4g} ({hist_end}, history) to '
                          f'{u:.4g} ({first}, your input); allowed change ±{allowed:.3g}. '
                          'Check that your series continues from the historical data.')
