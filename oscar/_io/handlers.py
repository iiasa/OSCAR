"""
OSCAR Data Handlers
Architecture: Specialized Variable Handlers + Specialized Regional Handlers
"""
import pandas as pd
import xarray as xr
import numpy as np
import re
import yaml
from .paths import PACKAGE_ROOT

# --- SHARED UTILS ---

def load_var_mapping(format_type):
    """Loads the translation registry from resources."""
    map_name = f"vars_{format_type}_map.yaml"
    path = PACKAGE_ROOT / "oscar" / "_resources" / map_name
    if not path.exists():
        raise FileNotFoundError(f"[OSCAR] Missing mapping registry: {map_name}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _melt_and_clean(csv_path):
    """
    Standardizes Wide/Long CSV with robust cleaning.
    Normalizes 'Variable' strings to prevent whitespace/case mismatches.
    """
    # 1. Read the file (Auto-separator, Comma-decimal)
    df = pd.read_csv(csv_path, sep=None, decimal=',', engine='python')

    # 2. Clean column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    # 3. IDENTIFY YEAR COLUMNS (Numeric headers)
    year_cols = [c for c in df.columns if str(c).replace('.', '').isdigit()]
    
    if year_cols:
        # Wide Format
        id_vars = [c for c in df.columns if c not in year_cols]
        df = df.melt(id_vars=id_vars, value_vars=year_cols, var_name='Year', value_name='Value')
    else:
        # Long Format: ensure standardized capitalization for core columns only
        # We DON'T capitalize 'Variable' content here, only the column name
        rename_map = {c: c.title() for c in df.columns if c.lower() in ['scenario', 'variable', 'region', 'unit']}
        df = df.rename(columns=rename_map)

    # --- STEP 4: CLEAN ALL TEXT COLUMNS ---
    text_cols = [c for c in df.columns if c not in ['Year', 'Value']]
    
    for col in text_cols:
        # 1. Convert to string
        df[col] = df[col].astype(str)
        
        # 2. REMOVE ALL WHITESPACE (Newlines, tabs, spaces)
        # We use .str.replace and remove the trailing .strip() 
        # because \s+ already catches leading/trailing whitespace.
        df[col] = df[col].str.replace(r'\s+', '', regex=True)

    # Special handling for decimal artifacts in Regions (e.g., '1.0' -> '1')
    if 'Region' in df.columns:
        df['Region'] = df['Region'].str.replace(r'\.0$', '', regex=True)

    # Ensure Value is float and Year is int
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce').fillna(0.0)
    
    return df

# --- STEP 1: VARIABLE TRANSLATORS ---

def handle_atmospheric_vars(csv_path, format_type):
    """
    Step 1A: Chemical and Unit Translation for Emissions and RF.
    Includes diagnostic info for non-matching variables.
    """
    import re
    df = _melt_and_clean(csv_path)
    mapping = load_var_mapping(format_type)

    df['oscar_var'] = None
    df['oscar_id'] = None

    # 1. Normalize CSV Variable names (Squeeze whitespace/newlines)
    df['Variable'] = df['Variable'].astype(str).str.replace(r'\s+', '', regex=True)

    # Track which registry keys were successfully matched
    matched_keys = set()

    for cat in ['emissions', 'radiative_forcing']:
        if cat not in mapping: continue
        
        for registry_key, data in mapping[cat].items():
            # 2. Normalize Registry Key (Squeeze whitespace/newlines)
            clean_reg_key = re.sub(r'\s+', '', str(registry_key))
            
            mask = df['Variable'] == clean_reg_key
            
            if mask.any():
                o_var, o_id, math_str = data[0], data[1], data[2]
                df.loc[mask, 'Value'] = pd.to_numeric(df.loc[mask, 'Value']) * pd.eval(math_str)
                df.loc[mask, 'oscar_var'] = o_var
                df.loc[mask, 'oscar_id'] = o_id
                matched_keys.add(registry_key)
                #print(df[['Variable', 'Year', 'Value','oscar_var', 'oscar_id']].head(15))  # Debug: Show the first few matches for this registry key

    # --- 3. INFORMATION BLOCK: IDENTIFY MISMATCHES ---
    '''
    all_registry_keys = set()
    for cat in ['emissions', 'radiative_forcing']:
        if cat in mapping:
            all_registry_keys.update(mapping[cat].keys())

    missing = all_registry_keys - matched_keys
    if missing:
        print(f"\n[OSCAR INFO] The following {len(missing)} registry variables were NOT found in the CSV:")
        # Print first 5 missing for brevity
        for m in sorted(list(missing)):
            print(f"  - Missing: {m}")
        if len(missing) > 5:
            print(f"  ... and {len(missing)-5} others.")
    '''

    # 4. Filter and Split
    df = df.dropna(subset=['oscar_var'])
    if df.empty:
        return xr.Dataset()

    # --- SEPARATION LOGIC ---
    # Group A: Standard (Eff, E_CH4, RF_solar, etc.) - oscar_var and oscar_id are the same
    # These should NOT have a species dimension in the final Dataset
    df_std = df[df['oscar_var'] == df['oscar_id']]

    # Group B: Halogens (E_Xhalo) - oscar_var is 'E_Xhalo', oscar_id is the compound name
    # These MUST have the spc_halo dimension
    df_halo = df[df['oscar_var'] == 'E_Xhalo']

    datasets = []

    # Process Standard variables
    if not df_std.empty:
        # We pivot using 'oscar_var' as both column and index to keep it 3D (Scen, Year, Reg)
        ds_std = df_std.pivot_table(
            index=['Scenario', 'Year', 'Region'], 
            columns='oscar_var', 
            values='Value',
            aggfunc='sum'
        ).to_xarray()
        datasets.append(ds_std)

    # Process Halogens
    if not df_halo.empty:
        ds_halo = df_halo.pivot_table(
            index=['Scenario', 'Year', 'Region', 'oscar_id'], 
            columns='oscar_var', 
            values='Value',
            aggfunc='sum'
        ).to_xarray()
        # Rename the extra dimension to spc_halo
        ds_halo = ds_halo.rename({'oscar_id': 'spc_halo'})
        datasets.append(ds_halo)

    # Merge the two groups
    if not datasets:
        return xr.Dataset()
        
    ds = xr.merge(datasets)
    
    # Standardize remaining dimension names
    ds = ds.rename({'Scenario': 'scen', 'Year': 'year', 'Region': 'reg_input'})
    
    # --- DEBUG PRINT ---
    '''
    print("\n--- XARRAY STRUCTURE CHECK ---")
    for v in ds.data_vars:
        print(f"Variable: {v:10} | Dimensions: {ds[v].dims}")
    '''
        
    return ds

def handle_lulcc_vars(csv_path):
    """
    Step 1B: Translates LULCC variables into a 5D pivot structure.
    """
    df = _melt_and_clean(csv_path)
    ds = df.pivot_table(index=['Scenario', 'Year', 'Region', 'Bio_from', 'Bio_to'], 
                        columns='Variable', values='Value').to_xarray()
    
    return ds.rename({'Scenario': 'scen', 'Year': 'year', 'Region': 'reg_input'})

# --- STEP 2: REGIONAL HANDLERS ---

# --- SHARED REGIONAL UTILS ---

def _load_allowed_regions(mod_region):
    """
    Retrieves the official list of region names for a specific resolution.
    Preserves the 0, 1, 2... order from the CSV rows.
    """
    import csv
    
    reg_meta_path =  PACKAGE_ROOT / "oscar" / "_resources" / "regions_long_name.csv"
    with open(reg_meta_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        if mod_region not in header:
            raise ValueError(f"[OSCAR] Resolution '{mod_region}' not found in metadata.")
        
        col_idx = header.index(mod_region)
        allowed = []
        for row in reader:
            name = row[col_idx].strip()
            # Preserve order and ensure uniqueness
            if name and name not in allowed:
                allowed.append(name)
    return allowed

# --- REGIONAL HANDLERS ---

def handle_atmospheric_reg(ds, mod_region):
    """
    Step 2A: Regional aggregation for Atmospheric drivers.
    Logic: Matches squeezed CSV regions to official model regions.
    """
    import re
    import numpy as np

    # 1. Load official regions (e.g., ['World', 'Middle East', ...])
    raw_allowed = _load_allowed_regions(mod_region)
    
    # 2. CREATE A SQUEEZED MAP
    # Maps "MiddleEast" (CSV) -> "Middle East" (Official)
    clean_to_official = {re.sub(r'\s+', '', str(a)): str(a) for a in raw_allowed}
    clean_allowed_names = list(clean_to_official.keys())
    
    # Standardize global aliases
    world_aliases = ['World', 'Global', 'Globe', 'Unknown']

    # 3. IDENTIFY MATCHES
    # ds.reg_input contains the regions from your pivot_table
    user_regions = [str(r) for r in ds.reg_input.values]
    valid_user_names = [r for r in user_regions if r in clean_allowed_names or r in world_aliases]

    if not valid_user_names:
        print(f"\n[!] REGIONAL ERROR for {mod_region}:")
        print(f"    Regions found in Data: {user_regions}")
        print(f"    Expected (squeezed):   {clean_allowed_names[:5]}...")
        raise ValueError(f"[OSCAR] No valid regions found for {mod_region}.")

    # 4. FILTER AND SUM
    # This aggregates all valid input regions into a single Global total
    ds_valid = ds.sel(reg_input=valid_user_names)
    #print(ds_valid[['Eff']].isel(year=0, reg_input=0))  # Debug: Check the first few values of the valid dataset
    ds_glob = ds_valid.sum('reg_input', min_count=1)

    # 5. ALIGN WITH MODEL RESOLUTION
    # OSCAR expects a 'reg_land' dimension with specific length
    full_indices = np.arange(len(raw_allowed))
    ds_out = ds_glob.expand_dims(reg_land=full_indices).copy()
    
    for var in ds_out.data_vars:
        # Place the total in Index 0 (Global/First Region), set others to 0.0
        mask = ds_out.reg_land != 0
        ds_out[var] = ds_out[var].where(~mask, 0.0)

    # --- DEBUG: CHECK IF DATA SURVIVED ---
    '''
    if 'Eff' in ds_out:
        test_val = ds_out['Eff'].sel(year=ds_out.year.min(), reg_land=0).values
        print(f"[DEBUG] handle_atmospheric_reg: Year {ds_out.year.min().values} Eff = {test_val} PgC")
    '''

    # 6. TEMPORAL INTERPOLATION
    # Fills gaps (like the 2015 gap) and ensures yearly resolution
    year_range = np.arange(int(ds_out.year.min()), int(ds_out.year.max()) + 1)
    return ds_out.interp(year=year_range, method="linear").fillna(0.0)


def handle_lulcc_reg(ds, mod_region):
    """
    Step 2B: Regional alignment for LULCC drivers.
    Logic: STRICT MATCHING. Reindexes to full model shape.
    """
    allowed_list = _load_allowed_regions(mod_region)
    user_regions = [str(r) for r in ds.reg_input.values]
    
    # --- MISMATCH CHECK ---
    valid_names = [r for r in user_regions if r in allowed_list]
    mismatched = [r for r in user_regions if r not in allowed_list]

    if mismatched:
        print(f"\n[!] WARNING: LULCC data for these regions was NOT used:")
        print(f"    >> {', '.join(mismatched)}")
        print(f"    Reason: Names do not match the '{mod_region}' official list.")

    if not valid_names:
        raise ValueError(f"[OSCAR ERROR] No valid regions found in LULCC data for {mod_region}.")

    # 1. Map names to official numeric indices
    name_to_idx = {name: i for i, name in enumerate(allowed_list)}
    numeric_coords = [name_to_idx[r] for r in valid_names]
    
    # 2. Slice and reindex
    ds_final = (ds.sel(reg_input=valid_names)
                  .assign_coords(reg_input=numeric_coords)
                  .rename({'reg_input': 'reg_land'})
                  .reindex(reg_land=np.arange(len(allowed_list)), fill_value=0.0))

    # 3. Yearly Interpolation
    year_range = np.arange(int(ds_final.year.min()), int(ds_final.year.max()) + 1)
    return ds_final.interp(year=year_range, method="linear").fillna(0.0)

# --- MASTER COMPILER ---

# --- MASTER COMPILER ---

def compile_custom_forcing(project_path, user_csv_map, mod_region):
    """
    Master orchestrator for Level-2 runs.
    Strictly identifies format (ceds vs iamc-ar6-public).
    """
    ds_final = xr.Dataset()

    # 1. Process Atmosphere
    atmo_cfg = user_csv_map.get('atmospheric', {})
    if atmo_cfg.get('file'):
        # DYNAMIC FORMAT DETECTION:
        # Check if 'format' exists in the configuration
        atmo_format = atmo_cfg.get('format')
        
        if not atmo_format:
            print("\n[!] ERROR: Atmospheric forcing format is missing in the configuration.")
            print("Please specify 'format: ceds' or 'format: iamc-ar6-public'.")
            raise ValueError("Atmospheric forcing format missing.")

        print(f"- Compiling atmospheric forcing [Format: {atmo_format}]...")
        
        ds_atmo_raw = handle_atmospheric_vars(
            project_path / atmo_cfg['file'],
            format_type=atmo_format
        )
        
        ds_atmo_final = handle_atmospheric_reg(ds_atmo_raw, mod_region)
        ds_final = xr.merge([ds_final, ds_atmo_final])

    # 2. Process Land-use (Assuming lulcc doesn't require strict format check yet)
    lulcc_cfg = user_csv_map.get('lulcc', {})
    if lulcc_cfg.get('file'):
        print(f"- Compiling land-use forcing...")
        ds_lulcc_raw = handle_lulcc_vars(project_path / lulcc_cfg['file'])
        ds_lulcc_final = handle_lulcc_reg(ds_lulcc_raw, mod_region)
        ds_final = xr.merge([ds_final, ds_lulcc_final])

    return ds_final