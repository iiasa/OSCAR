"""
OSCAR Help Utility
Location: oscar/_utils/help.py

Provides contextual information about run modes, official scientific 
scenarios, and regional configurations.
"""
from .load_config import load_config

def show_info(mode=None):
    """Router for the OSCAR information system."""
    cfg_full = load_config()
    
    # Normalize mode string
    m = str(mode).lower() if mode else None

    if m is None or m == 'none':
        _print_general()
    elif m == 'standard':
        _print_standard(cfg_full)
    elif m == 'configured':
        _print_configured(cfg_full)
    elif m == 'customized':
        _print_customized(cfg_full)
    elif m == 'advanced':
        _print_advanced()
    else:
        print(f"\n[!] Unknown mode: '{mode}'")
        print("Available modes are: 'standard', 'configured', 'customized', 'advanced'")

def _print_general():
    """Displays general overview and entry points for OSCAR."""
    width = 105
    print("\n" + "=" * width)
    print(f"{'OSCAR MODEL - GENERAL OVERVIEW':^105}")
    print("=" * width)
    print(f"{'A reduced-complexity Earth system model for climate research.':^105}")

    print("\nAvailable Run Modes ([T] = Terminal, [P] = Python):")

    # Column Width Definitions
    mode_w = 10
    desc_w = 30
    term_w = 25

    print(f"  {'standard':<{mode_w}} : {'Fast verification (no setup).':<{desc_w}} | "f"{'[T] oscar info standard':<{term_w}} | [P] oscar.info('standard')")
    print(f"  {'configured':<{mode_w}} : {'Official CMIP runs.':<{desc_w}} | "f"{'[T] oscar info configured':<{term_w}} | [P] oscar.info('configured')")
    print(f"  {'customized':<{mode_w}} : {'User-defined research.':<{desc_w}} | "f"{'[T] oscar info customized':<{term_w}} | [P] oscar.info('customized')")
    print(f"  {'advanced':<{mode_w}} : {'[DEV] Advanced features.':<{desc_w}} | "f"{'(Status: Under Development)':^53} ")

    print("=" * width + "\n")

def _print_standard(cfg_full):
    """Displays information for the Standard verification mode."""
    cfg = cfg_full['standard_mode']
    width = 90
    print("\n" + "="*width)
    print(f"{'MODE: STANDARD (Verification)':^90}")
    print("="*width)
    print("Goal:        Fast proof-of-concept run for the model installation.")
    print(f"Data source: Internal package bootstrap ({cfg['nMC']} members).")
    print(f"Timeline:    {cfg['run_range'][0]} to {cfg['run_range'][1]}.")
    print(f"History:     {cfg['hist_type']}")
    print(f"Region:      Fixed ({cfg['region']}).")
    
    print("\nExample Commands:")
    print("  [Terminal] : oscar run")
    print("  [Python]   : import oscar; oscar.run()")
    print("="*width + "\n")

def _print_configured(cfg_full):
    """Displays official scientific library options."""
    from oscar._io.paths import PACKAGE_ROOT
    from oscar._utils.metadata import load_var_registry

    var_path = (PACKAGE_ROOT / "oscar" / "_utils" / "_resources" / "variables.yaml")

    cfg = cfg_full['configured_mode']
    reg = cfg_full.get("registry", {})
    defaults = cfg.get("defaults", {})

    width = 105
    print("\n" + "="*width)
    print(f"{'MODE: CONFIGURED (Scientific Library)':^105}")
    print("="*width)
    print("Official scientific experiments using curated forcing and parameter libraries.")
    
    print("\nAvailable Options :")
    print(f"  {'Histories':<12} : {', '.join(cfg['allowed_hist'])}")
    print(f"  {'Regions':<12} : {', '.join(cfg['allowed_regions'])}")

    # --- CMIP6 & CMIP7 Scenarios ---
    all_scens = reg.get("all_scens", {})
    cmip6_scens = all_scens.get("scen6", [])
    cmip7_scens = all_scens.get("scen7", [])

    if cmip6_scens:
        print(f"  {'CMIP6 Scen':<12} : {', '.join(cmip6_scens)}")
    if cmip7_scens:
        print(f"  {'CMIP7 Scen':<12} : {', '.join(cmip7_scens)}")
    
    # --- Output Variables & Core Metadata Lookup ---
    var_meta = load_var_registry()
    v_core = reg.get("v_core", [])


    print(f"  {'Output Vars':<12} : Full list in {var_path}")
    print(f"{'':<17}[ Core Variables: ]")

    for var in v_core:
        meta = var_meta.get(var, {})
        desc = meta.get("long_name", "Core variable")
        unit = meta.get("unit", "")
        unit_str = f" [{unit}]" if unit else ""
        print(f"{'':<15}  - {var:<10}: {desc}{unit_str}")

    #print(f"  {'MC Ensemble':<12} : {cfg.get('official_nMC', 'N/A')} members (Pre-validated)")

    # --- Extract Dynamic Example Options from Config ---
    ex_scen = defaults.get("scenario", ["scen7-VL"])
    ex_scen = ex_scen[0] if isinstance(ex_scen, list) else ex_scen

    ex_reg = defaults.get("region", "IAMC_R5")
    ex_reg = ex_reg[0] if isinstance(ex_reg, list) else ex_reg

    ex_var = defaults.get("var_select", ["D_Tg"])
    ex_var = ex_var[0] if isinstance(ex_var, list) else ex_var

    print("\nExample Commands:")
    print(f"  [Terminal] : oscar run -m configured -s {ex_scen} -r {ex_reg} -v {ex_var}")
    print(f"  [Python]   : oscar.run(mode='configured', scenario='{ex_scen}', variables=['{ex_var}'])")

    print("=" * width + "\n")

def _print_customized(cfg_full):
    """Displays instructions for research mode (Tier 2)."""
    cfg = cfg_full['customized_mode']
    width = 95
    print("\n" + "="*width)
    print(f"{'MODE: CUSTOMIZED (User Research)':^95}")
    print("="*width)
    print("Run OSCAR with custom experimental forcing and extended horizons.")
    
    print("\n[ SCIENTIFIC BOUNDARIES ]")
    y_min = cfg['allowed_years']['min']
    y_max = cfg['allowed_years']['max']
    print(f"  {'Timeline':<17}: {y_min} to {y_max}")
    print(f"  {'Supported Hist':<17}: {', '.join(cfg['allowed_hist'])}")
    print(f"  {'Output Regions':<17}: {', '.join(cfg['allowed_regions'])}")
    print(f"  {'Baseline Forcing':<17}: ")
    print(f"  {'    - CMIP6' }[{', '.join(cfg_full['registry']['all_scens']['scen6'])}]")
    print(f"  {'    - CMIP7' }[{', '.join(cfg_full['registry']['all_scens']['scen7'])}]")

    print("\n[ OUTPUT THEMES ]")
    themes = cfg['output_themes']
    for theme, raw_variables in themes.items():
        if isinstance(raw_variables, list):
            # --- FIX: FLATTEN THE THEME LIST ---
            flat_theme_vars = []
            for item in raw_variables:
                if isinstance(item, list):
                    flat_theme_vars.extend(item)
                else:
                    flat_theme_vars.append(item)
            # -----------------------------------
            var_str = ", ".join(flat_theme_vars)
        else:
            var_str = "(User-defined selection)"
        print(f"  - {theme:<15}: {var_str}")

    from oscar._io.paths import PACKAGE_ROOT
    var_path = (PACKAGE_ROOT / "oscar" / "_utils" / "_resources" / "variables.yaml")
    print(f"  {'':<19}(Full list in {var_path})")

    print("\n[ USAGE - FOLLOW THESE STEPS ]")
    print("  0. Download Templates    : oscar.download(mode='customized')")
    print("  1. Create Project Folder : oscar.create_project('my-project')")
    print("  2. Edit setting file in the created folder (e.g., settings_my-experiment.yaml)")
    print("  3. Run Simulation        : oscar.run(mode='customized', project='my-project', experiment='my-experiment')")
    print("="*width + "\n")

def _print_advanced():
    """Information for Advanced development mode."""
    width = 90
    print("\n" + "="*width)
    print(f"{'MODE: ADVANCED (Model Development)':^90}")
    print("="*width)
    print("Direct control over model sub-modules and calibration internals.")
    print("\n( ! ) STATUS: Internal development only.")
    print("\nFuture Capabilities:")
    print("  - Selective Execution (Land only, Climate only)")
    print("  - Custom Calibration & Numerical Tuning")
    print("="*width + "\n")