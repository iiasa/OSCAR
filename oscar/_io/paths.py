"""
OSCAR Path Management Utility
Location: oscar/_io/paths.py
"""

import json
import shutil
import platform
from pathlib import Path
from typing import Optional, Dict


# 1. BASE REPO PATHS (Internal to the code structure)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent #package root path
SETTINGS_FILE = PACKAGE_ROOT / ".oscar_settings.json" #user settings file for data directory
INTERNAL_BOOTSTRAP_DIR = PACKAGE_ROOT / "oscar" / "_utils" / "_resources" / "bootstrap" #set bootstrap path, used for 'standard' mode
REGIONS_INFO_PATH = PACKAGE_ROOT / "oscar" / "_core" / "_regions"  #set region information path


# 2. DYNAMIC ROOT RESOLUTION
# set user data directory
def set_data_dir(path):
    """
    Permanently sets the data directory. 
    A specific path must be provided.
    """
    target = Path(path).expanduser().resolve()
    
    target.mkdir(parents=True, exist_ok=True)
    
    with open(SETTINGS_FILE, "w") as f:
        json.dump({"data_dir": str(target)}, f, indent=4)
        
    print(f"[OSCAR] User data directory set to: {target}")


# get user data directory
def get_user_data_dir() -> Optional[Path]:
    """Quietly retrieves the saved data directory, preferring OS-specific settings."""
    if not SETTINGS_FILE.exists():
        return None
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
        path = data.get(f"data_dir_{platform.system()}") or data.get("data_dir")
        if path:
            return Path(path).expanduser().resolve()
    except (KeyError, json.JSONDecodeError, OSError):
        pass    
    return None


def create_project(project_name: str) -> Optional[Path]:
    """Automates the creation of a research sandbox."""
    paths = get_paths()
    if not paths:
        print("[OSCAR] Cannot create project: No data directory configured.")
        return None

    p_path = paths["projects"] / project_name
    if p_path.exists():
        print(f"[OSCAR] Folder '{project_name}' already exists.")
        return p_path

    print(f"[OSCAR] Initializing project: {project_name}")
    p_path.mkdir(parents=True, exist_ok=True)
    (p_path / "results").mkdir(exist_ok=True)

    # Copy Templates from customized library
    tpl_file = paths["customized_library"] / "templates" / "settings_template.yaml"
    if tpl_file.exists():
        shutil.copy(tpl_file, p_path / "settings_my-experiment.yaml")
        print(f"[OSCAR] Setup complete! Edit files in: {p_path}")
    else:
        print("[OSCAR] Warning: Library templates not found. Manual setup required.")
    
    return None


def validate_config() -> bool:
    """
    The Single Check for UI/Startup.
    Handles three states:
    1. Path not set -> Returns False (Quietly, handled by run.py Welcome Guide)
    2. Path set but does not exist -> Prints Warning, Returns False
    3. Path OK -> Returns True
    """
    path = get_user_data_dir()
    
    # State 1: Not set at all
    if path is None:
        return False
    
    # State 2: Set but path was deleted or drive unplugged
    if not path.exists():
        print(f"\n{'!'*60}")
        print(f"[!] WARNING: Your saved data directory is no longer accessible:")
        print(f"    Path: {path}")
        print("\nPlease review the path or set a new one by running:")
        print("    oscar.set_data_dir('/new/path')")
        print(f"{'!'*60}\n")
        return False
    
    # State 3: Everything is fine
    return True


# 3. PATH REGISTRY (The Source of Truth)
def get_paths() -> Optional[Dict[str, Path]]:
    """
    Returns a dictionary of all standard paths.
    Returns None if no data directory is configured.
    """
    data_root = get_user_data_dir()
    if not data_root:
        return None
    
    lib = data_root / "library"
    lib_core_data = lib / "core_data"
    return {
        "data_root": data_root,
        "library": lib,
        # Scientific Library Core Data Paths
        "core_data": lib_core_data,
        "drivers_fixed": lib_core_data / "drivers_fixed",
        "drivers_latest": lib_core_data / "drivers_latest",
        "observations": lib_core_data / "observations",
        "params_precalib": lib_core_data / "params_precalib",
        "precalib_plots": lib_core_data / "precalib_control_plots",
        "precalib_data": lib_core_data / "precalib_data",
        # Scientific Library User Assets Paths
        "configured_library": lib / "run_mode" / "tier1_configured",
        "customized_library": lib / "run_mode" / "tier2_customized",
        # Research & Results
        "projects": data_root / "projects",
        "results": data_root / "results",
        # Developer folders
        "dev_data": data_root / "dev_data",
        "setup_data": data_root / "dev_data" / "setup_data",
    }

