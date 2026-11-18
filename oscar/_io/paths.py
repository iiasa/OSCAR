"""
OSCAR Path Management Utility
Location: oscar/_io/paths.py
"""

import json
import shutil
import platform
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Optional, Dict


# 1. BASE REPO PATHS (Internal to the code structure)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent #package root path
SETTINGS_FILE = PACKAGE_ROOT / ".oscar_settings.json"  # user settings: data folder and user paths (git-ignored)
_DATA_OVERRIDE = ContextVar("oscar_data_dir", default=None)
INTERNAL_BOOTSTRAP_DIR = PACKAGE_ROOT / "oscar" / "_utils" / "_resources" / "bootstrap" #set bootstrap path, used for 'standard' mode
REGIONS_INFO_PATH = PACKAGE_ROOT / "oscar" / "_core" / "_regions"  #set region information path
DEFAULT_DATA_DIR = PACKAGE_ROOT / "oscar_data"  # used when no data folder has been set (as in beta2)


# 2. DYNAMIC ROOT RESOLUTION
# set user data directory
# Folders users may move (their projects, a shared copy of an advanced library); saved in .oscar_settings.json by set_data_dir.
USER_PATHS = ("projects", "advanced_library")


def _user_paths(data_root):
    lib = Path(data_root) / "library" / "run_mode"
    return {"projects": Path(data_root) / "projects", "results": Path(data_root) / "results",
            "configured_library": lib / "configured", "customized_library": lib / "customized",
            "advanced_library": lib / "advanced"}


def set_data_dir(path=None):
    """
    Permanently sets the data directory, and saves it with the folders derived from it
    (USER_PATHS) in .oscar_settings.json; edit a folder there to use it instead (e.g. a shared library).
    Defaults to the oscar_data folder next to the oscar package. OSCAR_DATA_DIR
    overrides this saved choice, and data_dir= on a run overrides both for that call.
    """
    target = Path(path or DEFAULT_DATA_DIR).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    settings = {"data_dir": str(target), **{k: str(_user_paths(target)[k]) for k in USER_PATHS}}
    SETTINGS_FILE.write_text(json.dumps(settings, indent=4) + "\n")
    print(f"[OSCAR] User data directory set to: {target}\n        Paths saved in {SETTINGS_FILE}")
    return target


def _saved_settings():
    """Saved settings (data_dir, user paths), or {}; data_dir_<system> is preferred when present."""
    if not SETTINGS_FILE.exists():
        return {}
    try:
        data = json.loads(SETTINGS_FILE.read_text())
        path = data.get(f"data_dir_{platform.system()}") or data.get("data_dir")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid OSCAR settings: {SETTINGS_FILE}. Run oscar init --data-dir PATH.") from exc
    return {**data, "data_dir": path} if path else {}


def _chosen_data_dir() -> Optional[Path]:
    """Data directory chosen by the user (call override, OSCAR_DATA_DIR or saved settings)."""
    override = _DATA_OVERRIDE.get() or os.environ.get("OSCAR_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    saved = _saved_settings().get("data_dir")
    return Path(saved).expanduser().resolve() if saved else None


# get user data directory
def get_user_data_dir() -> Path:
    """The data directory: the user's choice, otherwise oscar_data next to the oscar package."""
    return _chosen_data_dir() or DEFAULT_DATA_DIR.resolve()


_NOTICE_SHOWN = False


def notice_default_data_dir():
    """Tell the user (once per session) when the default data folder is being used."""
    global _NOTICE_SHOWN
    if _chosen_data_dir() is None and not _NOTICE_SHOWN:
        print(f"[OSCAR] No data folder set; using {DEFAULT_DATA_DIR.resolve()}\n"
              "        Change it with: oscar init --data-dir PATH")
        _NOTICE_SHOWN = True


@contextmanager
def use_data_dir(path=None):
    token = _DATA_OVERRIDE.set(Path(path).expanduser().resolve() if path else _DATA_OVERRIDE.get())
    try:
        yield
    finally:
        _DATA_OVERRIDE.reset(token)


def validate_name(name, label="name"):
    import re
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"Invalid {label}: use letters, digits, dots, hyphens or underscores; start with a letter or digit.")
    return name


def create_project(project_name: str, experiment: str, data_dir=None) -> Path:
    """Create <project-id>/ (a starting kit) with settings_<experiment-id>.yaml; an existing
    project gains a new experiment."""
    validate_name(project_name, "project name")
    validate_name(experiment, "experiment name")
    with use_data_dir(data_dir):
        paths = get_paths()
    p_path = paths["projects"] / project_name
    target = p_path / f"settings_{experiment}.yaml"
    if target.exists():
        raise FileExistsError(f"Project settings already exist: {target}")
    from oscar._io._download import ensure_customized_library
    with use_data_dir(data_dir):
        library = ensure_customized_library()
    p_path.mkdir(parents=True, exist_ok=True)
    (p_path / "results").mkdir(exist_ok=True)
    (p_path / "inputs").mkdir(exist_ok=True)
    # Copy the template as text so its comments reach the user.
    shutil.copyfile(library / "templates" / "settings_template.yaml", target)
    info = library / "templates" / "INFO.md"
    if info.is_file() and not (p_path / "INFO.md").exists():
        shutil.copy(info, p_path / "INFO.md")
    print(f"[OSCAR] Templates and examples: {library}")
    print(f"[OSCAR] Project ready: {p_path}. Edit {target.name}; see INFO.md for all options.")
    return p_path


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
    user = _user_paths(data_root)
    saved = _saved_settings()
    if saved and Path(saved["data_dir"]).expanduser().resolve() == data_root:
        # Folders edited in .oscar_settings.json apply to the saved data directory only.
        user.update({k: Path(saved[k]).expanduser().resolve() for k in USER_PATHS if saved.get(k)})
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
        # User folders: run mode libraries, projects and results (see USER_PATHS)
        **user,
        # Developer folders
        "dev_data": data_root / "dev_data",
        "setup_data": data_root / "dev_data" / "setup_data",
    }

