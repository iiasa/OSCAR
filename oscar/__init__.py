# oscar/__init__.py
from importlib.metadata import version, PackageNotFoundError

# 1. Core Functions
from .run import run, info, download
from ._io.paths import set_data_dir, get_user_data_dir, create_project

# 2. Version Management
try:
    # This pulls the version from pyproject.toml after 'pip install -e .'
    __version__ = version("oscar")
except PackageNotFoundError:
    # Fallback if the package is not installed in the environment
    __version__ = "unknown"

# Display version with a 'OSCAR_v' prefix
__display_version__ = f"OSCAR_{__version__}"