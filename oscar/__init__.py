# oscar/__init__.py
from oscar._version import get_version

__version__ = get_version()
__display_version__ = f"OSCAR_{__version__}"

# 1. Core Functions
from oscar.run import run, info, download, validate_project
from oscar._io.paths import set_data_dir, get_user_data_dir, create_project
