"""Version shared by source checkouts, installed wheels and scientific metadata."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def get_version():
    package = Path(__file__).resolve().parent
    # Wheels carry a build-time copy of the root VERSION file. Editable/source
    # checkouts read the root directly, so no second value is maintained.
    for path in (package / "VERSION", package.parent / "VERSION"):
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            if not value:
                raise ValueError(f"OSCAR version file is empty: {path}")
            return value
    # Compatibility with older installations that did not bundle VERSION.
    try:
        return version("oscar")
    except PackageNotFoundError:
        return "unknown"
