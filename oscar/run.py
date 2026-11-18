"""Public workflows: standard, configured and customized modes."""
from oscar._io.paths import use_data_dir

# Scientific and solver choices that the standard demonstration does not accept.
STANDARD_FIXED = {'scenario', 'region', 'hist_type', 'variables', 'library_dir',
                  'project', 'experiment', 'nt', 'scheme', 'adapt_nt'}
# Configured runs use the library's ensemble and solver settings.
SOLVER_FIXED = {'nt', 'scheme', 'adapt_nt', 'dtype', 'members', 'n_config', 'nMC'}


def info(mode=None, **kwargs):
    from oscar._utils.help import show_info
    return show_info(mode, **kwargs)


def download(mode='configured', hist_type=None, region=None, data_dir=None, option=None):
    """Prepare CMIP data without running, fetch the customized templates and examples, or
    an advanced run option region (option='SCI-lite'; region= for options with several regions)."""
    from oscar._io._download import ensure_configured_library, ensure_customized_library
    with use_data_dir(data_dir):
        if mode == 'customized':
            return ensure_customized_library()
        if mode == 'advanced':
            from oscar._workflows.advanced_runs import download_option
            return download_option(option, region)
        if mode != 'configured':
            raise ValueError("Download mode must be 'configured', 'customized' or 'advanced'.")
        from oscar._workflows.configured_runs import resolve_selection
        sel = resolve_selection(region=region, hist_type=hist_type)
        return ensure_configured_library(sel.hist_type, sel.region)


def run(mode='standard', data_dir=None, **kwargs):
    """Return an xarray.Dataset. data_dir overrides settings for this call only."""
    if mode not in ('standard', 'configured', 'customized', 'advanced'):
        raise ValueError(f'Unknown mode {mode!r}. Choose standard, configured or customized.')
    with use_data_dir(data_dir):
        if mode == 'standard':
            chosen = sorted(set(kwargs) & STANDARD_FIXED)
            if chosen:
                raise ValueError(f'Standard mode is a fixed demonstration; remove {", ".join(chosen)}. '
                                 'Use configured or customized mode for selections.')
            from oscar._workflows.standard_run import run_standard
            return run_standard(**kwargs)
        if mode == 'configured':
            chosen = sorted(set(kwargs) & SOLVER_FIXED)
            if chosen:
                raise ValueError(f'Configured runs use the library ensemble and solver settings; remove '
                                 f'{", ".join(chosen)}.')
            from oscar._workflows.configured_runs import run_configured
            return run_configured(**kwargs)
        if mode == 'advanced':
            # For close collaborators: settings file, inputs (path or Dataset), output path or memory.
            from oscar._workflows.advanced_runs import run_advanced
            return run_advanced(**kwargs)
        from oscar._workflows.customized_runs import run_customized
        return run_customized(**kwargs)


def validate_project(project, experiment, data_dir=None):
    """Check settings and optional forcing without downloads or simulation."""
    from oscar._workflows.customized_runs import validate_project as validate
    with use_data_dir(data_dir):
        return validate(project, experiment)
