"""Compact, readable discovery pages for OSCAR's user workflows."""
from itertools import zip_longest
from pathlib import Path
from textwrap import fill, wrap

import xarray as xr

from oscar._io.paths import INTERNAL_BOOTSTRAP_DIR, get_paths
from oscar._utils.load_config import load_config
from oscar._utils.metadata import load_var_registry

WIDTH = 88
LABEL = 9
SECOND_COLUMN = 38


def format_run_help():
    """Task-first terminal help, without the parser's implementation details."""
    width = 104
    lines = [
        '', '=' * width, 'RUN OSCAR'.center(width).rstrip(), '=' * width, '',
        f"  {'What would you like to run?':<38}  {'Mode':<12}  Command",
        '  ' + '-' * (width - 4),
    ]
    for task, mode, command in (
        ('A demonstration', 'Standard', 'oscar run'),
        ('An available CMIP scenario', 'Configured', 'oscar run -m configured -s scen7-VL'),
        ('An experiment with your own forcing', 'Customized',
         'oscar run -m customized --project <project-id> \\\n  --experiment <experiment-id>'),
    ):
        first, *rest = command.split('\n')
        lines.append(f'  {task:<38}  {mode:<12}  {first}')
        lines.extend(f"  {'':<38}  {'':<12}  {line}" for line in rest)
    lines.extend([
        '', '  Learn more about each mode', '  ' + '-' * (width - 4),
        '  Standard   : oscar info standard',
        '  Configured : oscar info configured',
        '  Customized : oscar info customized',
        '=' * width,
    ])
    return '\n'.join(lines)


def _heading(title):
    print('\n' + '=' * WIDTH)
    print(title.center(WIDTH).rstrip())
    print('=' * WIDTH)


def _end(note=None):
    if note:
        _text(note)
    print('=' * WIDTH + '\n')


def _text(text):
    print(fill(text, width=WIDTH, initial_indent='  ', subsequent_indent='  ',
               break_long_words=False, break_on_hyphens=False))


def _field(label, value):
    prefix = f'  {label:<{LABEL}} : ' if label else ' ' * (LABEL + 5)
    print(fill(str(value), width=WIDTH, initial_indent=prefix,
               subsequent_indent=' ' * len(prefix), break_long_words=False,
               break_on_hyphens=False))


def _pair(left, right):
    left_label, left_value = left
    right_label, right_value = right
    first = f'  {left_label:<{LABEL}} : {left_value}'
    # The right-hand labels and colons start at fixed columns on every row.
    print(f'{first:<{SECOND_COLUMN}}{right_label:<11} : {right_value}')


def _section(title):
    print(f'\n  {title}')
    print('  ' + '-' * (WIDTH - 4))


def _cell(value, width):
    lines = []
    for line in str(value).splitlines() or ['']:
        indent = len(line) - len(line.lstrip())
        lines.extend(wrap(line, width=width, subsequent_indent=' ' * indent,
                          break_long_words=False, break_on_hyphens=False) or [''])
    return lines


def _table(rows, headers=('Action', 'Terminal', 'Python (import oscar)'), widths=(21, 25, 36)):
    print()
    if len(headers) == 2:
        print(f'  {headers[0]:<{widths[0]}}  {headers[1]}')
    else:
        print(f'  {headers[0]:<{widths[0]}}  {headers[1]:<{widths[1]}}  {headers[2]}')
    print('  ' + '-' * (WIDTH - 4))
    for action, terminal, python in rows:
        if python is None:
            # Instructions such as adding a file apply to both interfaces.
            cells = [_cell(action, widths[0]), _cell(terminal, WIDTH - widths[0] - 6)]
            for left, instruction in zip_longest(*cells, fillvalue=''):
                print(f'  {left:<{widths[0]}}  {instruction}'.rstrip())
        else:
            cells = [_cell(value, width) for value, width in zip((action, terminal, python), widths)]
            for left, middle, right in zip_longest(*cells, fillvalue=''):
                print(f'  {left:<{widths[0]}}  {middle:<{widths[1]}}  {right}'.rstrip())
    print('  ' + '-' * (WIDTH - 4))


def _variables():
    registry = load_var_registry()
    _heading('OSCAR — OUTPUT VARIABLES')
    for name, metadata in registry.items():
        _field(name, f"{metadata.get('long_name', '')} [{metadata.get('unit', '')}]")
    _end()
    return registry


def _general():
    _heading('OSCAR — EARTH SYSTEM MODEL')
    _text('Choose a mode based on what you want to do.')
    _table([
        ('Standard', 'Run a demonstration.', 'oscar info standard'),
        ('Configured', 'Select available scenarios and settings.', 'oscar info configured'),
        ('Customized', 'Run with your own prepared forcing.', 'oscar info customized'),
    ], headers=('Mode', 'Purpose', 'More information'))
    _end('Get started: oscar run. For command options: oscar --help.')
    return {'modes': ['standard', 'configured', 'customized']}


def _standard(cfg):
    from oscar._io._download import Library, BOOTSTRAP_FILES
    std = cfg['standard_mode']
    lib = Library(INTERNAL_BOOTSTRAP_DIR, std['hist_type'], std['region'], files=BOOTSTRAP_FILES)
    with xr.open_dataset(lib.files['params']) as ds:
        members = ds.sizes['config']
    result = {'mode': 'standard', 'hist_type': lib.hist_type, 'region': lib.region,
              'members': members, 'hist_start': lib.hist_start, 'hist_end': lib.hist_end,
              'projection_end': lib.end_year, 'scenarios': lib.scenarios}
    _heading('OSCAR — STANDARD MODE')
    _text('Run the bundled demonstration. No download or initialization required.')
    print()
    _pair(('History', lib.hist_type), ('Historical', f"{lib.hist_start}–{lib.hist_end}"))
    _pair(('Region', lib.region), ('Projections', f"{lib.hist_end + 1}–{lib.end_year}"))
    _field('Scenarios', ', '.join(lib.scenarios))
    _output_names(std['var_select'])
    _table([
        ('Run demonstration', 'oscar run', 'out = oscar.run()'),
        ('Save elsewhere', 'Add --output-dir DIR', 'Add output_dir="DIR"'),
        ('Run again', 'Add --overwrite', 'Add overwrite=True'),
        ('Skip plot / saving', 'Add --no-plot / --no-save', 'Add plot=False / save=False'),
    ])
    _end(f"Results and a temperature plot are saved in {get_paths()['results'] / 'standard'}/.")
    return result


def _output_names(variables, label='Outputs'):
    registry = load_var_registry()
    for i, var in enumerate(variables):
        meta = registry.get(var, {})
        _field(label if i == 0 else '', f"{meta.get('long_name', var)} ({var}, {meta.get('unit', 'n/a')})")


def _configured(cfg, hist_type, region, library_dir):
    from oscar._io._download import library_location, open_library
    from oscar._workflows.configured_runs import resolve_selection
    opt = cfg['configured_mode']
    defaults = opt['defaults']
    sel = resolve_selection(region=region, hist_type=hist_type)
    hist_type, region = sel.hist_type, sel.region
    location = Path(library_dir).expanduser().resolve() if library_dir else library_location(hist_type, region)
    lib = open_library(hist_type, region, location) if location.exists() else None
    result = {'mode': 'configured', 'hist_type': hist_type, 'region': region,
              'histories': opt['allowed_hist'], 'regions': opt['allowed_regions'], 'scenarios': opt['scenarios'],
              'library': str(location), 'library_found': lib is not None,
              'results': str(get_paths()['results'] / 'configured')}
    if lib:
        result.update(hist_end=lib.hist_end, projection_end=lib.end_year)
    _heading('OSCAR — CONFIGURED MODE')
    _text('Run available CMIP scenarios with your choice of history, region and outputs.')
    print()
    mark = lambda name, default: f'{name} [default]' if name == default else name
    _field('History', ', '.join(mark(h, defaults['hist_type']) for h in sorted(opt['allowed_hist'], reverse=True)))
    _field('Region', ', '.join(mark(r, defaults['region']) for r in opt['allowed_regions']))
    _field('Scenario', f"{defaults['scenario'][hist_type]} [default]")
    _output_names(defaults['var_select'], 'Outputs')
    _section('Available scenarios')
    for name in sorted(opt['scenarios'], reverse=True):
        _field(name, ', '.join(opt['scenarios'][name]))
    _section('Examples (terminal)')
    for task, command in (
            ('Run with all defaults', ''),
            ('Use CMIP6 history', ' --history CMIP6 -s SSP2-4.5'),
            ('Use the 10-region set', ' -r IAMC_R10'),
            ('Run two scenarios', ' -s scen7-H -s scen7-L'),
            ('Save other outputs', ' -v D_Tg -v D_CO2'),
            ('Save every output', ' -v all')):
        print(f'  {task:<24}oscar run -m configured{command}')
    print(f"  {'List output names':<24}oscar info --variables")
    _section('Same in Python')
    print('  import oscar')
    print('  out = oscar.run(mode="configured")')
    print('  out = oscar.run(mode="configured", hist_type="CMIP6", region="IAMC_R10",')
    print('                  scenario=["SSP2-4.5"], variables=["D_Tg", "D_CO2"])')
    _section('Where files go')
    data = get_paths()['data_root']
    inside = lambda path: (str(Path(path).relative_to(data)) if Path(path).is_relative_to(data) else str(path)) + '/'
    _field('Data', str(data))
    _field('', 'To use another folder: oscar init --data-dir PATH')
    _field('Library', inside(location) + ('' if lib else '  (not downloaded yet)'))
    if lib:
        _field('Years', f'history {lib.hist_start}–{lib.hist_end}, projections {lib.hist_end + 1}–{lib.end_year}')
    _field('Results', inside(result['results']) + f'configured_{hist_type}_{region}_results.nc')
    _field('Plots', 'Same folder, one image per output.')
    _end('Library and Results are inside the data folder.')
    return result


def _projection_years(hist_type, region, library_dir=None):
    """'2023–2099' from a downloaded library, or None."""
    from oscar._io._download import library_location, open_library
    location = Path(library_dir).expanduser() if library_dir else library_location(hist_type, region)
    if not location.exists():
        return None
    lib = open_library(hist_type, region, location)
    return f'{lib.hist_end + 1}–{lib.end_year}'


def _customized(cfg, hist_type, region, library_dir):
    from oscar._workflows.configured_runs import resolve_selection
    opt = cfg['configured_mode']
    defaults = opt['defaults']
    sel = resolve_selection(region=region, hist_type=hist_type)
    themes = cfg['customized_mode']['output_themes']
    years = {h: _projection_years(h, sel.region, library_dir if h == sel.hist_type else None)
             for h in sorted(opt['allowed_hist'], reverse=True)}
    projects = get_paths()['projects']
    result = {'mode': 'customized', 'histories': opt['allowed_hist'], 'regions': opt['allowed_regions'],
              'projection_years': years, 'output_themes': list(themes), 'projects': str(projects)}
    _heading('OSCAR — CUSTOMIZED MODE')
    _text('Run your own research experiment: your emissions, radiative forcing or land use '
          'applied to a CMIP baseline scenario.')
    print()
    mark = lambda name, default: f'{name} [default]' if name == default else name
    _field('History', ', '.join(mark(h, defaults['hist_type']) for h in years))
    _field('Region', ', '.join(mark(r, defaults['region']) for r in opt['allowed_regions']))
    _field('Baseline', f"{defaults['scenario'][sel.hist_type]} [default], or any scenario of configured mode")
    _field('Inputs', 'Emissions, radiative forcing and land use: continuous annual time series '
                     'covering every projection year')
    _output_names(themes['climate'], 'Outputs')
    _field('', f"Other output themes: {', '.join(t for t in themes if t != 'climate')}")
    _section('Steps (terminal)')
    for step, command in (('1. Create a project', 'oscar project <project-id> --experiment <experiment-id>'),
                          ('2. Add input files', '<project-id>/inputs/'),
                          ('3. Edit the settings', '<project-id>/settings_<experiment-id>.yaml'),
                          ('4. Check them', 'oscar validate <project-id> --experiment <experiment-id>'),
                          ('5. Run', 'oscar run -m customized --project <project-id> \\\n'
                                     '  --experiment <experiment-id>')):
        first, *rest = command.split('\n')
        print(f'  {step:<24}{first}')
        for line in rest:
            print(f"  {'':<24}{line}")
    _section('Same in Python')
    print('  import oscar')
    print('  oscar.create_project("<project-id>", experiment="<experiment-id>")')
    print('  oscar.validate_project("<project-id>", experiment="<experiment-id>")')
    print('  out = oscar.run(mode="customized", project="<project-id>",')
    print('                  experiment="<experiment-id>")')
    _section('Where files go')
    data = get_paths()['data_root']
    inside = lambda path: (str(Path(path).relative_to(data)) if Path(path).is_relative_to(data) else str(path)) + '/'
    _field('Data', str(data))
    _field('Templates', inside(get_paths()['customized_library']) + '  (downloaded with your first project)')
    _field('Projects', inside(projects))
    _field('', 'Each project holds inputs/, settings_<experiment-id>.yaml, forcing/ (compiled '
               'forcing) and results/ (results and plots).')
    _end('All settings, input formats and rules: INFO.md in each project folder.')
    return result


def show_info(mode=None, hist_type=None, region=None, library_dir=None, variables=False):
    """Print a readable page and return structured information for Python callers."""
    cfg = load_config()
    if mode not in (None, 'standard', 'configured', 'customized', 'advanced'):
        raise ValueError('Choose standard, configured or customized.')
    if variables:
        return _variables()
    if mode is None:
        return _general()
    if mode == 'advanced':
        from oscar._workflows.advanced_runs import OPTIONS
        _heading('OSCAR — ADVANCED MODE')
        _field('Purpose', 'Research workflows for collaborators')
        _field('Options', ', '.join(OPTIONS))
        _pair(('History', 'CMIP7'), ('Projections', '2024–2100'))
        _field('Ensemble', f"{cfg['configured_mode']['official_nMC']} members")
        _table([(name, ', '.join(spec['regions']), None) for name, spec in OPTIONS.items()],
               headers=('Option', 'Model regions'), widths=(12, 68))
        _section('Download a run option')
        _field('SCI-lite', 'oscar download -m advanced --option SCI-lite')
        _field('SCI-luc', 'oscar download -m advanced --option SCI-luc --region IAMC_R5')
        _end('After downloading, follow INFO_SCI-lite.md or INFO_SCI-luc.md in the '
             'advanced library folder for input requirements and Python examples.')
        return {'mode': 'advanced', 'options': {name: list(spec['regions']) for name, spec in OPTIONS.items()}}
    if mode == 'standard':
        return _standard(cfg)
    if mode == 'configured':
        return _configured(cfg, hist_type, region, library_dir)
    return _customized(cfg, hist_type, region, library_dir)
