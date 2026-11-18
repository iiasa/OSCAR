"""Terminal interface to the public CMIP workflows."""
from functools import wraps
import json
import click
from oscar.run import run as _run, info as _info, download as _download, validate_project
from oscar._io.paths import set_data_dir, create_project
from oscar._utils.metadata import get_oscar_version


class RunCommand(click.Command):
    """Keep parsing compatible while presenting the IAM-oriented workflow."""
    def get_help(self, ctx):
        from oscar._utils.help import format_run_help
        return format_run_help()


def friendly_errors(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            raise click.ClickException(str(exc)) from exc
    return wrapped


@click.group(invoke_without_command=True)
@click.version_option(version=get_oscar_version(), prog_name='OSCAR')
@click.pass_context
def main(ctx):
    """OSCAR Earth system model: CMIP demonstration, scenarios and research projects."""
    if ctx.invoked_subcommand is None:
        _info()


@main.command(hidden=True)
@click.option('--data-dir', type=click.Path(path_type=str), default=None, help='Data directory; defaults to the platform user location.')
@friendly_errors
def init(data_dir):
    """Save a data-directory preference outside the installed package."""
    set_data_dir(data_dir)


@main.command(cls=RunCommand)
@click.option('--mode', '-m', default='standard', type=click.Choice(['standard', 'configured', 'customized', 'advanced']))
@click.option('--option', 'advanced_option', hidden=True)
@click.option('--baseline', 'baseline_forcing', hidden=True)
@click.option('--end-year', 'projection_end_year', type=int, hidden=True)
@click.option('--reference', 'reference_scenarios', multiple=True, hidden=True)
@click.option('--input', 'input_file', type=click.Path(path_type=str), hidden=True)
@click.option('--output', 'output_file', type=click.Path(path_type=str), hidden=True)
@click.option('--scenario', '-s', multiple=True, help='Configured: select scenarios; repeat for multiple.')
@click.option('--region', '-r', help='Configured: select a model region.')
@click.option('--history', '--hist-type', 'hist_type', type=click.Choice(['CMIP6', 'CMIP7']), help='Configured: select the historical forcing.')
@click.option('--variables', '-v', multiple=True, help='Configured: select output variables; repeat as needed.')
@click.option('--data-dir', type=click.Path(path_type=str), hidden=True)
@click.option('--library-dir', type=click.Path(exists=True, file_okay=False, path_type=str), hidden=True, help='Local regional/history CMIP library, including compiled beta3 libraries.')
@click.option('--output-dir', type=click.Path(path_type=str), hidden=True)
@click.option('--run-name', hidden=True)
@click.option('--project', 'project_name', help='Customized: project containing your forcing and settings.')
@click.option('--experiment', help='Customized: experiment id (runs settings_<experiment-id>.yaml).')
@click.option('--plot/--no-plot', default=None, hidden=True, help='Generate summary plots; default off except where project settings enable them.')
@click.option('--show-plot', is_flag=True, hidden=True, help='Also request interactive display using the selected Matplotlib backend.')
@click.option('--save/--no-save', default=True, hidden=True)
@click.option('--overwrite', is_flag=True, hidden=True)
@click.option('--plot-only', is_flag=True, hidden=True, help='Load saved results instead of running the model.')
@friendly_errors
def run(mode, advanced_option, baseline_forcing, projection_end_year, reference_scenarios, input_file, output_file, scenario, region, hist_type, variables, data_dir, library_dir,
        output_dir, run_name, project_name, experiment, plot, show_plot, save, overwrite, plot_only):
    """Run a demonstration, an available CMIP scenario, or your own forcing."""
    if mode == 'advanced':
        if not advanced_option:
            raise ValueError('Advanced mode requires --option <run option>.')
        result = _run(mode='advanced', data_dir=data_dir, option=advanced_option, region=region, variables=list(variables),
                      forcing=input_file, output=output_file, baseline_forcing=baseline_forcing,
                      projection_end_year=projection_end_year, reference_scenarios=list(reference_scenarios),
                      overwrite=overwrite)
        click.echo(f'Completed advanced run: {result.sizes.get("year", 0)} years, {len(result.data_vars)} variables.')
        return
    if advanced_option or baseline_forcing or projection_end_year or reference_scenarios or input_file or output_file:
        raise ValueError('--option, --baseline, --end-year, --reference, --input and --output are for advanced mode.')
    options = dict(output_dir=output_dir, save=save, show_plot=show_plot, overwrite=overwrite)
    if plot is not None:
        options['plot'] = plot
    if plot_only:
        options.update(run_model=False, plot=True, save=False)
    if mode == 'standard':
        if any((scenario, region, hist_type, variables, library_dir, project_name)):
            raise ValueError('Standard mode has fixed scientific settings. Use configured or customized mode for selections.')
        options['run_name'] = run_name
    elif mode == 'configured':
        if project_name:
            raise ValueError('--project is for customized mode.')
        options.update(scenario=scenario or None, region=region, hist_type=hist_type,
                       variables=variables or None, library_dir=library_dir, run_name=run_name)
    else:
        if not project_name or not experiment:
            raise ValueError('Customized mode requires --project <project-id> --experiment <experiment-id>.')
        if any((scenario, region, hist_type, variables, run_name)):
            raise ValueError('Customized scientific selections and output names belong in project settings.')
        options.update(project=project_name, experiment=experiment, library_dir=library_dir)
    result = _run(mode=mode, data_dir=data_dir, **options)
    click.echo(f'Completed {mode} run: {result.sizes.get("year", 0)} years, {len(result.data_vars)} variables.')


@main.command()
@click.argument('mode', required=False, type=click.Choice(['standard', 'configured', 'customized', 'advanced']))
@click.option('--history', 'hist_type', type=click.Choice(['CMIP6', 'CMIP7']))
@click.option('--region')
@click.option('--library-dir', type=click.Path(exists=True, file_okay=False), hidden=True)
@click.option('--variables', is_flag=True, help='List output variables and units.')
@friendly_errors
def info(mode, **kwargs):
    """Inspect modes, scientific choices and output variables."""
    _info(mode, **kwargs)


@main.command(hidden=True)
@click.option('--mode', '-m', default='configured', type=click.Choice(['configured', 'customized', 'advanced']))
@click.option('--history', 'hist_type', type=click.Choice(['CMIP6', 'CMIP7']))
@click.option('--region')
@click.option('--option', hidden=True)
@click.option('--data-dir', type=click.Path(path_type=str), hidden=True)
@friendly_errors
def download(**kwargs):
    """Prepare data for later/offline runs, or install bundled customized examples."""
    click.echo(str(_download(**kwargs)))


@main.command()
@click.argument('name')
@click.option('--experiment', required=True, help='Id of the first experiment (settings_<experiment-id>.yaml).')
@click.option('--data-dir', type=click.Path(path_type=str), hidden=True)
@friendly_errors
def project(name, experiment, data_dir):
    """Create a project folder (a starting kit) with the settings of its first experiment."""
    create_project(name, experiment=experiment, data_dir=data_dir)


@main.command('validate')
@click.argument('project')
@click.option('--experiment', required=True, help='Experiment id (settings_<experiment-id>.yaml).')
@click.option('--data-dir', type=click.Path(path_type=str), hidden=True)
@friendly_errors
def validate(project, experiment, data_dir):
    """Check project settings and inputs without downloads or simulation."""
    result = validate_project(project, experiment, data_dir=data_dir)
    click.echo(json.dumps({k: v for k, v in result.items() if k not in ('forcing', 'settings')}, indent=2))
    click.echo('Project/input validation passed. Library compatibility is checked when running.')


if __name__ == '__main__':
    main()
