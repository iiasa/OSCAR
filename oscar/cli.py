"""
OSCAR Command Line Interface
"""
import click
from .run import run as _run
from .run import info as _info

@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """
    OSCAR: One-Line Simple Climate Artifact Research model.
    
    If no command is provided, displays general model information.
    """
    if ctx.invoked_subcommand is None:
        # If the user just types 'oscar', show the general info page
        _info()
    else:
        pass

@main.command()
@click.option('--mode', '-m', default='standard', 
              type=click.Choice(['standard', 'configured']),
              help='Run mode (Standard by default).')
@click.option('--scenario', '-s', multiple=True, 
              help='Library scenario name(s). Repeat for multiple.')
@click.option('--region', '-r', 
              help='Library region name.')
@click.option('--variables', '-v', multiple=True,
              help='Output variable IDs. Repeat for multiple.')
def run(mode, **kwargs):
    """
    Execute an OSCAR simulation (Standard or Configured modes only).
    
    For Customized or Advanced research, please use the Python API.
    """
    # Simply forward the arguments to run.py
    _run(mode=mode, **kwargs)

@main.command()
@click.argument('mode', required=False, type=click.Choice(['standard', 'configured']))
def info(mode):
    """
    Explore available scenarios, regions, and variable names.
    
    Usage: 'oscar info' or 'oscar info configured'
    """
    _info(mode)

if __name__ == "__main__":
    main()