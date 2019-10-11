import os
from glob import glob
import argparse
from argparse import RawTextHelpFormatter
from multiprocessing import cpu_count

from nipype import config as ncfg


def get_parser():
    """Build parser object"""
    parser = argparse.ArgumentParser(description='Complex Preproc BIDS arguments',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('bids_dir', help='The directory with the input dataset '
                        'formatted according to the BIDS standard.')
    parser.add_argument('output_dir', help='The directory where the output directory '
                        'and files should be stored. If you are running group level analysis '
                        'this folder should be prepopulated with the results of the'
                        'participant level analysis.')

    # preprocessing options
    proc_opts = parser.add_argument_group('Options for processing')
    proc_opts.add_argument('-w', '--work-dir', help='directory where temporary files '
                           'are stored (i.e. non-essential files). '
                           'This directory can be deleted once you are reasonably '
                           'certain nibs finished as expected.')

    # Image Selection options
    image_opts = parser.add_argument_group('Options for selecting images')
    parser.add_argument('--participant-label', nargs="+",
                        help='The label(s) of the participant(s) '
                             'that should be analyzed. The label '
                             'corresponds to sub-<participant_label> from the BIDS spec '
                             '(so it does not include "sub-"). If this parameter is not '
                             'provided all subjects should be analyzed. Multiple '
                             'participants can be specified with a space separated list.')
    image_opts.add_argument('--session-label', action='store',
                            default=None, help='select a session to analyze')
    image_opts.add_argument('-t', '--task-label', action='store',
                            default=None, help='select a specific task to be processed')
    image_opts.add_argument('--run-label', action='store',
                            default=None, help='select a run to analyze')

    # performance options
    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument('--nthreads', '-n-cpus', action='store', type=int,
                         help='maximum number of threads across all processes')
    g_perfm.add_argument('--use-plugin', action='store', default=None,
                         help='nipype plugin configuration file')

    # misc options
    misc = parser.add_argument_group('misc options')
    misc.add_argument('--graph', action='store_true', default=False,
                      help='generates a graph png of the workflow')

    return parser


def main(argv=None):
    from workflows import init_workflow

    # get commandline options
    opts = get_parser().parse_args(argv)

    # Set up directories
    # TODO: set up some sort of versioning system
    bids_dir = os.path.abspath(opts.bids_dir)

    output_dir = os.path.abspath(opts.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    if opts.work_dir:
        work_dir = os.path.abspath(opts.work_dir)
    else:
        work_dir = os.path.join(os.getcwd(), 'complexpreproc_work')

    os.makedirs(work_dir, exist_ok=True)

    if opts.participant_label:  # only for a subset of subjects
        subject_list = opts.participant_label
    else:  # for all subjects
        subject_dirs = glob(os.path.join(bids_dir, "sub-*"))
        subject_list = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]

    # Nipype plugin configuration
    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        from yaml import load as loadyml
        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
        plugin_settings.setdefault('plugin_args', {})
    else:
        # Defaults
        plugin_settings = {
            'plugin': 'MultiProc',
            'plugin_args': {
                'raise_insufficient': False,
                'maxtasksperchild': 1,
            }
        }

    # Resource management options
    # Note that we're making strong assumptions about valid plugin args
    # This may need to be revisited if people try to use batch plugins
    nthreads = plugin_settings['plugin_args'].get('n_procs')
    # Permit overriding plugin config with specific CLI options
    if nthreads is None or opts.nthreads is not None:
        nthreads = opts.nthreads
        if nthreads is None or nthreads < 1:
            nthreads = cpu_count()
        plugin_settings['plugin_args']['n_procs'] = nthreads

    # Nipype config (logs and execution)
    ncfg.update_config({
        'logging': {'log_directory': log_dir,
                    'log_to_file': True},
        'execution': {'crashdump_dir': log_dir,
                      'crashfile_format': 'txt',
                      'parameterize_dirs': False},
    })

    # running participant level
    participant_wf = init_workflow(
        bids_dir=bids_dir,
        output_dir=output_dir,
        work_dir=work_dir,
        subject_list=subject_list,
        session_label=opts.session_label,
        task_label=opts.task_label,
        run_label=opts.run_label,
    )

    if opts.graph:
        participant_wf.write_graph(graph2use='flat',
                                   format='svg',
                                   simple_form=False)

    try:
        participant_wf.run(**plugin_settings)
    except RuntimeError as e:
        if "Workflow did not execute cleanly" in str(e):
            print("Workflow did not execute cleanly")
        else:
            raise e


if __name__ == '__main__':
    main()
