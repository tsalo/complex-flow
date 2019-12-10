"""
"""
import os
from copy import deepcopy

import numpy as np
import nibabel as nib
from bids.layout import BIDSLayout

import nipype.pipeline.engine as pe  # pypeline engine
from nipype.interfaces import utility as niu
from nipype.interfaces.utility import Function
import nipype.interfaces.io as nio
from niflow.nipype1.workflows.dmri.fsl.utils import siemens2rads, rads2radsec

from sdcflows.workflows.docma import init_docma_wf
from utils import collect_data


def pick_first_two(func):
    """
    Use to grab first two echoes for multi-echo data
    """
    return func[:2]


def pick_first(func):
    """
    Used to grab first echo for multi-echo data
    """
    if isinstance(func, list):
        return func[0]
    else:
        return func


def pick_second(func):
    """
    Used to grab second echo for multi-echo data
    """
    if isinstance(func, list):
        return func[1]
    else:
        return func


def copy_files(*, output_dir, **kwargs):
    """
    Requires 3.6+
    """
    from os import mkdir
    import os.path as op
    from shutil import copyfile

    out_files = []
    for in_file in kwargs.values():
        fn = op.basename(in_file)
        out_file = op.join(output_dir, fn)
        if not op.isdir(output_dir):
            mkdir(output_dir)
        copyfile(in_file, out_file)
        out_files.append(out_file)
    return out_files


def init_preproc_workflow(bids_dir, output_dir, work_dir, subject_list,
                          session_label, task_label, run_label):
    """
    A workflow for preprocessing complex-valued multi-echo fMRI data with
    single-band reference images and available T1s.
    """
    # setup workflow
    participant_wf = pe.Workflow(name='participant_wf')
    participant_wf.base_dir = os.path.join(work_dir, 'complex_preprocessing')
    os.makedirs(participant_wf.base_dir, exist_ok=True)

    # Read in dataset, but don't validate because phase isn't supported yet
    layout = BIDSLayout(bids_dir, validate=False)

    for subject_label in subject_list:
        # collect the necessary inputs
        subject_data = collect_data(
            layout, subject_label, task=task_label, run=run_label, ses=session_label
        )

        single_subject_wf = init_single_subject_wf(
            name='single_subject_' + subject_label + '_wf',
            output_dir=output_dir,
            layout=layout,
            bold_files=subject_data['bold_mag_files'],
            bold_metadata=subject_data['bold_mag_metadata'],
            phase_files=subject_data['bold_phase_files'],
            phase_metadata=subject_data['bold_phase_metadata'],
        )
        single_subject_wf.config['execution']['crashdump_dir'] = os.path.join(
            output_dir, 'sub-' + subject_label, 'log'
        )

        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        participant_wf.add_nodes([single_subject_wf])

    return participant_wf


def init_single_subject_wf(name, output_dir, layout, bold_files, bold_metadata,
                           phase_files, phase_metadata):
    """
    Single-subject workflow
    """
    workflow = pe.Workflow(name=name)

    # name the nodes
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_files',
                'bold_metadata',
                'phase_files',
                'phase_metadata',
            ]
        ),
        name='inputnode',
        iterables=[
            ('bold_files', bold_files),
            ('bold_metadata', bold_metadata),
            ('phase_files', phase_files),
            ('phase_metadata', phase_metadata),
        ],
        synchronize=True)

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['fmap', 'fmap_ref', 'fmap_mask', 'reference', 'reference_brain', 'warp', 'mask']
        ),
        name='outputnode')

    # Generate single-band reference image-based field maps
    docma_wf = init_docma_wf(name='docma_wf', num_trs=20, omp_nthreads=1)

    # Output BOLD mag files
    derivativesnode = pe.Node(
        interface=Function(
            ['fmap', 'fmap_ref', 'fmap_mask', 'reference', 'reference_brain', 'warp', 'mask', 'output_dir'],
            ['fmap', 'fmap_ref', 'fmap_mask', 'reference', 'reference_brain', 'warp', 'mask'],
            copy_files),
        name='derivativesnode')
    derivativesnode.inputs.output_dir = output_dir
    workflow.connect([
        (inputnode, docma_wf, [(('bold_files', pick_first), 'inputnode.magnitude1'),
                               (('bold_files', pick_second), 'inputnode.magnitude2'),
                               (('phase_files', pick_first), 'inputnode.phase1'),
                               (('phase_files', pick_second), 'inputnode.phase2'),
                               (('phase_metadata', pick_first), 'inputnode.phase1_metadata'),
                               (('phase_metadata', pick_second), 'inputnode.phase2_metadata')]),
        (docma_wf, outputnode, [('outputnode.fmap', 'fmap'),
                                ('outputnode.fmap_mask', 'fmap_mask'),
                                ('outputnode.fmap_ref', 'fmap_ref'),
                                ('outputnode.reference', 'reference'),
                                ('outputnode.reference_brain', 'reference_brain'),
                                ('outputnode.warp', 'warp'),
                                ('outputnode.mask', 'mask')]),
        (outputnode, derivativesnode, [('fmap', 'fmap'),
                                       ('fmap_mask', 'fmap_mask'),
                                       ('fmap_ref', 'fmap_ref'),
                                       ('reference', 'reference'),
                                       ('reference_brain', 'reference_brain'),
                                       ('warp', 'warp'),
                                       ('mask', 'mask')]),
    ])
    return workflow
