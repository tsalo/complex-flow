"""
Perform minimal preprocessing.

Get functions appear to be working. They require pybids>='0.9.4'

A brief description of the steps:
For the T1w images:
1.  Estimate normalization transform to MNI space from T1w space.

For the EPI field map images:
1.  Generate field map with sdcflows.

For the single-band reference images:
1.  Process phase data (rescale and unwrap).
2.  Generate field map for SBRef from first two echoes' magnitude and phase data
    (with sdcflows). This is a single image, so it should be easy(ish).
3.  Apply field map to all echoes of SBRef (both magnitude and processed phase).
4.  Estimate coregistration transform to T1w space using unwarped SBRef from
    first echo.

For the BOLD data:
1.  Combine magnitude and phase data into complex-valued images.
2.  Run DWIDenoise on complex-valued data.
3.  Split denoised magnitude data out of denoised complex data.
    Discard denoised phase data.
4.  Process phase data (rescale and unwrap).
5.  Generate volume-specific field maps from first two echoes' magnitude and
    phase data (with sdcflows).
6.  Apply field maps to all echoes (both magnitude and processed phase).
7.  Estimate motion correction transform using unwarped SBRef's first echo as
    the reference image and the unwarped BOLD first echo as the moving data.
8.  Perform slice timing correction on motion-corrected and unwarped magnitude
    and phase data.
9.  Concatenate unwarping and motion correction transforms (pre-denoising/STC
    transforms).
10. Concatenate coregistration and normalization transforms (post-denoising/STC
    transforms).
11. Apply concatenated pre-denoising transform to all echoes (both magnitude and
    processed phase).
12. Estimate T2* map from undistorted, motion corrected, and slice timing
    corrected magnitude and phase data.
13. Run multi-echo denoising using T2* map and magnitude data.
14. Apply post-denoising transform to denoised data to warp to standard space.

Outputs:
1. Preprocessed multi-echo magnitude data in native space
2. Preprocessed multi-echo phase data in native space
3. Pre-denoising transform
4. Post-denoising transform
5. Preprocessed multi-echo single-band reference data in native space
6. Motion parameters
7. Multi-echo denoised magnitude data in native space
"""
import os
from copy import deepcopy

import numpy as np
import nibabel as nib
from bids.layout import BIDSLayout

from niworkflows.interfaces.itk import MCFLIRT2ITK
from niworkflows.interfaces.itk import MultiApplyTransforms
from niworkflows.func.util import init_skullstrip_bold_wf

import nipype.pipeline.engine as pe  # pypeline engine
from nipype.interfaces import fsl, afni, freesurfer
from nipype.interfaces import utility as niu
from nipype.interfaces.utility import Function
import nipype.interfaces.io as nio
from niflow.nipype1.workflows.dmri.fsl.utils import siemens2rads, rads2radsec

# Currently requires https://github.com/mattcieslak/sdcflows/tree/phase1phase2
from phaseprep.workflows import create_preprocess_phase_wf

from utils import *
from phase import init_phase_processing_wf


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
            bold_mag_files=subject_data['bold_mag_files'],
            bold_mag_metadata=subject_data['bold_mag_metadata'],
            bold_phase_files=subject_data['bold_phase_files'],
            bold_phase_metadata=subject_data['bold_phase_metadata'],
            sbref_mag_files=subject_data['sbref_mag_files'],
            sbref_mag_metadata=subject_data['sbref_mag_metadata'],
            sbref_phase_files=subject_data['sbref_phase_files'],
            sbref_phase_metadata=subject_data['sbref_phase_metadata'],
            t1w_files=subject_data['t1w_files'],
            t1w_metadata=subject_data['t1w_metadata'],
            t2w_files=subject_data['t2w_files'],
            t2w_metadata=subject_data['t2w_metadata'],
        )
        single_subject_wf.config['execution']['crashdump_dir'] = os.path.join(
            output_dir, 'sub-' + subject_label, 'log'
        )

        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        participant_wf.add_nodes([single_subject_wf])

    return participant_wf


def init_single_subject_wf(
    name,
    output_dir,
    layout,
    bold_mag_files,
    bold_mag_metadata,
    bold_phase_files,
    bold_phase_metadata,
    sbref_mag_files,
    sbref_mag_metadata,
    sbref_phase_files,
    sbref_phase_metadata,
    t1w_files,
    t1w_metadata,
    t2w_files,
    t2w_metadata,
):

    workflow = pe.Workflow(name=name)

    # name the nodes
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_mag_files',
                'bold_mag_metadata',
                'bold_phase_files',
                'bold_phase_metadata',
                'sbref_mag_files',
                'sbref_mag_metadata',
                'sbref_phase_files',
                'sbref_phase_metadata',
                't1w_files',
                't1w_metadata',
                't2w_files',
                't2w_metadata',
            ]
        ),
        name='inputnode',
        iterables=[
            ('bold_mag_files', bold_mag_files),
            ('bold_mag_metadata', bold_mag_metadata),
            ('bold_phase_files', bold_phase_files),
            ('bold_phase_metadata', bold_phase_metadata),
            ('sbref_mag_files', sbref_mag_files),
            ('sbref_mag_metadata', sbref_mag_metadata),
            ('sbref_phase_files', sbref_phase_files),
            ('sbref_phase_metadata', sbref_phase_metadata),
        ],
        synchronize=True,
    )
    inputnode.inputs.t1w_files = t1w_files
    inputnode.inputs.t1w_metadata = t1w_metadata
    inputnode.inputs.t2w_files = t2w_files
    inputnode.inputs.t2w_metadata = t2w_metadata

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['preproc_bold_files', 'preproc_phase_files', 'motion_parameters']
        ),
        name='outputnode',
    )

    # Generate single-band reference image-based field maps
    sbref_phdiff_wf = init_phdiff_wf(
        name='sbref_phdiff_wf', create_phasediff=True, omp_nthreads=1, fmap_bspline=None
    )
    workflow.connect(
        inputnode,
        ('sbref_phase_files', pick_first),
        sbref_phdiff_wf,
        'inputnode.phase1',
    )
    workflow.connect(
        inputnode,
        ('sbref_phase_files', pick_second),
        sbref_phdiff_wf,
        'inputnode.phase2',
    )
    workflow.connect(
        inputnode,
        ('sbref_mag_files', pick_first),
        sbref_phdiff_wf,
        'inputnode.magnitude',
    )
    workflow.connect(
        inputnode,
        ('sbref_phase_metadata', pick_first),
        sbref_phdiff_wf,
        'inputnode.phase1_metadata',
    )
    workflow.connect(
        inputnode,
        ('sbref_phase_metadata', pick_second),
        sbref_phdiff_wf,
        'inputnode.phase2_metadata',
    )

    # Generate dynamic field maps
    # Somehow select the first two echoes
    docma_wf = init_docma_wf(omp_nthreads=1, name='docma_wf')


    bold_mag_splitter = pe.MapNode(
        interface=fsl.Split(dimension='t'),
        name='bold_mag_splitter',
        iterfield=['in_file'],
    )
    bold_phase_splitter = pe.MapNode(
        interface=fsl.Split(dimension='t'),
        name='bold_phase_splitter',
        iterfield=['in_file'],
    )
    workflow.connect(inputnode, 'bold_mag_files', bold_mag_splitter, 'in_file')
    workflow.connect(inputnode, 'bold_phase_files', bold_phase_splitter, 'in_file')
    """
    bold_phdiff_wf = init_phdiff_wf(name='bold_phdiff_wf',
                                    create_phasediff=True,
                                    omp_nthreads=1,
                                    fmap_bspline=None)
    workflow.connect(inputnode, ('bold_phase_files', pick_first),
                     bold_phdiff_wf, 'inputnode.phase1')
    workflow.connect(inputnode, ('bold_phase_files', pick_second),
                     bold_phdiff_wf, 'inputnode.phase2')
    workflow.connect(inputnode, ('bold_mag_files', pick_first),
                     bold_phdiff_wf, 'inputnode.magnitude')
    workflow.connect(inputnode, ('bold_phase_metadata', pick_first),
                     bold_phdiff_wf, 'inputnode.phase1_metadata')
    workflow.connect(inputnode, ('bold_phase_metadata', pick_second),
                     bold_phdiff_wf, 'inputnode.phase2_metadata')

    """
    """
    # Skullstrip BOLD files on a volume-wise basis
    # Need to feed in 3D files from first echo
    bold_mag_buffer = pe.Node(
        interface=niu.IdentityInterface(fields=['bold_mag_files']),
        name='bold_mag_buffer')
    bold_mag_buffer.iterables = ('bold_mag_files', (bold_mag_splitter.outputs.out_files, pick_first))

    bold_skullstrip_wf = init_skullstrip_bold_wf(name='bold_skullstrip_wf')
    workflow.connect(bold_mag_buffer, 'bold_mag_files',
                     bold_skullstrip_wf, 'inputnode.in_file')

    # Apply volume-wise brain masks to corresponding volumes from all echoes
    bold_skullstrip_apply = pe.MapNode(
        fsl.ApplyMask(),
        name='bold_skullstrip_apply',
        iterfield=['in_file'],
    )
    workflow.connect(inputnode, 'bold_mag_files',
                     bold_skullstrip_apply, 'in_file')
    workflow.connect(bold_skullstrip_wf, 'outputnode.mask_file',
                     bold_skullstrip_apply, 'mask_file')
    """
    """
    # Unwarp BOLD data
    # Must be applied to each volume and each echo independently
    # Will also need to be done to the phase data, post preproc but pre-MC
    bold_unwarp_wf = init_sdc_unwarp_wf(name='bold_unwarp_wf',
                                        debug=False,
                                        omp_nthreads=1,
                                        fmap_demean=True)
    first_echo_metadata = pe.Node(interface=Function(['input'], ['output'], pick_first),
                                  name='first_echo_metadata')
    workflow.connect(bold_phdiff_wf, 'outputnode.fmap',
                     bold_unwarp_wf, 'inputnode.fmap')
    workflow.connect(bold_phdiff_wf, 'outputnode.fmap_mask',
                     bold_unwarp_wf, 'inputnode.fmap_mask')
    workflow.connect(bold_phdiff_wf, 'outputnode.fmap_ref',
                     bold_unwarp_wf, 'inputnode.fmap_ref')
    workflow.connect(inputnode, ('bold_mag_files', pick_first),
                     bold_unwarp_wf, 'inputnode.in_reference')
    workflow.connect(bold_skullstrip_apply, ('out_file', pick_first),
                     bold_unwarp_wf, 'inputnode.in_reference_brain')
    workflow.connect(bold_skullstrip_wf, ('outputnode.mask_file', pick_first),
                     bold_unwarp_wf, 'inputnode.in_mask')
    workflow.connect(inputnode, 'bold_mag_metadata',
                     first_echo_metadata, 'input')
    workflow.connect(first_echo_metadata, 'output',
                     bold_unwarp_wf, 'inputnode.metadata')
    """

    # Process BOLD phase data for distortion correction
    bold_phase_wf = init_phase_processing_wf(name='phase_processing_wf')
    workflow.connect(
        inputnode, 'bold_phase_files', bold_phase_wf, 'inputnode.phase_files'
    )
    workflow.connect(
        inputnode, 'bold_mag_files', bold_phase_wf, 'inputnode.magnitude_files'
    )

    # Phaseprep preprocessing workflow to prepare phase data for phase-denoising
    reference_wf = init_bold_reference_wf(name='reference_wf')
    workflow.connect(
        bold_motionCorrection_applyMag,
        ('out_file', pick_first),
        reference_wf,
        'inputnode.bold_file',
    )
    workflow.connect(
        inputnode, ('sbref_mag_files', pick_first), reference_wf, 'inputnode.sbref_file'
    )

    # This workflow is set up for single-echo data, so we need to
    # split or iterate over echoes here
    phaseprep_wf = create_preprocess_phase_wf(name='phaseprep_wf')
    workflow.connect(
        inputnode, 'bold_phase_files', phaseprep_wf, 'inputnode.input_phase'
    )
    workflow.connect(inputnode, 'bold_mag_files', phaseprep_wf, 'inputnode.input_mag')

    # These ones are constant across echoes
    workflow.connect(
        bold_motionCorrection_estimate,
        'oned_matrix_save',
        phaseprep_wf,
        'inputnode.motion_par',
    )
    workflow.connect(
        reference_wf, 'outputnode.bold_mask', phaseprep_wf, 'inputnode.mask_file'
    )

    # Perform motion correction for first echo only
    bold_motionCorrection_estimate = pe.Node(
        interface=afni.Volreg(outputtype='NIFTI_GZ'),
        name='bold_motionCorrection_estimate',
    )

    get_motpar_name_node = pe.Node(
        interface=Function(['source_file'], ['out_file'], get_motpar_name),
        name='get_motpar_name',
    )
    workflow.connect(
        inputnode,
        ('bold_mag_files', pick_first),
        bold_motionCorrection_estimate,
        'in_file',
    )
    workflow.connect(
        inputnode, ('bold_mag_files', pick_first), get_motpar_name_node, 'source_file'
    )
    workflow.connect(
        get_motpar_name_node,
        'out_file',
        bold_motionCorrection_estimate,
        'oned_matrix_save',
    )
    workflow.connect(
        inputnode,
        ('sbref_mag_files', pick_first),
        bold_motionCorrection_estimate,
        'basefile',
    )

    # Apply motion parameters to all echoes, for both magnitude and phase data
    bold_motionCorrection_applyMag = pe.MapNode(
        interface=afni.Allineate(outputtype='NIFTI_GZ'),
        name='bold_motionCorrection_applyMag',
        iterfield=['in_file'],
    )
    workflow.connect(
        bold_motionCorrection_estimate,
        'oned_matrix_save',
        bold_motionCorrection_applyMag,
        'in_matrix',
    )
    workflow.connect(
        inputnode, 'bold_mag_files', bold_motionCorrection_applyMag, 'in_file'
    )
    workflow.connect(
        inputnode,
        ('sbref_mag_files', pick_first),
        bold_motionCorrection_applyMag,
        'reference',
    )

    # Perform slice timing correction on magnitude and phase data
    bold_stc_getParams = pe.MapNode(
        interface=Function(['metadata'], ['slice_timing'], get_slice_timing),
        name='bold_stc_getParams',
        iterfield=['metadata'],
    )
    workflow.connect(inputnode, 'bold_mag_metadata', bold_stc_getParams, 'metadata')

    bold_magnitude_stc = pe.MapNode(
        interface=afni.TShift(outputtype='NIFTI_GZ'),
        name='bold_magnitude_stc',
        iterfield=['in_file', 'slice_timing'],
    )
    workflow.connect(
        bold_motionCorrection_applyMag, 'out_file', bold_magnitude_stc, 'in_file'
    )
    workflow.connect(
        bold_stc_getParams, 'slice_timing', bold_magnitude_stc, 'slice_timing'
    )

    # Use SBRef from first echo as reference image.
    # No need to coregister functional data to SBRef because it was used for
    # the motion correction.
    # Coregister reference image to structural
    coreg_est = pe.Node(
        interface=afni.Allineate(out_matrix='sbref2anat.1D'), name='sbref2anat_estimate'
    )
    workflow.connect(inputnode, ('sbref_mag_files', pick_first), coreg_est, 'in_file')
    workflow.connect(inputnode, ('t1w_files', pick_first), coreg_est, 'reference')

    # Apply xform to mag data
    coreg_apply_mag = pe.MapNode(
        interface=afni.Allineate(outputtype='NIFTI_GZ'),
        name='sbref2anat_apply_mag',
        iterfield=['in_file'],
    )
    workflow.connect(coreg_est, 'out_matrix', coreg_apply_mag, 'in_matrix')
    workflow.connect(bold_magnitude_stc, 'out_file', coreg_apply_mag, 'in_file')
    workflow.connect(
        inputnode, ('sbref_mag_files', pick_first), coreg_apply_mag, 'reference'
    )

    # Apply xform to phase data
    coreg_apply_phase = pe.MapNode(
        interface=afni.Allineate(outputtype='NIFTI_GZ'),
        name='sbref2anat_apply_phase',
        iterfield=['in_file'],
    )
    workflow.connect(coreg_est, 'out_matrix', coreg_apply_phase, 'in_matrix')
    workflow.connect(phaseprep_wf, 'outputnode.uw_phase', coreg_apply_phase, 'in_file')
    workflow.connect(
        inputnode, ('sbref_mag_files', pick_first), coreg_apply_phase, 'reference'
    )

    # Apply xform to magnitude sbref data
    coreg_apply_sbref = pe.MapNode(
        interface=afni.Allineate(outputtype='NIFTI_GZ'),
        name='sbref2anat_apply_sbref',
        iterfield=['in_file'],
    )
    workflow.connect(coreg_est, 'out_matrix', coreg_apply_sbref, 'in_matrix')
    workflow.connect(inputnode, 'sbref_mag_files', coreg_apply_sbref, 'in_file')
    workflow.connect(
        inputnode, ('sbref_mag_files', pick_first), coreg_apply_sbref, 'reference'
    )

    # Collect outputs
    workflow.connect(
        bold_motionCorrection_estimate, 'oned_file', outputnode, 'motion_parameters'
    )
    workflow.connect(coreg_apply_mag, 'out_file', outputnode, 'preproc_bold_files')
    workflow.connect(coreg_apply_phase, 'out_file', outputnode, 'preproc_phase_files')

    # Output BOLD mag files
    derivativesnode1 = pe.MapNode(
        interface=Function(['in_file', 'output_dir'], ['out_file'], copy_files),
        name='derivativesnode_bold_mag',
        iterfield=['in_file'],
    )
    derivativesnode1.inputs.output_dir = output_dir
    workflow.connect(outputnode, 'preproc_bold_files', derivativesnode1, 'in_file')
    # Output BOLD phase files
    derivativesnode2 = pe.MapNode(
        interface=Function(['in_file', 'output_dir'], ['out_file'], copy_files),
        name='derivativesnode_bold_phase',
        iterfield=['in_file'],
    )
    derivativesnode2.inputs.output_dir = output_dir
    workflow.connect(outputnode, 'preproc_phase_files', derivativesnode2, 'in_file')
    # Output motion parameters
    derivativesnode3 = pe.Node(
        interface=Function(['in_file', 'output_dir'], ['out_file'], copy_files),
        name='derivativesnode_motpars',
    )
    derivativesnode3.inputs.output_dir = output_dir
    workflow.connect(outputnode, 'motion_parameters', derivativesnode3, 'in_file')
    return workflow
