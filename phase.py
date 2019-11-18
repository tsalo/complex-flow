"""
Perform minimal preprocessing.

Get functions appear to be working. They require pybids>='0.9.4' (unreleased)

A brief description of the steps:
For the T1w images:
1. Estimate normalization transform to MNI space from T1w space.

For the GRE field map images:
1. Generate field map with sdcflows. Unknown if this will be used or not.

For the single-band reference images:
1. Process phase data (rescale and unwrap)
2. Generate field map for SBRef from first two echoes' magnitude and phase data
   (with sdcflows). This is a single image, so it should be easy(ish).
3. Apply field map to all echoes of SBRef (both magnitude and processed phase).
4. Estimate coregistration transform to T1w space using unwarped SBRef from
   first echo.

For the BOLD data:
1. Process phase data (rescale and unwrap)
2. Generate volume-specific field maps from first two echoes' magnitude and
   phase data (with sdcflows)
3. Apply field maps to all echoes (both magnitude and processed phase)
4. Estimate motion correction transform using unwarped SBRef's first echo as
   the reference image and the unwarped BOLD first echo as the moving data.
5. Perform slice timing correction on motion-corrected and unwarped magnitude
   and phase data.
6. Concatenate unwarping and motion correction transforms (pre-denoising/STC
   transforms).
7. Concatenate just coregistration and normalization transforms
   (post-denoising/STC transforms).
8. Apply concatenated pre-denoising transform to all echoes (both magnitude and
   processed phase)

Outputs:
1. Preprocessed multi-echo magnitude data in native space
2. Preprocessed multi-echo phase data in native space
3. Pre-denoising transform
4. Post-denoising transform
5. Preprocessed multi-echo single-band reference data in native space
6. Motion parameters
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


def init_phase_processing_wf(name='phase_processing_wf'):
    """
    A workflow for processing (rescaling + unwrapping) of phase data.
    Designed specifically for distortion correction
    """
    workflow = pe.Workflow(name=name)

    # name the nodes
    inputnode = pe.Node(niu.IdentityInterface(
            fields=['magnitude_files',
                    'phase_files']),
        name='inputnode')

    bold_phase_rescale = pe.MapNode(
        interface=Function(['phase_file'], ['out_file'], convert_to_radians),
        name='bold_phase_rescale',
        iterfield=['phase_file'],
    )
    workflow.connect(inputnode, 'phase_files', bold_phase_rescale, 'phase_file')

    # Default for num_partitions is 8
    # https://github.com/mshvartsman/FSL/blob/7aa2932949129f5c61af912ea677d4dbda843895/src/fugue/prelude.cc#L98
    bold_phase_unwrap = pe.MapNode(
        interface=Function(['magnitude_file', 'phase_file'], ['unwrapped_phase_file'], fake_unwrap),
        # interface=fsl.PRELUDE(),
        name='bold_phase_unwrap',
        iterfield=['magnitude_file', 'phase_file'],
    )
    workflow.connect(inputnode, 'magnitude_files', bold_phase_unwrap, 'magnitude_file')
    workflow.connect(bold_phase_rescale, 'out_file', bold_phase_unwrap, 'phase_file')

    outputnode = pe.Node(niu.IdentityInterface(
            fields=['unwrapped_phase_files']),
        name='outputnode')
    workflow.connect(bold_phase_unwrap, 'unwrapped_phase_file',
                     outputnode, 'unwrapped_phase_files')
    return workflow
