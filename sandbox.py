"""
Sandbox workflows
"""
import os
from glob import glob

import numpy as np
import nibabel as nb

import nipype.pipeline.engine as pe  # pypeline engine
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
import nipype.interfaces.io as nio
from nipype.interfaces.utility import Function
from niworkflows.func.util import init_skullstrip_bold_wf

from utils import copy_files


def split_file(in_file, volume):
    import os.path as op
    from nilearn.image import index_img
    from nipype.utils.filemanip import split_filename

    _, base, _ = split_filename(in_file)
    out_file = op.abspath(base + "_vol{0:05d}.nii.gz".format(volume))
    img = index_img(in_file, volume)
    img.to_filename(out_file)
    return out_file


def join_files(in_files):
    import os.path as op
    from nilearn.image import concat_imgs
    from nipype.utils.filemanip import split_filename

    _, base, _ = split_filename(in_files[0])
    base = '_'.join(base.split('_')[:-1])
    out_file = op.abspath(base + ".nii.gz")
    img = concat_imgs(in_files)
    img.to_filename(out_file)
    return out_file


def divide_files(in_file1, in_file2):
    import os.path as op
    import nibabel as nb
    from nipype.utils.filemanip import split_filename

    _, base, _ = split_filename(in_file1)
    out_file = op.abspath(base + "_divided.nii.gz")
    img1 = nb.load(in_file1)
    dat1 = img1.get_data()
    dat2 = nb.load(in_file2).get_data()
    div_dat = dat1 / dat2
    img = nb.Nifti1Image(div_dat, img1.affine, header=img1.header)
    img.to_filename(out_file)
    return out_file


def init_test_division_wf(name):
    workflow = pe.Workflow(name=name)
    # name the nodes
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["bold_file", "phase_file"]),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out_file"]),
        name="outputnode")
    dividenode = pe.Node(
        interface=Function(["in_file1", "in_file2"], ["out_file"], divide_files),
        name="dividenode")
    workflow.connect(inputnode, 'bold_file', dividenode, 'in_file1')
    workflow.connect(inputnode, 'phase_file', dividenode, 'in_file2')
    workflow.connect(dividenode, 'out_file', outputnode, 'out_file')
    return workflow


def init_skullstrip_wf_3d(name, num_trs):
    workflow = pe.Workflow(name=name)
    # name the nodes
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["bold_file", "phase_file"]),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["mask_file", "div_file"]),
        name="outputnode")

    buffernode = pe.Node(
        niu.IdentityInterface(fields=["bold_file", "phase_file", "volume"]),
        name="buffernode")
    buffernode.iterables = [('volume', np.arange(5, dtype=int))]
    workflow.connect(inputnode, 'bold_file', buffernode, 'bold_file')
    workflow.connect(inputnode, 'phase_file', buffernode, 'phase_file')

    split_bold = pe.Node(
        interface=Function(["in_file", "volume"], ["out_file"], split_file),
        name="split_bold")
    workflow.connect(buffernode, 'bold_file', split_bold, 'in_file')
    workflow.connect(buffernode, 'volume', split_bold, 'volume')

    split_phase = pe.Node(
        interface=Function(["in_file", "volume"], ["out_file"], split_file),
        name="split_phase")
    workflow.connect(buffernode, 'phase_file', split_phase, 'in_file')
    workflow.connect(buffernode, 'volume', split_phase, 'volume')

    divide_wf = init_test_division_wf(name='divide_wf')
    workflow.connect(split_bold, 'out_file', divide_wf, 'inputnode.bold_file')
    workflow.connect(split_phase, 'out_file', divide_wf, 'inputnode.phase_file')

    merge_divided = pe.JoinNode(
        interface=Function(["in_files"], ["out_file"], join_files),
        name="merge_divided",
        joinfield=["in_files"],
        joinsource="buffernode")
    workflow.connect(divide_wf, 'outputnode.out_file', merge_divided, 'in_files')
    workflow.connect(merge_divided, 'out_file', outputnode, 'div_file')

    bold_skullstrip_wf = init_skullstrip_bold_wf(name='bold_skullstrip_wf')
    workflow.connect(split_bold, 'out_file', bold_skullstrip_wf, 'inputnode.in_file')

    mask_merger = pe.JoinNode(
        interface=Function(["in_files"], ["out_file"], join_files),
        name="mask_merger",
        joinfield=["in_files"],
        joinsource="buffernode")
    workflow.connect(bold_skullstrip_wf, 'outputnode.mask_file', mask_merger, 'in_files')
    workflow.connect(mask_merger, 'out_file', outputnode, 'mask_file')
    return workflow


def init_sandbox_wf(name, output_dir, bold_mag_files, bold_phase_files, num_trs):
    workflow = pe.Workflow(name=name)
    # name the nodes
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["bold_mag_files", "bold_phase_files"]),
        name="inputnode",
        iterables=[
            ("bold_mag_files", bold_mag_files),
            ("bold_phase_files", bold_phase_files)],
        synchronize=True)
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["mask_file"]),
        name="outputnode")

    skullstrip_wf_3d = init_skullstrip_wf_3d(name='skullstrip_wf_3d', num_trs=num_trs)
    workflow.connect(inputnode, 'bold_mag_files', skullstrip_wf_3d, 'inputnode.bold_file')
    workflow.connect(inputnode, 'bold_phase_files', skullstrip_wf_3d, 'inputnode.phase_file')

    # Output mask files
    derivativesnode = pe.MapNode(
        interface=Function(["in_file", "output_dir"], ["out_file"], copy_files),
        name="derivativesnode",
        iterfield=["in_file"])
    derivativesnode.inputs.output_dir = output_dir
    workflow.connect(skullstrip_wf_3d, 'outputnode.mask_file', derivativesnode, 'in_file')

    derivativesnode2 = pe.MapNode(
        interface=Function(["in_file", "output_dir"], ["out_file"], copy_files),
        name="derivativesnode2",
        iterfield=["in_file"])
    derivativesnode2.inputs.output_dir = output_dir
    workflow.connect(skullstrip_wf_3d, 'outputnode.div_file', derivativesnode2, 'in_file')

    return workflow


if __name__ == "__main__":
    # Each input is a list of lists
    bold_mag_files = [
        sorted(glob('/bids_dataset/sub-PILOT/ses-01/func/sub-PILOT_ses-01_task-localizerDetection_run-01_echo-*_bold.nii.gz')),
        sorted(glob('/bids_dataset/sub-PILOT/ses-01/func/sub-PILOT_ses-01_task-localizerEstimation_run-01_echo-*_bold.nii.gz'))
        ]
    bold_phase_files = [
        sorted(glob('/bids_dataset/sub-PILOT/ses-01/func/sub-PILOT_ses-01_task-localizerDetection_run-01_echo-*_phase.nii.gz')),
        sorted(glob('/bids_dataset/sub-PILOT/ses-01/func/sub-PILOT_ses-01_task-localizerEstimation_run-01_echo-*_phase.nii.gz'))
        ]
    output_dir = '/outputs/'
    work_dir = '/work/'

    wf = pe.Workflow(name='sandbox_wf')
    wf.base_dir = work_dir
    for i_run, run_bold_files in enumerate(bold_mag_files):
        f = run_bold_files[0]
        num_trs = nb.load(f).shape[-1]
        run_wf = init_sandbox_wf('sandbox_wf_run-{}'.format(i_run),
                                 output_dir, run_bold_files,
                                 bold_phase_files[i_run],
                                 num_trs)
        wf.add_nodes([run_wf])
    wf.config["execution"]["crashdump_dir"] = os.path.join(
        output_dir, "log"
    )

    # Defaults
    plugin_settings = {
        "plugin": "MultiProc",
        "plugin_args": {"raise_insufficient": False, "maxtasksperchild": 1},
    }

    try:
        wf.run(**plugin_settings)
    except RuntimeError as e:
        if "Workflow did not execute cleanly" in str(e):
            print("Workflow did not execute cleanly")
        else:
            raise e
