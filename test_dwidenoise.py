"""
Run dwidenoise on either magnitude-only or complex multi-echo data.
"""
import os
import os.path as op
from glob import glob
import subprocess
import sys

from complex_utils import imgs_to_complex, split_complex


def run(command, env=None):
    """Run a command with specific environment information.

    Parameters
    ----------
    command: command to be sent to system
    env: parameters to be added to environment
    """
    merged_env = os.environ
    if env:
        merged_env.update(env)

    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True,
                               env=merged_env)
    while True:
        line = process.stdout.readline()
        line = line.decode('utf-8')
        sys.stdout.write(line)
        sys.stdout.flush()
        if line == '' and process.poll() is not None:
            break

    if process.returncode != 0:
        raise Exception("Non zero return code: {}\n"
                        "{}\n\n{}".format(
                            process.returncode, command, process.stdout.read()
                        ))
    return process.returncode


if __name__ == '__main__':

    in_dir = '/scratch/tsalo006/dset/sub-Blossom/ses-02/func/'

    magnitude_files = sorted(
        glob(
            op.join(in_dir, '*_task-localizer*_run-1_*_bold.nii.gz')
        )
    )
    phase_files = [f.replace('_bold', '_phase') for f in magnitude_files]

    for i_echo, mag_file in enumerate(magnitude_files):
        base_name = op.join('/scratch/tsalo006/', op.basename(mag_file))
        pha_file = phase_files[i_echo]
        cmplx_file = base_name.replace('_bold', '_complex')
        cmplx_img = imgs_to_complex(mag_file, pha_file)
        cmplx_img.to_filename(cmplx_file)
        print('Saved {}'.format(cmplx_file))
        # dwidenoise on magnitude-only data
        out_file = base_name.replace('_bold', '_desc-dwiDenoised_bold')
        cmd = ('singularity exec --cleanenv '
               '/scratch/tsalo006/brainlife_mrtrix3_3.0.0.sif dwidenoise '
               '-nthreads 4 {} {}').format(mag_file, out_file)
        run(cmd)

        # dwidenoise on complex data
        cmplx_denoised = base_name.replace('_bold', '_desc-complexDwiDenoised_complex')
        cmd = ('singularity exec --cleanenv '
               '/scratch/tsalo006/brainlife_mrtrix3_3.0.0.sif dwidenoise '
               '-nthreads 4 {} {}').format(cmplx_file, cmplx_denoised)
        run(cmd)

        # split denoised complex
        mag_denoised_cmplx, pha_denoised_cmplx = split_complex(cmplx_denoised)
        mag_denoised_cmplx_file = base_name.replace('_bold', '_desc-complexDwiDenoised_bold')
        mag_denoised_cmplx.to_filename(mag_denoised_cmplx_file)
        pha_denoised_cmplx_file = base_name.replace('_bold', '_desc-complexDwiDenoised_phase')
        pha_denoised_cmplx.to_filename(pha_denoised_cmplx_file)
        print('Saved {}'.format(mag_denoised_cmplx_file))
