"""
Split files for a test of ROMEO.

ROMEO files are split by volume and concatenated across echoes.
Magnitude and phase data are in separate files, both with ascending echo times.
"""
import os.path as op
from os import makedirs
from glob import glob

from complex_utils import split_multiecho_volumewise, to_radians

echo_times = [11.8, 28.04, 44.28, 60.52]
in_dir = '/Users/tsalo/Documents/Laird_DIVA/dset/sub-Blossom/ses-02/func/'
out_dir = op.abspath('temp/')

task = 'task-localizerEstimation'
mag_pattern = op.join(in_dir, '*{}*_bold.nii.gz'.format(task))
mag_files = sorted(glob(mag_pattern))
pha_pattern = op.join(in_dir, '*{}*_phase.nii.gz'.format(task))
pha_files = sorted(glob(pha_pattern))

makedirs(out_dir, exist_ok=True)

new_mag_imgs = split_multiecho_volumewise(mag_files)
for i, nf in enumerate(new_mag_imgs):
    out_file = op.join(out_dir, 'bold_v{0:04d}.nii'.format(i))
    nf.to_filename(out_file)

pha_imgs = [to_radians(img) for img in pha_files]
new_pha_imgs = split_multiecho_volumewise(pha_imgs)
for i, nf in enumerate(new_pha_imgs):
    out_file = op.join(out_dir, 'pha_v{0:04d}.nii'.format(i))
    nf.to_filename(out_file)
