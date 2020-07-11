from tedana.workflows import tedana_workflow
from glob import glob
import os.path as op

in_dirs = [
    '/Users/tsalo/Documents/Laird_DIVA/dwidenoised',
    '/Users/tsalo/Documents/Laird_DIVA/complex-dwidenoised',
    '/Users/tsalo/Documents/Laird_DIVA/dset/sub-Blossom/ses-02/func',
]
tasks = ['localizerDetection', 'localizerEstimation']
echo_times = [11.8, 28.04, 44.28, 60.52]

for in_dir in in_dirs:
    for task in tasks:
        pattern = op.join(in_dir, '*{}*_bold.nii.gz'.format(task))
        files = sorted(glob(pattern))
        out_dir = op.join(in_dir, 'tedana-{}'.format(task))
        if not op.isdir(out_dir):
            tedana_workflow(
                files, echo_times, fittype='curvefit',
                out_dir=out_dir,
                fixed_seed=1
            )
