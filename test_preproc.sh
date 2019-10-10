#!/bin/bash

# Build Dockerfile_devel (within fmriprep folder)
# docker build -f Dockerfile_devel -t fmriprep/test:latest .

# Run the Docker image
docker run -ti --rm \
  -v /Users/tsalo/Desktop/pilot_dset:/bids_dataset:ro \
  -v /Users/tsalo/Documents/complex-outputs:/outputs \
  -v /Users/tsalo/Documents/complex-work:/work \
  -v /Users/tsalo/Documents/freesurfer_license.txt:/freesurfer_license.txt \
  -v /Users/tsalo/Documents/nbc/diva-project:/home/diva-project \
  --entrypoint=bash \
  fmriprep/test:latest

# Run workflow
#pip install git+https://github.com/nipy/nipype
pip install git+https://github.com/bids-standard/pybids
cd /home/diva-project
python run_preproc.py /bids_dataset /outputs --participant-label TEST1 \
  -w /work --nthreads 1 --graph
