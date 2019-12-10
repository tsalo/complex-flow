#!/bin/bash

# Build Dockerfile_devel (within fmriprep folder)
# docker build -f Dockerfile_devel -t fmriprep/test:latest .

# Run the Docker image
docker run -ti --rm \
  -v /Users/tsalo/Documents/dset-pilot-reduced:/bids_dataset:ro \
  -v /Users/tsalo/Documents/docma-outputs:/outputs \
  -v /Users/tsalo/Documents/docma-work:/work \
  -v /Users/tsalo/Documents/freesurfer_license.txt:/freesurfer_license.txt \
  -v /Users/tsalo/Documents/tsalo/complex-flow:/home/complex-flow \
  -v /Users/tsalo/Documents/tsalo/sdcflows:/home/sdcflows \
  --entrypoint=bash \
  poldracklab/sdcflows:latest

# Run workflow
pip install niflow-nipype1-workflows
pip install git+https://github.com/bids-standard/pybids
pip uninstall sdcflows -y
cd /home/sdcflows
python setup.py develop
cd /home/complex-flow
python run.py /bids_dataset /outputs --participant-label PILOT \
  --task-label localizerDetection \
  -w /work --nthreads 2 --graph

docker run -ti --rm \
  -v /Users/tsalo/Documents/dset-for-openneuro:/bids_dataset:ro \
  -v /Users/tsalo/Documents/sandbox-outputs:/outputs \
  -v /Users/tsalo/Documents/sandbox-work:/work \
  -v /Users/tsalo/Documents/freesurfer_license.txt:/freesurfer_license.txt \
  -v /Users/tsalo/Documents/tsalo/complex-flow:/home/complex-flow \
  --entrypoint=bash \
  fmriprep/test:latest
cd /home/complex-flow
python test_docma.py
