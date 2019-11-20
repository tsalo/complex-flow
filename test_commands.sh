#!/bin/bash

# Build Dockerfile_devel (within fmriprep folder)
# docker build -f Dockerfile_devel -t fmriprep/test:latest .

# Run the Docker image
docker run -ti --rm \
  -v /Users/tsalo/Documents/heudiconv-outputs-reproin:/bids_dataset:ro \
  -v /Users/tsalo/Documents/complex-outputs:/outputs \
  -v /Users/tsalo/Documents/complex-work:/work \
  -v /Users/tsalo/Documents/freesurfer_license.txt:/freesurfer_license.txt \
  -v /Users/tsalo/Documents/tsalo/complex-flow:/home/complex-flow \
  -v /Users/tsalo/Documents/tsalo/sdcflows:/home/sdcflows \
  -v /Users/tsalo/Documents/tsalo/phaseprep:/home/phaseprep \
  --entrypoint=bash \
  fmriprep/test:latest

# Run workflow
pip install niflow-nipype1-workflows
pip install git+https://github.com/bids-standard/pybids
#pip install git+https://github.com/mattcieslak/sdcflows@phase1phase2
pip uninstall sdcflows -y
cd /home/sdcflows
python setup.py develop
cd /home/phaseprep
python setup.py develop
cd /home/complex-flow
python run.py /bids_dataset /outputs --participant-label PILOT \
  -w /work --nthreads 1 --graph

docker run -ti --rm \
  -v /Users/tsalo/Documents/dset-for-openneuro:/bids_dataset:ro \
  -v /Users/tsalo/Documents/sandbox-outputs:/outputs \
  -v /Users/tsalo/Documents/sandbox-work:/work \
  -v /Users/tsalo/Documents/freesurfer_license.txt:/freesurfer_license.txt \
  -v /Users/tsalo/Documents/tsalo/complex-flow:/home/complex-flow \
  --entrypoint=bash \
  fmriprep/test:latest
cd /home/complex-flow
python sandbox.py
