# complex-flow
Initial work on a complex-valued fMRI preprocessing workflow.

## Goals

- Processing of both magnitude and phase data for both single- and multi-echo data
- BIDS Derivatives-compatible outputs
- A user-friendly CLI
- Support for phase-based denoising within the workflow (when applicable), including phase regression
- Dynamic distortion correction, for multi-echo data, with DOCMA
- Support for field maps and single-band reference images
- Long-term, I want to move any workflows produced here into tools like SDCFlows and niworkflows
