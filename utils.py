"""
Miscellaneous utility functions.
"""


def recover_kspace(magnitude_file, phase_file, out_real_file=None, out_imag_file=None):
    """
    Convert raw magnitude and phase data into effective k-space data, split
    into real and imaginary components.
    """
    import numpy as np
    import nibabel as nib
    import os.path as op

    magnitude_img = nib.load(magnitude_file)
    phase_data = nib.load(phase_file).get_data()
    magnitude_data = magnitude_img.get_data()
    kspace_data = np.zeros(phase_data.shape, dtype=complex)

    cmplx_data = np.vectorize(complex)(magnitude_data, phase_data)
    for i_vol in range(cmplx_data.shape[3]):
        for j_slice in range(cmplx_data.shape[2]):
            slice_data = cmplx_data[:, :, j_slice, i_vol]
            slice_kspace = np.fft.ifft(slice_data)
            kspace_data[:, :, j_slice, i_vol] = slice_kspace

    kspace_real_data, kspace_imag_data = kspace_data.real, kspace_data.imag
    kspace_real_img = nib.Nifti1Image(kspace_real_data, magnitude_img.affine, magnitude_img.header)
    kspace_imag_img = nib.Nifti1Image(kspace_imag_data, magnitude_img.affine, magnitude_img.header)
    if out_real_file is None:
        out_real_file = out_file = op.abspath('complexReal_{}'.format(op.basename(magnitude_file)))
    if out_imag_file is None:
        out_imag_file = out_file = op.abspath('complexImag_{}'.format(op.basename(magnitude_file)))
    kspace_real_img.to_filename(out_real_file)
    kspace_imag_img.to_filename(out_imag_file)
    return out_real_file, out_imag_file


def convert_to_radians(phase_file, out_file=None):
    """
    Adapted from
    https://github.com/poldracklab/sdcflows/blob/
    659c2508ecef810c3acadbe808560b44d22801f9/sdcflows/interfaces/fmap.py#L94

    Ensure that phase images are in a usable range for unwrapping.

    From the FUGUE User guide::

        If you have seperate phase volumes that are in integer format then do:

        fslmaths orig_phase0 -mul 3.14159 -div 2048 phase0_rad -odt float
        fslmaths orig_phase1 -mul 3.14159 -div 2048 phase1_rad -odt float

        Note that the value of 2048 needs to be adjusted for each different
        site/scanner/sequence in order to be correct. The final range of the
        phase0_rad image should be approximately 0 to 6.28. If this is not the
        case then this scaling is wrong. If you have separate phase volumes are
        not in integer format, you must still check that the units are in radians,
        and if not scale them appropriately using fslmaths.
    """
    import os.path as op
    import numpy as np
    import nibabel as nib

    img = nib.load(phase_file)
    phase_data = img.get_data()
    imax = phase_data.max()
    imin = phase_data.min()
    scaled = (phase_data - imin) / (imax - imin)
    rad_data = 2 * np.pi * scaled
    out_img = nib.Nifti1Image(rad_data, img.affine, img.header)
    if out_file is None:
        out_file = op.abspath('rescaled_{}'.format(op.basename(phase_file)))
    out_img.to_filename(out_file)
    return out_file


def get_fmap_tediff(metadata):
    """
    Get difference in field map phase images' echo times.
    """
    delta_te = metadata['EchoTime2'] - metadata['EchoTime1']
    return delta_te


def compute_phasediff(phase_files, phase_metadata, out_file=None):
    """
    Compute phase-difference image in rad/s from two phase files in rad/s.
    """
    import os.path as op
    import numpy as np
    import nibabel as nib

    # Select first two echoes
    phase_files = phase_files[:2]
    phase_metadata = phase_metadata[:2]
    imgs = [nib.load(pf) for pf in phase_files]
    data = [img.get_data() for img in imgs]
    te_diff = 1000. * (phase_metadata[1]['EchoTime'] - phase_metadata[0]['EchoTime'])
    data = 1000. * (data[1] - data[0]) / te_diff
    out_img = nib.Nifti1Image(data, imgs[0].affine, imgs[0].header)
    if out_file is None:
        out_file = op.abspath('phasediffed_{}'.format(op.basename(phase_files[0])))
    out_img.to_filename(out_file)
    return out_file


def fake_unwrap(magnitude_file, phase_file):
    """
    An identity function used as a placeholder for PRELUDE,
    which can take a long time.
    """
    unwrapped_phase_file = phase_file
    return unwrapped_phase_file


def get_slice_timing(metadata):
    """
    Get slice timing information (in seconds) from metadata dictionary.
    """
    return metadata['SliceTiming']


def pick_first(func):
    """
    Used to grab first echo for multi-echo data
    """
    if isinstance(func, list):
        return func[0]
    else:
        return func


def pick_second(func):
    """
    Used to grab second echo for multi-echo data
    """
    if isinstance(func, list):
        return func[1]
    else:
        return func


def get_other_echoes(layout, func_obj):
    """
    Get full set of multi-echo fMRI files associated with one of the files.
    """
    entity_dict = func_obj.get_entities().copy()
    entity_dict.pop('echo')
    files = []
    for echo in sorted(layout.get_echoes(**entity_dict)):
        bold_mag_files = layout.get(echo=echo, **entity_dict)
        assert len(bold_mag_files) == 1
        files.append(bold_mag_files[0])
    return files


def get_phase(layout, func_obj):
    """
    Get phase file associated with a given BOLD file.
    """
    entity_dict = func_obj.get_entities().copy()
    entity_dict.pop('suffix')
    files = layout.get(suffix='phase', **entity_dict)
    assert len(files) <= 1
    if len(files) == 0:
        return None
    else:
        file_ = files[0]
    return file_


def get_sbref(layout, func_obj, reconstruction='magnitude'):
    """
    Get single-band reference image associated with a functional run.
    """
    entity_dict = func_obj.get_entities().copy()
    entity_dict.pop('suffix')
    files = layout.get(suffix='sbref', reconstruction=reconstruction, **entity_dict)
    assert len(files) <= 1
    if len(files) == 0:
        return None
    else:
        file_ = files[0]
    return file_


def collect_data(layout, participant_label, ses=None, task=None, run=None):
    """
    Collect required data from the dataset.
    """
    # get all the preprocessed fmri images.
    bold_query = {
        'subject': participant_label,
        'datatype': 'func',
        'suffix': 'bold',
        'extension': ['nii', 'nii.gz'],
        'echo': 1,
    }
    t1w_query = {
        'subject': participant_label,
        'datatype': 'anat',
        'suffix': 'T1w',
        'extension': ['nii', 'nii.gz'],
    }

    if task:
        bold_query['task'] = task
    if run:
        bold_query['run'] = run
    if ses:
        bold_query['session'] = ses

    first_echo_files = layout.get(**bold_query)
    bold_mag_files = [get_other_echoes(layout, f) for f in first_echo_files]
    bold_phase_files = [[get_phase(layout, f) for f in r] for r in bold_mag_files]
    sbref_mag_files = [[get_sbref(layout, f, reconstruction='magnitude') for f in r] for r in bold_mag_files]
    sbref_phase_files = [[get_sbref(layout, f, reconstruction='phase') for f in r] for r in bold_mag_files]
    fmaps = [layout.get_fieldmap(f.path) for f in first_echo_files]
    fmap_mag1_files = [d['magnitude1'] for d in fmaps]
    fmap_mag2_files = [d['magnitude2'] for d in fmaps]
    fmap_phasediff_files = [d['phasediff'] for d in fmaps]
    t1w_files = layout.get(**t1w_query)

    # Convert BIDS files to strings
    bold_mag_files = [[f.path for f in r] for r in bold_mag_files]
    bold_phase_files = [[f.path for f in r] for r in bold_phase_files]
    sbref_mag_files = [[f.path for f in r] for r in sbref_mag_files]
    sbref_phase_files = [[f.path for f in r] for r in sbref_phase_files]
    t1w_files = [f.path for f in t1w_files]

    bold_mag_metadata = [[layout.get_metadata(f) for f in r] for r in bold_mag_files]
    bold_phase_metadata = [[layout.get_metadata(f) for f in r] for r in bold_phase_files]
    sbref_mag_metadata = [[layout.get_metadata(f) for f in r] for r in sbref_mag_files]
    sbref_phase_metadata = [[layout.get_metadata(f) for f in r] for r in sbref_phase_files]
    fmap_mag1_metadata = [layout.get_metadata(f) for f in fmap_mag1_files]
    fmap_mag2_metadata = [layout.get_metadata(f) for f in fmap_mag2_files]
    fmap_phasediff_metadata = [layout.get_metadata(f) for f in fmap_phasediff_files]
    t1w_metadata = [layout.get_metadata(f) for f in t1w_files]

    # Compile into dictionary
    data = {'bold_mag_files': bold_mag_files,
            'bold_mag_metadata': bold_mag_metadata,
            'bold_phase_files': bold_phase_files,
            'bold_phase_metadata': bold_phase_metadata,
            'sbref_mag_files': sbref_mag_files,
            'sbref_mag_metadata': sbref_mag_metadata,
            'sbref_phase_files': sbref_phase_files,
            'sbref_phase_metadata': sbref_phase_metadata,
            'fmap_mag1_files': fmap_mag1_files,
            'fmap_mag1_metadata': fmap_mag1_metadata,
            'fmap_mag2_files': fmap_mag2_files,
            'fmap_mag2_metadata': fmap_mag2_metadata,
            'fmap_phasediff_files': fmap_phasediff_files,
            'fmap_phasediff_metadata': fmap_phasediff_metadata,
            't1w_files': t1w_files,
            't1w_metadata': t1w_metadata,
            }
    return data
