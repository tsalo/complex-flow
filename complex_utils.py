import numpy as np
import nibabel as nib
from nilearn._utils import check_niimg


def imgs_to_complex(mag, phase):
    """
    Combine magnitude and phase images into a complex-valued nifti image.
    """
    mag = check_niimg(mag)
    phase = check_niimg(phase)
    # Convert to radians to be extra safe
    phase = to_radians(phase)
    mag_data = mag.get_fdata()
    phase_data = phase.get_fdata()
    comp_data = to_complex(mag_data, phase_data)
    comp_img = nib.Nifti1Image(comp_data, mag.affine)
    return comp_img


def split_complex(comp_img):
    """
    Split a complex-valued nifti image into magnitude and phase images.
    """
    comp_img = check_niimg(comp_img)
    comp_data = comp_img.get_fdata(dtype=comp_img.get_data_dtype())
    real = comp_data.real
    imag = comp_data.imag
    mag = to_mag(real, imag)
    phase = to_phase(real, imag)
    mag = nib.Nifti1Image(mag, comp_img.affine)
    phase = nib.Nifti1Image(phase, comp_img.affine)
    return mag, phase


def to_complex(mag, phase):
    """
    Convert magnitude and phase data into complex real+imaginary data.
    """
    real = to_real(mag, phase)
    imag = to_imag(mag, phase)
    comp = real + (1j * imag)
    return comp


def to_mag(real, imag):
    """
    Convert real and imaginary data to magnitude data.

    https://www.eeweb.com/quizzes/convert-between-real-imaginary-and-magnitude-phase
    """
    mag = np.sqrt((real ** 2) + (imag ** 2))
    return mag


def to_phase(real, imag):
    """
    Convert real and imaginary data to phase data.

    https://www.eeweb.com/quizzes/convert-between-real-imaginary-and-magnitude-phase
    """
    phase = np.arctan2(imag, real)
    return phase


def to_real(mag, phase):
    """
    Convert magnitude and phase data to real data.

    Notes
    -----
    > # the canonical formula for magnitude from complex
    > mag = np.sqrt((real ** 2) + (imag ** 2))
    > # square both sides
    > mag ** 2 = (real ** 2) + (imag ** 2)
    > # subtract real from both sides
    > imag ** 2 = (mag ** 2) - (real ** 2)
    > # take sqrt of both sides
    > imag = np.sqrt((mag ** 2) - (real ** 2))
    > # the canonical formula for phase from complex
    > phase = np.arctan(imag / real)
    > # tan is inverse of arctan, so take tan of both sides
    > imag / real = np.tan(phase)
    > # multiply both sides by real
    > imag = real * np.tan(phase)
    > # substitute solution from mag calculation
    > real * np.tan(phase) = np.sqrt((mag ** 2) - (real ** 2))
    > # square both sides
    > (real * np.tan(phase)) ** 2 = (mag ** 2) - (real ** 2)
    > # add real ** 2 to both sides
    > ((np.tan(phase) * real) ** 2) + (real ** 2) = mag ** 2
    > (np.tan(phase) ** 2 + 1) * (real ** 2) = mag ** 2
    > # sqrt of both sides
    > np.sqrt(np.tan(phase) ** 2 + 1) * real = mag
    > # divide both sides by phase term
    > real2 = mag / np.sqrt(np.tan(phase) ** 2 + 1)
    """
    real = mag / np.sqrt(np.tan(phase) ** 2 + 1)
    return real


def to_imag(mag, phase):
    """
    Convert magnitude and phase data to imaginary data.
    """
    real = to_real(mag, phase)
    imag = np.tan(phase) * real
    return imag


def to_radians(phase):
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
        not in integer format, you must still check that the units are in
        radians, and if not scale them appropriately using fslmaths.
    """
    phase_img = check_niimg(phase)
    phase_data = phase_img.get_fdata()
    imax = phase_data.max()
    imin = phase_data.min()
    scaled = (phase_data - imin) / (imax - imin)
    rad_data = 2 * np.pi * scaled
    out_img = nib.Nifti1Image(rad_data, phase_img.affine, phase_img.header)
    return out_img
