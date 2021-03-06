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
    mag = abs(comp_data)
    phase = to_phase(real, imag)
    mag = nib.Nifti1Image(mag, comp_img.affine)
    phase = nib.Nifti1Image(phase, comp_img.affine)
    return mag, phase


def to_complex(mag, phase):
    """
    Convert magnitude and phase data into complex real+imaginary data.

    Should be equivalent to cmath.rect.
    """
    comp = mag * (np.cos(phase) + np.sin(phase)*1j)
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

    Equivalent to cmath.phase.

    https://www.eeweb.com/quizzes/convert-between-real-imaginary-and-magnitude-phase
    """
    phase = np.arctan2(imag, real)
    return phase


def to_real(mag, phase):
    """
    Convert magnitude and phase data to real data.
    """
    comp = to_complex(mag, phase)
    real = comp.real
    return real


def to_imag(mag, phase):
    """
    Convert magnitude and phase data to imaginary data.
    """
    comp = to_complex(mag, phase)
    imag = comp.imag
    return imag


def to_radians(phase):
    """
    Adapted from
    https://github.com/poldracklab/sdcflows/blob/
    659c2508ecef810c3acadbe808560b44d22801f9/sdcflows/interfaces/fmap.py#L94

    Ensure that phase images are in a usable range for unwrapping [0, 2pi).

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


def split_multiecho_volumewise(echo_imgs):
    """
    Take 4D echo-specific images, split them by volume, and concatenate each
    volume across echoes, as input for ROMEO.
    """
    from nilearn import image
    echo_imgs = [check_niimg(ei) for ei in echo_imgs]
    out_imgs = []
    for i in range(echo_imgs[0].shape[3]):
        vol_imgs = []
        for j_echo, echo_img in enumerate(echo_imgs):
            vol_imgs.append(image.index_img(echo_img, i))
        out_imgs.append(image.concat_imgs(vol_imgs))
    return out_imgs
