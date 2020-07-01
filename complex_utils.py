import numpy as np


def imgs_to_complex(mag, phase):
    mag = nib.load(mag)
    phase = nib.load(phase)
    mag_data = mag.get_fdata()
    phase_data = phase.get_fdata()
    comp_data = to_complex(mag_data, phase_data)
    comp_img = nib.Nifti1Image(comp_data, mag.affine)
    return comp_img


def to_complex(mag, phase):
    real = to_real(mag, phase)
    imag = to_imag(mag, phase)
    comp = mag + (1j * phase)
    return comp


def to_mag(real, imag):
    """
    https://www.eeweb.com/quizzes/convert-between-real-imaginary-and-magnitude-phase
    """
    mag = np.sqrt((real ** 2) + (imag ** 2))
    return mag


def to_phase(real, imag):
    """
    https://www.eeweb.com/quizzes/convert-between-real-imaginary-and-magnitude-phase
    """
    phase = np.arctan2(imag, real)
    return phase


def to_real(mag, phase):
    """
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
    real = to_real(mag, phase)
    imag = np.tan(phase) * real
