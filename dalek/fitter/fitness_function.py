from abc import ABCMeta, abstractmethod
import numpy as np
from specutils import Spectrum1D
from astropy import units as u, constants as const

class BaseFitnessFunction(object):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def get_spectrum(self, spectrum):
        """
        Function to make a spectrum out of a filename
        :param spectrum:
        :return:
        """


def w_histogram(nus, luminosities, frequencies):
    """
    Arguments:
    ----------
    nus --- a numpy array of nus
    luminosities --- a numpy array of luminosities
    frequencies --- a numpy array of frequencies

    Return:
    -------
    A two member list of bins containing nus and luminosities.
    """
    frequencies = sorted(frequencies)
    nu_bins = [[] for _ in range(len(frequencies) - 1)]
    luminosity_bins = [[] for _ in range(len(frequencies) - 1)]
    for index, _ in enumerate(frequencies[:-1]):
        for nu, luminosity in zip(nus, luminosities):
            if nu >= frequencies[index] and nu < frequencies[index + 1]:
                nu_bins[index].append(nu)
                luminosity_bins[index].append(luminosity)
    return nu_bins, luminosity_bins


def loglikelihood(nus1, luminosities1, nus2, luminosities2, bins):
    """
    Arguments:
    ----------
    nus1 --- nus of spectrum nr. 1
    luminosities1 --- luminosities of spectrum nr. 1
    nus2 --- nus of spectrum nr. 2
    luminosities2 --- luminosities of spectrum nr. 2
    bins --- bin boundaries or number of bins

    Return:
    -------
    Similarity measure of two spectra
    """
    if type(bins) == type(1):
        a = min(nus1)
        b = max(nus1)
        bins_ = [a + (b - a) / float(bins) * i for i in range(bins + 1)]
        bins = bins_
    hist1 = w_histogram(nus1, luminosities1, bins)
    hist2 = w_histogram(nus2, luminosities2, bins)
    sums1 = np.array([sum(bin_) for bin_ in hist1[0]])
    sums2 = np.array([sum(bin_) for bin_ in hist2[0]])
    stds1 = np.array([np.sqrt(np.var(np.array(bin_)) * len(bin_)) for bin_ in hist1[1]])
    stds2 = np.array([np.sqrt(np.var(np.array(bin_)) * len(bin_)) for bin_ in hist2[1]])
    return (((sums1 - sums2) ** 2 / (stds1 ** 2 + stds2 ** 2))[len(bin_) >= 5 for bin_ in hist2[0]]).sum()


class SimpleRMSFitnessFunction(BaseFitnessFunction):

    def __init__(self, spectrum):

        if hasattr(spectrum, '.flux'):
            self.observed_spectrum = spectrum
        else:
            wave, flux = np.loadtxt(spectrum, unpack=True)
            self.observed_spectrum = Spectrum1D.from_array(wave * u.angstrom,
                                                           flux * u.erg / u.s /
                                                           u.cm**2 / u.Angstrom)


        self.observed_spectrum_wavelength = self.observed_spectrum.wavelength.value
        self.observed_spectrum_flux = self.observed_spectrum.flux.value

    def __call__(self, radial1d_mdl):

        if radial1d_mdl.spectrum_virtual.flux_nu.sum() > 0:
            synth_spectrum = radial1d_mdl.spectrum_virtual
        else:
            synth_spectrum = radial1d_mdl.spectrum
        synth_spectrum_flux = np.interp(self.observed_spectrum_wavelength,
                                        synth_spectrum.wavelength.value[::-1],
                                        synth_spectrum.flux_lambda.value[::-1])

        fitness = np.sum((synth_spectrum_flux -
                          self.observed_spectrum_flux) ** 2)

        return fitness, synth_spectrum

class LogLikelihoodFitnessFunction(BaseFitnessFunction):
    def __init__(self, spectrum):
        if hasattr(spectrum, '.flux'):
            self.observed_spectrum = spectrum
        else:
            wave, flux = np.loadtxt(spectrum, unpack=True)
            self.observed_spectrum = Spectrum1D.from_array(wave * u.angstrom,
                                                           flux * u.erg / u.s /
                                                           u.cm**2 / u.Angstrom)
        self.observed_spectrum_wavelength = self.observed_spectrum.wavelength.value
        self.observed_spectrum_flux = self.observed_spectrum.flux.value

    def __call__(self, radial1d_mdl):
        if radial1d_mdl.spectrum_virtual.flux_nu.sum() > 0:
            synth_spectrum = radial1d_mdl.spectrum_virtual
        else:
            synth_spectrum = radial1d_mdl.spectrum
        fitness = loglikelihood(self.observed_spectrum_wavelength, 
                                self.observed_spectrum_flux,
                                synth_spectrum.wavelength.value[::-1], 
                                synth_spectrum.flux_lambda.value[::-1],
                                self.observed_spectrum_wavelength)
        return fitness, synth_spectrum


fitness_function_dict = {'simple_rms': SimpleRMSFitnessFunction}
