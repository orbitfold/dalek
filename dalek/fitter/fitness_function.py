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
    def __init__(self, observed_v_ld_filename, observed_unc_filename, mask_filename):
        """
        Parameters:
        -----------
        observed_v_ld_filename -- filename of a file containing observed luminosity density
        observed_unc_filename -- filename of a file containing observed uncertainty
        mask_filename -- filename of a file containing the mask for bins with more than 10 elts

        All files must be loadable with np.load and the arrays must be of the same dimensionality.

        Return:
        -------
        loglikelihood
        """
        self.observed_v_ld = np.load(observed_v_ld_filename)
        self.observed_unc = np.load(observed_unc_filename)
        self.mask = np.load(mask_filename)

    def loglikelihood(mdl):
        synth_v_ld = mdl.spectrum_virtual.luminosity_density_nu.value
        synth_unc = get_trivial_poisson_uncertainty(mdl).value
        uncs = (self.observed_unc[self.mask] / 5.0) ** 2 + (synth_unc[self.mask] / 5.0) ** 2
        term1 = ((self.observed_v_ld[self.mask] - synth_v_ld[self.mask]) ** 2 / uncs).sum()
        term2 = np.log(np.sqrt(uncs)).sum()
        return -0.5 * term1 - term2
    
    def __call__(self, mdl):
        fitness = loglikelihood(mdl)
        return fitness, mdl.spectrum_virtual


fitness_function_dict = {'simple_rms': SimpleRMSFitnessFunction}
