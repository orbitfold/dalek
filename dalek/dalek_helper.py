from tardis import run_tardis
from scipy.stats import gamma
import numpy as np
from dalek.fitter.fitness_function import LogLikelihoodFitnessFunction, SimpleRMSFitnessFunction
import yaml
from tardis.stats.base import get_trivial_poisson_uncertainty

def loglikelihood(mdl, observed_v_ld_filename, observed_unc_filename, mask_filename):
    observed_v_ld = np.load(observed_v_ld_filename)
    observed_unc = np.load(observed_unc_filename)
    mask = np.load(mask_filename)
    synth_v_ld = mdl.spectrum_virtual.luminosity_density_nu.value
    synth_unc = get_trivial_poisson_uncertainty(mdl).value
    uncs = (observed_unc[mask] / 5.0) ** 2 + (synth_unc[mask] / 5.0) ** 2
    term1 = ((observed_v_ld[mask] - synth_v_ld[mask]) ** 2 / uncs).sum()
    term2 = np.log(np.sqrt(uncs)).sum()
    return -0.5 * term1 - term2

def get_fitness(o, si, s, ca, fe, co, ni, mg, ti, cr, c, 
                luminosity_requested, velocity_start):
    luminosity_requested = 4e42 + (3e43 - 4e42) * luminosity_requested
    velocity_start = 7000.0 + (15000.0 - 7000.0) * velocity_start
    arr = np.array([o, si, s, ca, fe, co, ni, mg, ti, cr, c])
    o, si, s, ca, fe, co, ni, mg, ti, cr, c = gamma.pdf(arr, 1, scale=1, loc=0) / arr.sum()
    with open('/home/hpc/pr94se/di73kuj/optimization/PolyChord/tardis_02bo_kurucz.yml', 'r') as fd:
        conf_dict = yaml.load(fd)
    conf_dict['model']['abundances']['O'] = o
    conf_dict['model']['abundances']['Si'] = si
    conf_dict['model']['abundances']['S'] = s
    conf_dict['model']['abundances']['Ca'] = ca
    conf_dict['model']['abundances']['Fe'] = fe
    conf_dict['model']['abundances']['Co'] = co
    conf_dict['model']['abundances']['Ni'] = ni
    conf_dict['model']['abundances']['Mg'] = mg
    conf_dict['model']['abundances']['Ti'] = ti
    conf_dict['model']['abundances']['Cr'] = cr
    conf_dict['model']['abundances']['C'] = c
    conf_dict['supernova']['luminosity_requested'] = "%f erg/s" % luminosity_requested
    conf_dict['model']['structure']['velocity']['start'] = "%f km/s" % velocity_start
    mdl = run_tardis(conf_dict)
    value = loglikelihood(
        mdl, '/home/hpc/pr94se/di73kuj/optimization/PolyChord/observed_v_ld.npy',
        '/home/hpc/pr94se/di73kuj/optimization/PolyChord/observed_unc.npy',
        '/home/hpc/pr94se/di73kuj/optimization/PolyChord/mask.npy')
    return value

if __name__ == '__main__':
    print get_fitness(0.001574, 0.575, 0.115, 0.013333, 0.02, 0.023609, 0.03208, 0.2, 0.00016, 0.0008, 0.0005, 0.5, 0.5)
