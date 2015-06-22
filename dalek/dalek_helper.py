from tardis import run_tardis
from dalek.fitter.fitness_function import SimpleRMSFitnessFunction
import yaml

def get_fitness(o, si, s, ca, fe, co, ni, mg, ti, cr, c, 
                luminosity_requested, velocity_start):
    with open('tardis_02bo_kurucz.yml', 'r') as fd:
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
    fn = SimpleRMSFitnessFunction('spectrum.dat')
    value, _ = fn(mdl)
    return value

if __name__ == '__main__':
    print get_fitness(0.001574, 0.575, 0.115, 0.013333, 0.02, 0.023609, 0.03208, 0.2, 0.00016, 0.0008, 0.0005, 1.3265352399286673e+43, 10050.0)
