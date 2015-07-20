import pytest
import numpy as np
import matplotlib.pyplot as plt
from dalek.fitter.fitness_function import w_histogram, loglikelihood

def test_w_histogram():
    pass

def test_loglikelihood():
    spectrum = np.loadtxt('out.dat')
    assert(loglikelihood(spectrum[:,0], spectrum[:,1], spectrum[:,0], spectrum[:,1], 100) == 0)

if __name__ == '__main__':
    test_loglikelihood()
    
