from abc import ABCMeta, abstractmethod

from dalek.parallel import ParameterCollection
import numpy as np
import random

class BaseOptimizer(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @staticmethod
    def normalize_parameter_collection(parameter_collection):
        """
        Normalize the abundances, assuming all abundances have the prefix
        'model.abundances'.

        Parameters
        ----------

        parameter_collection: ~dalek.parallel.ParameterCollection

        Returns
        -------
            : ~dalek.parallel.ParameterCollection

        """

        abundance_columns = [item for item in parameter_collection.columns
                             if item.startswith('model.abundances')]

        parameter_collection[abundance_columns]
        parameter_collection[abundance_columns] = (parameter_collection[
                                                       abundance_columns].div(
            parameter_collection[abundance_columns].sum(axis=1), axis=0))

        return parameter_collection

    def split_parameter_collection(self, parameter_collection):
        fitness = parameter_collection['dalek.fitness']
        return fitness, parameter_collection[
            self.parameter_config.parameter_names]


class RandomSampling(BaseOptimizer):
    def __init__(self, fitter_configuration):
        raise NotImplementedError
        self.fitter_configuration = fitter_configuration
        self.lbounds = self.fitter_configuration.lbounds
        self.ubounds = self.fitter_configuration.ubounds
        self.n = self.fitter_configuration.number_of_samples

    def __call__(self, parameter_collection):
        parameters = []
        for _ in range(self.n):
            parameters.append(np.random.uniform(lbounds, ubounds))
        return ParameterCollection(np.array(parameters), columns=self.fitter_configuration.parameter_names)

class NoiseMeasurement(BaseOptimizer):
    def __init__(self, fitter_configuration):
        raise NotImplementedError
        self.fitter_configuration = fitter_configuration
        self.n = self.fitter_configuration.number_of_samples

    def __call__(self, parameter_collection):
        parameters = [random.randint(0, 2**16) for _ in range(self.n)]
        return ParameterCollection(np.array(parameters), columns=['montecarlo.seed'])

class LuusJaakolaOptimizer(BaseOptimizer):
    def __init__(self, parameter_conf, number_of_samples, **kwargs):
        self.parameter_config = parameter_conf
        self.x = (self.parameter_config.lbounds +
                  self.parameter_config.ubounds) * 0.5
        self.n = number_of_samples
        self.d = np.array(self.parameter_config.ubounds -
                          self.parameter_config.lbounds) * 0.5

    def __call__(self, parameter_collection):

        fitness, split_param_collection = self.split_parameter_collection(
            parameter_collection)
        best_fit = split_param_collection.ix[fitness.argmin()]

        best_x = best_fit[self.parameter_config.parameter_names].values
        new_parameters = [best_x]
        lbounds = self.parameter_config.lbounds
        ubounds = self.parameter_config.ubounds
        for _ in range(self.n):
            new_parameters.append(np.random.uniform(np.clip(best_x - self.d,
                                                            lbounds, ubounds),
                                                    np.clip(best_x + self.d,
                                                            lbounds, ubounds)))
        self.d *= 0.95
        new_parameter_collection = ParameterCollection(np.array(new_parameters),
                                   columns=self.parameter_config.parameter_names)
        return new_parameter_collection

class DEOptimizer(BaseOptimizer):
    def __init__(self, parameter_conf, population_size):
        self.population = None
        self.fitness = None
        self.parameter_config = parameter_conf
        self.dim = len(self.parameter_config.parameter_names)
        self.cr = 0.9
        self.f = 0.5
        if population_size < 4:
            raise ValueError('Need at least 4 samples for differential evolution')
        self.n = population_size

        self.lbounds = np.array(self.parameter_config.lbounds)
        self.ubounds = np.array(self.parameter_config.ubounds)

    def violates_bounds(self, x):
        return any(x < self.lbounds) or any(x > self.ubounds)
        
    def __call__(self, parameter_collection):
        fitness, split_param_collection = self.split_parameter_collection(
            parameter_collection)
        if self.population is None:
            self.population = np.array(split_param_collection.values)
            self.fitness = np.array(fitness.values)
        else:
            new_population = np.array(split_param_collection.values)
            for index, vector in enumerate(self.population):
                if fitness[index] < self.fitness[index]:
                    self.population[index] = np.array(new_population[index])
                    self.fitness[index] = fitness[index]
        candidates = self.population.copy()
        for index, vector in enumerate(self.population):
            indices = [i for i in range(self.n) if i != index]
            random.shuffle(indices)
            i1, i2, i3 = indices[:3]
            a, b, c = self.population[i1], self.population[i2], self.population[i3]
            r_ = np.random.randint(0, self.dim)
            for j, x in enumerate(vector):
                ri = np.random.random()
                if ri < self.cr or j == r_:
                    candidates[index][j] = a[j] + self.f * (b[j] - c[j])
                else:
                    candidates[index][j] = x
            if self.violates_bounds(candidates[index]):
                candidates[index] = np.array(vector)
	params = ParameterCollection(np.array(candidates),
                                 columns=self.parameter_config.parameter_names)
	return params

class PSOOptimizerGbest(BaseOptimizer):
    def __init__(self, parameter_conf, number_of_samples, **kwargs):
        self.parameter_config = parameter_conf
        self.x = None
        self.px = None
        self.v = None
        self.c1 = 2.05
        self.c2 = 2.05
        self.chi = 2.0 / (self.c1 + self.c2 - 2.0 + np.sqrt((self.c1 + self.c2) ** 2 - 4.0 * (self.c1 + self.c2)))
        self.n = number_of_samples
        self.lbounds = np.array(self.parameter_config.lbounds)
        self.ubounds = np.array(self.parameter_config.ubounds)

    def neighbourhood(self, index):
        return [i for i in range(self.n) if i != index]

    def violates_bounds(self, x):
        return any(x < self.lbounds) or any(x > self.ubounds)
        
    def __call__(self, parameter_collection):
        candidates = np.array([x[:-1] for x in parameter_collection.values])
        if self.x is None:
            self.x = np.array(candidates)
            self.y = np.array([x[-1] for x in parameter_collection.values])
            self.px = np.array(candidates)
            self.py = np.array([x[-1] for x in parameter_collection.values])
            self.v = np.zeros(self.x.shape)
        else:
            for index, candidate in enumerate(parameter_collection.values):
                if candidate[-1] < self.py[index]:
                    self.px[index] = np.array(candidate[:-1])
                    self.py[index] = candidate[-1]
        gx = np.zeros(self.x.shape)
        for index, _ in enumerate(self.x):
            neighbours = self.neighbourhood(index)
            nx, ny = min(zip([self.px[i] for i in neighbours], 
                             [self.py[i] for i in neighbours]), 
                         key=lambda pair: pair[1])
            gx[index] = np.array(nx)
        self.v = self.chi * (self.v + 
                             self.c1 * np.random.sample(self.x.shape) * (self.px - self.x) + 
                             self.c2 * np.random.sample(self.x.shape) * (gx - self.x))
        candidates = np.array(self.x + self.v)
        for index, x in enumerate(candidates):
            if not self.violates_bounds(x):
                candidates[index] = np.array(x)
            else:
                candidates[index] = np.array(self.px[index])
        self.x += self.v
        params = ParameterCollection(
            candidates, columns=self.parameter_config.parameter_names)
        
        return params
        

optimizer_dict = {'random_sampling': RandomSampling,
                  'luus_jaakola': LuusJaakolaOptimizer,
                  'devolution': DEOptimizer,
                  'pso': PSOOptimizerGbest}
