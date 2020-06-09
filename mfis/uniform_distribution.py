import numpy as np

class IndependentUniformDistribution:

    def __init__(self, bounds, seed=None):
        self._bounds = bounds
        if seed is not None:
            np.random.seed(seed)

    def draw_samples(self, num_samples):

        samples = np.zeros((num_samples, len(self._bounds)))
        
        for i,bound in enumerate(self._bounds):
            samples[:,i] = np.random.uniform(bound[0], bound[1], num_samples)

        return samples
