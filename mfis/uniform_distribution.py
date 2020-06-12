import numpy as np
from .input_distribution import InputDistribution

class IndependentUniformDistribution(InputDistribution):

    def __init__(self, bounds, seed=None):
        self._bounds = bounds
        if seed is not None:
            np.random.seed(seed)

    def draw_samples(self, num_samples):

        samples = np.zeros((num_samples, len(self._bounds)))
        
        for i,bound in enumerate(self._bounds):
            samples[:,i] = np.random.uniform(bound[0], bound[1], num_samples)

        return samples

    def evaluate_pdf(self, samples):
        p = np.ones((samples.shape[0],1))
    
        for i,bound in enumerate(self._bounds):
            p *= 1/(bound[1] - bound[0])
        indices_outside_bounds_samples = \
                self.indices_of_samples_outside_bounds(samples)
        p[indices_outside_bounds_samples] = 0
        return p
    
    def check_all_samples_within_bounds(self, samples):
        for i,bound in enumerate(self._bounds):
            if not (np.min(samples[:,i]) >= bound[0] and 
            np.max(samples[:,i]) <= bound[1]):
                raise ValueError("Not all samples within bounds")
                
    def indices_of_samples_outside_bounds(self, samples):
        outside_sample_indices = []
        for i,bound in enumerate(self._bounds):
            outside_indicators = ((samples[:,i] <= bound[0]) | 
                                  (samples[:,i] >= bound[1]))
            indices = [i for i, x in enumerate(outside_indicators) if x]
            outside_sample_indices = outside_sample_indices + indices   
        
        return np.unique(outside_sample_indices)             
                
