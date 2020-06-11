"""
Created on Wed Jun 10 11:26:19 2020

@author: dacole2
"""

import numpy as np
from scipy.stats import multivariate_normal
from .input_distribution import InputDistribution

class MultivariateNormalDistribution(InputDistribution):

    def __init__(self, mean, cov, seed=None):
        self._mean = mean
        if isinstance(cov, np.ndarray) and cov.ndim > 1:
            self._cov = cov
        else:
            self._cov = np.diag(cov)
            
        self._validate_dimensions_of_inputs()
        if seed is not None:
            np.random.seed(seed)

    def draw_samples(self, num_samples):

        samples = multivariate_normal.rvs(self._mean, self._cov, num_samples)
        
        return samples

    def evaluate_pdf(self, samples):
        self._samples_within_bounds(samples)
        
        p = np.ones((len(samples),1))
    
        for i in range(len(samples)):
            p[i] = multivariate_normal.pdf(samples[i,:], self._mean, self._cov)

        return p
    
    def _validate_dimensions_of_inputs(self):
        if not self._cov.shape[0] == self._cov.shape[1]:
            raise ValueError("Covariance isn't a square array")
        if not len(self._mean) == self._cov.shape[0]:
            raise ValueError("Covariance and Mean dimensions don't match")
            
            
    def _samples_within_bounds(self, samples):
        for i,var in enumerate(self._cov):
            if not (np.min(samples[:,i]) > self._mean[i]-6*np.sqrt(var[i]) and
                    np.max(samples[:,i]) < self._mean[i]+6*np.sqrt(var[i])):
                raise ValueError("Not all samples within bounds")
    