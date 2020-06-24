"""
Created on Wed Jun 10 11:26:19 2020

@author: dacole2
"""
from scipy.stats import multivariate_normal
import numpy as np
from .input_distribution import InputDistribution

class MultivariateNormalDistribution(InputDistribution):
    """
    Coming soon.
    """
    def __init__(self, mean, cov, seed=None):
        self.mean_ = mean
        if isinstance(cov, np.ndarray) and cov.ndim > 1:
            self.cov_ = cov
        else:
            self.cov_ = np.diag(cov)

        self._validate_dimensions_of_inputs()
        if seed is not None:
            np.random.seed(seed)


    def draw_samples(self, num_samples):
        """
        Coming soon.

        Parameters
        ----------
        num_samples : int
            TYPE
            DESCRIPTION.

        Returns
        -------
        samples : array
            TYPE
            DESCRIPTION.

        """
        samples = multivariate_normal.rvs(self.mean_, self.cov_, num_samples)

        return samples


    def evaluate_pdf(self, samples):
        """
        Coming soon.

        Parameters
        ----------
        samples : array
            TYPE
            DESCRIPTION.

        Returns
        -------
        densities : array
            TYPE
            DESCRIPTION.

        """
        densities = np.ones((len(samples), 1))

        for i in range(len(samples)):
            densities[i] = multivariate_normal.pdf(samples[i, :],
                                                   self.mean_, self.cov_)

        return densities


    def _validate_dimensions_of_inputs(self):
        if not self.cov_.shape[0] == self.cov_.shape[1]:
            raise ValueError("Covariance isn't a square array")
        if not len(self.mean_) == self.cov_.shape[0]:
            raise ValueError("Covariance and Mean dimensions don't match")
            