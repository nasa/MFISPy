"""
The :mod:'mfis.multivariate_normal_distribution' wraps a
Multivariate Normal Distribution to be used as an input distribution.

@author:    D. Austin Cole <david.a.cole@nasa.gov>
            James E. Warner <james.e.warner@nasa.gov>
"""

from scipy.stats import multivariate_normal
import numpy as np
from mfis.input_distribution import InputDistribution


class MVNormalDistribution(InputDistribution):
    """
    A multivariate normal distribution used to describe the distribution
    of inputs.

    Parameters
    ----------
    mean : array
        The mean vector of length d (dimensions) for the distribution.

    cov : 2D array
        The d by d positive-definite covariance matrix for the distribution.
    """
    def __init__(self, mean, cov):
        self.mean_ = mean
        if isinstance(cov, np.ndarray) and cov.ndim > 1:
            self.cov_ = cov
        else:
            self.cov_ = np.diag(cov)
        self._validate_dimensions_of_inputs()


    def draw_samples(self, n_samples):
        """
        Draws random input samples from the Multivariate Normal distribution.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : array
            An n_samples by d array of sample inputs from the Multivariate
            Normal distribution.

        """
        samples = multivariate_normal.rvs(self.mean_, self.cov_, n_samples)

        return samples


    def evaluate_pdf(self, samples):
        """
        Evaluates the probability density function of Multivariate Normal
        distribution.

        Parameters
        ----------
        samples : array
            An n_samples by d array of sample inputs.

        Returns
        -------
        densities : array
            The probability densities of each sample from the Multivariate
            Normal distribution's pdf.

        """
        densities = np.ones((len(samples), ))

        for i in range(len(samples)):
            densities[i] = multivariate_normal.pdf(samples[i, :],
                                                   self.mean_, self.cov_,
                                                   allow_singular=True)

        return densities


    def _validate_dimensions_of_inputs(self):
        if not self.cov_.shape[0] == self.cov_.shape[1]:
            raise ValueError("Covariance isn't a square array")
        if not len(self.mean_) == self.cov_.shape[0]:
            raise ValueError("Covariance and Mean dimensions don't match")
            