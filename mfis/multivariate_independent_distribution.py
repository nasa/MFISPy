"""
The :mod:'mfis.multivariate_independent_distribution' combines a series of
indepedent continuous variable distributions to be used as an
input distribution.

@author:    D. Austin Cole <david.a.cole@nasa.gov>
            James E. Warner <james.e.warner@nasa.gov>
"""
import numpy as np
from mfis.input_distribution import InputDistribution


class MultivariateIndependentDistribution(InputDistribution):
    """
    A multivariate independent distribution consisting of a series of
    indepdendent continuous distributions. It is used to describe the
    distribution of inputs.
    
    Parameters
    ----------
    distributions: list
        A series of continuous distribution instances.
   
    seed: int
        The seed number
    """
    def __init__(self, distributions, seed=None):
        self.distributions_list_ = distributions
        if seed is not None:
            np.random.seed(seed)

    def draw_samples(self, n_samples):
        """
        Draws and combines random input samples from the separate 
        continuous distributions.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw

        Returns
        -------
        samples : array
            An n_samples by d (number of distributions) array of sample
            inputs from the Multivariate Independent distribution

        """
        samples = np.zeros((n_samples, len(self.distributions_list_)))

        for i in range(len(self.distributions_list_)):
            samples[:, i] = self.distributions_list_[i].rvs(n_samples)

        return samples

    def evaluate_pdf(self, samples):
        """
        Evaluates the probability density function of Multivariate
        Indepdendent distribution

        Parameters
        ----------
        samples : array
            An n_samples by d array of sample inputs

        Returns
        -------
        densities : array
            The probability densities of each sample from the Multivariate
            Independent distribution's pdf

        """
        densities = np.ones((samples.shape[0],))

        for i in range(len(self.distributions_list_)):
            densities *= self.distributions_list_[i].pdf(samples)

        return densities
