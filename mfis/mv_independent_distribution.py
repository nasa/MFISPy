"""
The :mod:'mfis.multivariate_independent_distribution' combines a series of
indepedent continuous variable distributions to be used as an
input distribution.

@author:    D. Austin Cole <david.a.cole@nasa.gov>
            James E. Warner <james.e.warner@nasa.gov>
"""
import numpy as np
from mfis.input_distribution import InputDistribution


class MVIndependentDistribution(InputDistribution):
    """
    A multivariate independent distribution consisting of a series of
    independent continuous distributions. It is used to describe the
    distribution of inputs.

    Parameters
    ----------
    distributions : list
        A series of continuous distribution instances from the scipy.stats
        module. Each marginal distribution much have .rvs and .pdf functions.
    """
    def __init__(self, distributions):
        self.distributions_ = distributions


    def draw_samples(self, n_samples):
        """
        Draws and combines random input samples from the separate
        continuous distributions.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : array
            An n_samples by d (number of distributions) array of sample
            inputs from the Multivariate Independent distribution.
        """
        samples = np.zeros((n_samples, len(self.distributions_)))

        for i in range(len(self.distributions_)):
            samples[:, i] = self.distributions_[i].rvs(n_samples)

        return samples

    def evaluate_pdf(self, samples):
        """
        Evaluates the probability density function of Multivariate
        Indepdendent distribution.

        Parameters
        ----------
        samples : array
            An n_samples by d array of sample inputs.

        Returns
        -------
        densities : array
            The probability densities of each sample from the Multivariate
            Independent distribution's pdf.

        """
        densities = np.ones((samples.shape[0],))

        for i in range(len(self.distributions_)):
            densities *= self.distributions_[i].pdf(samples[:, i])

        return densities

    def ppf(self, q):
        inv_cdf = np.zeros((len(q), len(self.distributions_)))

        for i in range(len(self.distributions_)):
            inv_cdf[:, i] = self.distributions_[i].ppf(q[:,i])

        return inv_cdf 