"""
The :mod:'mfis.input_distribution' hold the Abstract Base Class for an
input distribution

@author:    D. Austin Cole <david.a.cole@nasa.gov>
            James E. Warner <james.e.warner@nasa.gov>
"""
from abc import ABCMeta, abstractmethod

class InputDistribution(metaclass=ABCMeta):
    """
    Creates a probability distribution that serves two functions:
        1) Draws a specified number of random samples from the distribution
            and returns the samples in an array.
        2) Evaluates the distribution's density at a given array of samples
            and returns the densities.
    """

    @abstractmethod
    def draw_samples(self, n_samples):
        """
        Performs independent random draws from the distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw

        Returns
        -------
        samples: array
            An n_samples by d (number of input dimensions) array of sample
            inputs from the Input Distribution
        """


    @abstractmethod
    def evaluate_pdf(self, samples):
        """
        Evaluates the probability density function of the distribution

        Parameters
        ----------
        samples : array
            An n_samples by d array of sample inputs

        Returns
        -------
        densities : array
            The probability densities of each sample from the Input
            distribution's pdf

        """
