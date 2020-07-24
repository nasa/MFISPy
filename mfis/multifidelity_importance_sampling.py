"""
The :mod:'mfis.multifidelity_importance_sampling' calculates an unbiased
estimator for the failure probability based on samples from a high-fidelity
model. The samples' inputs must have probability densities from the original
input distribution and a biasing distribution.

@author:    D. Austin Cole <david.a.cole@nasa.gov>
            James E. Warner <james.e.warner@nasa.gov>
"""
from inspect import isfunction
import numpy as np
from mfis.input_distribution import InputDistribution



class MultiFidelityIS:
    """
    Calculates the multi-fidelity importance sampling failure of probability
    estimate. Uses a batch of samples, their high-fidelity responses,
    and input and biasing distributions of the inputs.

    Parameters
    ----------
    limit_state : float, int, or function
        A scalar or function applied to the response(s) that distinguishes
        failures from non-failures. If a scalar is provided, outputs less than
        the limit state are considered failures. If a function is provided,
        values of the function less than zero are considered failures.

    input_distribution : instance of a probability distribution; optional
        A distribution of one or more random variables that reflects the
        distribution of the input(s). Should have InputDistribution as it's
        base class. The default is None.

    biasing_distribution : instance of a probability distribution; optional
        A distribution of one of more random variables that reflects the
        biased distribution of the input(s). Usually, an instance of
        BiasingDistribution. Should have InputDistribution as it's
        base class. The default is None.
    """
    def  __init__(self, limit_state, input_distribution=None,
                  biasing_distribution=None):
        self._limit_state = limit_state
        self._input_distribution = input_distribution
        # if isinstance(biasing_distribution, InputDistribution):
        self._biasing_distribution = biasing_distribution
        # else:
        #     self._biasing_distribution = None


    def calc_importance_weights(self, inputs, only_nonzero_weights=False):
        """
        Calculates the importance weights of each input based on the ratio of
        probability densities between the input and biasing distributions.

        Parameters
        ----------
        inputs : array
            An array of inputs used to evaluate the high-fidelity model.

        Raises
        ------
        ValueError
            If either of or both the input distribution and biasing
            distribution are not provided at initialization, an error
            is raised.

        Returns
        -------
        importance weights : array
            A series of importance weights of same length as the inputs.
        """
        if self._input_distribution is None or \
            self._biasing_distribution is None:
            raise ValueError("Probability distributions are not supplied.")

        input_density = self._input_distribution.evaluate_pdf(inputs)
        bd_density = self._biasing_distribution.evaluate_pdf(inputs)
        if only_nonzero_weights:
            nonzero_density_ind = (input_density > 0).flatten()
            input_nonzero = input_density[nonzero_density_ind]

            bd_nonzero = bd_density[nonzero_density_ind]
            
            log_import_weights = np.log(input_nonzero) - np.log(bd_nonzero)
            
            return np.exp(log_import_weights), nonzero_density_ind
        
        else:
            return input_density.flatten()/bd_density.flatten()     


    def get_failure_prob_estimate(self, inputs, outputs,
                                  importance_weights=None):
        """
        Calculates the probability of failure estimate.

        Parameters
        ----------
        inputs : array
            An n_samples by d array of inputs used to evaluate the
            high-fidelity model.

        outputs : array
            An 2D array with n_samples rows that contains the outputs from the
            high-fidelity model.

        importance_weights : array; optional
            A 1D array of length n_samples that contains the importance weights
            (input density / biasing density) of the inputs. The default
            is None.

        Returns
        -------
        probability_of_failure : float
            The mean estimate of the probabilty of the input resulting in a
            failure output from the high-fidelity model.

        rmse : float
            The root mean squared error of the probability of failure estimate.
        """
        if importance_weights is None:
            importance_weights, nonzero_indices = \
                self.calc_importance_weights(inputs, only_nonzero_weights=True)

            failure_indicators = \
                self._find_failure_indicators(outputs[nonzero_indices])
        else: 
            failure_indicators = self._find_failure_indicators(outputs) 
        failure_weights = \
            importance_weights * failure_indicators

        probability_of_failure = np.sum(failure_weights)/inputs.shape[0]
        sqaured_errors = (failure_weights-probability_of_failure)**2
        rmse = np.sqrt(np.mean(sqaured_errors)/len(inputs))
        
        return probability_of_failure, rmse


    def _evaluate_pdfs(self, inputs, input_densities, biasing_densities):
        if input_densities is None:
            if self._input_distribution is not None:
                input_densities = \
                    self._input_distribution.evaluate_pdf(inputs)
            else:
                raise ValueError("No input probability distribution supplied.")

        if biasing_densities is None:
            if self._biasing_distribution is not None:
                biasing_densities = \
                    self._biasing_distribution.evaluate_pdf(inputs)
            else:
                raise ValueError("No biasing distribution supplied.")

        return input_densities, biasing_densities


    def _calc_importance_weights_with_densities(self, input_densities,
                                                biasing_densities):
        importance_weights = \
            input_densities.flatten()/biasing_densities.flatten()

        return importance_weights


    def _find_failure_indicators(self, outputs):
        if isfunction(self._limit_state):
            failure_indicators = (self._limit_state(outputs) < 0)*1
        else:
            failure_indicators = (outputs < self._limit_state)*1

        return failure_indicators
