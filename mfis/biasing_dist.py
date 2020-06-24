"""
Created on Wed Jun 24 11:39:22 2020

@author: dacole2
"""
from inspect import isfunction
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from mfis.input_distribution import InputDistribution

POSSIBLE_COVARIANCES = ['full', 'spherical', 'tied', 'diag']


class BiasingDist(InputDistribution):
    """
    BiasingDist:
        Creates a distribution of the input variables biased towards
        the failure region
    """
    def __init__(self, trained_surrogate=None, limit_state=None,
                 input_distribution=None, seed=None):

        self._surrogate = trained_surrogate
        self._limit_state = limit_state
        if isinstance(input_distribution, InputDistribution):
            self._input_distribution = input_distribution
        else:
            self._input_distribution = None
        if seed is not None:
            np.random.seed(seed)
        self._surrogate_inputs = None
        self.mixture_model_ = None



    def fit(self, num_samples, max_clusters=10,
            covariance_type=POSSIBLE_COVARIANCES,
            min_failures=3, max_sample_attempts=10):
        """

        Parameters
        ----------
        num_samples : TYPE
            DESCRIPTION.
        max_clusters : TYPE, optional
            DESCRIPTION. The default is 10.
        covariance_type : TYPE, optional
            DESCRIPTION. The default is POSSIBLE_COVARIANCES.
        min_failures : TYPE, optional
            DESCRIPTION. The default is 3.
        max_sample_attempts : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        None.

        """
        failure_inputs = self.get_m_failed_inputs_from_surrogate_draws(
            num_samples, min_failures, max_sample_attempts)

        self.fit_from_failed_inputs(failure_inputs, max_clusters,
                                    covariance_type)


    def get_m_failed_inputs_from_surrogate_draws(self, num_samples,
                                                 min_failures=3,
                                                 max_sample_attempts=10):
        """

        Parameters
        ----------
        num_samples : TYPE
            DESCRIPTION.
        min_failures : TYPE, optional
            DESCRIPTION. The default is 3.
        max_sample_attempts : TYPE, optional
            DESCRIPTION. The default is 10.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        failure_inputs : TYPE
            DESCRIPTION.

        """
        failure_inputs = []
        attempts = 1
        while (len(failure_inputs) < min_failures and
               attempts <= max_sample_attempts):

            new_fail_inputs = \
                self.get_failed_inputs_from_surrogate_draws(num_samples)

            if new_fail_inputs is not None:
                if len(failure_inputs) == 0:
                    failure_inputs = new_fail_inputs
                else:
                    new_fail_inputs = new_fail_inputs.reshape(
                        len(new_fail_inputs), -1)
                    failure_inputs = np.vstack((failure_inputs,
                                                new_fail_inputs))

            attempts = attempts + 1

        if len(failure_inputs) < min_failures:
            print(failure_inputs)
            raise ValueError(f"Less than {min_failures} failures found in "
                             f"{max_sample_attempts*num_samples} surrogate"
                             " draws")
        return failure_inputs


    def get_failed_inputs_from_surrogate_draws(self, num_samples):
        """

        Parameters
        ----------
        num_samples : TYPE
            DESCRIPTION.

        Returns
        -------
        failure_inputs : TYPE
            DESCRIPTION.

        """
        surrogate_predictions = self._evaluate_surrogate(num_samples)
        failure_inputs = self._find_failures(self._surrogate_inputs,
                                             surrogate_predictions)
        return failure_inputs


    def _evaluate_surrogate_decorator(func):
        def wrapper(bias_dist, num_samples):
            if bias_dist._input_distribution is None:
                raise ValueError("No input distribution exists.")
            if bias_dist._surrogate is None:
                raise ValueError("Biasing Distribution not initialized"
                                 " with surrogate.")
            return func(bias_dist, num_samples)
        return wrapper


    @_evaluate_surrogate_decorator
    def _evaluate_surrogate(self, num_samples):
        self._surrogate_inputs = \
                    self._input_distribution.draw_samples(num_samples)
        surrogate_predictions = \
                    self._surrogate.predict(self._surrogate_inputs)

        return surrogate_predictions


    def _find_failures(self, inputs, outputs):
        if self._limit_state is not None:
            if isfunction(self._limit_state):
                failure_indexes = self._limit_state(outputs) < 0
            else:
                failure_indexes = outputs < self._limit_state

            if len(failure_indexes.flatten()) > 0:
                failure_inputs = inputs[failure_indexes.flatten(), :]
            else:
                failure_inputs = None
        else:
            raise ValueError("No limit state found to determine failures.")

        return failure_inputs


    def fit_from_failed_inputs(self, failed_inputs, max_clusters=10,
                               covariance_type=POSSIBLE_COVARIANCES):
        """

        Parameters
        ----------
        failed_inputs : TYPE
            DESCRIPTION.
        max_clusters : TYPE, optional
            DESCRIPTION. The default is 10.
        covariance_type : TYPE, optional
            DESCRIPTION. The default is POSSIBLE_COVARIANCES.

        Returns
        -------
        None.

        """
        covariance_type = self._check_covariance_types_are_valid(
            covariance_type)

        self.mixture_model_ = \
            self._mixture_model_grid_search(failed_inputs, max_clusters,
                                            covariance_type)


    def _check_covariance_types_are_valid(self, covariance_type):
        if covariance_type != POSSIBLE_COVARIANCES:
            if isinstance(covariance_type, (list, tuple)):
                for i, covar_type in enumerate(covariance_type):
                    self._check_covariance_type_is_valid(covar_type)
            else:
                self._check_covariance_type_is_valid(covariance_type)

        return covariance_type

    def _check_covariance_type_is_valid(self, covariance_type):
        if covariance_type not in POSSIBLE_COVARIANCES:
            raise ValueError("covariance_type not found in: "
                             f"{POSSIBLE_COVARIANCES}")


    def _mixture_model_grid_search(self, train_data, max_clusters,
                                   covariance_type):

        mix_model = GridSearchCV(GaussianMixture(), cv=10,
                                 param_grid={'n_components':
                                             list(range(1, max_clusters+1)),
                                             'covariance_type':
                                                 covariance_type})
        mix_model.fit(train_data)

        return mix_model.best_estimator_




    def _check_distribution_exists_decorator(func):
        def wrapper(bias_dist, *args):
            if bias_dist.mixture_model_ is None and \
                bias_dist._input_distribution is None:
                raise ValueError("No mixture model or "
                                 "input distribution exists.")
            return func(bias_dist, *args)
        return wrapper


    @_check_distribution_exists_decorator
    def draw_samples(self, num_samples):
        if self.mixture_model_ is not None:
            input_samples = self.mixture_model_.sample(num_samples)[0]
        else:
            input_samples = self._input_distribution.draw_samples(num_samples)
        return input_samples


    @_check_distribution_exists_decorator
    def evaluate_pdf(self, samples):
        if self.mixture_model_ is not None:
            samples_densities = self.evaluate_mixture_model_pdf(samples)
        else:
            samples_densities = self._input_distribution.evaluate_pdf(samples)

        return samples_densities


    def evaluate_mixture_model_pdf(self, samples):
        """

        Parameters
        ----------
        samples : TYPE
            DESCRIPTION.

        Returns
        -------
        samples_densities : TYPE
            DESCRIPTION.

        """
        log_densities = self.mixture_model_.score_samples(samples)
        samples_densities = np.exp(log_densities)

        return samples_densities


    def save(self, filename):
        """

        Parameters
        ----------
        filename : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        with open(filename, 'wb') as file_object:
            pickle.dump(self.__dict__, file_object)


    def load(self, filename):
        """

        Parameters
        ----------
        filename : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        with open(filename, 'rb') as file_object:
            self.__dict__.update(pickle.load(file_object))
