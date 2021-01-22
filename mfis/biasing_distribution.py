"""
The :mod:'mfis.biasing_distribution' builds a probability distribution for
the inputs that is biased towards the failure region.

@author:    D. Austin Cole <david.a.cole@nasa.gov>
            James E. Warner <james.e.warner@nasa.gov>
"""
from inspect import isfunction
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from mfis.input_distribution import InputDistribution

POSSIBLE_COVARIANCES = ['full', 'spherical', 'tied', 'diag']


class BiasingDistribution(InputDistribution):
    """
    Creates a distribution of the input variables biased towards
    the failure region using a Gaussian Mixture Model.

    Parameters
    ----------
    trained_surrogate : a model fit that contains a .perdict call; optional
        A surrogate model that is trained to a series of inputs, outputs
        from a high-fidelity model. The default is None.

    limit_state : float, int, or function; optional
        A scalar or function applied to the response that distinguishes
        failures from non-failures. If a scalar is provided, outputs less than
        the limit state are considered failures. If a function is provided,
        values of the function less than zero are considered failures.
        The default is None.

    input_distribution: instance of a probability distribution; optional
        A distribution of one or more random variables that reflects the
        distribution of the input(s). Should have InputDistribution as it's
        base class. The default is None.
    """
    def __init__(self, trained_surrogate=None, limit_state=None,
                 input_distribution=None):

        self._surrogate = trained_surrogate
        self._limit_state = limit_state
        self._input_distribution = input_distribution
        if input_distribution is not None:
            self._input_dim = input_distribution.draw_samples(2).shape[1]
        self._surrogate_inputs = None
        self.mixture_model_ = None


    def fit(self, n_samples, max_clusters=10,
            covariance_type=POSSIBLE_COVARIANCES,
            min_failures=3, max_sample_batches=10):
        """
        Draws n_samples inputs from the input distribution, makes output
        predictions with the surrogate, and determines the failure inputs.
        Multiple Gaussian mixture models are fit to the failure inputs. Uses
        cross-validation for various numbers of clusters and/or covariance
        types. Assigns the mixture model with the highest average
        log-likelihood to the attribute 'mixture_model_'.

        Parameters
        ----------
        n_samples : int
            The number of samples to be drawn from the input distribution and
            then evaluated with the trained surrogate model.

        max_clusters : int; optional
            The maximum number of clusters used to train the Gaussian mixture
            model. The default is 10.

        covariance_type : str or list; optional
            One or multiple types of covariance structures to use when finding
            the best Gaussian mixture model. The default is all possible types:
            ['full', 'spherical', 'tied', 'diag'].

        min_failures : int; optional
            The minimum number of failures needed before proceeding to fit
            Gaussian Mixture Models. The default is 3.

        max_sample_batches : int; optional
            The maximum number of sample batches to draw. A new batch is
            drawn if the minimum number of failures is not reached from
            previous batches of samples. The default is 10.

        Returns
        -------
        None.
        """
        failure_inputs = self.get_m_failed_inputs_from_surrogate_draws(
            n_samples, min_failures, max_sample_batches)

        self.fit_from_failed_inputs(failure_inputs, max_clusters,
                                    covariance_type)


    def get_m_failed_inputs_from_surrogate_draws(self, n_samples,
                                                 min_failures=3,
                                                 max_sample_batches=10):
        """
        Draws batches of samples from the input distribution, predicts the
        outputs using the surrogate, and determines which outputs are failures.
        Batches are drawn until m failures are found or the maximum number of
        sample batches is reached.

        Parameters
        ----------
        n_samples : int
            The number of samples to be drawn from the input distribution and
            then evaluated with the trained surrogate model.

        min_failures : int; optional
            The minimum number of failures needed before proceeding to fit
            the Gaussian mixture model. The default is 3.

        max_sample_batches : int; optional
            The maximum number of sample batches to draw. A new batch will be
            drawn if the minimum number of failures is not reached from
            previous batches of samples. The default is 10.

        Raises
        ------
        ValueError
            If m failures are not found within the maximum number of batches,
            an error is raised.

        Returns
        -------
        failure_inputs : array
            The set of at least m inputs from the input distribution that
            have surrogate predictions in the failure region.
        """
        failure_inputs = np.empty((0, self._input_dim))
        batches = 1
        while (len(failure_inputs) < min_failures and
               batches <= max_sample_batches):

            new_fail_inputs = \
                self.get_failed_inputs_from_surrogate_draws(n_samples)

            failure_inputs = np.vstack(failure_inputs, new_fail_inputs, axis=0)

            batches = batches + 1

        if failure_inputs.shape[0] < min_failures:
            raise ValueError(f"Less than {min_failures} failures found in "
                             f"{max_sample_batches*n_samples} surrogate"
                             " draws")
        return failure_inputs


    def get_failed_inputs_from_surrogate_draws(self, n_samples):
        """
        Draws a set of samples from the input distribution, predicts the
        outputs using the surrogate, and determines the inputs that correspond
        to outputs in the failure region.

        Parameters
        ----------
        n_samples : int
            The number of samples to be drawn from the input distribution and
            evaluated with the trained surrogate model.

        Returns
        -------
        failure_inputs : array
            The set of inputs from the input distribution that
            have surrogate predictions in the failure region.
        """
        input_samples = self._draw_input_samples(n_samples)
        surrogate_predictions = self._evaluate_surrogate(input_samples)
        failure_inputs = self.find_failures(input_samples,
                                            surrogate_predictions)
        return failure_inputs


    def _draw_input_samples_decorator(func):
        def wrapper(bias_dist, n_samples):
            if bias_dist._input_distribution is None:
                raise ValueError("No input distribution exists.")
            return func(bias_dist, n_samples)
        return wrapper


    @_draw_input_samples_decorator
    def _draw_input_samples(self, n_samples):
        input_samples = self._input_distribution.draw_samples(n_samples)

        return input_samples


    def _evaluate_surrogate_decorator(func):
        def wrapper(bias_dist, samples):
            if bias_dist._surrogate is None:
                raise ValueError("Biasing Distribution not initialized"
                                 " with surrogate.")
            return func(bias_dist, samples)
        return wrapper


    @_evaluate_surrogate_decorator
    def _evaluate_surrogate(self, samples):
        surrogate_predictions = \
                    self._surrogate.predict(samples)
        if surrogate_predictions.ndim == 1:
            surrogate_predictions = \
                surrogate_predictions.reshape((samples.shape[0], 1))

        return surrogate_predictions


    def _find_failures_decorator(func):
        def wrapper(bias_dist, inputs, outputs):
            if bias_dist._limit_state is None:
                raise ValueError("No limit state provided.")
            return func(bias_dist, inputs, outputs)
        return wrapper


    @_find_failures_decorator
    def find_failures(self, inputs, outputs):
        if isfunction(self._limit_state):
            failure_indexes = self._limit_state(outputs) < 0
        else:
            failure_indexes = np.all(outputs < self._limit_state, axis=1)

        failure_inputs = inputs[failure_indexes.flatten(), :]

        return failure_inputs


    def fit_from_failed_inputs(self, failed_inputs, max_clusters=10,
                               covariance_type=POSSIBLE_COVARIANCES):
        """
        Fits a series of Gaussian mixture models to a set of inputs that
        correspond to predictions in the failure region. Uses cross-validation
        for various numbers of clusters and/or covariance types. Assigns the
        mixture model with the highest average log-likelihood to the attribute
        'mixture_model_'.

        Parameters
        ----------
        failed_inputs : array
            The set of inputs from the input distribution that have
            predictions in the failure region.

        max_clusters : int; optional
            The maximum number of clusters used to train the Gaussian mixture
            model. The default is 10.

        covariance_type : str or list; optional
            One or multiple types of covariance structures to use when finding
            the best Gaussian mixture model. The default is all possible types:
            ['full', 'spherical', 'tied', 'diag'].

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
        if isinstance(covariance_type, (list, tuple)):
            for i, covar_type in enumerate(covariance_type):
                self._check_covariance_type_is_valid(covar_type)
        else:
            self._check_covariance_type_is_valid(covariance_type)
            covariance_type = [covariance_type]

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


    def _check_bias_distribution_exists_decorator(func):
        def wrapper(bias_dist, *args):
            if bias_dist.mixture_model_ is None:
                raise ValueError("No mixture model distribution exists.")
            return func(bias_dist, *args)
        return wrapper


    @_check_bias_distribution_exists_decorator
    def draw_samples(self, n_samples):
        """
        Draws random samples from the Gaussian mixture model.

        Parameters
        ----------
        n_samples : array
            Number of samples to draw.

        Returns
        -------
        samples : array
            An n_samples by d array of sample inputs from the Gaussian
            Mixture Model.
        """
        input_samples = self.mixture_model_.sample(n_samples)[0]

        return input_samples


    @_check_bias_distribution_exists_decorator
    def evaluate_pdf(self, samples):
        """
        Evaluates the probability density function of the Gaussian mixture
        model for a set of samples.

        Parameters
        ----------
        samples : array
            A set of inputs from the input distribution.

        Returns
        -------
        samples_densities : array
            The probability densities of each of the inputs based on the
            Gaussian mixture model.
        """
        samples_densities = self._evaluate_mixture_model_pdf(samples)

        return samples_densities


    def _evaluate_mixture_model_pdf(self, samples):
        log_densities = self.mixture_model_.score_samples(samples)
        samples_densities = np.exp(log_densities)

        return samples_densities


    def save(self, filename):
        """
        Saves the object with all it's attributes.

        Parameters
        ----------
        filename : filename
            A pathname for the pickle object.

        Returns
        -------
        None.
        """
        with open(filename, 'wb') as file_object:
            pickle.dump(self.__dict__, file_object)


    def load(self, filename):
        """
        Loads the attributes from a BiasingDistribution instance.

        Parameters
        ----------
        filename : filename
            The pathname for the pickle object's location.

        Returns
        -------
        None.
        """
        with open(filename, 'rb') as file_object:
            self.__dict__.update(pickle.load(file_object))
