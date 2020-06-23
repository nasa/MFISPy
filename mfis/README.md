# InputDistribution class:
> An abstract base class for a distribution of input variables

## Parameters:
* None

## Methods:
* **draw_samples**(*self, num_samples*)
	* Draws and returns an array of *num_samples* from the input distribution
	* num_samples: int of samples to draw
* **evaluate_pdf**(*self, samples*)
	* Evaluates the input distribution for the samples provided and returns an array of probability densities
	* samples: an array of input samples

# MultivariateIndependentDistribution class:
> A subclass of InputDistribution that stitches together multiple continuouse probability distribution instances from the *scipy.stats* library.

## Parameters:
* **list_of_distributions**: a list of instances of continuous distributions from the *scipy.stats* library.
* **seed**: *optional* attribute to set the random seed
## Methods:
* **draw_samples**(*self, num_samples*)
	* Draws and returns and array of *num_samples* by length of *list_of_distributions* from the various marginal input distributions
	* num_samples: int of samples to draw
* **evaluate_pdf**(*self, samples*)
	* Evaluates the marginal input distributions for the samples provided and returns an array of the probability densities
	* samples: an array of input sample (number of columns matches the length of *list_of_distributions*)

# MultivariateNormalDistribution class:
> A subclass of InputDistribution that builds and draws from a Multivariate Normal Distribution

## Parameters:
* **mean**: an array of means of the distribution
* **cov**: a square 2D array that consists of the covariance matrix. The matrix should have the same length as *mean* and be positive-definite
* **seed**: *optional* attribute to set the random seed
## Methods:
* **draw_samples**(*self, num_samples*)
        * Draws and returns an array of *num_samples* from a multivariate normal distribution
        * num_samples: int of samples to draw
* **evaluate_pdf**(*self, samples*)
        * Evaluates the multivariate normal distribution for the samples provided and returns an array of probability densities
        * samples: an array of input samples


# BiasingDist class:
> A subclass of InputDistribution that builds a Biasing Distribution from a Gaussian Mixture model trained on a series of data points that produce failures based on a limit state function.

## Parameters:
* **trained_surrogate**: *optional* surrogate model that includes predict(self, inputs) function
* **limit_state**: *optional* attribute that is either a scalar or function (applied to the outputs to determine failures)
* **input_distribution**: *optional* attribute that describes the distribution of the inputs for the process. Must have InputDistribution as it's abstract base class
* **seed**: *optional* attribute to set the random seed
## Methods:
* **fit**(*self, num_samples, max_clusters = 10, covariance_type =['diag','full','spherical','tied'], min_failures = 3, max_sample_attempts = 10*)
	* Fits a Gaussian mixture model based on a set of inputs evaluated by a surrogate model that produced failures based on the limit state function
	* num_samples: number of samples to draw from the input distribution to then be evaluated by the surrogate model
	* max_clusters: maximum number of clusters to fit in the Gaussian mixture model
	* covariance_type: a string or list of the covariance structures to evaluate when fitting the Gaussian mixture model. Possible inputs include: *diag, full, spherical, and tied*. For more information see the documentation for *sklearn.mixture.GaussianMixture*
	* min_failures: a minimum number of failures that must be found before trying to fit the Gaussian mixture model. 
	* max_sample_attempts: a maximum number of samples of size *num_samples* that will be evaluated with the surrogate model in order to find at least *min_failures*.
* **get_m_failed_inputs_from_surrogate_draws**(*self, num_samples, min_failures = 3, max_sample_attempts = 10*) 
	* Draws samples from the input distribution to then evaluate with the surrogate model. It will continue to draw samples and evaluate the surrogate until a minimum number of failures is found or it reached a maximum number of samples. Returns the inputs that resulted in failures based on the limit state function.
	* num_samples: number of samples to draw from the input distribution to then be evaluated by the surrogate model
	* min_failures: a minimum number of failures that must be found before trying to fit the Gaussia
n mixture model.
	* max_sample_attempts: a maximum number of samples of size *num_samples* that will be evaluated
with the surrogate model in order to find at least *min_failures*.
* **get_m_failed_inputs_from_surrogate_draws**(*self, num_samples*)
	* Draws samples from the input distribution to then evaluate with the surrogate model. Returns the inputs that resulted in failures based on the limit state function.
        * num_samples: number of samples to draw from the input distribution to then be evaluated by the surrogate model
* **fit_from_failed_inputs**(*self, failed_inputs, max_clusters = 10, covariance_type = ['diag','full','spherical','tied']*) 
	* Fits a Gaussian mixture model to a set of inputs that are assumed to produce failures based on the limit state function
	* failed_inputs: a set of inputs assumed to produce failures based on the limit state function
	* max_clusters: maximum number of clusters to fit in the Gaussian mixture model
	* covariance_type: a string or list of the covariance structures to evaluate when fitting the Gaussian mixture model. Possible inputs include: *diag, full, spherical, and tied*. For more information s
ee the documentation for *sklearn.mixture.GaussianMixture*
* **draw_samples**(*self, num_samples*) 
	* Draws samples from the biasing distribution (Gaussian mixture model)
	* num_samples: number of samples to draw from the biasing distribution
* **evaluate_pdf**(*self, samples*) 
	* Evaluates and returns the probability densities of the samples provided in the biasing distribution
	* samples: the set of inputs for which probability densities are desired
* **save**(*self, filename*)
	* Saves the state of the BiasingDist to a file that can later be loaded
	* filename: file pathname to save the BiasingDist object
* **load**(*self, filename*)
	* Loads the attributes and objects of a previously saved BiasingDist class into the current class
	* filename: file pathname where the BiasingDist object is locate
# multiIS class:
> Uses an input distribution, biasing distribution, and high-fidelity inputs and outputs to estimate the probability of failure based on a limit state function
## Parameters:
* **limit_state**: *optional* attribute that is either a scalar or function (applied to the outputs)
* **input_distribution**: *optional* attribute that contains the distribution of the inputs; has InputDistribution as it's abstract base class
* **biasing_distribution**: *optional* attribute that contains the biasing distribution of the inputs; an instance of BiasingDist
## Methods:
* **calc_importance_weights**(*self, inputs*)
	* Calculates the probability densities of the inputs from the input and biasing distributions and returns their ratio (importance weights)
	* inputs: an array of inputs
* **mfis_estimate**(*self, inputs, outputs, input_densities=None, biasing densities=None*)
	* Calculates the multi-fidelity importance sampling probability of failure estimate. Returns the probability of failure and it's root mean squared error.
	* inputs: an array of *n* rows of inputs fed into a high-fidelity model
	* outputs: an array of length *n* of the outputs of the high-fidelity model
	* input_densities: *optional* array of *n* probability densities of the inputs from an input distribution
	* biasing_densities: *optional* array of *n* probability densities of the inputs from a biasing distribution
