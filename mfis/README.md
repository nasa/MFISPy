# BiasingDist class:
> A subclass of InputDistribution that builds a Biasing Distribution from a Gaussian Mixture model trained on a series of data points that produce failures based on a limit state function.

## Parameters:
* **trained_surrogate**: surrogate model that includes .predict(inputs) function
* **limit_state**: *optional* attribute that is either a scalar or function (applied to the outputs)
* **input_distribution**: *optional* attribute that also has InputDistribution as it's abstract base class
* **seed**: *optional* attribute to set the random seed
## Methods:
* **fit**(self, N, max_clusters, covariance_type)
	* Fits a Gaussian mixture model based on a set of inputs evaluated by a surrogate model that produced failures based on the limit state function
	* N: number of samples to draw from the input distribution to then be evaluated by the surrogate model
	* max_clusters: maximum number of clusters to fit in the Gaussian mixture model *(default is 10)*
	* covariance_type: a description of the covariance structure to be used when fitting the Gaussian mixture model. Possible inputs include: *full (default), spherical, tied, and diagonal*
* **get_failed_inputs_from_surrogate_draws**(self, N) 
	* Draws samples from the input distribution to then evaluate with the surrogate model. Returns the inputs that resulted in failures based on the limit state function.
	* N: number of samples to draw from the input distribution to then be evaluated by the surrogate model
* **fit_from_failed_inputs**(self, failed_inputs, max_clusters, covariance_type) 
	* Fits a Gaussian mixture model to a set of inputs that are assumed to produce failures based on the limit state function
	* failed_inputs: a set of inputs assumed to produce failures based on the limit state function
	* max_clusters: maximum number of clusters to fit in the Gaussian mixture model *(default is 10)*
	* covariance_type: a description of the covariance structure to be used when fitting the Gaussian mixture model. Possible inputs include: *full (default), spherical, tied, and diagonal*
* **draw_samples**(self, num_samples) 
	* Draws samples from the biasing distribution (Gaussian mixture model)
	* num_samples: number of samples to draw from the biasing distribution
* **evaluate_pdf**(self, samples) 
	* Evaluates and returns the probability densities of the samples provided in the biasing distribution
	* samples: the set of inputs for which probability densities are desired
* **save**(self, filename)
	* Saves the state of the BiasingDist to a file that can later be loaded
	* filename: file pathname to save the BiasingDist object
* **load**(self, filename)
	* Loads the attributes and objects of a previously saved BiasingDist class into the current class
	* filename: file pathname where the BiasingDist object is locate
# multiIS class:
> Uses an inputs distribution, biasing distribution, and high-fidelity inputs and outputs to estimate the probability of failure based on a limit state function
## Parameters:
* **limit_state**: *optional* attribute that is either a scalar or function (applied to the outputs)
* **input_distribution**: *optional* attribute that contains the distribution of the inputs; has InputDistribution as it's abstract base class
* **biasing_distribution**: *optional* attribute that contains the biasing distribution of the inputs; an instance of BiasingDist
## Methods:
* **calc_importance_weights**(self, failure_inputs)
	* Calculates the probability densities of the failure inputs from the input and biasing distributions and returns their ratio (importance weights)
	* failure_inputs: an array of inputs that produced failures in the high-fidelity model
* **mfis_estimate**(self, inputs, outputs, input_densities, biasing densities)
	* Calculates the multi-fidelity importance sampling probability of failure estimate. Returns the probability of failure and it's root mean squared error.
	* inputs: an array of *n* rows of inputs fed into a high-fidelity model
	* outputs: an array of length *n* of the outputs of the high-fidelity model
	* input_densities: *optional* array of *n* probability densities of the inputs from an input distribution
	* biasing_densities: *optional* array of *n* probability densities of the inputs from a biasing distribution
