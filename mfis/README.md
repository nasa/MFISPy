# BiasingDist class:
> written with InputDistribution as it's abstract base class
* __init__:
	* trained_surrogate: surrogate model that includes .predict(inputs) function
	* limit_state: *optional* attribute that is either a scalar or function (applied to the outputs)
	* input_distribution: *optional* attribute that also has InputDistribution as it's abstract base class
	* seed: *optional* attribute to set the random seed
* fit: fits a Gaussian mixture model based on a set of inputs evaluated by a surrogate model that produced failures based on the limit state function
	* N: number of samples to draw from the input distribution to then be evaluated by the surrogate model
	* max_clusters: maximum number of clusters to fit in the Gaussian mixture model *(default is 10)*
	* covariance_type: a description of the covariance structure to be used when fitting the Gaussian mixture model. Possible inputs include: *full (default), spherical, tied, and diagonal*
* get_failed_inputs_from_surrogate_draws: draws samples from the input distribution to then evaluate with the surrogate model. Returns the inputs that resulted in failures based on the limit state function.
	* N: number of samples to draw from the input distribution to then be evaluated by the surrogate model
* fit_from_failed_inputs: fits a Gaussian mixture model to a set of inputs that are assumed to produce failures based on the limit state function
	* failed_inputs: a set of inputs assumed to produce failures based on the limit state function
	* max_clusters: maximum number of clusters to fit in the Gaussian mixture model *(default is 10)*
	* covariance_type: a description of the covariance structure to be used when fitting the Gaussian mixture model. Possible inputs include: *full (default), spherical, tied, and diagonal*
* draw_samples: draws samples from the biasing distribution (Gaussian mixture model)
	* num_samples: number of samples to draw from the biasing distribution
* evaluate_pdf: returns the probability densities of the samples provided in the biasing distribution
	* samples: the set of inputs for which probability densities are desired
* save: saves the state of the BiasingDist to a file that can later be loaded
	* filename: file pathname to save the BiasingDist object
* load: loads the attributes and objects of a previously saved BiasingDist class into the current class
	* filename: file pathname where the BiasingDist object is locate
