# InputDistribution class:
> An abstract base class for a probability distribution of input variables. The distribution is characterized by performing two functions: 
> > 1) draw_samples: drawing a number of random samples from the distribution 
> > 2) evaluate_pdf: evaluating the distribution's density for a set of samples

## Parameters:
* None

## Methods:
* **draw_samples**(*self, n_samples*)
	* Performs independent random draws from the distribution
	* n_samples: number of samples to draw
	* **returns**: *n_samples* by *d* array of samples  
* **evaluate_pdf**(*self, samples*)
	* Evaluates the probability density function of the distribution
	* samples: an array of input samples
	* **returns**: array of probability densities for each of the samples (length corresponds to number of *samples*)

# MVIndependentDistribution class:
> A subclass of InputDistribution that stitches together multiple continuous probability distribution instances from the *scipy.stats* library (must have .rvs and .pdf functions).

## Parameters:
* **distributions**: a list of instances of continuous distributions from the *scipy.stats* library.
## Methods:
* **draw_samples**(*self, n_samples*)
	* Draws an array of *n_samples* by length of *distributions* from the various marginal input distributions
	* n_samples: number of samples to draw
	* **returns**: *n_samples* by *d* array of samples
* **evaluate_pdf**(*self, samples*)
	* Evaluates the marginal input distributions for the samples provided
	* samples: an array of input samples (number of columns matches the length of *distributions*)
	* **returns**: an array of probability densities for each of the samples (length corresponds to number of *samples*)

# MVNormalDistribution class:
> A subclass of InputDistribution that builds and draws from a Multivariate Normal Distribution

## Parameters:
* **mean**: an array of means of the distribution
* **cov**: a square 2D array that consists of the covariance matrix. The matrix should have the same length as *mean* and be positive-definite
## Methods:
* **draw_samples**(*self, n_samples*)
	* Draws and returns an array of *n_samples* from a multivariate normal distribution
	* n_samples: number of samples to draw
	* **returns**: *n_samples* by *d* array of samples
* **evaluate_pdf**(*self, samples*)
	* Evaluates the multivariate normal distribution for the samples provided
	* samples: an array of input samples
	* **returns**: an array of probability densities for each of the samples (length corresponds to number of *samples*)

# BiasingDistribution class:
> A subclass of InputDistribution that builds a biasing distribution from a Gaussian mixture model trained on a series of data points that produce failures based on a limit state function.

## Parameters:
* **trained_surrogate**: *optional* fit surrogate model that includes a .predict function
* **limit_state**: *optional* attribute that is either a scalar or function (applied to the outputs to determine failures)
* **input_distribution**: *optional* instance of a probability distribution. Represents the distribution of the inputs for the process. Must have InputDistribution as it's abstract base class
## Methods:
* **draw_samples**(*self, n_samples*) 
	* Draws random samples from the biasing distribution (Gaussian mixture model)
	* n_samples: number of samples to draw from the biasing distribution
	* **returns**: *n_samples* by *d* array of samples drawn from the biasing
* **evaluate_pdf**(*self, samples*) 
	* Evaluates and returns the probability densities of the samples provided in the biasing distribution
	* samples: the set of inputs for which probability densities are desired
	* **returns**: array of the probability densities of each of the inputs based on the mixture model
* **fit**(*self, n_samples, max_clusters = 10, covariance_type =['diag','full','spherical','tied'], min_failures = 3, max_sample_batches = 10*)
	* Draws *n_samples* inputs from the input distribution, generates output predictions with the surrogate, and determines the failure inputs. Multiple Gaussian mixture models are fit to the failure inputs. Uses cross-validation over mixture models with various numbers of clusters and/or covariance types. Assignes the mixture model with the highest average log-likelihood to the attribution 'mixture_model_'.
	* n_samples: number of samples to draw from the input distribution to then be evaluated by the surrogate model
	* max_clusters: maximum number of clusters to fit in the Gaussian mixture model
	* covariance_type: a string or list of the covariance structures to evaluate when fitting the Gaussian mixture model. Possible inputs include: *diag, full, spherical, and tied*. For more information see the documentation for *sklearn.mixture.GaussianMixture*
	* min_failures: a minimum number of failures that must be found before trying to fit the Gaussian mixture model. 
	* max_sample_batches: a maximum number of sample batches of size *n_samples* that will be evaluated with the surrogate model in order to find at least *min_failures*.
* **fit_from_failed_inputs**(*self, failed_inputs, max_clusters = 10, covariance_type = ['diag','full','spherical','tied']*) 
	* Fits a series of Gaussian mixture models to a set of inputs that correspond to predictions in the failure region. Uses cross-validation for various numbers of clusters and/or covariance types. Assigns the mixture model with the highest average log-likelihood to the attribute 'mixture_model_'.
	* failed_inputs: a set of inputs assumed to produce failures based on the limit state function
	* max_clusters: maximum number of clusters to fit in the Gaussian mixture model
	* covariance_type: a string or list of the covariance structures to evaluate when fitting the Gaussian mixture model. Possible inputs include: *diag, full, spherical, and tied*. For more information see the documentation for *sklearn.mixture.GaussianMixture*
* **get_failed_inputs_from_surrogate_draws**(*self, n_samples*)
	* Draws a set of samples from the input distribution, predicts the outputs using the surrogate model, and determines the inputs that correspond to the outputs in the failure region.
	* n_samples: number of samples to draw from the input distribution to then be evaluated by the surrogate model
	* **returns**: array of inputs from the input distribution that have surrogate predictions in the failure region
* **get_m_failed_inputs_from_surrogate_draws**(*self, n_samples, min_failures = 3, max_sample_batches = 10*) 
	* Draws batches of *n_samples* from the input distribution, predicts the outputs using the surrogate model, and determines which outputs are failures. Batches are drawn until *min_failures* are found or *max_sample_batches* is reached.
	* n_samples: number of samples to draw from the input distribution to then be evaluated by the surrogate model
	* min_failures: a minimum number of failures that must be found before trying to fit the Gaussian mixture model.
	* max_sample_batches: a maximum number of samples of size *n_samples* that will be evaluated with the surrogate model in order to find at least *min_failures*.
	* **returns**: array of at least *min_failures* inputs from the input distribution that have surrogate predictions in the failure region
* **load**(*self, filename*)
	* Loads the attributes of a previously saved BiasingDistribution class into the current class
	* filename: file pathname where the object is located
* **save**(*self, filename*)
	* Saves the attribute of the BiasingDistribution instance to a file that can later be loaded
	* filename: file pathname to place the saved object

# MultiFidelityIS class:
> Uses an input distribution, biasing distribution, and high-fidelity inputs and outputs to estimate the probability of failure based on a limit state function
## Parameters:
* **limit_state**: *optional* attribute that is either a scalar or function (applied to the outputs)
* **input_distribution**: *optional* attribute that contains the distribution of the inputs; has InputDistribution as it's abstract base class
* **biasing_distribution**: *optional* attribute that contains the biasing distribution of the inputs; an instance of BiasingDist
## Methods:
* **calc_importance_weights**(*self, inputs*)
	* Calculates the probability densities of the inputs from the input and biasing distributions and returns their ratio (importance weights)
	* inputs: an array of inputs
	* **returns**: array of importance weights (of same length as number of inputs)
* **get_failure_prob_estimate**(*self, inputs, outputs, importance_weights=None*)
	* Calculates the multi-fidelity importance sampling probability of failure estimate. Returns the probability of failure and it's root mean squared error.
	* inputs: an array of *n* rows of inputs fed into a high-fidelity model
	* outputs: an array of length *n* of the outputs of the high-fidelity model
	* importance_weights: *optional* 1D array that contained the importance weights (input density/biasing density) of the inputs. If not provided, the importance weights are calculates using the input and biasing distributions and the *inputs*.
	* **returns**: estimated probability of failure and the root mean squared error of the estimate

