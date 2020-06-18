# BiasingDist class:
> written with InputDistribution as it's abstract base class
* __init__:
	* trained_surrogate: surrogate model that includes .predict(inputs) function
	* limit_state: *optional* attribute that is either a scalar or function (applied to the outputs)
	* input_distribution: *optional* attribute that also has InputDistribution as it's abstract base class
