import numpy as np
from mfis.input_distribution import InputDistribution
from inspect import isfunction


class multiIS:
    def  __init__(self, limit_state = None, input_distribution = None,
                  biasing_distribution = None):
        self._limit_state = limit_state
        if isinstance(input_distribution, InputDistribution):
            self._input_distribution = input_distribution
        else:
            self._input_distribution = None
        if isinstance(biasing_distribution, InputDistribution):
            self._biasing_distribution = biasing_distribution
        else:
            self._biasing_distribution = None
                
    def calc_importance_weights(self, inputs):
        input_density = self._input_distribution.evaluate_pdf(inputs)
        mm_density = self._biasing_distribution.evaluate_pdf(inputs)
        
        return input_density.flatten()/mm_density.flatten()

    
    def mfis_estimate(self, inputs, outputs, input_densities = None,
                      biasing_densities = None):
        if input_densities is None or biasing_densities is None:
            input_densities, biasing_densities = \
                self._evaluate_pdfs(inputs, input_densities, biasing_densities)
        
        importance_weights = \
                  self._calc_importance_weights_with_densities(
                      input_densities, biasing_densities)
 
        failure_indicators = self._find_failure_indicators(outputs)
        failure_weights = importance_weights * failure_indicators
        
        probability_of_failure = np.sum(failure_weights)/len(inputs)
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
    
    
    def _find_failures(self, inputs, outputs):
        failure_indexes = self._find_failure_indices(outputs)
            
        failure_inputs = inputs[failure_indexes,:]

        return failure_inputs
    
