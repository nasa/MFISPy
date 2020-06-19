import numpy as np
from mfis.input_distribution import InputDistribution
from inspect import isfunction


class multiIS:
    def  __init__(self, limit_state = None, input_distribution = None,
                  biasing_distribution = None):
       if limit_state is not None:
            self._limit_state = limit_state
       if input_distribution is not None:
           if isinstance(input_distribution, InputDistribution):
               self._input_distribution = input_distribution
       if biasing_distribution is not None:
           if isinstance(biasing_distribution, InputDistribution):
               self._biasing_distribution = biasing_distribution
                
                
    def calc_importance_weights(self, failure_inputs):
        input_density = self._input_distribution.evaluate_pdf(failure_inputs)
        mm_density = self._biasing_distribution.evaluate_pdf(failure_inputs)
        
        return input_density.flatten()/mm_density.flatten()

    
    def mfis_estimate(self, inputs, outputs, input_densities = None,
                      biasing_densities = None):
        if (input_densities is not None and biasing_densities is not None):
              importance_weights = \
                  self._calc_importance_weights_with_supplied_densities(
                      inputs, outputs, input_densities, biasing_densities)
                
        elif (hasattr(self, '_biasing_distribution') and 
                hasattr(self, '_input_distribution')):
            importance_weights = \
                self._calc_importance_weights_with_distributions(inputs, 
                                                                 outputs)
        else:
            raise ValueError("No input probability distributions or "
                             "densities supplied.")
            
        probability_of_failure = np.sum(importance_weights)/len(inputs)
        sqaured_errors = (importance_weights-probability_of_failure)**2
        rmse = np.sqrt(np.mean(sqaured_errors)/len(inputs))
        
        return probability_of_failure, rmse
    
    
    def _calc_importance_weights_with_distributions(self, inputs, outputs):
        failure_inputs = self._find_failures(inputs, outputs)
        
        if len(failure_inputs) > 0:
            importance_weights = self.calc_importance_weights(failure_inputs)
        else: 
            raise ValueError("No failures found in data supplied.")
        
        return importance_weights
    
    
    def _calc_importance_weights_with_supplied_densities(self, inputs, outputs,
                                                    input_densities, 
                                                    biasing_densities):
        
        failures = self._find_failures_and_densities(inputs, outputs,
                                                     input_densities, 
                                                     biasing_densities)
        failure_inputs = failures[0]
        
        if len(failure_inputs) > 0:
            importance_weights = failures[1].flatten()/failures[2].flatten()
        else: 
            raise ValueError("No failures found in data supplied.")
        
        return importance_weights
        
    
    def _find_failures(self, inputs, outputs, input_densities = None,
                      biasing_densities = None):
        failure_indexes = self._find_failure_indices(outputs)
            
        failure_inputs = inputs[failure_indexes,:]

        return failure_inputs
    
    
    def _find_failures_and_densities(self, inputs, outputs, input_densities,
                      biasing_densities):
        failure_indices = self._find_failure_indices(outputs)
        failure_inputs = inputs[failure_indices,:]
        failure_input_densities = input_densities[failure_indices]
        failure_bias_densities = biasing_densities[failure_indices]
        
        return failure_inputs, failure_input_densities, failure_bias_densities
    
    
    def _find_failure_indices(self, outputs):
        if isfunction(self._limit_state):
            failure_indices = (self._limit_state(outputs) < 0).flatten()
        else:
            failure_indices = (outputs < self._limit_state).flatten()
            
        return failure_indices