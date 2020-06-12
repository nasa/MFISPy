import numpy as np
from mfis.input_distribution import InputDistribution
from inspect import isfunction


class multiIS:
    def  __init__(self, limit_state = None, biasing_distribution = None,
                  input_distribution = None):
       if limit_state is not None:
            self._limit_state = limit_state
       if biasing_distribution is not None:
           if issubclass(biasing_distribution, InputDistribution):
               self.biasing_distribution = biasing_distribution
       if input_distribution is not None:
            if issubclass(input_distribution, InputDistribution):
                self.input_distribution = input_distribution
                
                
    def calc_importance_weights(self, failure_inputs):
        input_prob = self.input_distribution.evaluate_pdf(failure_inputs)
        mm_prob = self.biasing_distribution.evaluate_pdf(failure_inputs)

        return input_prob/mm_prob

    
    def mfip_estimate(self, high_fidelity_inputs, high_fidelity_outputs):
        failure_inputs = self._find_failures(high_fidelity_inputs, 
                                       high_fidelity_outputs)
        if len(failure_inputs) > 0:
            importance_weights = self.calc_importance_weights(failure_inputs)
        else: 
            raise ValueError("No failures found in data supplied.")
            
        return np.sum(importance_weights)/len(failure_inputs)
    
    
    def _find_failures(self, inputs, outputs):
        if isfunction(self._limit_state):
            failure_indexes = self._limit_state(outputs) < 0
        else:
            failure_indexes = outputs < self._limit_state
            
        failure_inputs = inputs[failure_indexes.flatten(),:]
    
        return failure_inputs