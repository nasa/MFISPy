import numpy as np
from mfis.input_distribution import InputDistribution
from inspect import isfunction


class multiIS:
    def  __init__(self, limit_state = None, biasing_distribution = None,
                  input_distribution = None):
       if limit_state is not None:
            self._limit_state = limit_state
       import pdb; pdb.set_trace()
       if biasing_distribution is not None:
           if isinstance(biasing_distribution, InputDistribution):
               self.biasing_distribution = biasing_distribution
       if input_distribution is not None:
           if isinstance(input_distribution, InputDistribution):
               self.input_distribution = input_distribution
                
                
    def calc_importance_weights(self, failure_inputs):
        input_density = self.input_distribution.evaluate_pdf(failure_inputs)
        mm_density = self.biasing_distribution.evaluate_pdf(failure_inputs)

        return input_density/mm_density

    
    def mfis_estimate(self, high_fidelity_inputs, high_fidelity_outputs):
        failure_inputs = self._find_failures(high_fidelity_inputs, 
                                       high_fidelity_outputs)
        if len(failure_inputs) > 0:
            importance_weights = self.calc_importance_weights(failure_inputs)
        else: 
            raise ValueError("No failures found in data supplied.")
        
        probability_of_failure = np.add(importance_weights)/ \
                                    len(high_fidelity_inputs)
        
        return probability_of_failure
    
    
    def _find_failures(self, inputs, outputs):
        if isfunction(self._limit_state):
            failure_indexes = self._limit_state(outputs) < 0
        else:
            failure_indexes = outputs < self._limit_state
            
        failure_inputs = inputs[failure_indexes.flatten(),:]
    
        return failure_inputs