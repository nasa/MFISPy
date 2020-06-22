import numpy as np
from sklearn.mixture import GaussianMixture
from mfis.input_distribution import InputDistribution
from inspect import isfunction
import pickle
from sklearn.model_selection import GridSearchCV

POSSIBLE_COVARIANCES = ['full','spherical','tied','diag']


class BiasingDist(InputDistribution):
    def __init__(self, trained_surrogate = None, limit_state = None, 
                 input_distribution = None, seed = None):
        
        self._surrogate = trained_surrogate              
        self._limit_state = limit_state
        if isinstance(input_distribution, InputDistribution):
            self._input_distribution = input_distribution
        else:
            self._input_distribution = None
        if seed is not None:
            np.random.seed(seed)
        self._surrogate_inputs = None
        self.gmm_ = None

    

    def fit(self, num_samples, max_clusters = 10,
            covariance_type = POSSIBLE_COVARIANCES, 
            min_failures = 3, max_sample_attempts = 10):
                
        failure_inputs = self.get_m_failed_inputs_from_surrogate_draws( 
            num_samples, min_failures, max_sample_attempts) 
              
        self.fit_from_failed_inputs(failure_inputs, max_clusters,
                                        covariance_type)

    
    def get_m_failed_inputs_from_surrogate_draws(self, num_samples, 
                            min_failures = 3, max_sample_attempts = 10):
        failure_inputs = []
        attempts = 1
        while (len(failure_inputs) < min_failures and 
               attempts <= max_sample_attempts):
            new_failure_inputs = \
                self.get_failed_inputs_from_surrogate_draws(num_samples)
            #import pdb; pdb.set_trace()
            if new_failure_inputs is not None:
                if failure_inputs is None: 
                    failure_inputs = new_failure_inputs
                else: 
                    failure_inputs = np.vstack((failure_inputs,
                     new_failure_inputs.reshape(len(new_failure_inputs),-1)))
                    
            attempts = attempts + 1 
            
        if len(failure_inputs) < min_failures:
            print(failure_inputs)
            raise ValueError(f"Less than {min_failures} failures found in "
                    f"{max_sample_attempts}*{num_samples} surrogate draws")
    
    def get_failed_inputs_from_surrogate_draws(self, num_samples):
         surrogate_predictions = self._evaluate_surrogate(num_samples)
         failure_inputs = self._find_failures(self._surrogate_inputs,
                                                 surrogate_predictions)
         return failure_inputs
    
    
    def _evaluate_surrogate_decorator(func):
        def wrapper(bd, num_samples):
            if bd._input_distribution is None:
                raise ValueError("No input distribution exists.")
            if bd._surrogate is None:
                raise ValueError("Biasing Distribution not initialized"
                                 " with surrogate.")
            return func(bd, num_samples)
        return wrapper
    
    
    @_evaluate_surrogate_decorator
    def _evaluate_surrogate(self, num_samples):
        self._surrogate_inputs = \
                    self._input_distribution.draw_samples(num_samples)
        surrogate_predictions = \
                    self._surrogate.predict(self._surrogate_inputs)
            
        return surrogate_predictions
    
    
    def _find_failures(self, inputs, outputs):
        if hasattr(self, '_limit_state'):
            if isfunction(self._limit_state):
                failure_indexes = self._limit_state(outputs) < 0
            else:
                failure_indexes = outputs < self._limit_state 
            
            if (len(failure_indexes.flatten()) > 0):
                failure_inputs = inputs[failure_indexes.flatten(),:]
            else:
                failure_inputs = None
        else:
            raise ValueError("No limit state found to determine failures.")
    
        return(failure_inputs)
    
    
    def fit_from_failed_inputs(self, failed_inputs, max_clusters = 10,
                               covariance_type = POSSIBLE_COVARIANCES):
        covariance_type = self._check_covariance_type_is_valid(covariance_type)
        
        self.gmm_ = self._gmm_GridSearch(failed_inputs, max_clusters,
                                         covariance_type)


    def _check_covariance_type_is_valid(self, covariance_type):
        if covariance_type != POSSIBLE_COVARIANCES:
            if not isinstance(covariance_type, list):
                covariance_type = [covariance_type]
            
            for i in range(len(covariance_type)):
                 
                if covariance_type[i] not in POSSIBLE_COVARIANCES:
                    print(covariance_type[i])
                    
                    raise ValueError("covariance_type not found in: "
                          f"{POSSIBLE_COVARIANCES}")
        
        return covariance_type


    def _gmm_GridSearch(self, train_data, max_clusters, covariance_type):

        mm = GridSearchCV(GaussianMixture(), cv = 10,
             param_grid ={'n_components': list(range(1, max_clusters+1)),
                          'covariance_type': covariance_type})
        mm.fit(train_data)

        return mm.best_estimator_
         
    
    def _check_distribution_exists_decorator(func):
        def wrapper(bd, *args):
            if bd.gmm_ is None and bd._input_distribution is None:
                raise ValueError("No mixture model or "
                                 "input distribution exists.")
            return func(bd, *args)
        return wrapper
    
    
    @_check_distribution_exists_decorator
    def draw_samples(self, num_samples):
        if self.gmm_ is not None:
            input_samples = self.gmm_.sample(num_samples)[0]
        else:
            input_samples = self._input_distribution.draw_samples(num_samples)
        return input_samples
    
    
    @_check_distribution_exists_decorator
    def evaluate_pdf(self, samples):
        if self.gmm_ is not None:
            samples_densities = self.evaluate_mixture_model_pdf(samples)
        else:
            samples_densities = self._input_distribution.evaluate_pdf(samples)
        
        return samples_densities


    def evaluate_mixture_model_pdf(self, samples):
        log_densities = self.gmm_.score_samples(samples)
        samples_densities = np.exp(log_densities)
        
        return samples_densities
            

    def save(self, filename):
        with open(filename, 'wb') as fObj: 
            pickle.dump(self.__dict__, fObj)
    
    
    def load(self, filename):
        with open(filename, 'rb') as fObj: 
            self.__dict__.update(pickle.load(fObj))