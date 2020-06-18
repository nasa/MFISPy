import numpy as np
from sklearn.mixture import GaussianMixture
from mfis.input_distribution import InputDistribution
from inspect import isfunction
import pickle

class BiasingDist(InputDistribution):
    def __init__(self, trained_surrogate, limit_state = None, 
                 input_distribution = None, seed = None):
        if trained_surrogate is not None:
            self._surrogate = trained_surrogate              
        if limit_state is not None:
            self._limit_state = limit_state
        if input_distribution is not None:
            if isinstance(input_distribution, InputDistribution):
                self._input_distribution = input_distribution
        self._surrogate_inputs = None      
        if seed is not None:
            np.random.seed(seed)
    
    
    def fit(self, N, max_clusters = 10, covariance_type = 'full'):
        failure_inputs = []
        max_attempts = 10; attempts = 1
        while (len(failure_inputs) == 0 and attempts <= max_attempts):
            failure_inputs = self.get_failed_inputs_from_surrogate_draws(N)
            attempts = attempts + 1
        
        if len(failure_inputs) > 0:
            self.fit_from_failed_inputs(failure_inputs, max_clusters, 
                                         covariance_type)
        else:
            raise ValueError(f"No failures found in 10*{N} surrogate draws")
    
    
    def get_failed_inputs_from_surrogate_draws(self, N):
         surrogate_predictions = self._evaluate_surrogate(N)
         failure_inputs = self._find_failures(self._surrogate_inputs,
                                                 surrogate_predictions)
         return failure_inputs
    
    
    def _evaluate_surrogate(self, N):
        if hasattr(self, '_input_distribution'):
            self._surrogate_inputs = self._input_distribution.draw_samples(N)
            if hasattr(self, '_surrogate'):
                surrogate_predictions = \
                    self._surrogate.predict(self._surrogate_inputs)
        else:
            raise ValueError("Biasing Distribution not initialized"
                                 " with surrogate.")
        return surrogate_predictions
    
    
    def _find_failures(self, inputs, outputs):
        if hasattr(self, '_limit_state'):
            if isfunction(self._limit_state):
                failure_indexes = self._limit_state(outputs) < 0
            else:
                failure_indexes = outputs < self._limit_state 
            
            failure_inputs = inputs[failure_indexes.flatten(),:]
        else:
            raise ValueError("No limit state found to determine failures.")
    
        return(failure_inputs)
    
    
    def fit_from_failed_inputs(self, failed_inputs, max_clusters = 10, 
                           covariance_type = 'full'): 
        self._gmm = self._lowest_bic_gmm(failed_inputs, max_clusters, 
                                         covariance_type)
        
        self.__dict__.update(self._gmm.__dict__.copy()) 

    
    def _lowest_bic_gmm(self, train_data, max_clusters, covariance_type):
        lowest_bic = np.infty 
        
        for n_components in range(1,max_clusters):
            mixmodel = GaussianMixture(n_components, 
                                       covariance_type = covariance_type)
            mixmodel.fit(train_data)
            bic = mixmodel.bic(train_data)
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = mixmodel
               
        return best_gmm
    
    
    def draw_samples(self, num_samples):
        mixture_model_samples = self._gmm.sample(num_samples)[0]
        return mixture_model_samples
    
    
    def evaluate_pdf(self, samples):
        if hasattr(self, '_gmm'):
            samples_densities = self.evaluate_mixture_model_pdf(samples)
        elif hasattr(self, '_input_distribution'):
            samples_densities = self._input_distribution.evaluate_pdf(samples)
        else: 
            raise ValueError("No mixture model or input distribution exists.")
        
        return samples_densities


    def evaluate_mixture_model_pdf(self, samples):
        log_densities = self._gmm.score_samples(samples)
        samples_densities = np.exp(log_densities)
        
        return samples_densities
            

    def save(self, filename):
        with open(filename, 'wb')as fObj: 
            #pickle.dump(self._input_distribution, fObj)
            #pickle.dump(self._gmm, fObj)
            pickle.dump(self.__dict__, fObj)
    
    
    def load(self, filename):
        with open(filename, 'rb') as fObj: 
            self.__dict__.update(pickle.load(fObj))
            #self.__dict__.update(pickle.load(fObj))