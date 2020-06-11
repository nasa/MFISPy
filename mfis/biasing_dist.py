import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from mfis.input_distribution import InputDistribution
import h5py

class BiasingDist:
    def __init__(self, trained_surrogate, threshold, 
                 input_distribution = None, input_samples = None, seed = None):
        self._surrogate = trained_surrogate
        self._threshold = threshold
        if hasattr(self,'input_distribution'):
            if issubclass(input_distribution, InputDistribution):
                self._input_distribution = input_distribution
        elif isinstance(input_samples, np.ndarray):
            self._input_samples = input_samples
        if seed is not None:
            np.random.seed(seed)

    
    def train(self, num_input_samples, max_clusters = 10, covar_type = 'full'):
        failure_inputs = self._surrogate_failures(num_input_samples)
        if len(failure_inputs > 0):
            self.train_mixture_model(failure_inputs, max_clusters, covar_type)

    
    def _surrogate_failures(self, num_input_samples):
         surrogate_predictions = self._evaluate_surrogate(num_input_samples)
         failure_inputs = self._find_failures(self._input_samples,
                                                 surrogate_predictions)
         return failure_inputs
    
    
    def _evaluate_surrogate(self, num_input_samples):
        if hasattr(self, '_input_distribution'):
            self._input_samples = self.input_distribution. \
                draw_samples(num_input_samples)
        
        return self._surrogate.predict(self._input_samples)
    
    
    def _find_failures(self, inputs, outputs):
        failure_indexes = outputs < self._threshold
        failure_inputs = inputs[failure_indexes.flatten(),:]
    
        return(failure_inputs)
    
    
    def train_mixture_model(self, train_data, max_clusters = 10, 
                           covar_type = 'full'): 
        self._gmm = self._lowest_bic_gmm(train_data, max_clusters, covar_type)
        
        self.n_components = self._gmm.n_components
        self.covariance_type = self._gmm.covariance_type
        self.covariances_ = self._gmm.covariances_
        self.means_ = self._gmm.means_
        self.weights_ = self._gmm.weights_
    
    
    def _lowest_bic_gmm(self, train_data, max_clusters, covar_type):
        lowest_bic = np.infty 
        
        for n_components in range(1,max_clusters):
            mixmodel = GaussianMixture(n_components, 
                                       covariance_type = covar_type)
            mixmodel.fit(train_data)
            if mixmodel.bic(train_data) < lowest_bic:
                lowest_bic = mixmodel.bic(train_data)
                best_gmm = mixmodel
                
        return best_gmm
    
    
    def draw_samples(self, num_samples):
        mixture_model_samples = self._gmm.sample(num_samples)
        
        return mixture_model_samples
    
    
    def evaluate_pdf(self, samples):
        if hasattr(self, '_gmm'):
            samples_densities = self.evaluate_mixture_model_pdf(samples)
        elif hasattr(self, '_input_distribution'):
            samples_densities = self.input_distribution.evaluate_pdf(samples)
        else: 
            raise ValueError("No mixture model or input distribution exists.")
        
        return samples_densities


    def evaluate_mixture_model_pdf(self, samples):
        densities_unweighted = np.zeros((samples.shape[0],self.n_components))
    
        for i in range(self.n_components):
            densities_unweighted[:,i] = self._cluster_pdf(self, samples, i)
             
        samples_densities = np.dot(densities_unweighted, self.weights_)
        
        return(samples_densities)


    def _cluster_pdf(self, samples, cluster_num):
        cluster_covariance = self._get_cluster_covariance(self, cluster_num)
        
        densities_for_clust = multivariate_normal.pdf(samples, 
                                mean = self.means_[cluster_num], 
                                cov = cluster_covariance)
        
        return densities_for_clust

    
    def _get_cluster_covariance(self, cluster_num):
        if self.covariance_type =='tied':
            covariance = self.covariances_
        else:
            covariance = self.covariances_[cluster_num] 
        
        return covariance
            

    def write_to_file(path, samples, sample_densities):
        with h5py.File(path, 'w') as fObj:
            exp_model_group = fObj.create_group("Biasing_Distribution_Samples")
            exp_model_group.create_dataset("samples", data=samples)
            exp_model_group.create_dataset("densities", data=sample_densities)

