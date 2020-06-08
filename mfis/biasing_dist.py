import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

class BiasingDist:
    def __init__(self):
        self.n_components = 3
        
    def train(self, Z, max_clusters):
        best_gmm = self._lowest_bic_gmm(Z, max_clusters = 10)
                
        self.n_components = best_gmm.n_components
        self.covariance_type = best_gmm.covariance_type
        self.covariances_ = best_gmm.covariances_
        self.means_ = best_gmm.means_
        self.weights_ = best_gmm.weights_
    
    def _lowest_bic_gmm(Z, max_clusters):
        lowest_bic = np.infty 
        
        for n_components in range(1,max_clusters):
            mixmodel = GaussianMixture(n_components, covariance_type='full')
            mixmodel.fit(Z)
            if mixmodel.bic(Z) < lowest_bic:
                lowest_bic = mixmodel.bic(Z)
                best_gmm = mixmodel
                
        return best_gmm
    
    def density(self, Z):
        densities_unweighted = np.zeros((Z.shape[0],self.n_components))
    
        for i in range(self.n_components):
            densities_unweighted[:,i] = self._density_in_cluster(self, Z, i)
             
        densities_Z = np.dot(densities_unweighted, self.weights_)
        
        return(densities_Z)

    def _density_in_cluster(self, Z, cluster):
        cov_matrix = self._build_covar_matrix(self, cluster)
        densities_for_clust = multivariate_normal.pdf(Z, 
                                mean = self.means_[cluster], cov = cov_matrix)
        
        return densities_for_clust
    
    def _build_covar_matrix(self, cluster):
        if self.covariance_type in ['tied','full']:
            cov_matrix = self.covariances_[cluster]
        else:
            cov_matrix = self._build_diag_covar_matrix(self, cluster)
            
        return cov_matrix
    
    def _build_diag_covar_matrix(self, cluster):
        if self.covariance_type == 'spherical':
            diagonal = np.repeat(self.covariances_[cluster],
                                     self.means_.shape[1])
        else: 
            diagonal = self.covariances_[cluster]
        diagonal_cov_matrix = np.diag(diagonal)   
            
        return diagonal_cov_matrix
        

    
