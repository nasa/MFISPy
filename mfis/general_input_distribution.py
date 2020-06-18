import numpy as np
from mfis.input_distribution import InputDistribution


class GeneralInputDistribution(InputDistribution):

    def __init__(self, list_of_distributions, seed=None):
        self.distributions_list_ = list_of_distributions
        if seed is not None:
            np.random.seed(seed)
    
    def draw_samples(self, num_samples):
        samples = np.zeros((num_samples, len(self.distributions_list_)))
        
        for i in range(len(self.distributions_list_)):
            samples[:,i] = self.distributions_list_[i].rvs(num_samples)
        
        return samples
    
    def evaluate_pdf(self, samples):
        densities = np.ones((samples.shape[0],))

        for i in range(len(self.distributions_list_)):
            densities *= self.distributions_list_[i].pdf(samples)
        
        return densities