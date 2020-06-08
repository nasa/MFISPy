import numpy as np

class multiIS:
    def  __init__(self):
       self.fail_thresh = .439
            
    def calc_importance_weights(z_p, p_dist, q_dist):
        p_zp = p_dist(z_p)
        q_zp = q_dist.density(z_p)

        return p_zp/q_zp
    
    def mfip_estimate(self, z_p, high_fidelity_Y, p_dist, q_dist):
        z_p_fail = z_p[high_fidelity_Y < self.fail_thresh,:]
        importance_weights = self.calc_importance_weights(z_p_fail, 
                                                          p_dist, q_dist)
        
        return np.sum(importance_weights)/z_p.shape[0]