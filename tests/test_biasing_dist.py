import pytest
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.stats import norm
from mfis import BiasingDist
from mock import Mock



#def test_samples_drawn_are_correct_shape():
 #   num_samples = 100
 #   mock_surrogate = Mock()
 #   bd = BiasingDist(mock_surrogate, 1)
 #   samples = bd.draw_samples(num_samples)
 #   shape_expected = np.array([num_samples, bd.means_.shape[1]])
 #   np.testing.assert_array_equal(shape_expected, samples.shape)

#def test_train_mixture_model_executed():
    # check that mixture model was trained
 #   bd = Mock()
 #   bd.train_mixture_model = Mock()
 #   attrs={'method.return_value': 3}
 #   bd.configure_mock(**attrs)
    
    
    
    
    
#def test_evaluate_pdf_valid_probability():
    #Test that probabilities are valid
    
    
#def test_number_of_values_evaluate_pdf():
    #Test that probabilities are valid     
    
    
#def test_cluster_pdf_valid_probability():
    #Test that probabilities are valid 
    
    
#def test_number_of_values_cluster_pdf():
    #Test that probabilities are valid     
    
#@patch('BiasingDist._evaluate_surrogate')    
#def test_shape_of_surrogate_failures(mock_eval_surrogate):
    # check that output matches dimensions of inputs
 #   num_input_samples = 100
 #   attrs={'return_value': np.ones((num_input_samples,1))}
 #   mock_surrogate.configure_mock(**attrs)
    
    
def test_find_failures_returns_failures():    
    # check that only inputs are returned that correspond to a failure
    mock_surrogate = Mock()
    failure_threshold = -0.1
    bd = BiasingDist(mock_surrogate, failure_threshold)

    num_inputs = 100
    inputs = np.array(range(1,num_inputs)).reshape(num_inputs-1,1)
    outputs = -1/inputs

    failure_inputs = bd._find_failures(inputs, outputs)
    
    np.testing.assert_almost_equal(np.array(range(1,10)).reshape(9,1), 
                                   failure_inputs)
            

def test_number_of_predictions_from_evaluate_surrogate():
    # check that number of predictions matches number of inputs
    
    num_input_samples = 100
    mock_surrogate = Mock()
    attrs={'predict.return_value': np.ones((num_input_samples,1))}
    mock_surrogate.configure_mock(**attrs)
    
    bd = BiasingDist(mock_surrogate, 1)
    bd._input_samples = 1
    surrogate_predictions = bd._evaluate_surrogate(num_input_samples)
    
    assert num_input_samples == len(surrogate_predictions)

    
@pytest.mark.parametrize("cov_type, covariances, clust_covariance",
                          [('tied',np.array([[2, .4],[.4, 3]]), 
                            np.array([[2, .4],[.4, 3]])),
                           ('full',np.array([[[2, .4],[.4, 3]],
                                             [[1, .3],[.3, 1.5]]]), 
                            np.array([[1, .3],[.3, 1.5]])),
                           ('diag',np.array([[2,.4],[.5,3]]), 
                            np.array([.5,3]))])
def test_shape_get_cluster_covariance(cov_type, covariances, clust_covariance):
    mock_surrogate = Mock()
    bd = BiasingDist(mock_surrogate, 1)
    bd.covariance_type = cov_type
    bd.covariances_ = covariances
    covariance = bd._get_cluster_covariance(1)

    np.testing.assert_almost_equal(clust_covariance.shape, covariance.shape)

def test_shape_spherical_covariance(cov_type, covariances, clust_covariance):
    mock_surrogate = Mock()
    bd = BiasingDist(mock_surrogate, 1)
    bd.covariance_type = 'spherical'
    bd.covariances_ = np.array([2,3])
    covariance = bd._get_cluster_covariance(1)

    np.testing.assert_almost_equal(3, covariance)