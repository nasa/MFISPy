import pytest
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.stats import norm
from mfis import BiasingDist
from mock import Mock



#def test_samples_drawn_are_correct_shape():
    #num_samples = 100
    #mock_surrogate = Mock()
    #bd = BiasingDist(mock_surrogate, 1)
    ## how do I mock gmm?
    #samples = bd.draw_samples(num_samples)
    #shape_expected = np.array([num_samples, bd.means_.shape[1]])
    #np.testing.assert_array_equal(shape_expected, samples.shape)

def test_fit_calls_gmm_fit_from_failures():
    # check that mixture model was trained
    mock_surrogate = Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    
    mock_get_surrogate_failed_inputs = Mock(return_value=[[2,3]])
    bd.get_surrogate_failed_inputs = mock_get_surrogate_failed_inputs
    
    mock_fit_from_failed_inputs = Mock()
    bd.fit_from_failed_inputs = mock_fit_from_failed_inputs
    
    bd.fit(N=50)
    
    mock_fit_from_failed_inputs.assert_called()


def test_fit_increases_N_surrogate_evals():
    # check that mixture model was not trained
    mock_surrogate = Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    
    mock_get_surrogate_failed_inputs = Mock(return_value=[])
    bd.get_surrogate_failed_inputs = mock_get_surrogate_failed_inputs
    
    with pytest.raises(ValueError):
        message = bd.fit(N = 100, max_clusters = 5, 
                         covariance_type = 'full')
 
    
def test_surrogate_failed_inputs_returned():
    mock_surrogate = Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    
    mock_evaluate_surrogate = Mock(return_value=1)
    bd._evaluate_surrogate = mock_evaluate_surrogate
    
    failures = np.array([1,2,3])
    mock_find_failures = Mock(return_value=failures)
    bd._find_failures = mock_find_failures
    
    failed_inputs_received = bd.get_surrogate_failed_inputs(N = 10)
    
    assert (failed_inputs_received == failures).all
    
    
def test_evaluate_surrogate_returns_predictions():
    num_samples = 5
    predictions = np.array(range(num_samples))
    attrs = {'predict.return_value': [predictions]}
    mock_surrogate = Mock(**attrs)
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    surrogate_output = bd._evaluate_surrogate(N = num_samples)
    
    assert (surrogate_output == predictions).all
    
## not sure if this is needed
#def test_samples_drawn_from_input_distribution():
    
#    samples = [1,2,3,4,5]
#    attrs = {'draw_samples.return_value': [samples], 
#             'evaluate_pdf.return_value': 1/5}
#    mock_input_distribution= Mock(**attrs)   
#    mock_surrogate = Mock()
#    failure_threshold = -0.1
#    bd = BiasingDist(trained_surrogate = mock_surrogate, 
#                     limit_state = failure_threshold,
#                     input_distribution = mock_input_distribution)   
#    drawn_samples = bd.input_distribution.draw_samples(len(samples))   
#   assert (drawn_samples == samples).all
    
    
def test_find_failures_returns_failures():    
    # check that only inputs are returned that correspond to a failure
    mock_surrogate = Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)

    num_inputs = 11
    inputs = np.array(range(1,num_inputs+1)).reshape(num_inputs,1)
    outputs = -1/inputs
    expected_failure_inputs = inputs[0:9,]
    
    failure_inputs = bd._find_failures(inputs, outputs)
    
    np.testing.assert_almost_equal(expected_failure_inputs, failure_inputs)
            

@pytest.mark.parametrize("gmm_attribute, attribute_example",
                         [('covariance_type','full'),('means_',[3,4]),
                          ('covariances_',[[3, .6],[.6,1]])])
def test_attributes_from_gmm_copied(gmm_attribute, attribute_example):
    mock_surrogate = Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    
    mock_gmm = Mock()                                 
    #mock_gmm.covariance_type = attribute_example
    setattr(mock_gmm,gmm_attribute,attribute_example)
    bd._lowest_bic_gmm = Mock(return_value = mock_gmm)
    
    bd.fit_from_failed_inputs(range(10))
    
    assert hasattr(bd, gmm_attribute)
    assert getattr(bd, gmm_attribute, attribute_example) == attribute_example
    

#import mock    
#@mock.patch('scipy.stats.GaussianMixture')
#def test_returns_min_bic_gmm(self, mock_gmm):
#    class FakeGmm:
#        def __init__(self, n_components, covariance_type):
#            self.n_components = n_components

#        def fit(self, inputs):
#            pass
    
#        def bic(self, inputs):
#            return np.abs(self.n_components - 5)
    
    
#    mock_surrogate = Mock()
#    failure_threshold = -0.1
#    bd = BiasingDist(trained_surrogate = mock_surrogate, 
#                     limit_state = failure_threshold)
#    mock_gmm.return_value = FakeGmm()    
    
     
    
#@patch('BiasingDist._evaluate_surrogate')    
#def test_shape_of_surrogate_failures(mock_eval_surrogate):
    # check that output matches dimensions of inputs
 #   num_input_samples = 100
 #   attrs={'return_value': np.ones((num_input_samples,1))}
 #   mock_surrogate.configure_mock(**attrs)
    
    
def test_number_of_predictions_from_evaluate_surrogate():
    # check that number of predictions matches number of inputs
    
    num_input_samples = 10
    mock_surrogate = Mock()
    attrs={'predict.return_value': np.ones((num_input_samples,1))}
    mock_surrogate.configure_mock(**attrs)
    
    bd = BiasingDist(trained_surrogate = mock_surrogate, limit_state = 1)
    bd._input_samples = 1
    surrogate_predictions = bd._evaluate_surrogate(num_input_samples)
    
    assert num_input_samples == len(surrogate_predictions)
    
    
#def test_saved_file(tmpdir):
#    save_path = "bias_dist.pbj"
#    mock_surrogate = Mock()
#    bd = BiasingDist(trained_surrogate = mock_surrogate, limit_state = 1)
#    bd.covariance_type = 'spherical'
## Not working b/c can't pickle mock object
#    bd.save(save_path)
    
#    bd_loaded = BiasingDist(trained_surrogate = mock_surrogate, 
#                            limit_state = 2)
#    bd_loaded.load(save_path)
