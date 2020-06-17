import pytest
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.stats import norm
from mfis import BiasingDist
from mock import Mock, patch


def test_fit_calls_gmm_fit_from_failures():
    # check that mixture model was trained
    mock_surrogate = Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    
    mock_surrogate_failed_inputs = Mock(return_value=[[2,3]])
    bd.get_failed_inputs_from_surrogate_draws = mock_surrogate_failed_inputs
    
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
    
    mock_surrogate_failed_inputs = Mock(return_value=[])
    bd.get_failed_inputs_from_surrogate_draws = mock_surrogate_failed_inputs
    
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
    
    failed_inputs_received = bd.get_failed_inputs_from_surrogate_draws(N = 10)
    
    assert (failed_inputs_received == failures).all
    
    
#def test_evaluate_surrogate_returns_predictions():
#    num_samples = 5
#    predictions = np.array(range(num_samples))
#    attrs = {'predict.return_value': [predictions]}
#    mock_surrogate = Mock(**attrs)
#    failure_threshold = -0.1
#    bd = BiasingDist(trained_surrogate = mock_surrogate, 
#                     limit_state = failure_threshold)
#    surrogate_output = bd._evaluate_surrogate(N = num_samples)
#    
#    assert (surrogate_output == predictions).all
    
    
    
#def test_find_failures_returns_failures():    
    # check that only inputs are returned that correspond to a failure
#    mock_surrogate = Mock()
#    failure_threshold = -0.1
#    bd = BiasingDist(trained_surrogate = mock_surrogate, 
#                     limit_state = failure_threshold)
#
#    num_inputs = 11
#    inputs = np.array(range(1,num_inputs+1)).reshape(num_inputs,1)
#    outputs = -1/inputs
#    expected_failure_inputs = inputs[0:9,]
#   
#    failure_inputs = bd._find_failures(inputs, outputs)
#    
#    np.testing.assert_almost_equal(expected_failure_inputs, failure_inputs)
            

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
    

@patch('mfis.biasing_dist.GaussianMixture', autospec = True)
def test_returns_min_bic_gmm(mock_gmm):
    mock_gmm.return_value.bic.side_effect = [3,2,1]
    
    mock_surrogate = Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    dummy_train_data = np.ones((100,3))
    mocked_gmm = bd._lowest_bic_gmm(train_data = dummy_train_data, 
                                    max_clusters = 3, 
                                    covariance_type = 'full')

    assert mocked_gmm.bic(dummy_train_data) == 1
    
 
    
@patch('mfis.biasing_dist.GaussianMixture', autospec = True)
def test_samples_drawn_are_correct_shape(mock_gmm):
    num_samples = 100
    dummy_train_data = np.ones((200,3))
    mock_gmm.return_value.sample.return_value = (np.ones((num_samples, 3)),0)
    mock_gmm.return_value.bic.side_effect = [1, 2]
    
    mock_surrogate = Mock()
    bd = BiasingDist(mock_surrogate, 1)
    bd.fit_from_failed_inputs(train_data = dummy_train_data, 
                              max_clusters = 2,
                              covariance_type = 'full')

    samples = bd.draw_samples(num_samples)
    
    assert samples.shape == (num_samples, 3)    
    
    
#def test_number_of_predictions_from_evaluate_surrogate():
#    # check that number of predictions matches number of inputs
#    
#    num_input_samples = 10
#    mock_surrogate = Mock()
#    attrs={'predict.return_value': np.ones((num_input_samples,1))}
#    mock_surrogate.configure_mock(**attrs)
#    
#    bd = BiasingDist(trained_surrogate = mock_surrogate, limit_state = 1)
#    bd._input_samples = 1
#    surrogate_predictions = bd._evaluate_surrogate(num_input_samples)
#    
#    assert num_input_samples == len(surrogate_predictions)
    
@patch('mfis.biasing_dist.pickle.dump')   
def test_saved_file(mock_pickle):
    save_path = "bias_dist.obj"
    mock_surrogate = Mock()
    bd = BiasingDist(trained_surrogate = mock_surrogate, limit_state = 1)
  
    bd.save(save_path)
    mock_pickle.assert_called_once()


@patch('mfis.biasing_dist.pickle.load')   
def test_loaded_file(mock_pickle):
    save_path = "bias_dist.obj"
    mock_surrogate = Mock()
    bd = BiasingDist(trained_surrogate = mock_surrogate, limit_state = 1)
  
    bd.load(save_path)
    mock_pickle.assert_called_once() 

