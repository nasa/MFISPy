import pytest
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.stats import norm
from mfis import BiasingDist
from mock import Mock, patch


def test_fit_calls_gmm_fit_from_failures(mocker):
    # check that mixture model was trained
    mock_surrogate = mocker.Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    
    mock_surrogate_failed_inputs = mocker.Mock(return_value=[[2,3]])
    bd.get_failed_inputs_from_surrogate_draws = mock_surrogate_failed_inputs
    
    mock_fit_from_failed_inputs = mocker.Mock()
    bd.fit_from_failed_inputs = mock_fit_from_failed_inputs
    
    bd.fit(num_samples = 50)
    
    mock_fit_from_failed_inputs.assert_called()


def test_fit_increases_number_of_surrogate_evals(mocker):
    # check that mixture model was not trained
    mock_surrogate = mocker.Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    
    mock_surrogate_failed_inputs = mocker.Mock(return_value=[])
    bd.get_failed_inputs_from_surrogate_draws = mock_surrogate_failed_inputs
    
    with pytest.raises(ValueError):
        message = bd.fit(num_samples = 100, max_clusters = 5, 
                         covariance_type = 'full')
 
    
def test_surrogate_failed_inputs_returned(mocker):
    mock_surrogate = mocker.Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    
    mock_evaluate_surrogate = mocker.Mock(return_value=1)
    bd._evaluate_surrogate = mock_evaluate_surrogate
    
    failures = np.array([1,2,3])
    mock_find_failures = mocker.Mock(return_value=failures)
    bd._find_failures = mock_find_failures
    
    failed_inputs_received = \
            bd.get_failed_inputs_from_surrogate_draws(num_samples = 10)
    
    assert (failed_inputs_received == failures).all
            

@pytest.mark.parametrize("gmm_attribute, attribute_example",
                         [('covariance_type','full'),('means_',[3,4]),
                          ('covariances_',[[3, .6],[.6,1]])])
def test_attributes_from_gmm_copied(mocker, gmm_attribute, attribute_example):
    mock_surrogate = mocker.Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    
    mock_gmm = mocker.Mock()                                 
    #mock_gmm.covariance_type = attribute_example
    setattr(mock_gmm,gmm_attribute,attribute_example)
    bd._lowest_bic_gmm = mocker.Mock(return_value = mock_gmm)
    
    bd.fit_from_failed_inputs(range(10))
    
    assert hasattr(bd, gmm_attribute)
    assert getattr(bd, gmm_attribute, attribute_example) == attribute_example
    

def test_returns_min_bic_gmm(mocker):
    mock_gmm = mocker.patch('mfis.biasing_dist.GaussianMixture', 
                            autospec = True)
    mock_gmm.return_value.bic.side_effect = [3,2,1]
    
    mock_surrogate = mocker.Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    dummy_train_data = np.ones((100,3))
    mocked_gmm = bd._lowest_bic_gmm(train_data = dummy_train_data, 
                                    max_clusters = 3, 
                                    covariance_type = 'full')

    assert mocked_gmm.bic(dummy_train_data) == 1
 

def test_samples_drawn_are_correct_shape(mocker):
    mock_gmm = mocker.patch('mfis.biasing_dist.GaussianMixture', 
                            autospec = True)
    num_samples = 100
    dummy_train_data = np.ones((200,3))
    mock_gmm.return_value.sample.return_value = (np.ones((num_samples, 3)),0)
    mock_gmm.return_value.bic.side_effect = [1, 2]
    
    mock_surrogate = mocker.Mock()
    bd = BiasingDist(mock_surrogate, 1)
    bd.fit_from_failed_inputs(failed_inputs = dummy_train_data, 
                              max_clusters = 2,
                              covariance_type = 'full')

    samples = bd.draw_samples(num_samples)
    
    assert samples.shape == (num_samples, 3)    
    
       
def test_saved_file(mocker):
    mock_pickle = mocker.patch('mfis.biasing_dist.pickle.dump') 
    save_path = "bias_dist.obj"
    mock_surrogate = mocker.Mock()
    bd = BiasingDist(trained_surrogate = mock_surrogate, limit_state = 1)
  
    bd.save(save_path)
    mock_pickle.assert_called_once()

  
def test_loaded_file(mocker):
    mock_pickle = mocker.patch('mfis.biasing_dist.pickle.load')
    save_path = "bias_dist.obj"
    mock_surrogate = mocker.Mock()
    bd = BiasingDist(trained_surrogate = mock_surrogate, limit_state = 1)
  
    bd.load(save_path)
    mock_pickle.assert_called_once() 