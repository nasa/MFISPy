import pytest
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.stats import norm
from mfis import BiasingDist
from mock import Mock, patch


@pytest.fixture
def mocked_BiasingDist():
    mock_surrogate = Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    return bd

def test_fit_calls_gmm_GridSearch(mocked_BiasingDist):
    mock_surrogate_failed_inputs = Mock(return_value=np.array([2,3]))
    mocked_BiasingDist.get_m_failed_inputs_from_surrogate_draws = \
                mock_surrogate_failed_inputs
    
    mock_fit_from_failed_inputs = Mock()
    mocked_BiasingDist.fit_from_failed_inputs = mock_fit_from_failed_inputs
    
    mocked_BiasingDist.fit(num_samples = 50, covariance_type = 'full')
    
    mock_fit_from_failed_inputs.assert_called()


def test_fit_increases_number_of_surrogate_evals(mocked_BiasingDist):
    side_effects =[None,np.ones((1,3)),None,
                   np.ones((2,3)),np.ones((4,3))]
    mock_surrogate_failed_inputs = mock.Mock(side_effect = side_effects)
    mocked_BiasingDist.get_failed_inputs_from_surrogate_draws = \
                mock_surrogate_failed_inputs
    
    with pytest.raises(ValueError):
        message = mocked_BiasingDist.fit(num_samples = 10,  min_failures = 8, 
                         covariance_type = 'full', max_sample_attempts = 5)
 
    
def test_surrogate_failed_inputs_returned(mocked_BiasingDist):
    mock_evaluate_surrogate = Mock(return_value=1)
    mocked_BiasingDist._evaluate_surrogate = mock_evaluate_surrogate
    
    failures = np.array([1,2,3])
    mock_find_failures = Mock(return_value=failures)
    mocked_BiasingDist._find_failures = mock_find_failures
    
    failed_inputs_received = \
            mocked_BiasingDist.get_failed_inputs_from_surrogate_draws(num_samples = 10)
    
    assert (failed_inputs_received == failures).all
            
    
def test_returns_min_bic_gmm():
    mock_gmm = mocker.patch('mfis.biasing_dist.GaussianMixture', 
                            autospec = True)

    mock_gmm.return_value.score.side_effect = [3,2,1]
    
    mock_surrogate = mocker.Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    dummy_train_data = np.ones((100,3))
    mocked_gmm = bd._gmm_GridSearch(train_data = dummy_train_data, 
                                    max_clusters = 3, 
                                    covariance_type = 'full')

    assert mocked_gmm.bic(dummy_train_data) == 1
 

def test_samples_drawn_are_correct_shape(mocker):
    mock_gmm = mocker.patch('mfis.biasing_dist.GaussianMixture', 
                            autospec = True)
    num_samples = 100
    dummy_train_data = np.ones((200,3))
    mock_gmm().sample.return_value = (np.ones((num_samples, 3)),0)
    mock_gmm().bic.side_effect = [1, 2]
    
    mock_surrogate = mocker.Mock()
    bd = BiasingDist(mock_surrogate, 1)
    bd.fit_from_failed_inputs(dummy_train_data, 
                              covariance_type = 'full',
                              max_clusters = 2)

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