import pytest
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.stats import norm
from mfis import BiasingDist
from mock import Mock, patch


@pytest.fixture
def mocked_BiasingDist(mocker):
    mock_surrogate = mocker.Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    return bd

def test_fit_calls_gmm_GridSearch(mocker, mocked_BiasingDist):
    mock_surrogate_failed_inputs = mocker.Mock(return_value=np.array([2,3]))
    mocked_BiasingDist.get_m_failed_inputs_from_surrogate_draws = \
                mock_surrogate_failed_inputs
    
    mock_fit_from_failed_inputs = Mock()
    mocked_BiasingDist.fit_from_failed_inputs = mock_fit_from_failed_inputs
    
    mocked_BiasingDist.fit(num_samples = 50, covariance_type = 'full')
    
    mock_fit_from_failed_inputs.assert_called()


def test_get_m_failed_inputs_ends_with_error(mocker, mocked_BiasingDist):
    side_effects =[None,np.ones((1,3)),None,
                   np.ones((2,3)),np.ones((4,3))]
    mock_surrogate_failed_inputs = mocker.Mock(side_effect = side_effects)
    mocked_BiasingDist.get_failed_inputs_from_surrogate_draws = \
                mock_surrogate_failed_inputs
    
    with pytest.raises(ValueError):
        mocked_BiasingDist.get_m_failed_inputs_from_surrogate_draws(
            num_samples = 10,  min_failures = 8, max_sample_attempts = 5)


def test_get_m_failed_inputs(mocker, mocked_BiasingDist):
    side_effects =[None,np.ones((1,3)),None,
                   np.ones((2,3)),np.ones((4,3)),np.ones((10,3))]
    mock_surrogate_failed_inputs = mocker.Mock(side_effect = side_effects)
    mocked_BiasingDist.get_failed_inputs_from_surrogate_draws = \
                mock_surrogate_failed_inputs
    
    min_failures = 6
    failed_inputs = \
        mocked_BiasingDist.get_m_failed_inputs_from_surrogate_draws(
            num_samples = 10,  min_failures = min_failures, 
            max_sample_attempts = len(side_effects))
 
    assert len(failed_inputs) >= min_failures
    
    
def test_surrogate_failed_inputs_returned(mocker, mocked_BiasingDist):
    mock_evaluate_surrogate = mocker.Mock(return_value=1)
    mocked_BiasingDist._evaluate_surrogate = mock_evaluate_surrogate
    
    failures = np.array([1,2,3])
    mock_find_failures = mocker.Mock(return_value=failures)
    mocked_BiasingDist._find_failures = mock_find_failures
    
    failed_inputs_received = \
            mocked_BiasingDist.get_failed_inputs_from_surrogate_draws(
                num_samples = 10)
    
    assert (failed_inputs_received == failures).all
            

def test_evaluate_surrogate_raises_error_with_no_surrogate(mocker):
    bd = BiasingDist(limit_state = -1)
    
    mocked_input_distribution = Mock()
    mocked_input_distribution().draw_samples.return_value = np.ones((3,2))
    mocked_BiasingDist._input_distribution = mocked_input_distribution
    
    with pytest.raises(ValueError):
        bd._evaluate_surrogate(num_samples = 4)
        
        
def test_evaluate_surrogate_raises_error_with_no_input_dist(mocker, 
                                                           mocked_BiasingDist):
      with pytest.raises(ValueError):
        mocked_BiasingDist._evaluate_surrogate(num_samples = 4)


def test_evaluate_surrogate_runs(mocker):
    mock_surrogate = Mock()
    mock_surrogate().predict.return_value = np.array([-2,-1,0])
    
    mocked_input_distribution = Mock()
    mocked_input_distribution().draw_samples.return_value = np.ones((3,2))
    
    failure_threshold = -0.1
    mocked_BiasingDist = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    mocked_BiasingDist._input_distribution = mocked_input_distribution
    
    surrogate_samples = mocked_BiasingDist._evaluate_surrogate(num_samples = 3)
    
    assert (surrogate_samples == np.array([-2,-1,0])).all
        

@pytest.mark.parametrize("cov_type", ["dia", -1])    
def test_bad_covariance_type_raises_error_in_fit_from_failed_inputs(
        mocked_BiasingDist, cov_type):
    dummy_inputs = np.ones((10,3))
    
    with pytest.raises(ValueError):
        mocked_BiasingDist.fit_from_failed_inputs(failed_inputs = dummy_inputs,
                           covariance_type = cov_type)


@pytest.mark.parametrize("cov_type", ["diag", ['spherical','tied']])  
def test_valid_covariance_type_in_fit_from_failed_inputs_runs(mocker,
        mocked_BiasingDist, cov_type):

    mock_gmm_grid_search = mocker.Mock(return_value = 1)
    mocked_BiasingDist._gmm_GridSearch = mock_gmm_grid_search

    dummy_inputs = np.ones((10,3))    
    mocked_BiasingDist.fit_from_failed_inputs(failed_inputs = dummy_inputs,
                                              covariance_type = cov_type)
    
    mock_gmm_grid_search.assert_called()
    
    
def test_GridSearch_called(mocker, mocked_BiasingDist):
    mock_gmm = mocker.patch('mfis.biasing_dist.GaussianMixture', 
                            autospec = True)
    mock_grid_search = mocker.patch('mfis.biasing_dist.GridSearchCV')
    mock_grid_search.return_value.best_estimator_ = mock_gmm
    
    dummy_train_data = np.ones((100,3))
    
    best_gmm = mocked_BiasingDist._gmm_GridSearch(
        train_data = dummy_train_data, max_clusters = 3, 
        covariance_type = ['full'])
    
    mock_grid_search.assert_called()



def test_check_distribution_exists_decorator_raises_error(mocked_BiasingDist):

    with pytest.raises(ValueError):
        mocked_BiasingDist.draw_samples(2)


def test_densities_from_input_dist_are_correct_length(mocker, 
                                                      mocked_BiasingDist):
    num_samples = 20
    dummy_samples = np.ones((num_samples, 3))
    
    mocked_input_distribution = mocker.Mock()
    mocked_input_distribution.evaluate_pdf.return_value = \
        np.ones((num_samples,1))
    mocked_BiasingDist._input_distribution = mocked_input_distribution

    densities = mocked_BiasingDist.evaluate_pdf(dummy_samples)
    
    assert len(densities) == num_samples  


def test_densities_from_gmm_are_correct_length(mocker, mocked_BiasingDist):
    mock_gmm = Mock()
    num_samples = 20
    dummy_samples = np.ones((num_samples, 3))
    mock_gmm.score_samples.return_value = np.ones((num_samples,1))
    
    mocked_BiasingDist.gmm_ = mock_gmm

    densities = mocked_BiasingDist.evaluate_mixture_model_pdf(
        samples = dummy_samples)
    
    assert len(densities) == num_samples  


def test_samples_drawn_from_input_dist_are_correct_shape(mocker, 
                                                      mocked_BiasingDist):
    num_samples = 20
    return_samples = np.ones((num_samples, 3))
    
    mocked_input_distribution = mocker.Mock()
    mocked_input_distribution.draw_samples.return_value = return_samples
    mocked_BiasingDist._input_distribution = mocked_input_distribution

    samples = mocked_BiasingDist.draw_samples(num_samples)
    
    assert (samples == return_samples).all()


def test_samples_drawn_from_gmm_are_correct_shape(mocker, mocked_BiasingDist):
    mock_gmm = Mock()
    num_samples = 20
    return_samples = np.ones((num_samples, 3))
    mock_gmm.sample.return_value = [return_samples,
                                      np.zeros((num_samples, 3))]

    mocked_BiasingDist.gmm_ = mock_gmm

    samples = mocked_BiasingDist.draw_samples(num_samples)
    assert (samples == return_samples).all()    
    
       
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