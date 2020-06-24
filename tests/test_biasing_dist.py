import pytest
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.stats import norm
from mfis import BiasingDist
#from mock import Mock, patch


@pytest.fixture
def mocked_BiasingDist(mocker):
    mock_surrogate = mocker.Mock()
    failure_threshold = -0.1
    bd = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    return bd

def test_fit_calls_mixture_moodel_grid_search(mocker, mocked_BiasingDist):
    mock_surrogate_failed_inputs = mocker.Mock(return_value=np.array([2,3]))
    mocked_BiasingDist.get_m_failed_inputs_from_surrogate_draws = \
                mock_surrogate_failed_inputs
    
    mock_fit_from_failed_inputs = mocker.Mock()
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
    
    mocked_input_distribution = mocker.Mock()
    mocked_input_distribution().draw_samples.return_value = np.ones((3,2))
    bd._input_distribution = mocked_input_distribution
    
    with pytest.raises(ValueError):
        bd._evaluate_surrogate(num_samples = 4)
        
        
def test_evaluate_surrogate_raises_error_with_no_input_dist(mocker, 
                                                           mocked_BiasingDist):
      with pytest.raises(ValueError):
        mocked_BiasingDist._evaluate_surrogate(num_samples = 4)


def test_evaluate_surrogate_runs(mocker):
    mock_surrogate = mocker.Mock()
    mock_surrogate().predict.return_value = np.array([-2,-1,0])
    
    mocked_input_distribution = mocker.Mock()
    mocked_input_distribution().draw_samples.return_value = np.ones((3,2))
    
    failure_threshold = -0.1
    mocked_BiasingDist = BiasingDist(trained_surrogate = mock_surrogate, 
                     limit_state = failure_threshold)
    mocked_BiasingDist._input_distribution = mocked_input_distribution
    
    surrogate_samples = mocked_BiasingDist._evaluate_surrogate(num_samples = 3)
    
    assert (surrogate_samples == np.array([-2,-1,0])).all


def test_find_failures_raise_error_with_no_limit_state():
    bd = BiasingDist()
    num_samples = 10
    dummy_inputs = np.ones((num_samples,3))
    dummy_outputs = np.zeros((num_samples,))
    
    with pytest.raises(ValueError):
        bd._find_failures(dummy_inputs, dummy_outputs)


def test_find_failures_with_threshold(mocker, mocked_BiasingDist):
    dummy_inputs = [ele for ele in [1,2,3,4,5,6] for i in range(3)]
    dummy_inputs = np.array(dummy_inputs).reshape((6,3))
    dummy_outputs = np.array([2,-3,-4,6,8,1])

    expected_failures = np.array([[2,2,2],[3,3,3]])
    failures = mocked_BiasingDist._find_failures(dummy_inputs, dummy_outputs)
    
    assert (expected_failures == failures).all()


def test_find_failures_with_limit_state_function():
    dummy_inputs = [ele for ele in [1,2,3,4,5,6] for i in range(3)]
    dummy_inputs = np.array(dummy_inputs).reshape((6,3))
    dummy_outputs = np.array([2,-3,-4,6,8,1])
    def limit_state(input):
        return input + .1
    
    bd = BiasingDist(limit_state = limit_state)
    
    expected_failures = np.array([[2,2,2],[3,3,3]])
    failures = bd._find_failures(dummy_inputs, dummy_outputs)
    
    assert (expected_failures == failures).all()


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

    mock_mixture_model_grid_search = mocker.Mock(return_value = 1)
    mocked_BiasingDist._mixture_model_grid_search = \
        mock_mixture_model_grid_search

    dummy_inputs = np.ones((10,3))    
    mocked_BiasingDist.fit_from_failed_inputs(failed_inputs = dummy_inputs,
                                              covariance_type = cov_type)
    
    mock_mixture_model_grid_search.assert_called()
    
    
def test_GridSearch_called(mocker, mocked_BiasingDist):
    mock_mixture_model = mocker.patch('mfis.biasing_dist.GaussianMixture', 
                            autospec = True)
    mock_grid_search = mocker.patch('mfis.biasing_dist.GridSearchCV')
    mock_grid_search.return_value.best_estimator_ = mock_mixture_model
    
    dummy_train_data = np.ones((100,3))
    
    best_mixture_model = mocked_BiasingDist._mixture_model_grid_search(
        train_data = dummy_train_data, max_clusters = 3, 
        covariance_type = ['full'])
    
    mock_grid_search.assert_called()


def test_evaluate_pdf_calls_evaluate_mixture_model_pdf(mocker, 
                                                       mocked_BiasingDist):
    mock_mixture_model = mocker.patch('mfis.biasing_dist.GaussianMixture', 
                            autospec = True)
    mocked_BiasingDist.mixture_model_ = mock_mixture_model
    
    mock_evaluate_mixed_model_pdf = mocker.patch("mfis.biasing_dist."
                                    "BiasingDist.evaluate_mixture_model_pdf")

    dummy_inputs = np.ones((10,3))
    mocked_BiasingDist.evaluate_pdf(dummy_inputs)
    
    mock_evaluate_mixed_model_pdf.assert_called_with(dummy_inputs)


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


def test_densities_from_mixture_model_are_correct_length(mocker, 
                                                         mocked_BiasingDist):
    mock_mixture_model = mocker.Mock()
    num_samples = 20
    dummy_samples = np.ones((num_samples, 3))
    mock_mixture_model.score_samples.return_value = np.ones((num_samples,1))
    
    mocked_BiasingDist.mixture_model_ = mock_mixture_model

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


def test_samples_drawn_from_mixture_model_are_correct_shape(mocker, 
                                                            mocked_BiasingDist):
    mock_mixture_model = mocker.Mock()
    num_samples = 20
    return_samples = np.ones((num_samples, 3))
    mock_mixture_model.sample.return_value = [return_samples,
                                      np.zeros((num_samples, 3))]

    mocked_BiasingDist.mixture_model_ = mock_mixture_model

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