import pytest
import numpy as np
from mfis import multiIS
from mfis.input_distribution import InputDistribution
#from mock import Mock

      
def test_calc_importance_weights(mocker):
    input_densities = np.array([.01, .1, 1])
    mock_input_distribution = mocker.create_autospec(InputDistribution)
    mock_input_distribution.evaluate_pdf.return_value = input_densities
    
    biasing_dist_densities = np.array([4, 2, .5])
    mock_bias_distribution = mocker.create_autospec(InputDistribution)
    mock_bias_distribution.evaluate_pdf.return_value = biasing_dist_densities
    
    importance_sampling = multiIS(biasing_distribution = mock_bias_distribution,
                                  input_distribution = mock_input_distribution)
    
    expected_weights = input_densities/biasing_dist_densities
    
    
    weights = importance_sampling.calc_importance_weights(1)
    
    assert (expected_weights == weights).all()


def test_calc_importance_weights_no_dists_raises_error(mocker):
    importance_sampling = multiIS()
    dummy_inputs = np.ones((10,3))
    
    with pytest.raises(ValueError):
        weights = importance_sampling.calc_importance_weights(dummy_inputs)


def test_calc_importance_weights_with_densities():
    input_densities = np.array([.01, .1, 1])
    biasing_dist_densities = np.array([4, 2, .5])
    
    importance_sampling = multiIS()
    
    expected_weights = input_densities/biasing_dist_densities
    
    weights = importance_sampling._calc_importance_weights_with_densities(
        input_densities = input_densities, 
        biasing_densities = biasing_dist_densities)
    
    assert (expected_weights == weights).all()


def test_find_failure_indicators_with_threshold():
    outputs = np.array([2,3,-4,6,8,1])
    mIS = multiIS(limit_state = 4)
    
    failure_indicators = mIS._find_failure_indicators(outputs)
    
    assert (failure_indicators == np.array([1,1,1,0,0,1])).all()


def test_find_failure_indicators_with_limit_state_function():
    outputs = np.array([2,3,-4,6,8,1])
    def limit_state(input):
        return input - 4
    
    mIS = multiIS(limit_state = limit_state)
    
    failure_indicators = mIS._find_failure_indicators(outputs)
    
    assert (failure_indicators == np.array([1,1,1,0,0,1])).all()


def test_mfis_estimate_is_correct(mocker):
    N = 10
    weights = 1/np.array(range(100, 100 + N))
    
    input_densities = weights
    mock_input_distribution = mocker.create_autospec(InputDistribution)
    mock_input_distribution.evaluate_pdf.return_value = input_densities
    
    biasing_dist_densities = np.ones((1,N))
    mock_bias_distribution = mocker.create_autospec(InputDistribution)
    mock_bias_distribution.evaluate_pdf.return_value = biasing_dist_densities
    
    mIS = multiIS(limit_state = 0.5, 
                  biasing_distribution = mock_bias_distribution,
                  input_distribution = mock_input_distribution)
    
    
    expected_mfis_estimate = np.sum(np.delete(weights,0))/N
    dummy_inputs = np.ones((N, 3))
    dummy_outputs = np.concatenate([np.ones((1,)),np.zeros((N-1,))])
    mfis_estimate = mIS.mfis_estimate(dummy_inputs, dummy_outputs)
    np.testing.assert_almost_equal(expected_mfis_estimate,  mfis_estimate[0])
    
    
def test_mfis_estimate_without_distributions(mocker):
    N = 10
    input_densities= 1/np.array(range(100, 100 + N))
    bias_densities = np.ones((N,1))

    mIS = multiIS(limit_state = 0.5)
    
    expected_mfis_estimate = np.mean(input_densities)
    mfis_estimate = mIS.mfis_estimate(inputs = np.ones((N, 3)), 
                                      outputs = np.zeros((N,)),
                                      input_densities = input_densities,
                                      biasing_densities = bias_densities)
    np.testing.assert_almost_equal(expected_mfis_estimate,  mfis_estimate[0])
    
    
def test_mfis_estimate_raises_error_no_bias_dist(mocker):
    #mock_mIS = mocker.Mock() 
    num_inputs = 10
    mock_input_distribution = mocker.create_autospec(InputDistribution)
    mock_input_distribution.evaluate_pdf.return_value = np.ones((num_inputs,))
    
    mIS = multiIS(limit_state = 0.5, 
                  input_distribution = mock_input_distribution)
    dummy_inputs = np.ones((num_inputs,3))
    dummy_outputs = np.zeros((num_inputs,))
    
    with pytest.raises(ValueError):
        mIS.mfis_estimate(inputs = dummy_inputs, outputs = dummy_outputs)
        
        
def test_mfis_estimate_raises_error_no_input_dist(mocker):
    #mock_mIS = mocker.Mock() 
    num_inputs = 10
    mock_bias_distribution = mocker.create_autospec(InputDistribution)
    mock_bias_distribution.evaluate_pdf.return_value = np.ones((num_inputs,))
    
    mIS = multiIS(limit_state = 0.5, 
                  biasing_distribution = mock_bias_distribution)
    dummy_inputs = np.ones((num_inputs,3))
    dummy_outputs = np.zeros((num_inputs,))
    
    with pytest.raises(ValueError):
        mIS.mfis_estimate(inputs = dummy_inputs, outputs = dummy_outputs)