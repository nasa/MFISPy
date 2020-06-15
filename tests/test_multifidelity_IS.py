import pytest
import numpy as np
from mfis import multiIS
from mock import Mock

      
def test_calc_importance_weights():
    input_densities = np.array([.01, .1, 1])
    attrs = {'evaluate_pdf.return_value': input_densities}
    mock_input_distribution= Mock(**attrs)
    
    biasing_dist_densities = np.array([4, 2, .5])
    attrs = {'evaluate_pdf.return_value': [biasing_dist_densities]}
    mock_bias_distribution= Mock(**attrs)
    
    importance_sampling = multiIS()
    importance_sampling.input_distribution = mock_input_distribution
    importance_sampling.biasing_distribution = mock_bias_distribution
    
    expected_weights = input_densities/biasing_dist_densities
    
    weights = importance_sampling.calc_importance_weights(1)
    
    assert (expected_weights == weights).all()


def test_mfip_estimate():
    N = 10
    mock_find_failures = Mock(return_value = np.ones((N, 3)))
    
    weights = 1/np.array(range(100, 100 + N))
    mock_importance_weights = Mock(return_value = weights)
    
    mIS = multiIS()
    mIS._find_failures = mock_find_failures
    mIS.calc_importance_weights = mock_importance_weights
    
    expected_mfis_estimate = np.mean(weights)
    mfis_estimate = mIS.mfis_estimate(np.zeros((N, 3)), np.zeros((N,)))
    
    np.testing.assert_almost_equal(expected_mfis_estimate,  mfis_estimate)
    
    
def test_find_failures_returns_failures():    
    # check that only inputs are returned that correspond to a failure
    failure_threshold = -0.1
    mIS = multiIS(limit_state = failure_threshold)

    num_inputs = 11
    inputs = np.array(range(1,num_inputs+1)).reshape(num_inputs,1)
    outputs = -1/inputs
    expected_failure_inputs = inputs[(outputs < failure_threshold).flatten(),]
    
    failure_inputs = mIS._find_failures(inputs, outputs)
    
    np.testing.assert_almost_equal(expected_failure_inputs, failure_inputs)