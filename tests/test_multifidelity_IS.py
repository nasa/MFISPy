import pytest
import numpy as np
from mfis import multiIS
from mfis.input_distribution import InputDistribution
from mock import Mock

      
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


def test_mfis_estimate(mocker):
    N = 10
    weights = 1/np.array(range(100, 100 + N))
    #mock_importance_weights = Mock(return_value = weights)
    
    input_densities = weights
    mock_input_distribution = mocker.create_autospec(InputDistribution)
    mock_input_distribution.evaluate_pdf.return_value = input_densities
    
    biasing_dist_densities = np.ones((1,N))
    mock_bias_distribution = mocker.create_autospec(InputDistribution)
    mock_bias_distribution.evaluate_pdf.return_value = biasing_dist_densities
    
    mIS = multiIS(limit_state = 0.5, 
                  biasing_distribution = mock_bias_distribution,
                  input_distribution = mock_input_distribution)
    
    #mIS.calc_importance_weights = mock_importance_weights
    
    expected_mfis_estimate = np.mean(weights)
    mfis_estimate = mIS.mfis_estimate(np.ones((N, 3)), np.zeros((N,)))
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