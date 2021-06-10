import numpy as np
import pytest
import scipy.stats as ss
from mfis import MVIndependentDistribution


def test_draw_samples_is_correct(mocker):
    normal_samples = np.array([.01, 1.14, -0.4])
    mock_normal_dist = mocker.Mock()
    mock_normal_dist.rvs.return_value = normal_samples
    
    uniform_samples = np.array([.1, .3, .81])
    mock_uniform_dist = mocker.Mock()
    mock_uniform_dist.rvs.return_value = uniform_samples

    input_dist = MVIndependentDistribution([mock_normal_dist, 
                                           mock_uniform_dist])
    num_samples = len(normal_samples)
    samples = input_dist.draw_samples(num_samples)
    
    expected_samples = np.concatenate((normal_samples.reshape((3, 1)), 
                                       uniform_samples.reshape((3, 1))),
                                      axis=1)
    
    np.testing.assert_array_equal(expected_samples, samples)
    
    
def test_evaluate_pdf(mocker):

    normal_densities = np.array([.01, 2, .4])
    mock_normal_dist = mocker.Mock()
    mock_normal_dist.pdf.return_value = normal_densities
    
    uniform_densities = np.array([.25, .25, .25])
    mock_uniform_dist = mocker.Mock()
    mock_uniform_dist.pdf.return_value = uniform_densities
    
    input_dist = MVIndependentDistribution([mock_normal_dist, 
                                           mock_uniform_dist])

    dummy_samples = np.ones((len(normal_densities), 2))
    densities = input_dist.evaluate_pdf(samples = dummy_samples)

    expected_densities = normal_densities * uniform_densities
    np.testing.assert_almost_equal(expected_densities, densities)
    
    
@pytest.mark.parametrize("normal_densities, uniform_densities",
                         [(np.array([0, 4]), np.array([1, 1])),
                          (np.array([1, .2]), np.array([.5, .5]))])    
def test_evaluate_pdf_valid_density(mocker, normal_densities, 
                                        uniform_densities):

    mock_normal_dist = mocker.Mock()
    mock_normal_dist.pdf.return_value = normal_densities
    
    mock_uniform_dist = mocker.Mock()
    mock_uniform_dist.pdf.return_value = uniform_densities
    
    input_dist = MVIndependentDistribution([mock_normal_dist, 
                                           mock_uniform_dist])

    dummy_samples = np.ones((len(normal_densities),2))
    densities = input_dist.evaluate_pdf(samples = dummy_samples)
    assert (densities >= 0).all()
