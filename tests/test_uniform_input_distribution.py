import numpy as np
import pytest

from mfis.uniform_distribution import IndependentUniformDistribution


@pytest.mark.parametrize("bounds,num_samples",[([[0,1]], 5),
                                              ([[0,1],[2,3],[3,5]], 10)])
def test_draw_samples_is_correct_shape(bounds, num_samples):

    uniform = IndependentUniformDistribution(bounds)
    samples = uniform.draw_samples(num_samples)
    
    shape_expected = np.array([num_samples, len(bounds)])
    np.testing.assert_array_equal(shape_expected, samples.shape)

def test_draw_sample_is_within_bounds():

    bounds = [[0,1],[2,3],[3,5]]
    num_samples = 10000
    uniform = IndependentUniformDistribution(bounds)
    samples = uniform.draw_samples(num_samples)

    for i,bound in enumerate(bounds):
        assert np.min(samples[:,i]) > bound[0]
        assert np.max(samples[:,i]) < bound[1]

def test_same_seed_gives_same_samples():

    bounds = [[0,1]]
    num_samples = 10
    seed = 1
    uniform_v1 = IndependentUniformDistribution(bounds, seed)
    samples_v1 = uniform_v1.draw_samples(num_samples)
    uniform_v2 = IndependentUniformDistribution(bounds, seed)
    samples_v2 = uniform_v2.draw_samples(num_samples)

    np.testing.assert_array_equal(samples_v1,
                                  samples_v2)

def test_evaluate_pdf():

    bounds = [[0,1],[2,3],[3,5]]
    uniform = IndependentUniformDistribution(bounds)
    samples = np.array([[1,2,3]])

    expected_pdf = np.array([0.5])
    np.testing.assert_almost_equal(expected_pdf, uniform.evaluate_pdf(samples))

    

