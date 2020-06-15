import numpy as np
import pytest

from mfis import IndependentUniformDistribution


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


    bounds = [[0,1]]
    num_samples = 10
    seed = 1
    uniform_v1 = IndependentUniformDistribution(bounds, seed)
    samples_v1 = uniform_v1.draw_samples(num_samples)
    uniform_v2 = IndependentUniformDistribution(bounds, seed)
    samples_v2 = uniform_v2.draw_samples(num_samples)

    np.testing.assert_array_equal(samples_v1,
                                  samples_v2)

def test_different_seeds_give_different_samples():

    bounds = [[0,1]]
    num_samples = 10
    seed = 1
    uniform_v1 = IndependentUniformDistribution(bounds, seed)
    samples_v1 = uniform_v1.draw_samples(num_samples)
    seed = 2
    uniform_v2 = IndependentUniformDistribution(bounds, seed)
    samples_v2 = uniform_v2.draw_samples(num_samples)

    assert not np.array_equal(samples_v1, samples_v2)


def test_evaluate_pdf():

    bounds = [[0,1],[2,3],[3,5]]
    uniform = IndependentUniformDistribution(bounds)
    samples = np.array([[1,2,3]])

    expected_pdf = np.array([[0.5]])
    np.testing.assert_almost_equal(expected_pdf, uniform.evaluate_pdf(samples))
    
    
@pytest.mark.parametrize("samples, probability",[(np.array([[1,4]]), .5),
                                          (np.array([[0,6]]), 0),
                                          (np.array([[-1,4]]), 0),
                                          (np.array([[-1,6]]), 0)])    
def test_evaluate_pdf_valid_probability(samples, probability):

    bounds = [[0,1],[3,5]]
    uniform = IndependentUniformDistribution(bounds)

    pdf_samples = uniform.evaluate_pdf(samples)
    assert pdf_samples == probability
    #assert pdf_samples >= 0
    
    
def test_nonnegative_uniform_density():
    bounds = [[0,2]]
    num_of_samples = 100
    uniform = IndependentUniformDistribution(bounds)

    assert uniform._get_uniform_density(num_of_samples).all() >=0
                          

def test_finds_samples_outside_bounds():

    bounds = [[0,1],[3,5]]
    uniform = IndependentUniformDistribution(bounds)

    samples = np.array([[.5, 4],[-1, 3],[-1, 2],[2, 2],[.5, 6], [.5, 2]])
    outside_samples_indices =  \
            uniform._indices_of_samples_outside_bounds(samples)
            
    assert np.array_equal(outside_samples_indices, np.array([1,2,3,4,5]))
