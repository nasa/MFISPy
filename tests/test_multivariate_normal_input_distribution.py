import numpy as np
import pytest
from scipy.stats import norm
from mfis import MultivariateNormalDistribution


@pytest.mark.parametrize("mean, cov,num_samples",
                         [([3,6,0.2],
                          np.array([[2,.7,.4], [.7,.5,.02], [.4,.02,1]]),
                          5), ([3,6], np.array([[1.5,.2], [.2,.045]]), 10)])
def test_draw_samples_is_correct_shape(mean, cov, num_samples):

    mvn = MultivariateNormalDistribution(mean, cov)
    samples = mvn.draw_samples(num_samples)
    
    shape_expected = np.array([num_samples, len(mvn._mean)])
    np.testing.assert_array_equal(shape_expected, samples.shape)

def test_draw_sample_is_within_bounds():
    
    mean = [3,6,0.2]
    cov = np.array([[2,0.7,0.4], [0.7,0.5,0.02], [0.4,0.02,1]])
    num_samples = 10000
    mvn = MultivariateNormalDistribution(mean, cov)
    samples = mvn.draw_samples(num_samples)

    for i,var in enumerate(cov):
        assert np.min(samples[:,i]) > mean[i] - 6 * np.sqrt(var[i])
        assert np.max(samples[:,i]) < mean[i] + 6 * np.sqrt(var[i])

def test_same_seed_gives_same_samples():

    mean = [3,6,0.2]
    cov = np.array([[2,0.7,0.4], [0.7,0.5,0.02], [0.4,0.02,1]])
    num_samples = 10
    seed = 1
    mvn_v1 = MultivariateNormalDistribution(mean, cov, seed)
    samples_v1 = mvn_v1.draw_samples(num_samples)
    mvn_v2 = MultivariateNormalDistribution(mean, cov, seed)
    samples_v2 = mvn_v2.draw_samples(num_samples)

    np.testing.assert_array_equal(samples_v1,
                                  samples_v2)

def test_different_seeds_give_different_samples():

    mean = [3,6,0.2]
    cov = np.array([[2,0.7,0.4], [0.7,0.5,0.02], [0.4,0.02,1]])
    num_samples = 10
    seed = 1
    mvn_v1 = MultivariateNormalDistribution(mean, cov, seed)
    samples_v1 = mvn_v1.draw_samples(num_samples)
    seed = 2
    mvn_v2 = MultivariateNormalDistribution(mean, cov, seed)
    samples_v2 = mvn_v2.draw_samples(num_samples)

    assert not np.array_equal(samples_v1, samples_v2)

def test_evaluate_pdf():

    mean = [3,6]
    cov = np.array([[1.5,0], [0,0.045]])
    mvn = MultivariateNormalDistribution(mean, cov)
    samples = np.array([[2,6.3]])

    expected_pdf = norm.pdf(samples[0,0], mean[0], np.sqrt(cov[0,0])) * \
                    norm.pdf(samples[0,1], mean[1], np.sqrt(cov[1,1]))
    np.testing.assert_almost_equal(expected_pdf.reshape(1,1), 
                                   mvn.evaluate_pdf(samples))
    
def test_evaluate_pdf_valid_probability():

    mean = [3,6]
    cov = np.array([[1.5,.2], [.2,0.045]])
    mvn = MultivariateNormalDistribution(mean, cov)
    samples = np.array([[2,6.3]])

    pdf_samples = mvn.evaluate_pdf(samples)
    assert pdf_samples <= 1
    assert pdf_samples >= 0
    

