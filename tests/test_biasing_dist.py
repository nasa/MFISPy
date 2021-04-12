import pytest
import numpy as np
from mfis import BiasingDistribution

@pytest.fixture
def mock_bias_dist(mocker):
    mock_surrogate = mocker.Mock()
    failure_threshold = 0.1
    bias_dist = BiasingDistribution(trained_surrogate=mock_surrogate,
                                    limit_state=failure_threshold)
    return bias_dist


def test_fit_calls_mixture_model_grid_search(mocker, mock_bias_dist):
    mock_surrogate_failed_inputs = mocker.Mock(return_value=np.array([2, 3]))
    mock_bias_dist.get_m_failed_inputs_from_surrogate_draws = \
        mock_surrogate_failed_inputs

    mock_fit_from_failed_inputs = mocker.Mock()
    mock_bias_dist.fit_from_failed_inputs = \
        mock_fit_from_failed_inputs

    mock_bias_dist.fit(n_samples=50, covariance_type='full')

    mock_fit_from_failed_inputs.assert_called()


def test_get_m_failed_inputs_ends_with_error(mocker, mock_bias_dist):
    side_effects = [np.empty((0, 3)), np.ones((1, 3)), np.empty((0, 3)),
                    np.ones((2, 3)), np.ones((4, 3))]
    mock_surrogate_failed_inputs = mocker.Mock(side_effect=side_effects)
    mock_bias_dist.get_failed_inputs_from_surrogate_draws = \
        mock_surrogate_failed_inputs
    mock_bias_dist._input_dim = 3

    with pytest.raises(ValueError):
        mock_bias_dist.get_m_failed_inputs_from_surrogate_draws(
            n_samples=10, min_failures=8, max_sample_batches=5)


def test_get_m_failed_inputs(mocker, mock_bias_dist):
    side_effects = [np.empty((0, 3)), np.ones((1, 3)), np.empty((0, 3)),
                    np.ones((2, 3)), np.ones((4, 3)), np.ones((10, 3))]
    mock_surrogate_failed_inputs = mocker.Mock(side_effect=side_effects)
    mock_bias_dist.get_failed_inputs_from_surrogate_draws = \
        mock_surrogate_failed_inputs
    mock_bias_dist._input_dim = 3
    
    min_failures = 6
    failed_inputs = \
        mock_bias_dist.get_m_failed_inputs_from_surrogate_draws(
            n_samples=10, min_failures=min_failures,
            max_sample_batches=len(side_effects))

    assert len(failed_inputs) >= min_failures


def test_surrogate_failed_inputs_returned(mocker, mock_bias_dist):
    mock_evaluate_surrogate = mocker.Mock(return_value=1)
    mock_bias_dist._evaluate_surrogate = mock_evaluate_surrogate
    mock_bias_dist._draw_input_samples = mock_evaluate_surrogate

    failures = np.array([1, 2, 3])
    mock_find_failures = mocker.Mock(return_value=failures)
    mock_bias_dist.find_failures = mock_find_failures

    failed_inputs_received = \
            mock_bias_dist.get_failed_inputs_from_surrogate_draws(
                n_samples=10)

    np.testing.assert_array_almost_equal(failed_inputs_received, failures)


def test_evaluate_surrogate_raises_error_with_no_surrogate(mocker):
    bias_dist = BiasingDistribution(limit_state=-1)

    mocked_input_distribution = mocker.Mock()
    mocked_input_distribution().draw_samples.return_value = np.ones((3, 2))
    bias_dist._input_distribution = mocked_input_distribution

    dummy_input_samples = np.ones((3, 2))
    with pytest.raises(ValueError):
        bias_dist._evaluate_surrogate(samples=dummy_input_samples)


def test_draw_input_samples_error_with_no_input_dist(mocker,
                                                     mock_bias_dist):
    with pytest.raises(ValueError):
        mock_bias_dist._draw_input_samples(n_samples=4)


def test_evaluate_surrogate_runs(mocker):
    mock_surrogate = mocker.Mock()
    mock_surrogate.predict.return_value = np.array([-2, -1, 0])

    failure_threshold = -0.1
    mocked_BiasingDist = BiasingDistribution(trained_surrogate=mock_surrogate,
                                             limit_state=failure_threshold)

    dummy_samples = np.ones((3, 2))
    surrogate_samples = mocked_BiasingDist._evaluate_surrogate(samples=
                                                               dummy_samples)
    
    np.testing.assert_array_almost_equal(surrogate_samples,
                                         np.array([-2, -1, 0]).reshape((-1,1)))


def test_find_failures_raise_error_with_no_limit_state():
    bias_dist = BiasingDistribution()
    n_samples = 10
    dummy_inputs = np.ones((n_samples, 3))
    dummy_outputs = np.zeros((n_samples,2))

    with pytest.raises(ValueError):
        bias_dist.find_failures(dummy_inputs, dummy_outputs)


def test_find_failures_with_threshold(mocker, mock_bias_dist):
    dummy_inputs = [ele for ele in [1, 2, 3, 4, 5, 6] for i in range(3)]
    dummy_inputs = np.array(dummy_inputs).reshape((6, 3))
    dummy_outputs = np.array([2, -3, -4, 6, 8, 1]).reshape((6,1))

    expected_failures = np.array([[2, 2, 2], [3, 3, 3]])
    failures = mock_bias_dist.find_failures(dummy_inputs, dummy_outputs)

    np.testing.assert_array_almost_equal(expected_failures, failures)


def test_find_failures_with_limit_state_function():
    dummy_inputs = [ele for ele in [1, 2, 3, 4, 5, 6] for i in range(3)]
    dummy_inputs = np.array(dummy_inputs).reshape((6, 3))
    dummy_outputs = np.array([2, -3, -4, 6, 8, 1]).reshape((6,1))
    def limit_state(outputs):
        return outputs + .1

    bias_dist = BiasingDistribution(limit_state=limit_state)

    expected_failures = np.array([[2, 2, 2], [3, 3, 3]])
    failures = bias_dist.find_failures(dummy_inputs, dummy_outputs)

    np.testing.assert_array_almost_equal(expected_failures, failures)


@pytest.mark.parametrize("cov_type", ["dia", -1])
def test_bad_covariance_type_raises_error_in_fit_from_failed_inputs(
        mock_bias_dist, cov_type):
    dummy_inputs = np.ones((10, 3))

    with pytest.raises(ValueError):
        mock_bias_dist.fit_from_failed_inputs(failed_inputs=dummy_inputs,
                                              covariance_type=cov_type)


@pytest.mark.parametrize("cov_type", ["diag", ['spherical', 'tied']])
def test_valid_covariance_type_in_fit_from_failed_inputs_runs(mocker,
                                                              mock_bias_dist,
                                                              cov_type):

    mock_mixture_model_grid_search = mocker.Mock(return_value=1)
    mock_bias_dist._mixture_model_grid_search = \
        mock_mixture_model_grid_search

    dummy_inputs = np.ones((10, 3))
    mock_bias_dist.fit_from_failed_inputs(failed_inputs=dummy_inputs,
                                          covariance_type=cov_type)

    mock_mixture_model_grid_search.assert_called()


def test_grid_search_called(mocker, mock_bias_dist):
    mock_mixture_model = mocker.patch('mfis.biasing_distribution.'
                                      'GaussianMixture',
                                      autospec=True)
    mock_grid_search = mocker.patch('mfis.biasing_distribution.GridSearchCV')
    mock_grid_search.return_value.best_estimator_ = mock_mixture_model

    dummy_train_data = np.ones((100, 3))

    best_mixture_model = mock_bias_dist._mixture_model_grid_search(
        train_data=dummy_train_data, max_clusters=3,
        covariance_type=['full'])

    mock_grid_search.assert_called()


def test_evaluate_pdf_calls_evaluate_mixture_model_pdf(mocker,
                                                       mock_bias_dist):
    mock_mixture_model = mocker.patch('mfis.biasing_distribution.'
                                      'GaussianMixture',
                                      autospec=True)
    mock_bias_dist.mixture_model_ = mock_mixture_model

    mock_evaluate_mixed_model_pdf = mocker.patch("mfis.biasing_distribution."
                                                 "BiasingDistribution."
                                                 "_evaluate_mixture_model_pdf")

    dummy_inputs = np.ones((10, 3))
    mock_bias_dist.evaluate_pdf(dummy_inputs)

    mock_evaluate_mixed_model_pdf.assert_called_with(dummy_inputs)


def test_check_distribution_exists_decorator_raises_error(mock_bias_dist):

    with pytest.raises(ValueError):
        mock_bias_dist.draw_samples(2)


def test_densities_from_mixture_model_are_correct_length(mocker,
                                                         mock_bias_dist):
    mock_mixture_model = mocker.Mock()
    n_samples = 20
    dummy_samples = np.ones((n_samples, 3))
    mock_mixture_model.score_samples.return_value = np.ones((n_samples, 1))

    mock_bias_dist.mixture_model_ = mock_mixture_model

    densities = mock_bias_dist._evaluate_mixture_model_pdf(
        samples=dummy_samples)

    assert len(densities) == n_samples


def test_samples_drawn_from_mixture_model_are_correct_shape(mocker,
                                                            mock_bias_dist):
    mock_mixture_model = mocker.Mock()
    n_samples = 20
    return_samples = np.ones((n_samples, 3))
    mock_mixture_model.sample.return_value = [return_samples,
                                              np.zeros((n_samples, 3))]
    mock_mixture_model.means_ = np.ones((2,3))
    mock_bias_dist.mixture_model_ = mock_mixture_model

    samples = mock_bias_dist.draw_samples(n_samples)
    np.testing.assert_array_almost_equal(samples, return_samples)


def test_saved_file(mocker):
    mock_pickle = mocker.patch('mfis.biasing_distribution.pickle.dump')
    save_path = "bias_dist.obj"
    mock_surrogate = mocker.Mock()
    bias_dist = BiasingDistribution(trained_surrogate=mock_surrogate,
                                    limit_state=1)

    bias_dist.save(save_path)
    mock_pickle.assert_called_once()


def test_loaded_file(mocker):
    mock_pickle = mocker.patch('mfis.biasing_distribution.pickle.load')
    save_path = "bias_dist.obj"
    mock_surrogate = mocker.Mock()
    bias_dist = BiasingDistribution(trained_surrogate=mock_surrogate,
                                    limit_state=1)

    bias_dist.load(save_path)
    mock_pickle.assert_called_once()
