import unittest

import numpy as np

from kalman import KalmanFilter
from sensors import measure_position


class KalmanTests(unittest.TestCase):
    def test_initial_state_uses_the_observation_once(self):
        position = np.array([90.0, -4.0, 3.0])
        tracker = KalmanFilter(position, 2.0, velocity_std=3.0)
        np.testing.assert_array_equal(tracker.state, [90, -4, 3, 0, 0, 0])
        np.testing.assert_array_equal(tracker.covariance, np.diag([4, 4, 4, 9, 9, 9]))
        position[0] = -100.0
        self.assertEqual(tracker.state[0], 90.0)

    def test_prediction_matches_calculated_motion_and_covariance(self):
        tracker = KalmanFilter([10, -4, 2], 2.0, acceleration_std=2.0, velocity_std=3.0)
        tracker.state[3:] = [3, -2, 1]
        tracker.predict(0.5)
        np.testing.assert_allclose(tracker.state, [11.5, -5, 2.5, 3, -2, 1])
        # calculates the 0.5 s position, cross + velocity covariance blocks
        identity = np.eye(3)
        expected = np.block([[6.3125 * identity, 4.75 * identity],
                             [4.75 * identity, 10.0 * identity]])
        np.testing.assert_allclose(tracker.covariance, expected)

    def test_correction_matches_a_hand_calculated_gain(self):
        tracker = KalmanFilter([0, 0, 0], 2.0, acceleration_std=0.0, velocity_std=3.0)
        tracker.predict(1.0)
        observation = np.array([2.0, -4.0, 6.0])
        tracker.update(observation)
        # calculates gains of 13/17 for position + 9/17 for velocity
        expected_state = np.concatenate((observation * 13 / 17, observation * 9 / 17))
        np.testing.assert_allclose(tracker.state, expected_state)
        identity = np.eye(3)
        expected_covariance = np.block([[52 / 17 * identity, 36 / 17 * identity],
                                        [36 / 17 * identity, 72 / 17 * identity]])
        np.testing.assert_allclose(tracker.covariance, expected_covariance)

    def tracking_errors(self, velocity, seed):
        random_generator = np.random.default_rng(seed)
        initial = np.array([90.0, -4.0, 3.0])
        tracker = KalmanFilter(measure_position(initial, 2.0, random_generator),
                               2.0, acceleration_std=1.0)
        raw_errors = []
        filtered_errors = []
        for sample in range(1, 301):
            actual = initial + np.asarray(velocity) * (sample * 0.1)
            measured = measure_position(actual, 2.0, random_generator)
            tracker.predict(0.1)
            tracker.update(measured)
            if sample > 20:
                raw_errors.append(measured - actual)
                filtered_errors.append(tracker.state[:3] - actual)
        # measures 3D position RMSE after the initial settling period
        raw_rmse = np.sqrt(np.mean(np.sum(np.square(raw_errors), axis=1)))
        filtered_rmse = np.sqrt(np.mean(np.sum(np.square(filtered_errors), axis=1)))
        return raw_rmse, filtered_rmse, tracker.state.copy()

    def test_stationary_tracking_reduces_measurement_error(self):
        raw_rmse, filtered_rmse, _ = self.tracking_errors([0, 0, 0], 71)
        self.assertLess(filtered_rmse, 0.65 * raw_rmse)

    def test_constant_velocity_tracking_is_accurate_and_repeatable(self):
        velocity = [6, -2, 1]
        first = self.tracking_errors(velocity, 72)
        repeated = self.tracking_errors(velocity, 72)
        np.testing.assert_array_equal(first[:2], repeated[:2])
        np.testing.assert_array_equal(first[2], repeated[2])
        self.assertLess(first[1], 0.65 * first[0])
        np.testing.assert_allclose(first[2][3:], velocity, atol=0.5)

    def test_covariance_stays_symmetric_and_nonnegative(self):
        random_generator = np.random.default_rng(73)
        tracker = KalmanFilter([0, 0, 0], 1.5)
        for _ in range(300):
            tracker.predict(random_generator.uniform(0.01, 0.2))
            tracker.update(measure_position([0, 0, 0], 1.5, random_generator))
            self.assertTrue(np.all(np.isfinite(tracker.state)))
            np.testing.assert_allclose(tracker.covariance, tracker.covariance.T, atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(tracker.covariance).min(), -1e-10)

    def test_zero_noise_handles_exact_and_degenerate_observations(self):
        for acceleration_std, velocity_std in [(10.0, 25.0), (0.0, 0.0)]:
            with self.subTest(acceleration_std=acceleration_std, velocity_std=velocity_std):
                tracker = KalmanFilter([1, 2, 3], 0.0, acceleration_std, velocity_std)
                for observation in [[2, 4, 6], [-1, 3, 5], [0, 0, 0]]:
                    tracker.predict(0.1)
                    tracker.update(observation)
                    np.testing.assert_array_equal(tracker.state[:3], observation)
                    np.testing.assert_array_equal(tracker.covariance[:3], np.zeros((3, 6)))
                    self.assertGreaterEqual(np.linalg.eigvalsh(tracker.covariance).min(), -1e-10)

    def test_prediction_without_observations_increases_position_uncertainty(self):
        tracker = KalmanFilter([0, 0, 0], 2.0, acceleration_std=1.0, velocity_std=4.0)
        tracker.state[3:] = [1, 2, 3]
        previous_variance = np.diag(tracker.covariance)[:3].copy()
        for _ in range(10):
            tracker.predict(0.1)
            variance = np.diag(tracker.covariance)[:3]
            self.assertTrue(np.all(variance > previous_variance))
            previous_variance = variance.copy()
        np.testing.assert_allclose(tracker.state[:3], [1, 2, 3], atol=1e-12)

    def test_invalid_initial_values_are_rejected(self):
        settings = {"position": [1, 2, 3], "noise_std": 1.0}
        for changes in [{"position": [1, 2]}, {"position": [[1, 2, 3]]},
                        {"position": [np.nan, 0, 0]}, {"position": [np.inf, 0, 0]},
                        {"noise_std": -1}, {"noise_std": np.inf}, {"noise_std": [1]},
                        {"acceleration_std": -1}, {"acceleration_std": np.nan},
                        {"velocity_std": -1}, {"velocity_std": np.inf}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                KalmanFilter(**dict(settings, **changes))

    def test_invalid_updates_and_timesteps_leave_the_filter_unchanged(self):
        tracker = KalmanFilter([1, 2, 3], 1.0)
        state = tracker.state.copy()
        covariance = tracker.covariance.copy()
        for dt in [0, -0.1, np.nan, np.inf, [0.1]]:
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                tracker.predict(dt)
        for observation in [[1, 2], [[1, 2, 3]], [np.nan, 0, 0], [0, np.inf, 0]]:
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                tracker.update(observation)
        np.testing.assert_array_equal(tracker.state, state)
        np.testing.assert_array_equal(tracker.covariance, covariance)


if __name__ == "__main__":
    unittest.main()
