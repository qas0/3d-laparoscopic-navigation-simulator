import unittest

import numpy as np

from sensors import measure_position


class SensorTests(unittest.TestCase):
    def test_zero_noise_preserves_position_without_sharing_memory(self):
        position = np.array([70.0, -12.0, 8.0])
        original = position.copy()
        measured = measure_position(position, 0.0, np.random.default_rng(10))
        np.testing.assert_array_equal(measured, original)
        self.assertFalse(np.shares_memory(measured, position))
        measured[0] = -100.0
        np.testing.assert_array_equal(position, original)

    def test_seed_reproduces_a_sequence_with_changing_readings(self):
        first = np.random.default_rng(23)
        second = np.random.default_rng(23)
        position = [90.0, 0.0, 0.0]
        readings = np.array([measure_position(position, 2.0, first) for _ in range(5)])
        repeated = np.array([measure_position(position, 2.0, second) for _ in range(5)])
        np.testing.assert_array_equal(readings, repeated)
        self.assertFalse(np.array_equal(readings[0], readings[1]))

    def test_noise_strength_scales_the_same_underlying_sequence(self):
        position = np.array([70.0, -12.0, 8.0])
        first = np.random.default_rng(38)
        second = np.random.default_rng(38)
        for _ in range(5):
            small = measure_position(position, 1.0, first) - position
            large = measure_position(position, 3.0, second) - position
            np.testing.assert_allclose(large, 3.0 * small, atol=1e-12)

        # preserves the measurement sequence when a reading uses zero noise
        first = np.random.default_rng(38)
        second = np.random.default_rng(38)
        measure_position(position, 0.0, first)
        measure_position(position, 1.0, second)
        np.testing.assert_array_equal(measure_position(position, 2.0, first),
                                      measure_position(position, 2.0, second))

    def test_sample_mean_and_covariance_match_the_noise_model(self):
        sigma = 2.5
        count = 10000
        random_generator = np.random.default_rng(61)
        position = np.array([90.0, -4.0, 3.0])
        errors = np.array([measure_position(position, sigma, random_generator) - position
                           for _ in range(count)])
        # checks six standard errors around the expected mean + covariance
        self.assertTrue(np.all(abs(errors.mean(axis=0)) < 6.0 * sigma / np.sqrt(count)))
        covariance = np.cov(errors, rowvar=False)
        expected = sigma ** 2 * np.eye(3)
        standard_error = sigma ** 2 * np.sqrt((np.ones((3, 3)) + np.eye(3)) / (count - 1))
        self.assertTrue(np.all(abs(covariance - expected) < 6.0 * standard_error))

    def test_invalid_positions_are_rejected(self):
        for position in [[], [1, 2], [1, 2, 3, 4], [[1, 2, 3]],
                         [np.nan, 0, 0], [np.inf, 0, 0], [-np.inf, 0, 0]]:
            with self.subTest(position=position), self.assertRaises(ValueError):
                measure_position(position, 1.0, np.random.default_rng(1))

    def test_invalid_noise_strengths_are_rejected(self):
        for sigma in [-1.0, np.nan, np.inf, -np.inf, [1.0], [[1.0]]]:
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                measure_position([1, 2, 3], sigma, np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main()
