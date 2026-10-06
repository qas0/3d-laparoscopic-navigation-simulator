import unittest

import numpy as np

from feedback import get_target_feedback
from kalman import KalmanFilter
from sensors import ApplyBias, ApplyDropout, measure_position


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

    def test_DropoutExtremesKeepReadingsAndDrawSequence(self):
        position = np.array([90.0, -4.0, 3.0])
        original = position.copy()
        random_generator = np.random.default_rng(62)
        repeated = np.random.default_rng(62)
        self.assertIs(ApplyDropout(position, 0.0, random_generator), position)
        self.assertIsNone(ApplyDropout(position, 1.0, random_generator))
        np.testing.assert_array_equal(position, original)
        repeated.random(2)
        self.assertEqual(random_generator.random(), repeated.random())

    def test_DropoutSeedRepeatsMasksWithMonotonicThresholds(self):
        position = np.array([90.0, -4.0, 3.0])
        low_generator = np.random.default_rng(63)
        repeated_generator = np.random.default_rng(63)
        high_generator = np.random.default_rng(63)
        for _ in range(200):
            low = ApplyDropout(position, 0.25, low_generator) is None
            repeated = ApplyDropout(position, 0.25, repeated_generator) is None
            high = ApplyDropout(position, 0.75, high_generator) is None
            self.assertEqual(low, repeated)
            if low:
                self.assertTrue(high)

    def test_DropoutAvailabilityMatchesProbability(self):
        probability = 0.3
        count = 10000
        random_generator = np.random.default_rng(64)
        position = np.array([90.0, -4.0, 3.0])
        available = sum(ApplyDropout(position, probability, random_generator) is not None
                        for _ in range(count))
        # checks the availability fraction within six Bernoulli standard errors
        standard_error = np.sqrt(probability * (1 - probability) / count)
        self.assertLess(abs(available / count - (1 - probability)), 6 * standard_error)

    def test_InvalidDropoutInputsAreRejected(self):
        for probability in [-0.1, 1.1, np.nan, np.inf, -np.inf, [0.5], [[0.5]], "invalid"]:
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                ApplyDropout([1, 2, 3], probability, np.random.default_rng(1))
        for position in [None, [], [1, 2], [[1, 2, 3]], [np.nan, 0, 0], [np.inf, 0, 0]]:
            with self.subTest(position=position), self.assertRaises(ValueError):
                ApplyDropout(position, 1.0, np.random.default_rng(1))

    def test_BiasDisabledPreservesInputsWithoutSharingMemory(self):
        position = np.array([90.0, -4.0, 3.0])
        bias = np.zeros(3)
        drift_rate = np.zeros(3)
        measured = ApplyBias(position, bias, drift_rate, 20.0)
        np.testing.assert_array_equal(measured, position)
        for value in (position, bias, drift_rate):
            self.assertFalse(np.shares_memory(measured, value))
        measured[0] = -100.0
        np.testing.assert_array_equal(position, [90.0, -4.0, 3.0])
        np.testing.assert_array_equal(bias, np.zeros(3))
        np.testing.assert_array_equal(drift_rate, np.zeros(3))

    def test_BiasAndDriftUseSignedOffsetsAndElapsedSeconds(self):
        position = np.array([90.0, -4.0, 3.0])
        bias = np.array([-3.0, 2.0, 0.5])
        drift_rate = np.array([0.2, -0.4, 0.0])
        np.testing.assert_array_equal(ApplyBias(position, bias, drift_rate, 0.0),
                                      [87.0, -2.0, 3.5])
        final_reading = ApplyBias(position, bias, drift_rate, 2.5)
        np.testing.assert_array_equal(final_reading, [87.5, -3.0, 3.5])

        # keeps the same final drift when sample intervals change
        elapsed_time = 0.0
        for interval in (0.5, 0.25, 0.75, 1.0):
            elapsed_time += interval
            measured = ApplyBias(position, bias, drift_rate, elapsed_time)
        np.testing.assert_array_equal(measured, final_reading)
        np.testing.assert_array_equal(position, [90.0, -4.0, 3.0])
        np.testing.assert_array_equal(bias, [-3.0, 2.0, 0.5])
        np.testing.assert_array_equal(drift_rate, [0.2, -0.4, 0.0])

    def test_BiasPreservesSeededNoiseAndDropoutSequences(self):
        position = np.array([90.0, -4.0, 3.0])
        noise_generator = np.random.default_rng(65)
        repeated_noise = np.random.default_rng(65)
        dropout_generator = np.random.default_rng(66)
        repeated_dropout = np.random.default_rng(66)
        for sample in range(30):
            elapsed_time = sample * 0.1
            biased = ApplyBias(position, [3.0, -2.0, 1.0], [0.2, 0.0, -0.1], elapsed_time)
            measured = measure_position(biased, 2.0, noise_generator)
            repeated = measure_position(position, 2.0, repeated_noise)
            received = ApplyDropout(measured, 0.4, dropout_generator)
            original = ApplyDropout(repeated, 0.4, repeated_dropout)
            self.assertEqual(received is None, original is None)
            if received is not None:
                np.testing.assert_allclose(received - original,
                                           [3.0 + 0.2 * elapsed_time, -2.0,
                                            1.0 - 0.1 * elapsed_time], atol=1e-12)
        self.assertEqual(noise_generator.random(), repeated_noise.random())
        self.assertEqual(dropout_generator.random(), repeated_dropout.random())

    def test_BiasRejectsInvalidVectorsTimesAndOverflow(self):
        for index in range(3):
            for value in (None, [], [1, 2], [[1, 2, 3]], [np.nan, 0, 0],
                          [np.inf, 0, 0], "invalid"):
                inputs = [[1, 2, 3], [0, 0, 0], [0, 0, 0]]
                inputs[index] = value
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    ApplyBias(*inputs, 1.0)
        for elapsed_time in (-1.0, np.nan, np.inf, -np.inf, [1.0], [[1.0]], "invalid"):
            with self.subTest(elapsed_time=elapsed_time), self.assertRaises(ValueError):
                ApplyBias([1, 2, 3], [0, 0, 0], [0, 0, 0], elapsed_time)
        with self.assertRaises(ValueError):
            ApplyBias([0, 0, 0], [0, 0, 0], [1e308, 0, 0], 2.0)

    def test_BiasCanCauseFalseArrivalDespiteSmallFilterCovariance(self):
        tool = np.array([0.0, 0.0, 0.0])
        target = np.array([5.0, 0.0, 0.0])
        tool_reading = ApplyBias(tool, [0, 0, 0], [0, 0, 0], 0.0)
        target_reading = ApplyBias(target, [-5, 0, 0], [0, 0, 0], 0.0)
        filters = [KalmanFilter(reading, 0.1, acceleration_std=0.0, velocity_std=0.0)
                   for reading in (tool_reading, target_reading)]
        for _ in range(30):
            for filter_, reading in zip(filters, (tool_reading, target_reading)):
                filter_.predict(0.1)
                filter_.update(reading)

        tool_filter, target_filter = filters
        feedback = get_target_feedback(tool_filter.state[:3], target_filter.state[:3],
                                       tool_covariance=tool_filter.covariance[:3, :3],
                                       target_covariance=target_filter.covariance[:3, :3])
        # checks that repeated biased readings reduce covariance without removing their error
        self.assertTrue(feedback["arrived"])
        self.assertLess(feedback["uncertainty"], 0.1)
        self.assertAlmostEqual(np.linalg.norm(target_filter.state[:3] - target), 5.0)
        self.assertFalse(get_target_feedback(tool, target)["arrived"])


if __name__ == "__main__":
    unittest.main()
