import unittest

import numpy as np

from motion import GetRespiratoryMotion


class MotionTests(unittest.TestCase):
    def test_CycleLandmarksRepeatAndDisabledMotionStaysStill(self):
        expected = [(0.0, 2 * np.pi), (4.0, 0.0), (0.0, -2 * np.pi),
                    (-4.0, 0.0), (0.0, 2 * np.pi)]
        for elapsed_time, result in enumerate(expected):
            with self.subTest(elapsed_time=elapsed_time):
                np.testing.assert_allclose(GetRespiratoryMotion(4.0, 0.25, elapsed_time),
                                           result, atol=1e-12)
                np.testing.assert_allclose(GetRespiratoryMotion(4.0, 0.25, elapsed_time + 4),
                                           result, atol=1e-12)
        for elapsed_time in (0.0, 100.0):
            self.assertEqual(GetRespiratoryMotion(0.0, 0.25, elapsed_time), (0.0, 0.0))
            self.assertEqual(GetRespiratoryMotion(4.0, 0.0, elapsed_time), (0.0, 0.0))

    def test_AmplitudeAndFrequencyScaleDisplacementAndVelocity(self):
        displacement, velocity = GetRespiratoryMotion(3.0, 0.2, 0.7)
        larger = GetRespiratoryMotion(6.0, 0.2, 0.7)
        faster = GetRespiratoryMotion(3.0, 0.4, 0.35)
        np.testing.assert_allclose(larger, [2 * displacement, 2 * velocity], atol=1e-12)
        np.testing.assert_allclose(faster, [displacement, 2 * velocity], atol=1e-12)

    def test_VelocityMatchesTheNumericalDisplacementDerivative(self):
        step = 1e-5
        for elapsed_time in (0.17, 0.75, 1.8, 2.7):
            with self.subTest(elapsed_time=elapsed_time):
                before, _ = GetRespiratoryMotion(6.0, 0.2, elapsed_time - step)
                after, _ = GetRespiratoryMotion(6.0, 0.2, elapsed_time + step)
                _, velocity = GetRespiratoryMotion(6.0, 0.2, elapsed_time)
                self.assertAlmostEqual((after - before) / (2 * step), velocity, delta=1e-7)

    def test_FrameIntervalsPreserveMotionAndSmallerStepsImproveIntegration(self):
        elapsed_time = 0.0
        for interval in (0.2, 0.35, 0.05, 0.9, 0.5):
            elapsed_time += interval
            final_motion = GetRespiratoryMotion(3.0, 0.25, elapsed_time)
        np.testing.assert_allclose(final_motion, GetRespiratoryMotion(3.0, 0.25, 2.0),
                                   atol=1e-12)

        # compares velocity integration with the exact position after one second
        exact_position, _ = GetRespiratoryMotion(3.0, 0.25, 1.0)
        errors = []
        for count in (10, 100):
            step = 1.0 / count
            position = sum(GetRespiratoryMotion(3.0, 0.25, sample * step)[1] * step
                           for sample in range(count))
            errors.append(abs(position - exact_position))
        self.assertGreater(errors[0], 0.0)
        self.assertLess(errors[1], errors[0] / 5)

    def test_InvalidInputsAndOverflowAreRejected(self):
        for index in range(3):
            for value in (-1.0, np.nan, np.inf, -np.inf, None, [], [1.0], [[1.0]], "invalid"):
                inputs = [3.0, 0.25, 1.0]
                inputs[index] = value
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    GetRespiratoryMotion(*inputs)
        for inputs in ((1e308, 1.0, 0.0), (1.0, 1e308, 1.0), (1.0, 1.0, 1e308)):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                GetRespiratoryMotion(*inputs)


if __name__ == "__main__":
    unittest.main()
