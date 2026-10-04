import unittest

import numpy as np

from feedback import get_target_feedback


class FeedbackTests(unittest.TestCase):
    def test_distance_matches_a_hand_calculated_norm(self):
        feedback = get_target_feedback([10, -2, 7], [13, 2, 19])
        self.assertEqual(feedback["distance"], 13.0)
        self.assertFalse(feedback["arrived"])
        self.assertIsNone(feedback["uncertainty"])

    def test_arrival_includes_the_boundary_and_rounding_tolerance(self):
        for distance in [0.0, 1.0, 2.0, 2.0 + 0.5e-9]:
            with self.subTest(distance=distance):
                self.assertTrue(get_target_feedback([0, 0, 0], [distance, 0, 0])["arrived"])
        self.assertFalse(get_target_feedback([0, 0, 0], [2.0 + 2e-9, 0, 0])["arrived"])
        self.assertTrue(get_target_feedback([3, 4, 5], [3, 4, 5], 0)["arrived"])
        self.assertFalse(get_target_feedback([0, 0, 0], [0.1, 0, 0], 0)["arrived"])

    def test_measured_arrival_can_be_false_against_the_actual_position(self):
        actual = get_target_feedback([85, 0, 0], [90, 0, 0])
        measured = get_target_feedback([89, 0, 0], [90, 0, 0])
        self.assertEqual(actual["distance"], 5.0)
        self.assertFalse(actual["arrived"])
        self.assertTrue(measured["arrived"])

    def test_measurements_can_miss_an_actual_arrival(self):
        actual = get_target_feedback([89, 0, 0], [90, 0, 0])
        measured = get_target_feedback([86, 0, 0], [90, 0, 0])
        self.assertTrue(actual["arrived"])
        self.assertFalse(measured["arrived"])

    def test_uncertainty_matches_the_relative_covariance_trace(self):
        tool_covariance = np.array([[4, 1, 0], [1, 9, 0], [0, 0, 16]])
        target_covariance = np.diag([1, 4, 9])
        feedback = get_target_feedback([0, 0, 0], [3, 4, 0], 2,
                                       tool_covariance, target_covariance)
        self.assertAlmostEqual(feedback["uncertainty"], np.sqrt(43.0))
        self.assertEqual(feedback["distance"], 5.0)
        self.assertFalse(feedback["arrived"])

    def test_zero_and_singular_covariances_are_valid(self):
        zero = np.zeros((3, 3))
        exact = get_target_feedback([0, 0, 0], [0, 0, 0], 2, zero, zero)
        self.assertEqual(exact["uncertainty"], 0.0)
        singular = get_target_feedback([0, 0, 0], [0, 0, 0], 2,
                                       np.diag([4, 0, 0]), zero)
        self.assertEqual(singular["uncertainty"], 2.0)
        self.assertTrue(singular["arrived"])

    def test_common_translation_preserves_distance_and_arrival(self):
        tool = np.array([10, -2, 7])
        target = np.array([13, 2, 19])
        translation = np.array([100, 15, -20])
        first = get_target_feedback(tool, target, 14)
        translated = get_target_feedback(tool + translation, target + translation, 14)
        self.assertEqual(first, translated)

    def test_inputs_are_not_mutated(self):
        tool = np.array([3.0, 4.0, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        tool_covariance = np.eye(3)
        target_covariance = 4 * np.eye(3)
        originals = [value.copy() for value in (tool, target, tool_covariance, target_covariance)]
        get_target_feedback(tool, target, 2, tool_covariance, target_covariance)
        for value, original in zip((tool, target, tool_covariance, target_covariance), originals):
            np.testing.assert_array_equal(value, original)

    def test_invalid_positions_are_rejected(self):
        for position in [[], [1, 2], [1, 2, 3, 4], [[1, 2, 3]],
                         [np.nan, 0, 0], [np.inf, 0, 0], ["invalid", 0, 0]]:
            for argument in ["tool_position", "target_position"]:
                with self.subTest(position=position, argument=argument), self.assertRaises(ValueError):
                    settings = {"tool_position": [0, 0, 0], "target_position": [1, 0, 0]}
                    settings[argument] = position
                    get_target_feedback(**settings)

    def test_invalid_target_tolerances_are_rejected(self):
        for tolerance in [-1, np.nan, np.inf, -np.inf, [2], [[2]], "invalid"]:
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                get_target_feedback([0, 0, 0], [1, 0, 0], tolerance)

    def test_covariances_must_be_supplied_together(self):
        for settings in [{"tool_covariance": np.eye(3)}, {"target_covariance": np.eye(3)}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                get_target_feedback([0, 0, 0], [1, 0, 0], **settings)

    def test_invalid_covariances_are_rejected(self):
        invalid = [np.eye(2), np.ones(3), np.ones((3, 3, 1)),
                   np.diag([np.nan, 1, 1]), np.diag([np.inf, 1, 1]),
                   np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
                   np.diag([-1, 1, 1]), np.array([[1, 2, 0], [2, 1, 0], [0, 0, 1]]),
                   [["invalid"] * 3] * 3]
        for covariance in invalid:
            for argument in ["tool_covariance", "target_covariance"]:
                with self.subTest(covariance=covariance, argument=argument), self.assertRaises(ValueError):
                    settings = {"tool_covariance": np.eye(3), "target_covariance": np.eye(3)}
                    settings[argument] = covariance
                    get_target_feedback([0, 0, 0], [1, 0, 0], **settings)


if __name__ == "__main__":
    unittest.main()
