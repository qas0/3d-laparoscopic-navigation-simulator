import unittest

import numpy as np

from feedback import GetRouteGuidance, get_target_feedback


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


class RouteGuidanceTests(unittest.TestCase):
    def test_ChangesRetainRadiansMillimetresAndSigns(self):
        current = [np.deg2rad(10), np.deg2rad(-5), 60]
        path = [[np.deg2rad(-2), np.deg2rad(3), 45]]
        guidance = GetRouteGuidance(current, path)
        self.assertEqual(guidance["waypoint"], 0)
        self.assertFalse(guidance["complete"])
        np.testing.assert_allclose(guidance["change"],
                                   [np.deg2rad(-12), np.deg2rad(8), -15])

    def test_RetractionPrecedesRotationAndReinsertion(self):
        angle = np.deg2rad(20)
        path = np.array([[0, 0, 80], [0, 0, 40],
                         [angle, 0, 40], [angle, 0, 80]])
        waypoint = 0
        for current, expected_index, expected_change in [
                (path[0], 1, [0, 0, -40]),
                (path[1], 2, [angle, 0, 0]),
                (path[2], 3, [0, 0, 40])]:
            with self.subTest(waypoint=expected_index):
                guidance = GetRouteGuidance(current, path, waypoint)
                self.assertEqual(guidance["waypoint"], expected_index)
                self.assertFalse(guidance["complete"])
                np.testing.assert_allclose(guidance["change"], expected_change)
                waypoint = guidance["waypoint"]

    def test_NearbyLaterWaypointDoesNotSkipEarlierMovement(self):
        angle = np.deg2rad(20)
        path = [[0, 0, 80], [0, 0, 40], [angle, 0, 40], [angle, 0, 80]]
        guidance = GetRouteGuidance(path[-1], path, 1)
        self.assertEqual(guidance["waypoint"], 1)
        self.assertFalse(guidance["complete"])
        np.testing.assert_allclose(guidance["change"], [-angle, 0, -40])

    def test_AllAxesMustBeWithinTheirTolerances(self):
        path = [[0, 0, 40], [0, 0, 60]]
        for current in [[np.deg2rad(0.6), 0, 40],
                        [0, np.deg2rad(0.6), 40], [0, 0, 40.6]]:
            with self.subTest(current=current):
                guidance = GetRouteGuidance(current, path)
                self.assertEqual(guidance["waypoint"], 0)
                np.testing.assert_allclose(guidance["change"], np.array(path[0]) - current)
        reached = GetRouteGuidance([np.deg2rad(0.4), np.deg2rad(-0.4), 40.4], path)
        self.assertEqual(reached["waypoint"], 1)

    def test_ExplicitTolerancesIncludeBoundaryAndRounding(self):
        path = [[0, 0, 40], [0, 0, 60]]
        guidance = GetRouteGuidance([0.1 + 0.5e-9, -0.1, 42], path,
                                    angle_tolerance=0.1, depth_tolerance=2)
        self.assertEqual(guidance["waypoint"], 1)
        for current in [[0.1 + 2e-9, 0, 40], [0, 0, 42 + 2e-9]]:
            with self.subTest(current=current):
                self.assertEqual(GetRouteGuidance(current, path, angle_tolerance=0.1,
                                                   depth_tolerance=2)["waypoint"], 0)
        self.assertTrue(GetRouteGuidance([0, 0, 40], path[:1],
                                         angle_tolerance=0, depth_tolerance=0)["complete"])
        self.assertFalse(GetRouteGuidance([0, 0, 40.01], path[:1],
                                          angle_tolerance=0, depth_tolerance=0)["complete"])

    def test_OnePointAndFinishedRoutesReturnZeroChanges(self):
        path = [[0, 0, 40]]
        for current, waypoint in [([0, 0, 40], 0), ([0.2, -0.1, 60], 1)]:
            with self.subTest(waypoint=waypoint):
                guidance = GetRouteGuidance(current, path, waypoint)
                self.assertEqual(guidance["waypoint"], 1)
                self.assertTrue(guidance["complete"])
                np.testing.assert_array_equal(guidance["change"], np.zeros(3))

    def test_ExistingProgressSurvivesTakeoverAndBacktracking(self):
        path = [[0, 0, 40], [0.2, 0, 40], [0.2, 0, 60]]
        takeover = GetRouteGuidance(path[1], path, np.int64(1))
        self.assertEqual(takeover["waypoint"], 2)
        backtracked = GetRouteGuidance(path[0], path, takeover["waypoint"])
        self.assertEqual(backtracked["waypoint"], 2)
        np.testing.assert_allclose(backtracked["change"], [0.2, 0, 20])

    def test_InputsAndReturnedChangesDoNotShareMutableState(self):
        current = np.array([0.1, -0.2, 40.0])
        path = np.array([[0.3, 0.1, 60.0]])
        original_current, original_path = current.copy(), path.copy()
        guidance = GetRouteGuidance(current, path)
        guidance["change"][:] = 0
        np.testing.assert_array_equal(current, original_current)
        np.testing.assert_array_equal(path, original_path)

    def test_InvalidConfigurationsAndPathsAreRejected(self):
        for current in [[], [0, 40], [[0, 0, 40]], [0, 0, -1],
                        [np.nan, 0, 40], [0, np.inf, 40], ["invalid", 0, 40]]:
            with self.subTest(current=current), self.assertRaises(ValueError):
                GetRouteGuidance(current, [[0, 0, 40]])
        for path in [[], [0, 0, 40], [[0, 40]], [[[0, 0, 40]]],
                     [[0, 0, 40], [0, 0, -1]], [[0, 0, np.inf]],
                     [[np.nan, 0, 40]], [["invalid", 0, 40]]]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                GetRouteGuidance([0, 0, 40], path)

    def test_InvalidIndicesAndTolerancesAreRejected(self):
        for waypoint in [-1, 2, 0.0, True, np.bool_(False), [0], None, np.nan]:
            with self.subTest(waypoint=waypoint), self.assertRaises(ValueError):
                GetRouteGuidance([0, 0, 40], [[0, 0, 40]], waypoint)
        for tolerance in [-1, np.nan, np.inf, -np.inf, [0.5], "invalid"]:
            for argument in ["angle_tolerance", "depth_tolerance"]:
                with self.subTest(argument=argument, tolerance=tolerance), self.assertRaises(ValueError):
                    GetRouteGuidance([0, 0, 40], [[0, 0, 40]], **{argument: tolerance})


if __name__ == "__main__":
    unittest.main()
