import unittest

import numpy as np

from geometry import (CheckRemainingRoute, InstrumentRotation, SegmentDistance, instrument_direction,
                      movement_is_clear, shaft_clearance, tip_position)


class SegmentDistanceTests(unittest.TestCase):
    def test_InteriorCrossingAndSkewSegments(self):
        self.assertEqual(SegmentDistance([0, 0, 0], [10, 0, 0],
                                          [5, -4, 0], [5, 4, 0]), 0)
        self.assertEqual(SegmentDistance([0, 0, 0], [10, 0, 0],
                                          [5, -4, 3], [5, 4, 3]), 3)

    def test_ParallelSegmentsAndClampedEndpoints(self):
        for start, end, expected in [
            ([2, 4, 0], [8, 4, 0], 4),
            ([8, 4, 0], [2, 4, 0], 4),
            ([15, 0, 0], [20, 0, 0], 5),
            ([11, -2, 0], [11, 2, 0], 1),
            ([0, 0, 0], [10, 0, 0], 0),
        ]:
            with self.subTest(start=start, end=end):
                self.assertAlmostEqual(SegmentDistance([0, 0, 0], [10, 0, 0], start, end), expected)

    def test_ZeroLengthSegments(self):
        self.assertEqual(SegmentDistance([0, 0, 0], [10, 0, 0], [5, 3, 0], [5, 3, 0]), 3)
        self.assertEqual(SegmentDistance([5, 3, 0], [5, 3, 0], [0, 0, 0], [10, 0, 0]), 3)
        self.assertEqual(SegmentDistance([0, 0, 0], [0, 0, 0], [3, 4, 12], [3, 4, 12]), 13)

    def test_NearlyParallelSegmentsStillDetectInteriorCrossings(self):
        # avoids treating a shallow crossing as parallel just because its angle is small
        for length in (1.0, 1e6):
            height = length * 1e-12
            self.assertEqual(SegmentDistance([0, 0, 0], [length, 0, 0],
                                             [0, -height, 0], [length, height, 0]), 0)

    def test_BroadcastSegmentsAndEndpointOrder(self):
        starts = np.array([[0, 0, 0], [0, 0, 5]])[:, None, :]
        ends = np.array([[10, 0, 0], [10, 0, 5]])[:, None, :]
        other_starts = np.array([[5, -4, 0], [5, -4, 3], [15, -4, 0]])
        other_ends = other_starts + [0, 8, 0]
        expected = [[0, 3, 5], [5, 2, np.sqrt(50)]]
        np.testing.assert_allclose(SegmentDistance(starts, ends, other_starts, other_ends), expected)
        np.testing.assert_allclose(SegmentDistance(ends, starts, other_ends, other_starts), expected)

    def test_InvalidSegmentPoints(self):
        for values in ([0, 0], [0, np.inf, 0], [[0, 0, 0], [1, 2, np.nan]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                SegmentDistance(values, [1, 0, 0], [0, 1, 0], [1, 1, 0])
        with self.assertRaises(ValueError):
            SegmentDistance(np.zeros((2, 3)), np.zeros((3, 3)), [0, 0, 0], [1, 0, 0])


class GeometryTests(unittest.TestCase):
    def test_InstrumentRotationPreservesDirectionAndOrientation(self):
        for yaw, pitch in [(0, 0), (0.4, 0.2), (-0.7, -0.3)]:
            with self.subTest(yaw=yaw, pitch=pitch):
                rotation = InstrumentRotation(yaw, pitch)
                np.testing.assert_allclose(rotation @ [1, 0, 0], instrument_direction(yaw, pitch))
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                np.testing.assert_allclose(np.cross(rotation[:, 0], rotation[:, 1]),
                                           rotation[:, 2], atol=1e-12)

    def test_direction_convention_and_fixed_port(self):
        np.testing.assert_allclose(instrument_direction(0, 0), [1, 0, 0])
        np.testing.assert_allclose(instrument_direction(np.pi / 2, 0), [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(instrument_direction(0, np.pi / 2), [0, 0, 1], atol=1e-12)
        port = np.array([4.0, -7.0, 3.0])
        for yaw, pitch, depth in [(0.4, 0.2, 60), (-0.7, -0.3, 25)]:
            tip = tip_position(port, yaw, pitch, depth)
            direction = instrument_direction(yaw, pitch)
            self.assertAlmostEqual(np.linalg.norm(tip - port), depth)
            np.testing.assert_allclose(tip - depth * direction, port)

    def test_whole_shaft_collision_despite_clear_tip(self):
        port, tip, centre = [0, 0, 0], [100, 0, 0], [50, 2, 0]
        self.assertGreater(np.linalg.norm(np.array(tip) - centre), 40)
        self.assertAlmostEqual(shaft_clearance(port, tip, 1, centre, 3), -2)

    def test_projection_is_clamped_to_segment(self):
        self.assertAlmostEqual(shaft_clearance([0, 0, 0], [10, 0, 0], 1, [-5, 0, 0], 2), 2)
        self.assertAlmostEqual(shaft_clearance([0, 0, 0], [10, 0, 0], 1, [15, 0, 0], 2), 2)
        self.assertAlmostEqual(shaft_clearance([0, 0, 0], [0, 0, 0], 1, [5, 0, 0], 2), 2)

    def test_collision_during_rotation_with_clear_endpoints(self):
        port, centre = [0, 0, 0], [50, 0, 0]
        start = np.array([-0.3, 0.0, 100.0])
        end = np.array([0.3, 0.0, 100.0])
        for configuration in (start, end):
            self.assertGreater(shaft_clearance(port, tip_position(port, *configuration),
                                              1, centre, 3), 0)
        self.assertFalse(movement_is_clear(port, start, end, 1, centre, 3))

    def test_movement_rejects_invalid_geometry(self):
        settings = {
            "port": [0, 0, 0], "start": [-0.3, 0, 100], "end": [0.3, 0, 100],
            "tool_radius": 1, "structure_points": [50, 0, 0], "structure_radius": 3,
        }
        self.assertFalse(movement_is_clear(**settings))
        # rejects non-finite geometry before checking shaft movement
        for values in [
            {"structure_points": [50, np.nan, 0]}, {"structure_radius": np.nan},
            {"tool_radius": np.inf}, {"required_clearance": np.nan},
            {"max_step": np.nan}, {"port": [np.inf, 0, 0]}, {"start": [0, 100]},
            {"tool_radius": -1},
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                movement_is_clear(**dict(settings, **values))

    def test_guard_detects_collision_between_midpoint_samples(self):
        # checks a crossing at yaw zero between clear samples at +/-0.01 radians
        start, end = [-0.02, 0, 100], [0.02, 0, 100]
        for yaw in [-0.01, 0.01]:
            self.assertGreater(shaft_clearance([0, 0, 0], tip_position([0, 0, 0], yaw, 0, 100),
                                              0.1, [50, 0, 0], 0.1), 0)
        self.assertFalse(movement_is_clear([0, 0, 0], start, end, 0.1,
                                          [50, 0, 0], 0.1, max_step=2.0))

    def test_clear_motion_and_required_margin(self):
        port, centre = [0, 0, 0], [50, 15, 0]
        self.assertTrue(movement_is_clear(port, [0, 0, 30], [0, 0, 90], 1,
                                         centre, 3, required_clearance=2))
        self.assertFalse(movement_is_clear(port, [0, 0, 90], [0, 0, 90], 1,
                                          centre, 3, required_clearance=12))

    def test_retraction_at_required_margin_remains_possible(self):
        self.assertTrue(movement_is_clear([0, 0, 0], [0, 0, 80], [0, 0, 20],
                                         1, [50, 6, 0], 3, required_clearance=2))

    def test_pitch_motion_is_checked_in_three_dimensions(self):
        self.assertFalse(movement_is_clear([0, 0, 0], [0, -0.3, 100],
                                          [0, 0.3, 100], 1, [50, 0, 0], 3))
        self.assertTrue(movement_is_clear([0, 0, 0], [0, -0.3, 100],
                                         [0, 0.3, 100], 1, [50, 20, 0], 3))

    def test_guarded_acceptance_against_dense_clearance_reference(self):
        random = np.random.default_rng(24)
        port, centre = [0, 0, 0], [50, 12, 5]
        accepted = 0
        for _ in range(30):
            start = random.uniform([-0.5, -0.4, 10], [0.5, 0.4, 100])
            end = random.uniform([-0.5, -0.4, 10], [0.5, 0.4, 100])
            if movement_is_clear(port, start, end, 1.5, centre, 6, required_clearance=2):
                accepted += 1
                for fraction in np.linspace(0, 1, 501):
                    tip = tip_position(port, *(start + fraction * (end - start)))
                    self.assertGreaterEqual(shaft_clearance(port, tip, 1.5, centre, 6), 2)
        self.assertGreater(accepted, 0)

    def test_VesselMidpointBlocksTheWholeShaftDespiteClearVerticesAndTip(self):
        vessel = np.array([[50, -20, 0], [50, 20, 0]])
        self.assertTrue(np.all(np.linalg.norm(vessel - [50, 0, 0], axis=1) > 10))
        self.assertTrue(np.all(np.linalg.norm(vessel - [100, 0, 0], axis=1) > 40))
        self.assertEqual(shaft_clearance([0, 0, 0], [100, 0, 0], 1.5, vessel, 2), -3.5)

    def test_PolylineClearanceAndBatchTips(self):
        vessel = [[40, 8, -10], [40, 8, 10], [60, 8, 10]]
        tips = np.array([[30, 0, 0], [80, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(shaft_clearance([0, 0, 0], tips, 1, vessel, 2),
                                   [np.sqrt(164) - 3, 5, np.sqrt(1664) - 3])
        np.testing.assert_allclose(shaft_clearance([0, 0, 0], tips, 1, [40, 8, 0], 2),
                                   [np.sqrt(164) - 3, 5, np.sqrt(1664) - 3])
        self.assertAlmostEqual(shaft_clearance([0, 0, 0], tips[1], 1, vessel, 2), 5)
        self.assertEqual(shaft_clearance([0, 0, 0], [80, 0, 0], 1,
                                        [[40, 8, 0], [40, 8, 0]], 2), 5)

    def test_VesselSweepAndGuardRejectBetweenClearMidpointSamples(self):
        vessel = [[50, 0, -10], [50, 0, 10]]
        for yaw in (-0.01, 0.01):
            self.assertGreater(shaft_clearance([0, 0, 0], tip_position([0, 0, 0], yaw, 0, 100),
                                              0.1, vessel, 0.1), 0)
        self.assertFalse(movement_is_clear([0, 0, 0], [-0.02, 0, 100], [0.02, 0, 100],
                                          0.1, vessel, 0.1, max_step=2.0))

    def test_VesselRetractionAndSamplesUseWholeSegmentClearance(self):
        vessel = [[50, 6, -10], [50, 6, 10]]
        samples = []
        self.assertTrue(movement_is_clear([0, 0, 0], [0, 0, 80], [0, 0, 20], 1,
                                         vessel, 3, 2, clearance_samples=samples))
        self.assertGreater(len(samples), 0)
        self.assertGreaterEqual(min(samples), 2)
        self.assertAlmostEqual(min(samples), 2)
        self.assertFalse(movement_is_clear([0, 0, 0], [0, 0, 80], [0, 0, 20], 1,
                                          vessel, 3, 2.01))

    def test_VesselGuardedAcceptanceAgainstDenseMovement(self):
        random = np.random.default_rng(41)
        vessel = [[35, 12, -10], [50, 14, 0], [65, 12, 10]]
        accepted = 0
        for _ in range(20):
            start = random.uniform([-0.5, -0.4, 10], [0.5, 0.4, 100])
            end = random.uniform([-0.5, -0.4, 10], [0.5, 0.4, 100])
            if movement_is_clear([0, 0, 0], start, end, 1.5, vessel, 2, 2):
                accepted += 1
                fractions = np.linspace(0, 1, 1001)
                positions = start + fractions[:, None] * (end - start)
                directions = instrument_direction(positions[:, 0], positions[:, 1]).T
                tips = positions[:, 2, None] * directions
                self.assertGreaterEqual(np.min(shaft_clearance([0, 0, 0], tips, 1.5, vessel, 2)), 2)
        self.assertGreater(accepted, 0)

    def test_InvalidVesselCentrelines(self):
        for vessel in ([], [[1, 2, 3]], [[1, 2], [3, 4]],
                       [[1, 2, 3], [4, np.nan, 6]], np.zeros((2, 2, 3))):
            with self.subTest(vessel=vessel):
                with self.assertRaises(ValueError):
                    shaft_clearance([0, 0, 0], [100, 0, 0], 1, vessel, 2)
                with self.assertRaises(ValueError):
                    movement_is_clear([0, 0, 0], [0, 0, 30], [0, 0, 80], 1, vessel, 2)


class RemainingRouteTests(unittest.TestCase):
    def test_RemainingRouteChecksVesselSegments(self):
        vessel = [[65, 6, -10], [65, 6, 10]]
        path = np.array([[0, 0, 35], [0, 0, 50], [0, 0, 90]])
        self.assertTrue(CheckRemainingRoute([0, 0, 0], path[0], path, 1, 1, vessel, 2, 3))
        self.assertFalse(CheckRemainingRoute([0, 0, 0], path[0], path, 1, 1, vessel, 2, 3.01))

    def test_SpareClearanceAllowsAnIncreasedRequirementWithoutChangingThePath(self):
        path = np.array([[0.0, 0.0, 35.0], [0.0, 0.0, 50.0], [0.0, 0.0, 90.0]])
        original = path.copy()
        self.assertTrue(CheckRemainingRoute(
            [0, 0, 0], path[0], path, 1, 1.5, [65, 11, 0], 6, 3.4))
        np.testing.assert_array_equal(path, original)

    def test_AFutureBottleneckFailsDespiteAClearCurrentShaftAndNextEdge(self):
        path = np.array([[0.0, 0.0, 35.0], [0.0, 0.0, 50.0], [0.0, 0.0, 90.0]])
        self.assertGreater(shaft_clearance([0, 0, 0], tip_position([0, 0, 0], *path[0]),
                                           1.5, [65, 11, 0], 6), 4.0)
        self.assertTrue(movement_is_clear([0, 0, 0], path[0], path[1],
                                          1.5, [65, 11, 0], 6, 4.0))
        self.assertFalse(CheckRemainingRoute(
            [0, 0, 0], path[0], path, 1, 1.5, [65, 11, 0], 6, 4.0))

    def test_CompletedNarrowEdgesDoNotInvalidateTheRemainingRoute(self):
        path = np.array([[0.0, 0.0, 90.0], [0.0, 0.0, 35.0], [0.3, 0.0, 35.0]])
        self.assertFalse(CheckRemainingRoute(
            [0, 0, 0], path[0], path, 1, 1.5, [65, 11, 0], 6, 4.0))
        self.assertTrue(CheckRemainingRoute(
            [0, 0, 0], path[1], path, 2, 1.5, [65, 11, 0], 6, 4.0))

    def test_TheCurrentShaftIsCheckedWhenAllWaypointsAreComplete(self):
        path = np.array([[0.0, 0.0, 90.0]])
        self.assertTrue(CheckRemainingRoute(
            [0, 0, 0], path[0], path, len(path), 1.5, [65, 11, 0], 6, 3.5))
        self.assertFalse(CheckRemainingRoute(
            [0, 0, 0], path[0], path, len(path), 1.5, [65, 11, 0], 6, 3.50001))

    def test_ClearWaypointsStillRejectAnUnsafeShaftSweep(self):
        path = np.array([[-0.3, 0.0, 100.0], [0.3, 0.0, 100.0]])
        for configuration in path:
            self.assertGreater(shaft_clearance(
                [0, 0, 0], tip_position([0, 0, 0], *configuration), 1, [50, 0, 0], 3), 2.0)
        self.assertFalse(CheckRemainingRoute(
            [0, 0, 0], path[0], path, 1, 1, [50, 0, 0], 3, 2.0))


if __name__ == "__main__":
    unittest.main()
