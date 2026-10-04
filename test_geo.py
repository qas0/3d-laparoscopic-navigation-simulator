import unittest

import numpy as np

from geometry import instrument_direction, movement_is_clear, shaft_clearance, tip_position


class GeometryTests(unittest.TestCase):
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
            "tool_radius": 1, "structure_centre": [50, 0, 0], "structure_radius": 3,
        }
        self.assertFalse(movement_is_clear(**settings))
        # rejects non-finite geometry before checking shaft movement
        for values in [
            {"structure_centre": [50, np.nan, 0]}, {"structure_radius": np.nan},
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


if __name__ == "__main__":
    unittest.main()
