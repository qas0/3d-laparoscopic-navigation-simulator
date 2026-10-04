import unittest
from threading import Event

import numpy as np

from astar import find_path
from geometry import movement_is_clear, shaft_clearance, tip_position
from scenarios import SCENARIOS


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "target": np.array([90.0, 0.0, 0.0]), "port": np.zeros(3),
            "tool_radius": 1.5, "structure_centre": np.array([65.0, 11.0, 0.0]),
            "structure_radius": 6.0,
            "lower_limits": np.array([np.deg2rad(-45), np.deg2rad(-35), 10.0]),
            "upper_limits": np.array([np.deg2rad(45), np.deg2rad(35), 110.0]),
        }

    def check_route(self, result, start, settings):
        self.assertEqual(result["status"], "found")
        path = result["path"]
        np.testing.assert_array_equal(path[0], start)
        self.assertLessEqual(np.linalg.norm(tip_position(settings["port"], *path[-1])
                                           - settings["target"]),
                             settings.get("target_tolerance", 2.0) + 1e-9)
        for first, second in zip(path, path[1:]):
            self.assertEqual(np.count_nonzero(abs(second - first) > 1e-12), 1)
            self.assertTrue(movement_is_clear(
                settings["port"], first, second, settings["tool_radius"],
                settings["structure_centre"], settings["structure_radius"],
                settings.get("required_clearance", 2.0)))
            for fraction in np.linspace(0, 1, 41):
                tip = tip_position(settings["port"], *(first + fraction * (second - first)))
                self.assertGreaterEqual(shaft_clearance(
                    settings["port"], tip, settings["tool_radius"],
                    settings["structure_centre"], settings["structure_radius"]),
                    settings.get("required_clearance", 2.0) - 1e-9)

    def scenario_settings(self, name):
        scenario = SCENARIOS[name]
        settings = dict(self.settings,
                        target=tip_position(self.settings["port"], *scenario["target_configuration"]),
                        structure_centre=scenario["structure_centre"].copy(),
                        structure_radius=scenario["structure_radius"])
        return scenario["start"].copy(), settings

    def test_direct_route_has_expected_length(self):
        start, settings = self.scenario_settings("Direct insertion")
        result = find_path(start=start, **settings)
        self.check_route(result, start, settings)
        self.assertAlmostEqual(result["cost"], 55.0)
        np.testing.assert_array_equal(result["path"][:, :2], np.zeros((len(result["path"]), 2)))

    def test_off_grid_start_is_preserved_without_diagonal_movement(self):
        start = np.array([np.deg2rad(1.3), np.deg2rad(-2.1), 36.7])
        result = find_path(start=start, **self.settings)
        self.check_route(result, start, self.settings)

    def test_retraction_route_matches_shortest_discrete_cost(self):
        settings = dict(self.settings, tool_radius=1.0, structure_radius=3.0,
                        structure_centre=np.array([30.0, 0.0, 0.0]),
                        lower_limits=np.array([np.deg2rad(-20), 0, 10.0]),
                        upper_limits=np.array([np.deg2rad(20), 0, 40.0]),
                        grid_step=np.array([np.deg2rad(10), np.deg2rad(5), 10.0]),
                        target_tolerance=0.01)
        start = np.array([np.deg2rad(-20), 0, 40.0])
        settings["target"] = tip_position(settings["port"], np.deg2rad(20), 0, 40.0)
        result = find_path(start=start, **settings)
        self.check_route(result, start, settings)
        self.assertAlmostEqual(min(result["path"][:, 2]), 20.0)
        # checks the shortest route: retract 20 mm, rotate at depth 20, insert 20 mm
        self.assertAlmostEqual(result["cost"], 40.0 + 20.0 * np.deg2rad(40))

    def test_clear_endpoints_do_not_allow_a_shaft_sweep_through_an_obstacle(self):
        settings = dict(self.settings, tool_radius=1.0, structure_radius=3.0,
                        structure_centre=np.array([50.0, 0.0, 0.0]),
                        lower_limits=np.array([-0.3, 0, 100.0]),
                        upper_limits=np.array([0.3, 0, 100.0]),
                        grid_step=np.array([0.6, 0.1, 5.0]), required_clearance=0.0)
        start = np.array([-0.3, 0, 100.0])
        settings["target"] = tip_position(settings["port"], 0.3, 0, 100.0)
        result = find_path(start=start, **settings)
        self.assertEqual(result["status"], "no_path")
        self.assertIsNone(result["path"])

    def test_three_dimensional_route_uses_pitch_to_pass_the_obstacle(self):
        start, settings = self.scenario_settings("3D detour")
        goal = SCENARIOS["3D detour"]["target_configuration"]
        self.assertFalse(movement_is_clear(
            settings["port"], start, goal, settings["tool_radius"],
            settings["structure_centre"], settings["structure_radius"], 2.0))
        result = find_path(start=start, **settings)
        self.check_route(result, start, settings)
        self.assertTrue(np.any(abs(result["path"][:, 1]) > 1e-9))

    def test_fixed_scenarios_have_valid_starting_configurations(self):
        for name in SCENARIOS:
            with self.subTest(scenario=name):
                start, settings = self.scenario_settings(name)
                self.assertTrue(np.all(start >= settings["lower_limits"]))
                self.assertTrue(np.all(start <= settings["upper_limits"]))
                self.assertGreaterEqual(shaft_clearance(
                    settings["port"], tip_position(settings["port"], *start),
                    settings["tool_radius"], settings["structure_centre"],
                    settings["structure_radius"]), 2.0)

    def test_retraction_preset_requires_depth_change_with_full_pitch_range(self):
        start, settings = self.scenario_settings("Retraction required")
        maximum_pitch = max(abs(settings["lower_limits"][1]), abs(settings["upper_limits"][1]))
        # checks that every allowed pitch remains blocked at yaw zero while fully inserted
        maximum_clearance = (settings["structure_centre"][0] * np.sin(maximum_pitch)
                             - settings["structure_radius"] - settings["tool_radius"])
        self.assertLess(maximum_clearance, 2.0)
        result = find_path(start=start, **settings)
        self.check_route(result, start, settings)
        self.assertLessEqual(min(result["path"][:, 2]), 20.0)

    def test_inaccessible_preset_has_a_clear_tip_but_an_obstructed_final_shaft(self):
        start, settings = self.scenario_settings("Inaccessible target")
        tip_clearance = (np.linalg.norm(settings["target"] - settings["structure_centre"])
                         - settings["structure_radius"] - settings["tool_radius"])
        self.assertGreater(tip_clearance, 2.0)
        result = find_path(start=start, **settings)
        self.assertEqual(result["status"], "invalid_target")
        self.assertIsNone(result["path"])

    def test_coarse_grid_failure_is_distinct_from_an_invalid_target(self):
        settings = dict(self.settings, structure_centre=np.array([500.0, 500.0, 500.0]),
                        target_tolerance=0.2)
        settings["target"] = tip_position(settings["port"], np.deg2rad(2.5), 0, 90.0)
        result = find_path(start=[0, 0, 35], **settings)
        self.assertEqual(result["status"], "no_grid_goal")
        self.assertEqual(find_path(start=[0, 0, 35], **dict(settings, target=[200, 0, 0]))
                         ["status"], "invalid_target")

    def test_optional_target_insertion_reaches_an_off_grid_observation(self):
        settings = dict(self.settings, target=np.array([90.0, -2.5, 1.5]))
        start = np.array([0.0, 0.0, 35.0])
        self.assertEqual(find_path(start=start, **settings)["status"], "no_grid_goal")
        result = find_path(start=start, include_target=True, **settings)
        self.check_route(result, start, settings)
        # follows supplied observation rather than the original scene target
        self.assertLess(tip_position(settings["port"], *result["path"][-1])[1], -1.0)

    def test_optional_target_insertion_preserves_an_off_grid_start(self):
        settings = dict(self.settings, target=np.array([90.0, -2.5, 1.5]))
        start = np.array([np.deg2rad(1.3), np.deg2rad(-2.1), 36.7])
        result = find_path(start=start, include_target=True, **settings)
        self.check_route(result, start, settings)

    def test_optional_target_insertion_handles_an_exact_off_grid_goal(self):
        settings = dict(self.settings, target_tolerance=0.0,
                        structure_centre=np.array([500.0, 500.0, 500.0]))
        configuration = np.array([np.deg2rad(-1.3), np.deg2rad(0.7), 91.4])
        settings["target"] = tip_position(settings["port"], *configuration)
        start = np.array([0.0, 0.0, 35.0])
        self.assertEqual(find_path(start=start, **settings)["status"], "no_grid_goal")
        result = find_path(start=start, include_target=True, **settings)
        self.check_route(result, start, settings)
        np.testing.assert_allclose(result["path"][-1], configuration, atol=1e-12, rtol=0)

    def test_optional_target_insertion_uses_the_nearest_legal_depth(self):
        settings = dict(self.settings)
        settings["target"] = tip_position(settings["port"], np.deg2rad(-1.6),
                                          np.deg2rad(1.1), 111.5)
        start = np.array([0.0, 0.0, 35.0])
        self.assertEqual(find_path(start=start, **settings)["status"], "no_grid_goal")
        result = find_path(start=start, include_target=True, **settings)
        self.check_route(result, start, settings)
        self.assertTrue(np.all(result["path"] >= settings["lower_limits"]))
        self.assertTrue(np.all(result["path"] <= settings["upper_limits"]))
        self.assertAlmostEqual(result["path"][-1, 2], 110.0)

    def test_optional_target_insertion_still_rejects_infeasible_goals(self):
        start = np.array([0.0, 0.0, 35.0])
        for changes in [{"target": np.array([200.0, 0.0, 0.0])},
                        {"structure_centre": np.array([65.0, 0.0, 0.0])}]:
            with self.subTest(changes=changes):
                result = find_path(start=start, include_target=True,
                                   **dict(self.settings, **changes))
                self.assertEqual(result["status"], "invalid_target")
                self.assertIsNone(result["path"])

    def test_optional_target_insertion_does_not_skip_a_blocked_sweep(self):
        settings = dict(self.settings, tool_radius=1.0, structure_radius=3.0,
                        structure_centre=np.array([50.0, 0.0, 0.0]),
                        lower_limits=np.array([-0.3, 0, 100.0]),
                        upper_limits=np.array([0.3, 0, 100.0]),
                        grid_step=np.array([0.6, 0.1, 5.0]), required_clearance=0.0)
        start = np.array([-0.3, 0, 100.0])
        settings["target"] = tip_position(settings["port"], 0.27, 0, 100.0)
        self.assertGreater(shaft_clearance(settings["port"], settings["target"],
                                          settings["tool_radius"], settings["structure_centre"],
                                          settings["structure_radius"]), 0)
        result = find_path(start=start, include_target=True, **settings)
        self.assertEqual(result["status"], "no_path")
        self.assertIsNone(result["path"])

    def test_target_tolerance_can_include_a_point_outside_insertion_limits(self):
        settings = dict(self.settings, target=np.array([112.0, 0, 0]))
        start = np.array([0.0, 0.0, 35.0])
        result = find_path(start=start, **settings)
        self.check_route(result, start, settings)
        self.assertAlmostEqual(result["path"][-1, 2], 110.0)

    def test_invalid_start_and_whole_shaft_target_obstruction(self):
        result = find_path(start=[np.deg2rad(10), 0, 90], **self.settings)
        self.assertEqual(result["status"], "invalid_start")
        result = find_path(start=[0, 0, 35],
                           **dict(self.settings, structure_centre=np.array([65.0, 0, 0])))
        self.assertEqual(result["status"], "invalid_target")

    def test_budget_and_cancellation_are_not_reported_as_no_path(self):
        result = find_path(start=[0, 0, 35], max_expansions=1, **self.settings)
        self.assertEqual(result["status"], "budget_exceeded")
        result = find_path(start=[0, 0, 35], max_expansions=1,
                           **dict(self.settings, target=np.array([40.0, 0, 0])))
        self.assertEqual(result["status"], "found")
        event = Event()
        event.set()
        result = find_path(start=[0, 0, 35], cancel_event=event, **self.settings)
        self.assertEqual(result["status"], "cancelled")

    def test_target_already_reached_requires_no_movement(self):
        result = find_path(start=[0, 0, 90], **self.settings)
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["path"]), 1)
        self.assertEqual(result["cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
