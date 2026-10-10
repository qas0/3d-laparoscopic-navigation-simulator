import unittest
from threading import Event
from unittest.mock import patch

import numpy as np

from astar import find_path
from experiments import CompareMethods, GetUncertaintyCoverage, RunTrial
from geometry import tip_position
from motion import GetRespiratoryMotion
from scenarios import SCENARIOS
from sensors import ApplyDropout


class UncertaintyCoverageTests(unittest.TestCase):
    def test_CorrelatedCovarianceUsesTheWholeEllipsoid(self):
        covariance = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.3], [0.0, 0.3, 1.0]])
        factor = np.linalg.cholesky(covariance)
        direction = np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0)
        inside = factor @ (direction * np.sqrt(7.81))
        outside = factor @ (direction * np.sqrt(7.82))
        self.assertTrue(GetUncertaintyCoverage(inside, covariance))
        self.assertFalse(GetUncertaintyCoverage(outside, covariance))

    def test_RotationAndScalingKeepTheSameCoverage(self):
        covariance = np.diag([1.0, 4.0, 9.0])
        rotation = np.array([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
        for error, expected in (([1.0, 2.0, 3.0], True), ([1.0, 4.0, 6.0], False)):
            error = np.asarray(error)
            self.assertEqual(GetUncertaintyCoverage(error, covariance), expected)
            self.assertEqual(GetUncertaintyCoverage(rotation @ error,
                             rotation @ covariance @ rotation.T), expected)
            self.assertEqual(GetUncertaintyCoverage(error * 0.01,
                             covariance * 0.01 ** 2), expected)

    def test_TinyPositiveCovarianceRetainsItsStandardizedError(self):
        covariance = np.eye(3) * 1e-24
        self.assertTrue(GetUncertaintyCoverage([1e-12, 0.0, 0.0], covariance))
        self.assertFalse(GetUncertaintyCoverage([3e-12, 0.0, 0.0], covariance))

    def test_ZeroCovarianceCoversOnlyNumericallyExactPositions(self):
        covariance = np.zeros((3, 3))
        self.assertTrue(GetUncertaintyCoverage([0.0, 0.0, 0.0], covariance))
        self.assertTrue(GetUncertaintyCoverage([1e-10, 0.0, 0.0], covariance))
        self.assertFalse(GetUncertaintyCoverage([2e-9, 0.0, 0.0], covariance))
        self.assertFalse(GetUncertaintyCoverage([5.0, 0.0, 0.0], covariance))

    def test_SingularCovarianceUsesItsRankAndChecksZeroVarianceDirections(self):
        for covariance, inside, outside in (
                (np.diag([4.0, 0.0, 0.0]), np.sqrt(3.84), np.sqrt(3.85)),
                (np.diag([4.0, 9.0, 0.0]), np.sqrt(5.99), np.sqrt(6.00))):
            self.assertTrue(GetUncertaintyCoverage([2 * inside, 0.0, 0.0], covariance))
            self.assertFalse(GetUncertaintyCoverage([2 * outside, 0.0, 0.0], covariance))
            self.assertTrue(GetUncertaintyCoverage([0.0, 0.0, 1e-10], covariance))
            self.assertFalse(GetUncertaintyCoverage([0.0, 0.0, 2e-9], covariance))

        rotation = np.array([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
        covariance = rotation @ np.diag([4.0, 0.0, 0.0]) @ rotation.T
        self.assertTrue(GetUncertaintyCoverage(rotation @ [2.0, 0.0, 0.0], covariance))
        self.assertFalse(GetUncertaintyCoverage(rotation @ [0.0, 1.0, 0.0], covariance))

    def test_GaussianErrorsHaveApproximatelyNominalCoverage(self):
        covariance = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.3], [0.0, 0.3, 1.0]])
        errors = np.random.default_rng(24).multivariate_normal(np.zeros(3), covariance, 5000)
        fraction = np.mean([GetUncertaintyCoverage(error, covariance) for error in errors])
        self.assertAlmostEqual(fraction, 0.95, delta=0.015)

    def test_InvalidErrorsAndCovariancesAreRejected(self):
        for error in ([0.0, 0.0], [[0.0, 0.0, 0.0]], [np.nan, 0.0, 0.0],
                      [0.0, np.inf, 0.0], ["bad", 0.0, 0.0]):
            with self.subTest(error=error), self.assertRaises(ValueError):
                GetUncertaintyCoverage(error, np.eye(3))
        for covariance in (np.eye(2), np.ones((3, 2)), np.eye(3) * np.nan,
                           [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                           np.diag([1.0, -0.1, 1.0]), [[1.0], [0.0, 1.0], [0.0]]):
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                GetUncertaintyCoverage(np.zeros(3), covariance)


class ExperimentTests(unittest.TestCase):
    def test_ZeroNoiseReachesTheTrueTargetWithExpectedTravel(self):
        trial = RunTrial(tool_noise=0.0, target_noise=0.0)
        self.assertEqual(trial["status"], "confirmed_arrival")
        self.assertTrue(trial["success"])
        self.assertTrue(trial["reported_arrival"])
        self.assertTrue(trial["actual_arrived"])
        self.assertFalse(trial["false_arrival"])
        self.assertAlmostEqual(trial["travel"], 55.0)
        self.assertAlmostEqual(trial["actual_error"], 0.0)
        self.assertAlmostEqual(trial["minimum_clearance"], 3.5)
        self.assertGreater(trial["completion_time"], 55.0 / 15.0)
        for error in trial["rmse"].values():
            self.assertAlmostEqual(error, 0.0)
        for sensor in ("tool", "target"):
            self.assertEqual(trial["predicted_rms"][sensor], 0.0)
            self.assertEqual(trial["coverage"][sensor], 100.0)

    def test_BiasedRawTargetCanReportArrivalAwayFromTheTrueTarget(self):
        trial = RunTrial(tracking="raw", tool_noise=0.0, target_noise=0.0,
                         target_bias=(8.0, 0.0, 0.0))
        self.assertEqual(trial["status"], "false_arrival")
        self.assertTrue(trial["reported_arrival"])
        self.assertTrue(trial["false_arrival"])
        self.assertFalse(trial["success"])
        self.assertFalse(trial["actual_arrived"])
        self.assertIsNone(trial["completion_time"])
        self.assertGreater(trial["actual_error"], 2.0)
        self.assertAlmostEqual(trial["rmse"]["target_raw"], 8.0)
        self.assertAlmostEqual(trial["rmse"]["target_filtered"], 8.0)
        self.assertEqual(trial["predicted_rms"]["target"], 0.0)
        self.assertEqual(trial["coverage"]["target"], 0.0)
        np.testing.assert_array_equal(trial["planned_target"], [98.0, 0.0, 0.0])

    def test_MissingTrackingPreventsFilteredPlanningButIdealCanMove(self):
        settings = dict(tool_noise=0.0, target_noise=0.0, tool_dropout=1.0)
        unavailable = RunTrial(**settings)
        self.assertEqual(unavailable["status"], "tracking_unavailable")
        self.assertIsNone(unavailable["plan"])
        self.assertFalse(unavailable["success"])
        self.assertEqual(unavailable["travel"], 0.0)

        ideal = RunTrial(tracking="ideal", **settings)
        self.assertTrue(ideal["success"])
        self.assertEqual(ideal["dropped"]["tool"], ideal["sensor_samples"])
        for source in ("tool_raw", "tool_filtered"):
            self.assertEqual(ideal["samples"][source], 0)
            self.assertIsNone(ideal["rmse"][source])
        self.assertIsNone(ideal["predicted_rms"]["tool"])
        self.assertIsNone(ideal["coverage"]["tool"])
        for source in ("target_raw", "target_filtered"):
            self.assertEqual(ideal["samples"][source], ideal["sensor_samples"])
            self.assertAlmostEqual(ideal["rmse"][source], 0.0)

    def test_UncertaintyCanRejectAPathThatConventionalPlanningAccepts(self):
        settings = dict(tracking="ideal", tool_noise=0.0, target_noise=0.0,
                        tool_bias=(5.0, 0.0, 0.0))
        conventional = RunTrial(**settings)
        cautious = RunTrial(mode="uncert_aware", **settings)
        self.assertTrue(conventional["success"])
        self.assertEqual(cautious["status"], "invalid_target")
        self.assertEqual(cautious["plan"]["status"], "invalid_target")
        self.assertAlmostEqual(cautious["plan"]["extra_clearance"], 5.0)
        self.assertFalse(cautious["success"])
        self.assertEqual(cautious["travel"], 0.0)

    def test_MovingTargetKeepsThePlanningSnapshotAndRepeatsWithTheSeed(self):
        original = SCENARIOS["Direct insertion"]["start"].copy()
        settings = dict(seed=19, tracking="ideal", tool_noise=0.5, target_noise=0.5,
                        motion_amplitude=6.0, motion_frequency=0.2)
        first = RunTrial(**settings)
        repeated = RunTrial(**settings)
        planned_motion, _ = GetRespiratoryMotion(6.0, 0.2, 1.0)
        np.testing.assert_allclose(first["planned_target"], [90.0, 0.0, planned_motion],
                                   atol=1e-12)
        final_motion, _ = GetRespiratoryMotion(6.0, 0.2, first["elapsed_time"])
        true_target = np.array([90.0, 0.0, final_motion])
        self.assertAlmostEqual(first["actual_error"], np.linalg.norm(
            tip_position(np.zeros(3), *first["configuration"]) - true_target))
        self.assertLessEqual(np.linalg.norm(tip_position(
            np.zeros(3), *first["plan"]["path"][-1]) - first["planned_target"]),
            2.0 + 1e-9)

        # checks repeated simulation results while leaving wall-clock planning time separate
        for key in ("status", "success", "false_arrival", "reported_arrival", "actual_arrived",
                    "actual_error", "elapsed_time", "completion_time", "travel",
                    "minimum_clearance", "sensor_samples", "dropped", "rmse", "samples",
                    "predicted_rms", "coverage"):
            self.assertEqual(first[key], repeated[key], key)
        for key in ("configuration", "planned_target"):
            np.testing.assert_array_equal(first[key], repeated[key])
        np.testing.assert_array_equal(first["plan"]["path"], repeated["plan"]["path"])
        np.testing.assert_array_equal(SCENARIOS["Direct insertion"]["start"], original)

    def test_ExecutionDeadlineRetainsAnUnfinishedTrial(self):
        trial = RunTrial(tool_noise=0.0, target_noise=0.0, warmup=0.0, duration=0.2)
        self.assertEqual(trial["status"], "timeout")
        self.assertFalse(trial["success"])
        self.assertIsNone(trial["completion_time"])
        self.assertLessEqual(trial["elapsed_time"], 0.2 + 1e-9)
        self.assertGreater(trial["travel"], 0.0)
        self.assertLess(trial["travel"], 55.0)
        self.assertGreater(trial["actual_error"], 2.0)

    def test_ComparisonKeepsFailedSearchesAndUsesOnlySuccessfulCompletionMetrics(self):
        comparison = CompareMethods("Inaccessible target", seeds=[0, 1],
                                    tool_noise=0.0, target_noise=0.0)
        self.assertEqual(len(comparison["trials"]), 6)
        self.assertEqual(comparison["seeds"], [0, 1])
        for summary in comparison["summary"].values():
            self.assertEqual(summary["trials"], 2)
            self.assertEqual(summary["successes"], 0)
            self.assertEqual(summary["success_rate"], 0.0)
            self.assertEqual(summary["statuses"], {"invalid_target": 2})
            self.assertEqual(summary["metrics"]["actual_error"]["count"], 2)
            self.assertAlmostEqual(summary["metrics"]["actual_error"]["mean"], 55.0)
            self.assertAlmostEqual(summary["metrics"]["actual_error"]["sd"], 0.0)
            for name in ("successful_travel", "successful_completion_time"):
                self.assertEqual(summary["metrics"][name]["count"], 0)
                self.assertIsNone(summary["metrics"][name]["mean"])
                self.assertIsNone(summary["metrics"][name]["sd"])

    def test_OneTrialDoesNotInventASampleStandardDeviation(self):
        comparison = CompareMethods(seeds=[23], tool_noise=0.0, target_noise=0.0)
        self.assertEqual(comparison["settings"]["tool_noise"], 0.0)
        for summary in comparison["summary"].values():
            self.assertEqual(summary["successes"], 1)
            self.assertEqual(summary["success_rate"], 1.0)
            for metric in summary["metrics"].values():
                self.assertEqual(metric["count"], 1)
                self.assertIsNone(metric["sd"])
            self.assertGreaterEqual(summary["metrics"]["successful_travel"]["mean"],
                                    55.0 - 1e-9)
        self.assertAlmostEqual(comparison["summary"]["conventional"]["metrics"][
            "successful_travel"]["mean"], 55.0)

    def test_ComparisonDoesNotTreatUnavailableReadingsAsZeroErrors(self):
        comparison = CompareMethods(seeds=[0], tool_dropout=1.0, target_dropout=1.0)
        for summary in comparison["summary"].values():
            self.assertEqual(summary["statuses"], {"tracking_unavailable": 1})
            for name in ("planning_time", "tool_raw_rmse", "target_raw_rmse",
                         "tool_filtered_rmse", "target_filtered_rmse", "tool_predicted_rms",
                         "target_predicted_rms", "tool_coverage", "target_coverage"):
                self.assertEqual(summary["metrics"][name]["count"], 0)
                self.assertIsNone(summary["metrics"][name]["mean"])

    def test_TwoIndependentTrialsUseTheSampleStandardDeviation(self):
        comparison = CompareMethods(seeds=[0, 1], tracking="ideal", tool_noise=0.5,
                                    target_noise=0.5, warmup=0.0, duration=0.5)
        for mode, summary in comparison["summary"].items():
            errors = [trial["rmse"]["tool_raw"] for trial in comparison["trials"]
                      if trial["mode"] == mode]
            metric = summary["metrics"]["tool_raw_rmse"]
            self.assertEqual(metric["count"], 2)
            self.assertGreater(abs(errors[0] - errors[1]), 0.01)
            self.assertAlmostEqual(metric["mean"], (errors[0] + errors[1]) / 2)
            self.assertAlmostEqual(metric["sd"], abs(errors[0] - errors[1]) / np.sqrt(2))
            self.assertEqual(summary["statuses"], {"timeout": 2})
            self.assertEqual(summary["success_rate"], 0.0)

    def test_PartialDropoutCountsOnlyAvailableReadingsAndInitializedPredictions(self):
        readings = []

        def RecordCoverage(error, covariance):
            covered = GetUncertaintyCoverage(error, covariance)
            readings.append((error.copy(), covariance.copy(), covered))
            return covered

        with patch("experiments.GetUncertaintyCoverage", side_effect=RecordCoverage):
            trial = RunTrial(seed=22, tracking="ideal", tool_noise=0.5, target_noise=0.5,
                             tool_dropout=0.35, target_dropout=0.2, warmup=0.5, duration=0.8)
        self.assertEqual(trial["status"], "timeout")
        self.assertEqual(trial["sensor_samples"], 9)
        for sensor in ("tool", "target"):
            self.assertEqual(trial["samples"][sensor + "_raw"],
                             trial["sensor_samples"] - trial["dropped"][sensor])
        # leaves the first three missing tool readings out of filter error calculations
        self.assertEqual(trial["samples"]["tool_raw"], 3)
        self.assertEqual(trial["samples"]["tool_filtered"], 6)
        self.assertEqual(trial["samples"]["target_filtered"], 9)
        self.assertEqual(len(readings), 15)
        sensor_readings = {"tool": readings[3::2], "target": readings[:3] + readings[4::2]}
        for sensor, observations in sensor_readings.items():
            errors, covariances, covered = zip(*observations)
            self.assertEqual(len(observations), trial["samples"][sensor + "_filtered"])
            self.assertAlmostEqual(trial["rmse"][sensor + "_filtered"],
                                   np.sqrt(np.mean(np.sum(np.asarray(errors) ** 2, axis=1))))
            self.assertAlmostEqual(trial["predicted_rms"][sensor],
                                   np.sqrt(np.mean([np.trace(value) for value in covariances])))
            self.assertAlmostEqual(trial["coverage"][sensor], 100 * np.mean(covered))
        for error in trial["rmse"].values():
            self.assertGreater(error, 0.0)

    def test_CalibrationSummaryWeightsTrialsEquallyDespiteDifferentSampleCounts(self):
        template = RunTrial(tool_noise=0.0, target_noise=0.0, warmup=0.0, duration=0.2)

        def UnequalTrial(scenario, mode, seed, **settings):
            trial = dict(template, scenario=scenario, mode=mode, seed=seed)
            trial["samples"] = dict.fromkeys(template["samples"], 1 if seed == 0 else 100)
            trial["coverage"] = dict.fromkeys(("tool", "target"), 100.0 if seed == 0 else 0.0)
            trial["predicted_rms"] = dict.fromkeys(("tool", "target"), 1.0 if seed == 0 else 3.0)
            return trial

        with patch("experiments.RunTrial", side_effect=UnequalTrial):
            comparison = CompareMethods(seeds=[0, 1])
        for summary in comparison["summary"].values():
            self.assertEqual(summary["statuses"], {"timeout": 2})
            for sensor in ("tool", "target"):
                coverage = summary["metrics"][sensor + "_coverage"]
                predicted = summary["metrics"][sensor + "_predicted_rms"]
                self.assertEqual(coverage["count"], 2)
                self.assertEqual(coverage["mean"], 50.0)
                self.assertAlmostEqual(coverage["sd"], 100.0 / np.sqrt(2))
                self.assertEqual(predicted["count"], 2)
                self.assertEqual(predicted["mean"], 2.0)
                self.assertAlmostEqual(predicted["sd"], np.sqrt(2))

    def test_SmallerTimestepsKeepArrivalAndCommonSensorSequences(self):
        coarse = RunTrial(tool_noise=0.0, target_noise=0.0, timestep=0.02)
        fine = RunTrial(tool_noise=0.0, target_noise=0.0, timestep=0.01)
        self.assertTrue(coarse["success"])
        self.assertTrue(fine["success"])
        self.assertAlmostEqual(coarse["travel"], fine["travel"])
        self.assertAlmostEqual(coarse["actual_error"], fine["actual_error"])
        self.assertLessEqual(abs(coarse["completion_time"] - fine["completion_time"]), 0.1)

        settings = dict(seeds=[0], tracking="ideal", tool_noise=0.5, target_noise=0.5,
                        warmup=0.0, duration=0.5)
        comparisons = [CompareMethods(timestep=step, **settings) for step in (0.02, 0.01)]
        expected = comparisons[0]["trials"][0]["rmse"]
        for comparison in comparisons:
            for trial in comparison["trials"]:
                self.assertEqual(trial["status"], "timeout")
                self.assertEqual(trial["sensor_samples"], 6)
                for source in ("tool_raw", "target_raw"):
                    self.assertAlmostEqual(trial["rmse"][source], expected[source], places=12)

    def test_InvalidSeedsAndSimulationSettingsAreRejected(self):
        for seeds in ([], [1, 1], [-1], [1.5]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                CompareMethods(seeds=seeds)
        for settings in ({"timestep": 0.0}, {"duration": -1.0}, {"warmup": -1.0},
                         {"tracking": "unknown"}, {"seed": -1}, {"mode": "unknown"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                RunTrial(**settings)

    def test_CancelledTrialDoesNotSampleMoveOrPlan(self):
        cancelled = Event()
        cancelled.set()
        with patch("experiments.find_path") as planner:
            trial = RunTrial(cancel_event=cancelled, motion_amplitude=6.0)
        planner.assert_not_called()
        self.assertEqual(trial["status"], "cancelled")
        self.assertEqual(trial["elapsed_time"], 0.0)
        self.assertEqual(trial["travel"], 0.0)
        self.assertEqual(trial["sensor_samples"], 0)
        self.assertIsNone(trial["plan"])
        self.assertFalse(trial["success"])
        np.testing.assert_array_equal(trial["true_target"], [90.0, 0.0, 0.0])
        self.assertEqual(trial["settings"]["motion_amplitude"], 6.0)

    def test_CancellationReachesThePlannerDuringSearch(self):
        cancelled = Event()

        def CancelPlanning(*args, **settings):
            cancelled.set()
            return find_path(*args, **settings)

        with patch("experiments.find_path", side_effect=CancelPlanning) as planner:
            trial = RunTrial(cancel_event=cancelled, warmup=0.0,
                             tool_noise=0.0, target_noise=0.0)
        self.assertIs(planner.call_args.kwargs["cancel_event"], cancelled)
        self.assertEqual(trial["status"], "cancelled")
        self.assertEqual(trial["plan"]["status"], "cancelled")
        self.assertEqual(trial["travel"], 0.0)
        self.assertFalse(trial["reported_arrival"])

    def test_ComparisonCancelledBeforeStartingHasNoInventedFailures(self):
        cancelled = Event()
        cancelled.set()
        progress = []
        comparison = CompareMethods(seeds=[2, 3], cancel_event=cancelled,
                                    progress=progress.append, target_noise=3.0)
        self.assertTrue(comparison["cancelled"])
        self.assertEqual(comparison["trials"], [])
        self.assertEqual(progress, [])
        self.assertEqual(comparison["settings"]["target_noise"], 3.0)
        self.assertEqual(comparison["settings"]["timestep"], 0.02)
        self.assertEqual(len(comparison["summary"]), 3)
        for summary in comparison["summary"].values():
            self.assertEqual(summary["trials"], 0)
            self.assertEqual(summary["successes"], 0)
            self.assertIsNone(summary["success_rate"])
            self.assertEqual(summary["statuses"], {})
            for metric in summary["metrics"].values():
                self.assertEqual(metric["count"], 0)
                self.assertIsNone(metric["mean"])
                self.assertIsNone(metric["sd"])

    def test_CancellationDuringMovementLeavesTheUnfinishedTrialOutOfResults(self):
        cancelled, progress = Event(), []

        def CancelDuringMovement(position, probability, generator):
            if 40.0 < position[0] < 80.0:
                cancelled.set()
            return ApplyDropout(position, probability, generator)

        with patch("experiments.ApplyDropout", side_effect=CancelDuringMovement):
            comparison = CompareMethods(seeds=[2], cancel_event=cancelled,
                                        progress=progress.append, tool_noise=0.0,
                                        target_noise=0.0)
        self.assertTrue(cancelled.is_set())
        self.assertTrue(comparison["cancelled"])
        self.assertEqual(comparison["trials"], [])
        self.assertEqual(progress, [])
        for summary in comparison["summary"].values():
            self.assertEqual(summary["trials"], 0)
            self.assertEqual(summary["statuses"], {})
            self.assertIsNone(summary["success_rate"])

    def test_CancellingAfterProgressRetainsOnlyCompletedTrials(self):
        cancelled, progress = Event(), []

        def StopComparison(update):
            progress.append(update)
            cancelled.set()

        comparison = CompareMethods(seeds=[7, 8], cancel_event=cancelled,
                                    progress=StopComparison, tool_noise=0.0, target_noise=0.0)
        self.assertTrue(comparison["cancelled"])
        self.assertEqual(progress, [(1, 6, "conventional", 7)])
        self.assertEqual(len(comparison["trials"]), 1)
        self.assertTrue(comparison["trials"][0]["success"])
        first = comparison["summary"]["conventional"]
        self.assertEqual(first["trials"], 1)
        self.assertEqual(first["success_rate"], 1.0)
        self.assertEqual(first["statuses"], {"confirmed_arrival": 1})
        self.assertEqual(first["metrics"]["successful_travel"]["count"], 1)
        for mode in ("prox_aware", "uncert_aware"):
            self.assertEqual(comparison["summary"][mode]["trials"], 0)
            self.assertIsNone(comparison["summary"][mode]["success_rate"])

    def test_CompletedComparisonReportsProgressForEveryRetainedTrial(self):
        progress = []
        comparison = CompareMethods("Inaccessible target", seeds=[4], progress=progress.append,
                                    tool_noise=0.0, target_noise=0.0)
        self.assertFalse(comparison["cancelled"])
        self.assertEqual(progress, [(1, 3, "conventional", 4), (2, 3, "prox_aware", 4),
                                    (3, 3, "uncert_aware", 4)])
        self.assertEqual(len(comparison["trials"]), 3)


if __name__ == "__main__":
    unittest.main()
