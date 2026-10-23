import threading
import unittest
from unittest.mock import Mock

import numpy as np

from calibration import CalibrationMetrics, FitCalibration


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        true_logits = rng.normal(0, 1.5, 10000)
        true_probabilities = 1 / (1 + np.exp(-true_logits))
        cls.labels = (rng.random(len(true_logits)) < true_probabilities).astype(float)
        cls.logits = 2 * true_logits + 1.5
        cls.fitted = FitCalibration(cls.logits, cls.labels)

    def test_FitCorrectsKnownSyntheticMiscalibration(self):
        raw = 1 / (1 + np.exp(-self.logits))
        calibrated = 1 / (1 + np.exp(-(
            self.logits / self.fitted["temperature"] + self.fitted["bias"])))
        before = CalibrationMetrics(raw, self.labels)
        after = CalibrationMetrics(calibrated, self.labels)
        self.assertAlmostEqual(self.fitted["temperature"], 2, delta=0.2)
        self.assertAlmostEqual(self.fitted["bias"], -0.75, delta=0.12)
        self.assertLess(after["nll"], before["nll"] - 0.05)
        self.assertLess(after["brier"], before["brier"])

    def test_PositiveTemperaturePreservesOrderingAndTransformedDecisions(self):
        temperature, bias = self.fitted["temperature"], self.fitted["bias"]
        logits = np.linspace(-8, 8, 101)
        probabilities = 1 / (1 + np.exp(-(logits / temperature + bias)))
        self.assertTrue(np.all(np.diff(probabilities) >= 0))
        self.assertTrue(0.1 <= temperature <= 20)
        self.assertTrue(-20 <= bias <= 20)
        threshold = np.log(0.7 / 0.3)
        transformed = 1 / (1 + np.exp(-(threshold / temperature + bias)))
        np.testing.assert_array_equal(probabilities >= transformed, logits >= threshold)

    def test_WeightedFitRestoresNaturalClassPrevalence(self):
        fitted = FitCalibration([0, 0], [0, 1], weights=[99, 1])
        probability = 1 / (1 + np.exp(-fitted["bias"]))
        self.assertAlmostEqual(probability, 0.01, delta=0.0001)
        self.assertAlmostEqual(fitted["temperature"], 1, places=10)
        unweighted = FitCalibration([0, 0], [0, 1])
        self.assertEqual(unweighted, {"temperature": 1.0, "bias": 0.0})

    def test_BiasStillFitsTheClassPriorAfterTemperatureReachesItsBound(self):
        for logit in [5.0, 35.0, -35.0]:
            for prevalence in [0.01, 0.5, 0.99]:
                with self.subTest(logit=logit, prevalence=prevalence):
                    fitted = FitCalibration([logit, logit], [0, 1],
                                            weights=[1 - prevalence, prevalence])
                    probability = 1 / (1 + np.exp(-(
                        logit / fitted["temperature"] + fitted["bias"])))
                    self.assertAlmostEqual(probability, prevalence, delta=0.0001)
                    self.assertTrue(0.1 <= fitted["temperature"] <= 20)

    def test_RescalingWeightsDoesNotChangeTheFit(self):
        weights = np.linspace(0.25, 2, len(self.logits))
        first = FitCalibration(self.logits, self.labels, weights)
        second = FitCalibration(self.logits, self.labels, weights * 17)
        self.assertAlmostEqual(first["temperature"], second["temperature"], places=8)
        self.assertAlmostEqual(first["bias"], second["bias"], places=8)

    def test_FitNeverWorsensItsCalibrationSetLogLoss(self):
        for logits, labels in [([-3, -1, 1, 3], [0, 0, 1, 1]),
                               ([-3, -1, 1, 3], [1, 1, 0, 0]),
                               ([0, 0, 0, 0], [0, 1, 0, 1])]:
            with self.subTest(labels=labels):
                logits = np.asarray(logits, dtype=float)
                fitted = FitCalibration(logits, labels)
                raw = 1 / (1 + np.exp(-logits))
                calibrated = 1 / (1 + np.exp(-(
                    logits / fitted["temperature"] + fitted["bias"])))
                self.assertLessEqual(CalibrationMetrics(calibrated, labels)["nll"],
                                     CalibrationMetrics(raw, labels)["nll"] + 1e-10)

    def test_AnalyticBinaryMetricsAndBinsMatchHandCalculations(self):
        metrics = CalibrationMetrics([0.1, 0.3, 0.7, 0.9], [0, 1, 0, 1], bins=2)
        self.assertEqual(metrics["pixels"], 4)
        self.assertEqual(metrics["positives"], 2)
        self.assertEqual(metrics["weight"], 4)
        self.assertEqual(metrics["positive_weight"], 2)
        self.assertEqual(metrics["prevalence"], 0.5)
        self.assertAlmostEqual(metrics["brier"], 0.25)
        self.assertAlmostEqual(metrics["nll"], -0.5 * np.log(0.27))
        self.assertAlmostEqual(metrics["ece"], 0.3)
        self.assertEqual(metrics["bin_counts"], [2, 2])
        self.assertEqual(metrics["bin_positive_counts"], [1, 1])
        np.testing.assert_allclose(metrics["bin_mean_probability"], [0.2, 0.8])
        self.assertEqual(metrics["bin_positive_fraction"], [0.5, 0.5])
        self.assertAlmostEqual(metrics["positive_brier"], 0.25)
        self.assertAlmostEqual(metrics["negative_brier"], 0.25)

    def test_WeightedBinsAndMetricsUsePixelPopulationWeights(self):
        metrics = CalibrationMetrics([0.2, 0.8], [0, 1], weights=[3, 1], bins=2)
        self.assertEqual(metrics["pixels"], 2)
        self.assertEqual(metrics["positives"], 1)
        self.assertEqual(metrics["weight"], 4)
        self.assertEqual(metrics["prevalence"], 0.25)
        self.assertEqual(metrics["bin_counts"], [3.0, 1.0])
        self.assertEqual(metrics["bin_positive_counts"], [0.0, 1.0])
        self.assertAlmostEqual(metrics["brier"], 0.04)
        self.assertAlmostEqual(metrics["nll"], -np.log(0.8))
        self.assertAlmostEqual(metrics["ece"], 0.2)
        scaled = CalibrationMetrics([0.2, 0.8], [0, 1], weights=[51, 17], bins=2)
        for key in ["prevalence", "brier", "nll", "ece", "positive_brier", "negative_brier"]:
            self.assertAlmostEqual(metrics[key], scaled[key])

    def test_TinyWeightsDoNotChangeBinMeansOrCalibrationError(self):
        ordinary = CalibrationMetrics([0.2, 0.8], [0, 1], weights=[1, 1], bins=2)
        tiny = CalibrationMetrics([0.2, 0.8], [0, 1], weights=[1e-323, 1e-323], bins=2)
        np.testing.assert_allclose(tiny["bin_mean_probability"], ordinary["bin_mean_probability"])
        np.testing.assert_allclose(tiny["bin_positive_fraction"], ordinary["bin_positive_fraction"])
        self.assertAlmostEqual(tiny["ece"], ordinary["ece"])

    def test_EmptyBinsStayUndefinedAndProbabilityOneUsesTheLastBin(self):
        metrics = CalibrationMetrics([0, 1], [0, 1], bins=4)
        self.assertEqual(metrics["bin_counts"], [1, 0, 0, 1])
        self.assertEqual(metrics["bin_positive_counts"], [0, 0, 0, 1])
        self.assertEqual(metrics["bin_mean_probability"], [0.0, None, None, 1.0])
        self.assertEqual(metrics["bin_positive_fraction"], [0.0, None, None, 1.0])
        self.assertEqual(metrics["brier"], 0)
        self.assertEqual(metrics["ece"], 0)
        self.assertLess(metrics["nll"], 1e-10)

    def test_AbsentWeightedClassesHaveUndefinedConditionalMetrics(self):
        metrics = CalibrationMetrics([0.1, 0.2], [0, 0])
        self.assertIsNone(metrics["positive_brier"])
        self.assertIsNone(metrics["positive_nll"])
        self.assertAlmostEqual(metrics["negative_brier"], 0.025)
        zero_weight = CalibrationMetrics([0.1, 0.9], [0, 1], weights=[1, 0])
        self.assertEqual(zero_weight["positives"], 1)
        self.assertEqual(zero_weight["positive_weight"], 0)
        self.assertEqual(zero_weight["prevalence"], 0)
        self.assertIsNone(zero_weight["positive_brier"])

    def test_ReadonlyInputsAreUnchangedAndRepeatedFitsAreDeterministic(self):
        logits = self.logits.copy()
        labels = self.labels.copy()
        weights = np.ones_like(logits)
        for value in (logits, labels, weights):
            value.flags.writeable = False
        first = FitCalibration(logits, labels, weights)
        second = FitCalibration(logits, labels, weights)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(logits, self.logits)
        np.testing.assert_array_equal(labels, self.labels)
        np.testing.assert_array_equal(weights, 1)

    def test_FitRejectsEmptyNonfiniteAndSingleWeightedClassData(self):
        cases = [([], [], None), ([0, np.nan], [0, 1], None),
                 ([0, np.inf], [0, 1], None), ([[0, 1]], [[0, 1]], None),
                 ([0, 1], [0], None), ([0, 1], [0, 2], None),
                 ([0, 1], [0, 0], None), ([0, 1], [0, 1], [1, 0]),
                 ([0, 1], [0, 1], [1, -1]), ([0, 1], [0, 1], [1, np.inf])]
        for logits, labels, weights in cases:
            with self.subTest(logits=logits, labels=labels, weights=weights), self.assertRaises(ValueError):
                FitCalibration(logits, labels, weights)

    def test_MetricsRejectEmptyNonfiniteAndInvalidProbabilityOrWeightInputs(self):
        cases = [([], [], None), ([0.1, np.nan], [0, 1], None),
                 ([-0.1, 0.5], [0, 1], None), ([0.1, 1.1], [0, 1], None),
                 ([[0.1, 0.9]], [[0, 1]], None), ([0.1, 0.9], [0, 2], None),
                 ([0.1, 0.9], [0], None), ([0.1, 0.9], [0, 1], [0, 0]),
                 ([0.1, 0.9], [0, 1], [1, -1]), ([0.1, 0.9], [0, 1], [1, np.inf])]
        for probabilities, labels, weights in cases:
            with self.subTest(probabilities=probabilities, weights=weights), self.assertRaises(ValueError):
                CalibrationMetrics(probabilities, labels, weights)
        for bins in [0, -1, 101, True, 2.0, [10]]:
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                CalibrationMetrics([0.1, 0.9], [0, 1], bins=bins)

    def test_CancellationWorksBeforeAndDuringOptimization(self):
        stopped = threading.Event()
        stopped.set()
        during = Mock()
        during.is_set.side_effect = [False, False, True]
        for cancel in (stopped, during):
            with self.subTest(cancel=cancel), self.assertRaises(InterruptedError):
                FitCalibration(self.logits, self.labels, cancel=cancel)


if __name__ == "__main__":
    unittest.main()
