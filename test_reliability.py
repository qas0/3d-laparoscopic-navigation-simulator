import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from perception import LoadInstrument, PredictInstrument, TrainInstrument
from reliability import EvaluateReliability, ReliabilityMetrics
from safesight import DegradeView


class ReliabilityMetricsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"tp": 8, "fp": 2, "fn": 2, "tn": 88,
             "quality_passed": True, "distance": 0.1, "appearance_limit": 0.2},
            {"tp": 0, "fp": 10, "fn": 0, "tn": 90,
             "quality_passed": True, "distance": 0.3, "appearance_limit": 0.2},
            {"tp": 1, "fp": 0, "fn": 9, "tn": 90,
             "quality_passed": False, "distance": 0.05, "appearance_limit": 0.2},
            {"tp": 0, "fp": 0, "fn": 0, "tn": 100,
             "quality_passed": True, "distance": 0.1, "appearance_limit": 0.2}]

    def test_AlwaysShowsAllMasksAndReportsPooledPixelErrors(self):
        result = ReliabilityMetrics(self.rows, policy="always")
        self.assertEqual((result["frames"], result["accepted"], result["withheld"]), (4, 4, 0))
        self.assertEqual(result["coverage"], 1)
        self.assertEqual([result[key] for key in ("tp", "fp", "fn", "tn")], [9, 12, 11, 368])
        self.assertAlmostEqual(result["precision"], 9 / 21)
        self.assertAlmostEqual(result["recall"], 9 / 20)
        self.assertAlmostEqual(result["iou"], 9 / 32)
        self.assertAlmostEqual(result["missed_tool_fraction"], 11 / 20)
        self.assertAlmostEqual(result["false_positive_fraction"], 12 / 380)

    def test_QualityAndAppearancePoliciesDiscardWholeMasksUsingOnlyTheirGate(self):
        quality = ReliabilityMetrics(self.rows, policy="quality")
        appearance = ReliabilityMetrics(self.rows)
        self.assertEqual((quality["accepted"], quality["withheld"]), (3, 1))
        self.assertEqual([quality[key] for key in ("tp", "fp", "fn", "tn")], [8, 12, 2, 278])
        self.assertEqual((appearance["accepted"], appearance["withheld"]), (2, 2))
        self.assertEqual([appearance[key] for key in ("tp", "fp", "fn", "tn")], [8, 2, 2, 188])
        self.assertEqual(appearance["coverage"], 0.5)
        self.assertEqual((appearance["positive_frames"], appearance["positive_accepted"],
                          appearance["withheld_positive_frames"]), (2, 1, 1))

    def test_FrameBalancedMaskErrorIsNotPixelPooledIntersectionOverUnion(self):
        result = ReliabilityMetrics(self.rows, policy="always")
        self.assertAlmostEqual(result["mean_mask_error"], (4 / 12 + 1 + 9 / 10 + 0) / 4)
        self.assertNotAlmostEqual(result["mean_mask_error"], 1 - result["iou"])
        accepted = ReliabilityMetrics(self.rows)
        self.assertAlmostEqual(accepted["mean_mask_error"], (4 / 12 + 0) / 2)

    def test_AbsentToolFalseDetectionsUseAcceptedNegativeFramesAsDenominator(self):
        always = ReliabilityMetrics(self.rows, policy="always")
        self.assertEqual((always["negative_frames"], always["negative_accepted"],
                          always["negative_false_detections"]), (2, 2, 1))
        self.assertEqual(always["negative_false_detection_fraction"], 0.5)
        accepted = ReliabilityMetrics(self.rows)
        self.assertEqual((accepted["negative_frames"], accepted["negative_accepted"],
                          accepted["negative_false_detections"]), (2, 1, 0))
        self.assertEqual(accepted["negative_false_detection_fraction"], 0)

    def test_AllWithheldIsUnmeasuredRatherThanPerfect(self):
        result = ReliabilityMetrics(self.rows, cutoff=0)
        self.assertEqual((result["accepted"], result["withheld"], result["coverage"]), (0, 4, 0))
        self.assertEqual([result[key] for key in ("tp", "fp", "fn", "tn")], [0, 0, 0, 0])
        for key in ("precision", "recall", "iou", "missed_tool_fraction", "false_positive_fraction",
                    "mean_mask_error", "negative_false_detection_fraction"):
            self.assertIsNone(result[key], key)
        self.assertEqual(result["withheld_positive_frames"], 2)

    def test_AllAbsentCorrectMasksHaveNoInventedRecallOrIntersectionOverUnion(self):
        result = ReliabilityMetrics([self.rows[-1]], policy="always")
        for key in ("precision", "recall", "iou", "missed_tool_fraction"):
            self.assertIsNone(result[key], key)
        self.assertEqual(result["false_positive_fraction"], 0)
        self.assertEqual(result["mean_mask_error"], 0)
        self.assertEqual(result["negative_false_detection_fraction"], 0)
        self.assertEqual(result["positive_frames"], 0)

    def test_EmptyEvaluationHasNoCoverageOrPerformanceEstimate(self):
        result = ReliabilityMetrics([])
        self.assertEqual((result["frames"], result["accepted"], result["withheld"]), (0, 0, 0))
        for key in ("coverage", "precision", "recall", "iou", "mean_mask_error",
                    "missed_tool_fraction", "false_positive_fraction", "negative_false_detection_fraction"):
            self.assertIsNone(result[key], key)

    def test_DescriptiveCutoffIncludesBoundaryAndKeepsQualityCheck(self):
        self.assertEqual(ReliabilityMetrics(self.rows, cutoff=0.1), ReliabilityMetrics(self.rows))
        self.assertEqual(ReliabilityMetrics(self.rows, cutoff=0.35),
                         ReliabilityMetrics(self.rows, policy="quality"))
        self.assertEqual(ReliabilityMetrics(self.rows, cutoff=1)["accepted"], 3)
        self.assertEqual(self.rows[0]["appearance_limit"], 0.2)

    def test_InvalidPoliciesCutoffsAndConfusionCountsFailClearly(self):
        for policy in ("unknown", None, 1):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                ReliabilityMetrics(self.rows, policy=policy)
        for cutoff in (-0.1, 1.1, float("nan"), float("inf"), True, "0.2"):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                ReliabilityMetrics(self.rows, cutoff=cutoff)
        for key, value in (("tp", -1), ("fp", 0.5), ("fn", True), ("tn", float("nan")),
                           ("quality_passed", 1), ("distance", -0.1),
                           ("distance", float("nan")), ("appearance_limit", 1.1)):
            invalid = dict(self.rows[0], **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ReliabilityMetrics([invalid])
        for invalid in (None, [None], [{"tp": 1}], "frames"):
            with self.subTest(rows=invalid), self.assertRaises(ValueError):
                ReliabilityMetrics(invalid)


class ReliabilityEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name) / "captures"
        rows, columns = np.indices((32, 40))
        for source, count in ((cls.folder, 8), (cls.folder / "calibration", 2)):
            for name in ("images", "labels", "degraded"):
                (source / name).mkdir(parents=True)
            for index in range(count):
                grain = ((3 * rows + columns + index) % 9).astype(np.uint8)
                image = np.empty((32, 40, 3), dtype=np.uint8)
                image[:] = [145, 70, 50] if source == cls.folder else [150, 73, 53]
                image += grain[:, :, None]
                mask = np.full((32, 40), 2, dtype=np.uint8)
                left = 5 + index * 2
                mask[5:27, left:left + 6] = 1
                image[mask == 1] = np.array([145, 153, 162]) + grain[mask == 1, None]
                stem = f"frame_{index + 1:06}"
                Image.fromarray(image).save(source / "images" / (stem + ".png"))
                Image.fromarray(DegradeView(image, exposure=-0.5, haze=0.1)).save(
                    source / "degraded" / (stem + ".png"))
                np.savez_compressed(source / "labels" / (stem + ".npz"),
                                    scenario=np.array("reliability fixture" if source == cls.folder
                                                      else "separate calibration fixture"),
                                    camera_configuration=np.array([index * 0.02, 0, 20]),
                                    tool_configuration=np.array([0, index * 0.01, 80 + index]),
                                    breathing_displacement_mm=np.array(float(index % 3 - 1)),
                                    semantic_mask=mask,
                                    class_ids=np.array([0, 1, 2], dtype=np.uint8),
                                    class_names=np.array(["background", "instrument", "tissue"]))
        cls.model, cls.training_report = TrainInstrument(cls.folder)
        cls.heldout_ids = sorted({frame["frame_id"] for frame in cls.training_report["frames"]
                                 if frame["split"] == "validation" and frame["variant"] == "clean"})
        cls.report = EvaluateReliability(cls.folder, cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_EvaluationUsesOnlyHeldOutPhysicalSnapshotsAndOneFixedCaseGrid(self):
        report = self.report
        self.assertEqual(report["heldout_groups"], len(self.model["validation_groups"]))
        self.assertEqual(report["evaluation_groups"], 0)
        self.assertEqual(report["snapshots"], len(self.heldout_ids))
        self.assertEqual(len(report["cases"]), 16)
        case_names = {case["name"] for case in report["cases"]}
        self.assertEqual(len(case_names), 16)
        frames = report["frames"]
        self.assertEqual(len(frames), len(self.heldout_ids) * len(case_names))
        self.assertEqual({frame["frame_id"] for frame in frames}, set(self.heldout_ids))
        self.assertEqual({frame["group_id"] for frame in frames}, set(self.model["validation_groups"]))
        self.assertEqual({frame["source"] for frame in frames}, {"heldout"})
        for frame_id in self.heldout_ids:
            selected = [frame for frame in frames if frame["frame_id"] == frame_id]
            self.assertEqual({frame["case"] for frame in selected}, case_names)
            self.assertEqual(len({frame["tp"] + frame["fn"] for frame in selected}), 1)
            self.assertEqual(len({frame["fp"] + frame["tn"] for frame in selected}), 1)
        fitted = set(self.model["train_groups"]) | set(self.model["calibration_groups"])
        self.assertFalse(fitted & {frame["group_id"] for frame in frames})

    def test_ThreePoliciesAndCaseSummariesReuseTheSameRecordedPredictions(self):
        for policy in ("always", "quality", "appearance"):
            self.assertEqual(self.report["policies"][policy],
                             ReliabilityMetrics(self.report["frames"], policy=policy))
        for case in self.report["by_case"]:
            rows = [frame for frame in self.report["frames"] if frame["case"] == case["name"]]
            for policy in ("always", "quality", "appearance"):
                self.assertEqual(case["policies"][policy], ReliabilityMetrics(rows, policy=policy))
        self.assertEqual(self.report["policies"]["always"]["coverage"], 1)
        self.assertLessEqual(self.report["policies"]["appearance"]["accepted"],
                             self.report["policies"]["quality"]["accepted"])

    def test_CutoffCurveIsDescriptiveAndContainsTheUnchangedCurrentGate(self):
        curve = self.report["curve"]
        self.assertTrue(curve)
        self.assertEqual(sum(row["current"] for row in curve), 1)
        current = next(row for row in curve if row["current"])
        self.assertEqual(current["cutoff"], float(self.model["appearance_limit"]))
        for key, value in self.report["policies"]["appearance"].items():
            self.assertEqual(current[key], value)
        for row in curve:
            expected = ReliabilityMetrics(self.report["frames"], cutoff=row["cutoff"])
            self.assertTrue(all(row[key] == value for key, value in expected.items()))
        ordered = sorted(curve, key=lambda row: row["cutoff"])
        self.assertEqual([row["accepted"] for row in ordered],
                         sorted(row["accepted"] for row in ordered))

    def test_EvaluationIsDeterministicDoesNotWriteFilesOrMutateTheModel(self):
        before_files = {str(path.relative_to(self.folder)): path.read_bytes()
                        for path in self.folder.rglob("*") if path.is_file()}
        before_model = {key: value.copy() for key, value in self.model.items()}
        result = EvaluateReliability(self.folder, self.model)
        self.assertEqual(result, self.report)
        after_files = {str(path.relative_to(self.folder)): path.read_bytes()
                       for path in self.folder.rglob("*") if path.is_file()}
        self.assertEqual(before_files, after_files)
        for key in self.model:
            np.testing.assert_array_equal(self.model[key], before_model[key])
        json.dumps(result, allow_nan=False)

    def test_RawPredictionRunsOncePerCaseAndDoesNotUseCalibratedProbabilities(self):
        with patch("reliability.PredictInstrument", wraps=PredictInstrument) as predicted:
            report = EvaluateReliability(self.folder, self.model)
        cases = [call for call in predicted.call_args_list if call.args[0].shape[:2] != (16, 16)]
        self.assertEqual(len(cases), len(report["frames"]))
        self.assertTrue(all(not call.kwargs.get("calibrated", False)
                            for call in predicted.call_args_list))
        self.assertEqual(report, self.report)

    def test_GroundTruthChangesErrorsWithoutChangingGateOrPredictedPixelCounts(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            for frame_id in self.heldout_ids:
                label_path = folder / "labels" / (frame_id + ".npz")
                with np.load(label_path, allow_pickle=False) as source:
                    labels = {key: source[key].copy() for key in source.files}
                labels["semantic_mask"][:] = 2
                np.savez_compressed(label_path, **labels)
            report = EvaluateReliability(folder, self.model)
        self.assertEqual(len(report["frames"]), len(self.report["frames"]))
        for original, changed in zip(self.report["frames"], report["frames"]):
            for key in ("frame_id", "group_id", "source", "case", "quality_passed", "distance", "reasons"):
                self.assertEqual(original[key], changed[key], key)
            self.assertEqual(original["tp"] + original["fp"], changed["tp"] + changed["fp"])
            self.assertEqual(changed["tp"] + changed["fn"], 0)
        self.assertEqual(report["policies"]["always"]["positive_frames"], 0)
        self.assertEqual(report["policies"]["always"]["negative_frames"], len(report["frames"]))

    def test_MissingModelProvenanceCapturesAndInvalidLabelsAreRejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            fitted = next(frame for frame in self.training_report["frames"]
                          if frame["split"] == "train" and frame["variant"] == "clean")
            missing = folder / "images" / (fitted["frame_id"] + ".png")
            original = missing.read_bytes()
            missing.unlink()
            with self.assertRaises(ValueError):
                EvaluateReliability(folder, self.model)
            missing.write_bytes(original)
            label_path = folder / "labels" / (self.heldout_ids[0] + ".npz")
            with np.load(label_path, allow_pickle=False) as source:
                labels = {key: source[key].copy() for key in source.files}
            labels["semantic_mask"] = labels["semantic_mask"].astype(float)
            np.savez_compressed(label_path, **labels)
            with self.assertRaises(ValueError):
                EvaluateReliability(folder, self.model)

    def test_FreshViewpointsAndAbsentToolCapturesStaySeparateFromTheOriginalHeldOutSet(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            evaluation = folder / "evaluation"
            for name in ("images", "labels"):
                (evaluation / name).mkdir(parents=True)
            for index, frame_id in enumerate(self.heldout_ids):
                with np.load(folder / "labels" / (frame_id + ".npz"), allow_pickle=False) as source:
                    labels = {key: source[key].copy() for key in source.files}
                labels["scenario"] = np.array("fresh evaluation viewpoint")
                labels["camera_configuration"][0] += 0.75
                with Image.open(folder / "images" / (frame_id + ".png")) as source:
                    image = np.asarray(source).copy()
                image[0, 0] = [1, 3, 5]
                if index == 0:
                    image[labels["semantic_mask"] == 1] = [160, 75, 55]
                    labels["semantic_mask"][:] = 2
                np.savez_compressed(evaluation / "labels" / (frame_id + ".npz"), **labels)
                Image.fromarray(image).save(evaluation / "images" / (frame_id + ".png"))
            report = EvaluateReliability(folder, self.model)
        self.assertEqual(report["heldout_groups"], len(self.model["validation_groups"]))
        self.assertEqual(report["evaluation_groups"], len(self.heldout_ids))
        fresh = [frame for frame in report["frames"] if frame["source"] == "evaluation"]
        original = [frame for frame in report["frames"] if frame["source"] == "heldout"]
        self.assertEqual(original, self.report["frames"])
        self.assertEqual(len(fresh), len(self.heldout_ids) * len(report["cases"]))
        self.assertTrue(all(frame["frame_id"].startswith("evaluation/") for frame in fresh))
        self.assertEqual(sum(frame["tp"] + frame["fn"] == 0 for frame in fresh), len(report["cases"]))
        model_groups = (set(self.model["train_groups"]) | set(self.model["calibration_groups"])
                        | set(self.model["validation_groups"]))
        self.assertFalse({frame["group_id"] for frame in fresh} & model_groups)

    def test_FreshEvaluationRejectsPhysicalOrRgbAliasesOfAnyModelSplit(self):
        fitted = next(frame for frame in self.training_report["frames"]
                      if frame["split"] == "train" and frame["variant"] == "clean")
        calibrated = next(frame for frame in self.training_report["frames"]
                          if frame["split"] == "calibration" and frame["variant"] == "clean")
        examples = [(fitted["frame_id"], "images", True),
                    (self.heldout_ids[0], "images", True),
                    (calibrated["frame_id"], "images", True),
                    (fitted["frame_id"], "images", False),
                    (calibrated["frame_id"], "images", False),
                    (fitted["frame_id"], "degraded", False)]
        for frame_id, variant, physical_alias in examples:
            with self.subTest(frame=frame_id, variant=variant, physical_alias=physical_alias), \
                    tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary) / "captures"
                shutil.copytree(self.folder, folder)
                evaluation = folder / "evaluation"
                for name in ("images", "labels"):
                    (evaluation / name).mkdir(parents=True)
                relative = Path(frame_id)
                capture_folder = folder / relative.parent
                with np.load(capture_folder / "labels" / (relative.name + ".npz"),
                             allow_pickle=False) as source:
                    labels = {key: source[key].copy() for key in source.files}
                with Image.open(capture_folder / variant / (relative.name + ".png")) as source:
                    image = np.asarray(source).copy()
                if physical_alias:
                    image[0, 0] = [1, 2, 3]
                else:
                    labels["scenario"] = np.array("fresh but duplicated RGB")
                    labels["camera_configuration"][0] += 0.75
                np.savez_compressed(evaluation / "labels" / "frame_000001.npz", **labels)
                Image.fromarray(image).save(evaluation / "images" / "frame_000001.png")
                with self.assertRaises(ValueError):
                    EvaluateReliability(folder, self.model)

    def test_RecapturesAndDifferentMetadataRgbAliasesDoNotMultiplyHeldOutSnapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            frame_id = self.heldout_ids[0]
            with np.load(folder / "labels" / (frame_id + ".npz"), allow_pickle=False) as source:
                labels = {key: source[key].copy() for key in source.files}
            for name, alias in (("frame_000100", False), ("frame_000101", True)):
                if alias:
                    labels["camera_configuration"][0] += 0.75
                np.savez_compressed(folder / "labels" / (name + ".npz"), **labels)
                shutil.copyfile(folder / "images" / (frame_id + ".png"),
                                folder / "images" / (name + ".png"))
            report = EvaluateReliability(folder, self.model)
        self.assertEqual(report, self.report)

    def test_TransitiveFreshPhysicalAndRgbAliasesCountAsOneSnapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            evaluation = folder / "evaluation"
            for name in ("images", "labels"):
                (evaluation / name).mkdir(parents=True)
            frame_id = self.heldout_ids[0]
            with np.load(folder / "labels" / (frame_id + ".npz"), allow_pickle=False) as source:
                original = {key: source[key].copy() for key in source.files}
            with Image.open(folder / "images" / (frame_id + ".png")) as source:
                image = np.asarray(source).copy()
            for index, (position, colour) in enumerate(((0.75, [1, 3, 5]),
                                                       (1.5, [2, 4, 6]), (0.75, [2, 4, 6]))):
                labels = {key: value.copy() for key, value in original.items()}
                labels["scenario"] = np.array("fresh evaluation aliases")
                labels["camera_configuration"][0] += position
                image[0, 0] = colour
                stem = f"frame_{index + 1:06}"
                np.savez_compressed(evaluation / "labels" / (stem + ".npz"), **labels)
                Image.fromarray(image).save(evaluation / "images" / (stem + ".png"))
            report = EvaluateReliability(folder, self.model)
        self.assertEqual(report["evaluation_groups"], 1)
        fresh = [frame for frame in report["frames"] if frame["source"] == "evaluation"]
        self.assertEqual(len(fresh), len(report["cases"]))
        self.assertEqual(len({frame["group_id"] for frame in fresh}), 1)

    def test_OptionalOcclusionLabelsDescribeSubgroupsWithoutAffectingPredictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            label_path = folder / "labels" / (self.heldout_ids[0] + ".npz")
            with np.load(label_path, allow_pickle=False) as source:
                labels = {key: source[key].copy() for key in source.files}
            labels["tool_projected_pixels"] = np.array(int(np.count_nonzero(labels["semantic_mask"] == 1)) + 100)
            labels["tool_occluded_pixels"] = np.array(100)
            np.savez_compressed(label_path, **labels)
            report = EvaluateReliability(folder, self.model)
        self.assertEqual(report["policies"], self.report["policies"])
        occluded = [frame for frame in report["frames"] if frame["visibility"] == "occluded"]
        self.assertEqual(len(occluded), len(report["cases"]))
        self.assertEqual(report["by_visibility"]["occluded"]["always"]["frames"], len(occluded))
        for before, after in zip(self.report["frames"], report["frames"]):
            for key in ("tp", "fp", "fn", "tn", "quality_passed", "distance", "reasons"):
                self.assertEqual(before[key], after[key])

    def test_VersionTwoAppearanceModelsWorkAndVersionOneFailsWithoutAPretendGate(self):
        model = {key: value.copy() for key, value in self.model.items()
                 if key not in {"calibration_temperature", "calibration_bias", "calibration_groups"}}
        report = json.loads(str(model["report_json"]))
        report["frames"] = [frame for frame in report["frames"] if frame["split"] != "calibration"]
        model["version"] = np.array(2)
        model["report_json"] = np.array(json.dumps(report))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "version_two.npz"
            np.savez_compressed(path, **model)
            legacy = LoadInstrument(path)
            result = EvaluateReliability(self.folder, legacy)
        self.assertEqual(result["policies"], self.report["policies"])
        self.assertEqual(result["calibration_groups"], 0)
        legacy = {key: value.copy() for key, value in legacy.items()
                  if key not in {"appearance_references", "appearance_groups", "appearance_limit"}}
        legacy["version"] = np.array(1)
        with self.assertRaises(ValueError):
            EvaluateReliability(self.folder, legacy)

    def test_RgbReplacementCannotMergeTrainingAndHeldOutGroups(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            fitted = next(frame for frame in self.training_report["frames"]
                          if frame["split"] == "train" and frame["variant"] == "clean")
            shutil.copyfile(folder / "images" / (self.heldout_ids[0] + ".png"),
                            folder / "images" / (fitted["frame_id"] + ".png"))
            with self.assertRaises(ValueError):
                EvaluateReliability(folder, self.model)

    def test_UniformDegradedRgbWithinTheSameTrainingSplitDoesNotCreateFalseLeakage(self):
        fitted = sorted({frame["frame_id"] for frame in self.training_report["frames"]
                         if frame["split"] == "train" and frame["variant"] == "degraded"})[:2]
        self.assertEqual(len(fitted), 2)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            uniform = np.empty((32, 40, 3), dtype=np.uint8)
            uniform[:] = [185, 190, 195]
            for frame_id in fitted:
                Image.fromarray(uniform).save(folder / "degraded" / (frame_id + ".png"))
            report = EvaluateReliability(folder, self.model)
        self.assertEqual(report, self.report)

    def test_UniformDegradedRgbSharedBetweenTrainingAndHeldOutSplitsIsRejected(self):
        fitted = next(frame["frame_id"] for frame in self.training_report["frames"]
                      if frame["split"] == "train" and frame["variant"] == "degraded")
        heldout = next(frame["frame_id"] for frame in self.training_report["frames"]
                       if frame["split"] == "validation" and frame["variant"] == "degraded")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            uniform = np.empty((32, 40, 3), dtype=np.uint8)
            uniform[:] = [185, 190, 195]
            for frame_id in (fitted, heldout):
                Image.fromarray(uniform).save(folder / "degraded" / (frame_id + ".png"))
            with self.assertRaises(ValueError):
                EvaluateReliability(folder, self.model)

    def test_MatchingFilenamesAndRgbDoNotReplaceMissingPhysicalModelAnchors(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "unrelated captures"
            shutil.copytree(self.folder, folder)
            for label_path in (folder / "labels").glob("frame_*.npz"):
                with np.load(label_path, allow_pickle=False) as source:
                    labels = {key: source[key].copy() for key in source.files}
                labels["scenario"] = np.array("unrelated physical capture")
                labels["camera_configuration"][0] += 0.75
                np.savez_compressed(label_path, **labels)
            with self.assertRaises(ValueError):
                EvaluateReliability(folder, self.model)

    def test_SwappedCompleteCapturesWithinOneSplitStillFailRecordedFrameProvenance(self):
        fitted = sorted({frame["frame_id"] for frame in self.training_report["frames"]
                         if frame["split"] == "train" and frame["variant"] == "clean"})[:2]
        self.assertEqual(len(fitted), 2)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            for name, suffix in (("images", ".png"), ("degraded", ".png"), ("labels", ".npz")):
                first, second = [folder / name / (frame_id + suffix) for frame_id in fitted]
                original = first.read_bytes()
                first.write_bytes(second.read_bytes())
                second.write_bytes(original)
            with self.assertRaises(ValueError):
                EvaluateReliability(folder, self.model)

    def test_RecordedDegradedAliasCanReferToItsCanonicalCleanRgbPhysicalGroup(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            original_id, alias_id = "frame_000001", "frame_000100"
            with np.load(folder / "labels" / (original_id + ".npz"), allow_pickle=False) as source:
                labels = {key: source[key].copy() for key in source.files}
            labels["scenario"] = np.array("legitimate recorded RGB alias")
            labels["camera_configuration"][0] += 0.75
            np.savez_compressed(folder / "labels" / (alias_id + ".npz"), **labels)
            shutil.copyfile(folder / "images" / (original_id + ".png"),
                            folder / "images" / (alias_id + ".png"))
            with Image.open(folder / "degraded" / (original_id + ".png")) as source:
                degraded = np.asarray(source).copy()
            degraded[0, 0] = [1, 3, 5]
            Image.fromarray(degraded).save(folder / "degraded" / (alias_id + ".png"))
            model, trained = TrainInstrument(folder)
            original = next(frame for frame in trained["frames"]
                            if frame["frame_id"] == original_id and frame["variant"] == "clean")
            recorded_alias = next(frame for frame in trained["frames"] if frame["frame_id"] == alias_id)
            self.assertEqual(recorded_alias["variant"], "degraded")
            self.assertEqual(recorded_alias["group_id"], original["group_id"])
            report = EvaluateReliability(folder, model)
        self.assertEqual(report["evaluation_groups"], 0)
        self.assertEqual(report["heldout_groups"], len(model["validation_groups"]))
        self.assertEqual(report["snapshots"], len(model["validation_groups"]))

    def test_TransitiveUnreportedAliasesCannotBridgeTrainingAndHeldOutGroups(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            fitted = next(frame for frame in self.training_report["frames"]
                          if frame["split"] == "train" and frame["variant"] == "clean")
            with np.load(folder / "labels" / (fitted["frame_id"] + ".npz"),
                         allow_pickle=False) as source:
                bridge = {key: source[key].copy() for key in source.files}
            bridge["scenario"] = np.array("unreported alias bridge")
            bridge["camera_configuration"][0] += 0.75
            with np.load(folder / "labels" / (self.heldout_ids[0] + ".npz"),
                         allow_pickle=False) as source:
                heldout = {key: source[key].copy() for key in source.files}
            with Image.open(folder / "images" / (fitted["frame_id"] + ".png")) as source:
                original = np.asarray(source).copy()
            new_rgb = original.copy()
            new_rgb[0, 0] = [1, 3, 5]
            for index, (labels, image) in enumerate(((bridge, original), (bridge, new_rgb), (heldout, new_rgb))):
                stem = f"frame_{index + 100:06}"
                np.savez_compressed(folder / "labels" / (stem + ".npz"), **labels)
                Image.fromarray(image).save(folder / "images" / (stem + ".png"))
            with self.assertRaises(ValueError):
                EvaluateReliability(folder, self.model)

    def test_CancellationBeforeLoadingAndBetweenCasesStopsWithoutWriting(self):
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(InterruptedError):
            EvaluateReliability(self.folder, self.model, cancel=cancelled)
        cancelled.clear()

        def stop_after_prediction(*args, **kwargs):
            result = PredictInstrument(*args, **kwargs)
            if args[0].shape[:2] != (16, 16):
                cancelled.set()
            return result

        before_files = {str(path.relative_to(self.folder)): path.read_bytes()
                        for path in self.folder.rglob("*") if path.is_file()}
        with patch("reliability.PredictInstrument", side_effect=stop_after_prediction) as predicted:
            with self.assertRaises(InterruptedError):
                EvaluateReliability(self.folder, self.model, cancel=cancelled)
        self.assertEqual(predicted.call_count, 2)
        after_files = {str(path.relative_to(self.folder)): path.read_bytes()
                       for path in self.folder.rglob("*") if path.is_file()}
        self.assertEqual(before_files, after_files)

    def test_CancellationOnTheLastCaseDoesNotPublishAnApparentlyCompleteReport(self):
        cancelled = threading.Event()
        count = 0

        def stop_on_last_prediction(*args, **kwargs):
            nonlocal count
            result = PredictInstrument(*args, **kwargs)
            if args[0].shape[:2] != (16, 16):
                count += 1
                if count == len(self.report["frames"]):
                    cancelled.set()
            return result

        with patch("reliability.PredictInstrument", side_effect=stop_on_last_prediction):
            with self.assertRaises(InterruptedError):
                EvaluateReliability(self.folder, self.model, cancel=cancelled)
        self.assertEqual(count, len(self.report["frames"]))


if __name__ == "__main__":
    unittest.main()
