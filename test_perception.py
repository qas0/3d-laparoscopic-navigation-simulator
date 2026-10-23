import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from capture import SaveCapture
from perception import (AppearanceFeatures, AssessPrediction, EvaluateInstrument,
                        FitAppearance, ImageFeatures, LoadInstrument,
                        PredictInstrument, TrainInstrument)
from safesight import AssessView, DegradeView


class InstrumentPerceptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name) / "captures"
        for name in ("images", "labels", "degraded"):
            (cls.folder / name).mkdir(parents=True)
        rows, columns = np.indices((48, 64))
        for index in range(8):
            grain = ((rows + columns + index) % 5).astype(np.uint8)
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:] = [155, 75, 60]
            image += grain[:, :, None]
            mask = np.full((48, 64), 2, dtype=np.uint8)
            left = 10 + (index % 4) * 5
            mask[8:40, left:left + 8] = 1
            image[mask == 1] = np.array([150, 157, 165]) + grain[mask == 1, None]
            name = f"frame_{index + 1:06}"
            Image.fromarray(image).save(cls.folder / "images" / (name + ".png"))
            Image.fromarray(DegradeView(image, exposure=-0.5, haze=0.1)).save(
                cls.folder / "degraded" / (name + ".png"))
            np.savez_compressed(cls.folder / "labels" / (name + ".npz"),
                                scenario=np.array("fixture"),
                                camera_configuration=np.array([0.01 * index, 0, 20]),
                                tool_configuration=np.array([0, 0.01 * index, 80 + index]),
                                breathing_displacement_mm=np.array(float(index % 3 - 1)),
                                semantic_mask=mask, class_ids=np.array([0, 1, 2], dtype=np.uint8),
                                class_names=np.array(["background", "instrument", "tissue"]))
        cls.output = Path(cls.temporary.name) / "instrument.npz"
        cls.model, cls.report = TrainInstrument(cls.folder, cls.output)
        cls.designated = Path(cls.temporary.name) / "designated"
        shutil.copytree(cls.folder, cls.designated)
        for name in ("images", "labels", "degraded"):
            (cls.designated / "calibration" / name).mkdir(parents=True)
        for index in range(2):
            stem = f"frame_{index + 1:06}"
            with np.load(cls.folder / "labels" / (stem + ".npz")) as saved:
                labels = {key: saved[key].copy() for key in saved.files}
            labels["scenario"] = np.array("designated calibration fixture")
            labels["camera_configuration"][0] += 0.25
            np.savez_compressed(cls.designated / "calibration" / "labels" / (stem + ".npz"), **labels)
            for variant in ("images", "degraded"):
                with Image.open(cls.folder / variant / (stem + ".png")) as source:
                    image = np.asarray(source).copy()
                image += np.array([5, 3, 2], dtype=np.uint8)
                Image.fromarray(image).save(cls.designated / "calibration" / variant / (stem + ".png"))
        cls.external_output = Path(cls.temporary.name) / "external.npz"
        cls.external_model, cls.external_report = TrainInstrument(cls.designated, cls.external_output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_FeaturesDependOnColourRatherThanPixelLocation(self):
        image = np.zeros((16, 32, 3), dtype=np.uint8)
        image[:, :16] = [150, 70, 40]
        image[:, 16:] = [90, 100, 110]
        original = image.copy()
        features = ImageFeatures(image).reshape(16, 32, 12)
        shifted = ImageFeatures(np.roll(image, 16, axis=1)).reshape(16, 32, 12)
        np.testing.assert_array_equal(shifted, np.roll(features, 16, axis=1))
        np.testing.assert_array_equal(image, original)
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(features)))
        np.testing.assert_array_equal(ImageFeatures(np.zeros_like(image)), 0)

    def test_SelectedFeaturesExactlyMatchFullFeaturesIncludingRepeatedAndEmptyPositions(self):
        image = np.random.default_rng(12).integers(0, 256, (35, 47, 3), dtype=np.uint8)
        full = ImageFeatures(image)
        for positions in (np.array([], dtype=int), np.array([1644, 0, 31, 31, 58]),
                          np.arange(image.shape[0] * image.shape[1])):
            with self.subTest(size=len(positions)):
                np.testing.assert_array_equal(ImageFeatures(image, positions=positions), full[positions])
        for positions in ([True], [-1], [1645], [0.5], [[1]], "invalid"):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                ImageFeatures(image, positions=positions)

    def test_ReusedLogitsAndScoresPreservePublicPredictionAndEvaluationResults(self):
        with Image.open(self.folder / "images" / "frame_000001.png") as source:
            image = np.asarray(source)
        with np.load(self.folder / "labels" / "frame_000001.npz") as labels:
            mask = labels["semantic_mask"]
        scores, logits = PredictInstrument(image, self.model, return_logits=True)
        np.testing.assert_array_equal(scores, PredictInstrument(image, self.model))
        np.testing.assert_array_equal(scores, 1 / (1 + np.exp(-logits)))
        calibrated, same_logits = PredictInstrument(image, self.model, calibrated=True, return_logits=True)
        np.testing.assert_array_equal(same_logits, logits)
        expected = 1 / (1 + np.exp(-np.clip(
            logits / float(self.model["calibration_temperature"]) + float(self.model["calibration_bias"]), -35, 35)))
        np.testing.assert_array_equal(calibrated, expected)
        with patch("perception.PredictInstrument", side_effect=AssertionError("recomputed prediction")):
            reused = EvaluateInstrument(image, mask, self.model, scores=scores)
        self.assertEqual(reused, EvaluateInstrument(image, mask, self.model))
        for invalid in (None, "yes", 1):
            with self.subTest(return_logits=invalid), self.assertRaises(ValueError):
                PredictInstrument(image, self.model, return_logits=invalid)
        for invalid in (scores[:15], scores.astype(int), np.full_like(scores, np.nan),
                        np.full_like(scores, -0.1), np.full_like(scores, 1.1)):
            with self.assertRaises(ValueError):
                EvaluateInstrument(image, mask, self.model, scores=invalid)

    def test_TrainingUsesSampledFeaturesAndReusesEachReportedFramePrediction(self):
        features = ImageFeatures
        prediction = PredictInstrument
        with patch("perception.ImageFeatures", wraps=features) as measured, \
                patch("perception.PredictInstrument", wraps=prediction) as predicted:
            model, report = TrainInstrument(self.folder)
        sampled = [call for call in measured.call_args_list if call.kwargs.get("positions") is not None]
        self.assertEqual(sum(len(call.kwargs["positions"]) for call in sampled), report["training_pixels"])
        self.assertTrue(all(len(call.kwargs["positions"]) <= 1000 for call in sampled))
        self.assertEqual(predicted.call_count, report["training_frames"] + len(report["frames"]))
        self.assertEqual(sum(call.kwargs.get("return_logits", False) for call in predicted.call_args_list),
                         len(report["frames"]))
        np.testing.assert_array_equal(model["weights"], self.model["weights"])
        self.assertEqual(report, self.report)

    def test_SampledTrainingMatchesTheFullFrameFeatureBaselineExactly(self):
        features = ImageFeatures

        def full_features(image, positions=None):
            full = features(image)
            return full if positions is None else full[positions]

        with patch("perception.ImageFeatures", side_effect=full_features):
            baseline, report = TrainInstrument(self.folder)
        for key in self.model:
            np.testing.assert_array_equal(baseline[key], self.model[key])
        self.assertEqual(report, self.report)

    def test_TrainingWhileCaptureWritesIgnoresItsPrivateIncompleteArchive(self):
        started = threading.Event()
        release = threading.Event()
        saved = np.savez_compressed
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            with Image.open(folder / "images" / "frame_000001.png") as source:
                image = np.asarray(source).copy()
            image[0, 0] = [1, 2, 3]
            with np.load(folder / "labels" / "frame_000001.npz") as source:
                labels = {key: source[key].copy() for key in source.files}
            labels["camera_configuration"][0] += 0.5

            def write_labels(output, **values):
                output.write(b"unfinished archive")
                started.set()
                if not release.wait(timeout=15):
                    raise TimeoutError("capture test stalled")
                output.seek(0)
                output.truncate()
                saved(output, **values)

            with patch("capture.np.savez_compressed", side_effect=write_labels), \
                    ThreadPoolExecutor(max_workers=1) as workers:
                future = workers.submit(SaveCapture, folder, image, labels)
                try:
                    self.assertTrue(started.wait(timeout=5))
                    model, report = TrainInstrument(folder)
                    self.assertEqual(report, self.report)
                    np.testing.assert_array_equal(model["weights"], self.model["weights"])
                finally:
                    release.set()
                _, label_path = future.result(timeout=5)
            self.assertEqual(label_path.stem, "frame_000009")
            model, report = TrainInstrument(folder)
            self.assertEqual(len(model["train_groups"]) + len(model["validation_groups"])
                             + len(model["calibration_groups"]), 9)

    def test_LearnedPredictionsSegmentAnUnseenFixtureFrame(self):
        validation = next(frame for frame in self.report["frames"]
                          if frame["split"] == "validation" and frame["variant"] == "clean")
        with Image.open(self.folder / "images" / (validation["frame_id"] + ".png")) as source:
            image = np.asarray(source)
        with np.load(self.folder / "labels" / (validation["frame_id"] + ".npz")) as labels:
            mask = labels["semantic_mask"]
        scores = PredictInstrument(image, self.model)
        self.assertEqual(scores.shape, mask.shape)
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))
        self.assertGreater(scores[mask == 1].mean(), scores[mask != 1].mean())
        metrics = EvaluateInstrument(image, mask, self.model)
        self.assertGreater(metrics["precision"], 0.95)
        self.assertGreater(metrics["recall"], 0.95)
        self.assertGreater(metrics["iou"], 0.95)
        self.assertEqual(sum(metrics[key] for key in ("tp", "fp", "fn", "tn")), mask.size)

    def test_CleanAndDegradedFramesShareDisjointSnapshotSplits(self):
        train = set(self.model["train_groups"])
        calibration = set(self.model["calibration_groups"])
        validation = set(self.model["validation_groups"])
        self.assertFalse(train & validation)
        self.assertFalse(train & calibration)
        self.assertFalse(calibration & validation)
        self.assertEqual((len(train), len(calibration), len(validation)), (4, 2, 2))
        frame_groups = {}
        for frame in self.report["frames"]:
            frame_groups.setdefault(frame["frame_id"], set()).add((frame["group_id"], frame["split"]))
        self.assertTrue(all(len(assignments) == 1 for assignments in frame_groups.values()))
        self.assertEqual(self.report["clean"]["frames"], 2)
        self.assertEqual(self.report["degraded"]["frames"], 2)
        self.assertLessEqual(self.report["training_pixels"], 4 * 2000)

    def test_RecapturedSnapshotWithDifferentStemStaysInItsExistingGroup(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            for subfolder, suffix in [("images", ".png"), ("labels", ".npz"), ("degraded", ".png")]:
                shutil.copyfile(folder / subfolder / ("frame_000001" + suffix),
                                folder / subfolder / ("frame_000009" + suffix))
            model, report = TrainInstrument(folder)
        self.assertEqual(report["training_groups"] + report["calibration"]["groups"] + report["validation_groups"], 8)
        self.assertEqual(len(report["frames"]), 16)
        np.testing.assert_array_equal(model["weights"], self.model["weights"])

    def test_IdenticalCleanRGBWithDifferentMetadataCannotLeakAcrossSplits(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            with np.load(folder / "labels" / "frame_000001.npz") as saved:
                labels = {key: saved[key] for key in saved.files}
            labels["tool_configuration"] = np.array([0.25, 0.3, 90])
            np.savez_compressed(folder / "labels" / "frame_000009.npz", **labels)
            shutil.copyfile(folder / "images" / "frame_000001.png",
                            folder / "images" / "frame_000009.png")
            model, report = TrainInstrument(folder)
        self.assertEqual(report["training_groups"] + report["calibration"]["groups"] + report["validation_groups"], 8)
        np.testing.assert_array_equal(model["weights"], self.model["weights"])

    def test_HeldOutImagesDoNotAffectNormalizerWeightsOrThreshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            for frame in self.report["frames"]:
                if frame["split"] == "validation":
                    subfolder = "images" if frame["variant"] == "clean" else "degraded"
                    path = folder / subfolder / (frame["frame_id"] + ".png")
                    with Image.open(path) as source:
                        image = np.asarray(source).copy()
                    image[:] = [2, 4, 6] if subfolder == "images" else [248, 250, 252]
                    image[0, 0, 0] = int(frame["frame_id"][-1])
                    Image.fromarray(image).save(path)
            changed, report = TrainInstrument(folder)
        for key in ("mean", "scale", "weights", "threshold", "train_groups", "validation_groups",
                    "appearance_references", "appearance_groups", "appearance_limit",
                    "calibration_groups", "calibration_temperature", "calibration_bias"):
            np.testing.assert_array_equal(changed[key], self.model[key])
        self.assertNotEqual(report["clean"], self.report["clean"])

    def test_EmptyTruthAndPredictionHaveUndefinedIoUAndZeroFalsePositiveFraction(self):
        model = {key: value.copy() for key, value in self.model.items()}
        model["weights"][:] = 0
        model["weights"][0] = -10
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        metrics = EvaluateInstrument(image, np.zeros((16, 16), dtype=np.uint8), model)
        self.assertIsNone(metrics["iou"])
        self.assertIsNone(metrics["recall"])
        self.assertEqual(metrics["false_positive_fraction"], 0)
        self.assertEqual(metrics["tn"], 256)

    def test_FalsePositivesOnAnEmptyFrameAreReportedAsFailures(self):
        model = {key: value.copy() for key, value in self.model.items()}
        model["weights"][:] = 0
        model["weights"][0] = 10
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        metrics = EvaluateInstrument(image, np.zeros((16, 16), dtype=np.uint8), model)
        self.assertEqual(metrics["iou"], 0)
        self.assertEqual(metrics["precision"], 0)
        self.assertIsNone(metrics["recall"])
        self.assertEqual(metrics["false_positive_fraction"], 1)
        self.assertEqual(metrics["fp"], 256)

    def test_ArchiveRoundTripPreservesPredictionsAndNumericProvenance(self):
        loaded = LoadInstrument(self.output)
        report = json.loads(str(loaded["report_json"]))
        self.assertEqual(report, self.report)
        self.assertEqual(report["threshold"], float(loaded["threshold"]))
        image = np.full((16, 16, 3), 128, dtype=np.uint8)
        np.testing.assert_array_equal(PredictInstrument(image, loaded),
                                      PredictInstrument(image, self.model))
        with np.load(self.output, allow_pickle=False) as archive:
            self.assertTrue(all(archive[key].dtype.kind != "O" for key in archive.files))

    def test_InvalidModelArchivesAreRejectedWithoutPickle(self):
        invalid = []
        for key, value in [("version", np.array(4)), ("weights", np.zeros(12)),
                           ("weights", np.full(13, np.nan)), ("scale", np.zeros(12)),
                           ("threshold", np.array(1.0)), ("mean", np.zeros(11)),
                           ("feature_names", np.array([{}], dtype=object)),
                           ("report_json", np.array("invalid")),
                           ("validation_groups", self.model["train_groups"][:2])]:
            model = {name: array.copy() for name, array in self.model.items()}
            model[key] = value
            invalid.append(model)
        extra = {**self.model, "unapproved": np.array(1)}
        missing = {key: value for key, value in self.model.items() if key != "mean"}
        invalid.extend([extra, missing])
        with tempfile.TemporaryDirectory() as temporary:
            for index, model in enumerate(invalid):
                path = Path(temporary) / f"invalid_{index}.npz"
                np.savez_compressed(path, **model)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    LoadInstrument(path)

    def test_EmptyJunkAndPlainArrayModelFilesRaiseValueError(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name, contents in [("empty.npz", b""), ("junk.npz", b"not an archive"),
                                   ("truncated.npz", b"PK\x03\x04junk")]:
                path = Path(temporary) / name
                path.write_bytes(contents)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    LoadInstrument(path)
            path = Path(temporary) / "plain.npz"
            with path.open("wb") as output:
                np.save(output, np.zeros(13))
            with self.assertRaises(ValueError):
                LoadInstrument(path)

    def test_ContradictoryReportAssignmentsCountsAndRatesAreRejected(self):
        reports = []
        for rate in ["invalid", True, float("nan"), 1.5]:
            report = json.loads(str(self.model["report_json"]))
            report["negative_false_positive_fraction"] = rate
            reports.append(report)
        report = json.loads(str(self.model["report_json"]))
        report["frames"][0]["group_id"] = "f" * 64
        reports.append(report)
        report = json.loads(str(self.model["report_json"]))
        report["frames"][0]["split"] = "unknown"
        reports.append(report)
        report = json.loads(str(self.model["report_json"]))
        report["frames"][0]["tp"] += 1
        reports.append(report)
        report = json.loads(str(self.model["report_json"]))
        report["clean"]["fp"] += 1
        reports.append(report)
        report = json.loads(str(self.model["report_json"]))
        report["threshold"] = 0.99
        reports.append(report)
        with tempfile.TemporaryDirectory() as temporary:
            for index, report in enumerate(reports):
                model = {**self.model, "report_json": np.array(json.dumps(report))}
                path = Path(temporary) / f"invalid_report_{index}.npz"
                np.savez_compressed(path, **model)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    LoadInstrument(path)

    def test_InvalidImagesMasksAndModelsRaiseValueError(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        for invalid in [None, [], image.astype(float), image[:, :, 0], image[:15]]:
            for operation in [ImageFeatures,
                              lambda value: PredictInstrument(value, self.model),
                              lambda value: EvaluateInstrument(value, np.zeros((16, 16), dtype=np.uint8), self.model)]:
                with self.subTest(operation=operation, shape=getattr(invalid, "shape", None)), self.assertRaises(ValueError):
                    operation(invalid)
        for mask in [None, np.zeros((16, 16), dtype=float), np.zeros((16, 15), dtype=np.uint8)]:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                EvaluateInstrument(image, mask, self.model)
        with self.assertRaises(ValueError):
            PredictInstrument(image, {})

    def test_TooFewGroupsAndAmbiguousClassMappingsAreRejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            path = folder / "labels" / "frame_000008.npz"
            path.unlink()
            with self.assertRaises(ValueError):
                TrainInstrument(folder)
            shutil.copyfile(self.folder / "labels" / path.name, path)
            with np.load(path) as saved:
                labels = {key: saved[key] for key in saved.files}
            labels["class_ids"] = np.array([1, 1, 2], dtype=np.uint8)
            np.savez_compressed(path, **labels)
            with self.assertRaises(ValueError):
                TrainInstrument(folder)

    def test_CancellationBeforeAndDuringTrainingPreservesAnExistingOutput(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "instrument.npz"
            output.write_bytes(b"previous model")
            stopped = threading.Event()
            stopped.set()
            during = Mock()
            during.is_set.side_effect = [False] * 30 + [True]
            for cancel in (stopped, during):
                with self.subTest(cancel=cancel), self.assertRaises(InterruptedError):
                    TrainInstrument(self.folder, output, cancel=cancel)
                self.assertEqual(output.read_bytes(), b"previous model")
            self.assertEqual(sorted(path.name for path in Path(temporary).iterdir()), ["instrument.npz"])

    def test_AppearanceUsesOnlyTheCentralSquareWithNormalizedChannelHistograms(self):
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        image[8:72, 28:92] = [16, 32, 48]
        features = AppearanceFeatures(image).reshape(3, 16)
        np.testing.assert_array_equal(features.sum(axis=1), 1)
        self.assertEqual(features[0, 1], 1)
        self.assertEqual(features[1, 2], 1)
        self.assertEqual(features[2, 3], 1)
        self.assertEqual(np.count_nonzero(features), 3)

    def test_AppearanceHistogramsDoNotRecognizeSpatialArrangement(self):
        image = np.full((80, 80, 3), [160, 70, 50], dtype=np.uint8)
        image[:, 40:] = [80, 90, 100]
        reversed_image = image[:, ::-1]
        np.testing.assert_array_equal(AppearanceFeatures(image), AppearanceFeatures(reversed_image))
        self.assertFalse(np.array_equal(image, reversed_image))

    def test_AppearanceCalibrationExcludesTheWholeSnapshotGroup(self):
        black = np.zeros((80, 80, 3), dtype=np.uint8)
        white = np.full_like(black, 255)
        profile = FitAppearance([black, black.copy(), white], ["a" * 64, "a" * 64, "b" * 64])
        self.assertEqual(float(profile["appearance_limit"]), 1)
        self.assertEqual(profile["appearance_references"].shape, (3, 48))

    def test_AppearanceUsesTheFixedNinetyFifthPercentileReferenceLimit(self):
        images = []
        for white_columns in [0, 16, 48]:
            image = np.zeros((80, 80, 3), dtype=np.uint8)
            image[8:72, 8:8 + white_columns] = 255
            images.append(image)
        profile = FitAppearance(images, ["a" * 64, "b" * 64, "c" * 64])
        # nearest other-group distances are 0.25, 0.25 and 0.5
        self.assertAlmostEqual(float(profile["appearance_limit"]), 0.475)

    def test_AppearanceFitRejectsMissingOrAmbiguousGroups(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        for images, groups in [([], []), ([image], ["a" * 64]),
                               ([image, image], ["a" * 64, "a" * 64]),
                               ([image, image], ["a" * 64]),
                               ([image, image], ["invalid", "b" * 64])]:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                FitAppearance(images, groups)

    def test_AssessmentSeparatesQualityWarningsFromAppearanceDistance(self):
        rows, columns = np.indices((80, 80))
        image = np.repeat((60 + 100 * ((rows // 8 + columns // 8) % 2))[:, :, None], 3, axis=2).astype(np.uint8)
        model = {key: value.copy() for key, value in self.model.items()}
        model.update(FitAppearance([image, image.copy()], model["train_groups"][:2]))
        accepted = AssessPrediction(image, model)
        self.assertTrue(accepted["supported"])
        self.assertEqual(accepted["reasons"], ())
        self.assertEqual(accepted["distance"], 0)
        hazy = AssessPrediction(DegradeView(image, haze=1), model)
        self.assertFalse(hazy["supported"])
        self.assertIn("low_detail", hazy["reasons"])
        self.assertIn("appearance_shift", hazy["reasons"])
        self.assertGreater(hazy["distance"], hazy["limit"])

    def test_AssessmentReusesOnlyAConsistentViewQualityResult(self):
        rows, columns = np.indices((80, 120))
        image = np.repeat((60 + 100 * ((rows // 8 + columns // 8) % 2))[:, :, None], 3, axis=2).astype(np.uint8)
        expected = AssessPrediction(image, self.model)
        quality = AssessView(image[8:72, 12:108])
        with patch("perception.AssessView", side_effect=AssertionError("recomputed quality")):
            self.assertEqual(AssessPrediction(image, self.model, quality=quality), expected)
        invalid = [{**quality, "status": "limited"}, {**quality, "issues": ("dark",)},
                   {**quality, "detail": np.nan}, {**quality, "dark_fraction": 2.0},
                   {**quality, "bright_fraction": True}, {**quality, "extra": 1}, {}, []]
        for result in invalid:
            with self.subTest(quality=result), self.assertRaises(ValueError):
                AssessPrediction(image, self.model, quality=result)

    def test_MinimumSizedAssessmentRemainsValid(self):
        result = AssessPrediction(np.full((16, 16, 3), 128, dtype=np.uint8), self.model)
        self.assertIsInstance(result["supported"], bool)
        self.assertIn("low_detail", result["reasons"])

    def test_LegacyModelPredictsButWithholdsWithoutAnAppearanceProfile(self):
        image = np.full((16, 16, 3), 128, dtype=np.uint8)
        for version in (1, 2):
            legacy = {key: value.copy() for key, value in self.external_model.items()
                      if not key.startswith("calibration_")
                      and (version == 2 or not key.startswith("appearance_"))}
            legacy["version"] = np.array(version)
            report = json.loads(str(legacy["report_json"]))
            report.pop("calibration")
            report["frames"] = [frame for frame in report["frames"] if frame["split"] != "calibration"]
            if version == 1:
                report.pop("support")
            legacy["report_json"] = np.array(json.dumps(report))
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "legacy.npz"
                np.savez_compressed(path, **legacy)
                loaded = LoadInstrument(path)
                np.testing.assert_array_equal(PredictInstrument(image, loaded),
                                              PredictInstrument(image, self.external_model))
                with self.assertRaises(ValueError):
                    PredictInstrument(image, loaded, calibrated=True)
                if version == 1:
                    assessment = AssessPrediction(image, loaded)
                    self.assertFalse(assessment["supported"])
                    self.assertIn("profile_missing", assessment["reasons"])
                    self.assertIsNone(assessment["distance"])
                    self.assertIsNone(assessment["limit"])

    def test_VersionTwoLoaderRejectsInvalidOrHeldOutAppearanceReferences(self):
        invalid = []
        references = self.model["appearance_references"].copy()
        references[0, 0] = -1
        nonnormalized = self.model["appearance_references"].copy()
        nonnormalized[0] = 0
        heldout = self.model["appearance_groups"].copy()
        heldout[0] = self.model["validation_groups"][0]
        for key, value in [("appearance_references", np.zeros((2, 47))),
                           ("appearance_references", np.full_like(references, np.nan)),
                           ("appearance_references", references),
                           ("appearance_references", nonnormalized),
                           ("appearance_groups", heldout),
                           ("appearance_groups", np.repeat(self.model["train_groups"][:1], len(heldout))),
                           ("appearance_limit", np.array(-0.1)),
                           ("appearance_limit", np.array(1.1)),
                           ("appearance_limit", np.array(np.nan))]:
            invalid.append({**self.model, key: value})
        with tempfile.TemporaryDirectory() as temporary:
            for index, model in enumerate(invalid):
                path = Path(temporary) / f"invalid_appearance_{index}.npz"
                np.savez_compressed(path, **model)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    LoadInstrument(path)

    def test_TrainingReportKeepsRawMetricsWhenAllPredictionsAreWithheld(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            for subfolder in ("images", "degraded"):
                for index, path in enumerate(sorted((folder / subfolder).glob("*.png"))):
                    colour = 100 + index + (20 if subfolder == "degraded" else 0)
                    Image.fromarray(np.full((48, 64, 3), colour, dtype=np.uint8)).save(path)
            model, report = TrainInstrument(folder)
        self.assertEqual(int(model["version"]), 3)
        self.assertEqual(set(model["appearance_groups"]), set(model["train_groups"]))
        for category in ("clean", "degraded"):
            self.assertEqual(report[category]["frames"], 2)
            self.assertIsNotNone(report[category]["iou"])
            self.assertEqual(report["support"][category]["accepted"], 0)
            self.assertEqual(report["support"][category]["withheld"], 2)
            self.assertIsNone(report["support"][category]["iou"])

    def test_VersionTwoReportCannotInventAcceptedMetricsOrCoverage(self):
        reports = []
        report = json.loads(str(self.model["report_json"]))
        report["support"]["clean"]["accepted"] += 1
        reports.append(report)
        report = json.loads(str(self.model["report_json"]))
        report["frames"][0]["supported"] = not report["frames"][0]["supported"]
        reports.append(report)
        report = json.loads(str(self.model["report_json"]))
        current = report["support"]["clean"]["iou"]
        report["support"]["clean"]["iou"] = 0.5 if current != 0.5 else 0.25
        reports.append(report)
        with tempfile.TemporaryDirectory() as temporary:
            for index, report in enumerate(reports):
                path = Path(temporary) / f"invalid_support_{index}.npz"
                np.savez_compressed(path, **{**self.model, "report_json": np.array(json.dumps(report))})
                with self.subTest(index=index), self.assertRaises(ValueError):
                    LoadInstrument(path)

    def test_CalibrationIsOptInAndCannotChangeRawMasksOrClearanceInputs(self):
        with Image.open(self.folder / "images" / "frame_000001.png") as source:
            image = np.asarray(source)
        raw = PredictInstrument(image, self.model)
        changed = {key: value.copy() for key, value in self.model.items()}
        changed["calibration_temperature"] = np.array(0.2)
        changed["calibration_bias"] = np.array(-8.0)
        np.testing.assert_array_equal(PredictInstrument(image, changed), raw)
        np.testing.assert_array_equal(PredictInstrument(image, changed, calibrated=False), raw)
        np.testing.assert_array_equal(PredictInstrument(image, changed) >= float(changed["threshold"]),
                                      raw >= float(self.model["threshold"]))
        probabilities = PredictInstrument(image, changed, calibrated=True)
        self.assertFalse(np.array_equal(probabilities, raw))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        with self.assertRaises(ValueError):
            PredictInstrument(image, self.model, calibrated="yes")

    def test_ReservedCalibrationPixelsCannotChangeTheClassifierOrAppearanceReference(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            for frame in self.report["frames"]:
                if frame["split"] != "calibration":
                    continue
                directory = "images" if frame["variant"] == "clean" else "degraded"
                path = folder / directory / (frame["frame_id"] + ".png")
                with Image.open(path) as source:
                    image = np.asarray(source).copy()
                image[:] = [75, 95, 110]
                image[0, 0, 0] = int(frame["frame_id"][-1])
                Image.fromarray(image).save(path)
            changed, _ = TrainInstrument(folder)
        for key in ("weights", "mean", "scale", "threshold", "train_groups", "validation_groups",
                    "calibration_groups", "appearance_references", "appearance_groups", "appearance_limit"):
            np.testing.assert_array_equal(changed[key], self.model[key])
        self.assertTrue(float(changed["calibration_temperature"]) != float(self.model["calibration_temperature"])
                        or float(changed["calibration_bias"]) != float(self.model["calibration_bias"]))

    def test_DesignatedCalibrationKeepsTheOriginalPoolAndHeldOutAssignment(self):
        model, report = self.external_model, self.external_report
        self.assertEqual(report["calibration"]["source"], "external")
        self.assertEqual((len(model["train_groups"]), len(model["calibration_groups"]),
                          len(model["validation_groups"])), (6, 2, 2))
        np.testing.assert_array_equal(model["validation_groups"], self.model["validation_groups"])
        self.assertEqual(set(model["train_groups"]), set(self.model["train_groups"]) | set(self.model["calibration_groups"]))
        self.assertFalse(set(model["calibration_groups"]) & set(model["train_groups"]))
        self.assertFalse(set(model["calibration_groups"]) & set(model["validation_groups"]))
        self.assertEqual(set(model["appearance_groups"]), set(model["train_groups"]))
        self.assertEqual(report["clean"]["frames"], 2)
        self.assertEqual(report["calibration"]["frames"], 4)

    def test_ChangingDesignatedCalibrationDoesNotAdvanceTheClassifierRandomStream(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.designated, folder)
            for directory in ("images", "degraded"):
                for index, path in enumerate(sorted((folder / "calibration" / directory).glob("*.png"))):
                    with Image.open(path) as source:
                        image = np.asarray(source).copy()
                    image[:] = [95, 110, 125]
                    image[0, 0, 0] = index + 1
                    Image.fromarray(image).save(path)
            changed, _ = TrainInstrument(folder)
        for key in ("weights", "mean", "scale", "threshold", "train_groups", "validation_groups",
                    "appearance_references", "appearance_groups", "appearance_limit"):
            np.testing.assert_array_equal(changed[key], self.external_model[key])

    def test_DesignatedCalibrationRejectsPhysicalAndExactRGBOverlapWithThePool(self):
        for overlap in ("physical", "rgb", "merged_physical"):
            with self.subTest(overlap=overlap), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary) / "captures"
                shutil.copytree(self.designated, folder)
                if overlap == "physical":
                    shutil.copyfile(folder / "labels" / "frame_000001.npz",
                                    folder / "calibration" / "labels" / "frame_000001.npz")
                elif overlap == "rgb":
                    shutil.copyfile(folder / "images" / "frame_000001.png",
                                    folder / "calibration" / "images" / "frame_000001.png")
                else:
                    # keeps the physical alias even when duplicate clean RGB is merged
                    shutil.copyfile(folder / "calibration" / "labels" / "frame_000001.npz",
                                    folder / "labels" / "frame_000009.npz")
                    shutil.copyfile(folder / "images" / "frame_000001.png",
                                    folder / "images" / "frame_000009.png")
                with self.assertRaisesRegex(ValueError, "overlap"):
                    TrainInstrument(folder)

    def test_CalibrationSamplingRestoresNaturalPrevalenceInsteadOfBalancingClasses(self):
        from calibration import FitCalibration
        with patch("perception.FitCalibration", wraps=FitCalibration) as fit:
            _, report = TrainInstrument(self.folder)
        _, targets = fit.call_args.args
        weights = fit.call_args.kwargs["weights"]
        expected = []
        for group in self.model["calibration_groups"]:
            fractions = []
            for frame in self.report["frames"]:
                if frame["group_id"] == group:
                    with np.load(self.folder / "labels" / (frame["frame_id"] + ".npz")) as saved:
                        fractions.append(float((saved["semantic_mask"] == 1).mean()))
            expected.append(np.mean(fractions))
        self.assertAlmostEqual(float(np.sum(weights * targets) / weights.sum()), float(np.mean(expected)))
        self.assertNotAlmostEqual(float(np.sum(weights * targets) / weights.sum()), 0.5)
        self.assertEqual(report["calibration"]["source"], "reserved")

    def test_CalibrationArchivesRejectGroupParameterAndReliabilityBinContradictions(self):
        invalid = []
        for key, value in [("calibration_temperature", np.array(0.0)),
                           ("calibration_temperature", np.array(np.nan)),
                           ("calibration_bias", np.array(np.inf)),
                           ("calibration_bias", np.array([0.0])),
                           ("calibration_groups", self.model["train_groups"][:2])]:
            invalid.append({**self.model, key: value})
        for corruption in ("groups", "temperature", "bins", "count", "ece", "frame"):
            report = json.loads(str(self.model["report_json"]))
            if corruption == "groups":
                report["calibration"]["groups"] += 1
            elif corruption == "temperature":
                report["calibration"]["temperature"] += 0.5
            elif corruption == "bins":
                report["calibration"]["clean"]["raw"]["bin_edges"][1] = 0.12
            elif corruption == "count":
                report["calibration"]["clean"]["calibrated"]["bin_counts"][0] += 1
            elif corruption == "ece":
                report["calibration"]["clean"]["raw"]["ece"] += 0.1
            else:
                next(frame for frame in report["frames"] if frame["split"] == "calibration")["split"] = "train"
            invalid.append({**self.model, "report_json": np.array(json.dumps(report))})
        with tempfile.TemporaryDirectory() as temporary:
            for index, model in enumerate(invalid):
                path = Path(temporary) / f"invalid_calibration_{index}.npz"
                np.savez_compressed(path, **model)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    LoadInstrument(path)

    def test_CleanOnlyCapturesReportMissingDegradedCalibrationMetrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            for path in (folder / "degraded").glob("*.png"):
                path.unlink()
            (folder / "degraded").rmdir()
            output = Path(temporary) / "instrument.npz"
            _, report = TrainInstrument(folder, output)
            LoadInstrument(output)
        self.assertIsNone(report["calibration"]["degraded"]["raw"])
        self.assertIsNone(report["calibration"]["degraded"]["calibrated"])
        self.assertGreater(report["calibration"]["clean"]["raw"]["pixels"], 0)


if __name__ == "__main__":
    unittest.main()
