import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from perception import (AppearanceFeatures, AssessPrediction, EvaluateInstrument,
                        FitAppearance, ImageFeatures, LoadInstrument,
                        PredictInstrument, TrainInstrument)
from safesight import DegradeView


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
        validation = set(self.model["validation_groups"])
        self.assertFalse(train & validation)
        self.assertEqual((len(train), len(validation)), (6, 2))
        frame_groups = {}
        for frame in self.report["frames"]:
            frame_groups.setdefault(frame["frame_id"], set()).add((frame["group_id"], frame["split"]))
        self.assertTrue(all(len(assignments) == 1 for assignments in frame_groups.values()))
        self.assertEqual(self.report["clean"]["frames"], 2)
        self.assertEqual(self.report["degraded"]["frames"], 2)
        self.assertLessEqual(self.report["training_pixels"], 6 * 2000)

    def test_RecapturedSnapshotWithDifferentStemStaysInItsExistingGroup(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "captures"
            shutil.copytree(self.folder, folder)
            for subfolder, suffix in [("images", ".png"), ("labels", ".npz"), ("degraded", ".png")]:
                shutil.copyfile(folder / subfolder / ("frame_000001" + suffix),
                                folder / subfolder / ("frame_000009" + suffix))
            model, report = TrainInstrument(folder)
        self.assertEqual(report["training_groups"] + report["validation_groups"], 8)
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
        self.assertEqual(report["training_groups"] + report["validation_groups"], 8)
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
                    "appearance_references", "appearance_groups", "appearance_limit"):
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
        for key, value in [("version", np.array(3)), ("weights", np.zeros(12)),
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

    def test_MinimumSizedAssessmentRemainsValid(self):
        result = AssessPrediction(np.full((16, 16, 3), 128, dtype=np.uint8), self.model)
        self.assertIsInstance(result["supported"], bool)
        self.assertIn("low_detail", result["reasons"])

    def test_LegacyModelPredictsButWithholdsWithoutAnAppearanceProfile(self):
        legacy = {key: value.copy() for key, value in self.model.items()
                  if not key.startswith("appearance_")}
        legacy["version"] = np.array(1)
        image = np.full((16, 16, 3), 128, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.npz"
            np.savez_compressed(path, **legacy)
            loaded = LoadInstrument(path)
        np.testing.assert_array_equal(PredictInstrument(image, loaded), PredictInstrument(image, self.model))
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
        self.assertEqual(int(model["version"]), 2)
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


if __name__ == "__main__":
    unittest.main()
