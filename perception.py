import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np
from PIL import Image

from safesight import AssessView


_FEATURE_NAMES = np.array([
    "red", "green", "blue", "red_fraction", "green_fraction", "blue_fraction",
    "red_squared", "green_squared", "blue_squared", "red_green", "red_blue", "green_blue"])
_MODEL_KEYS = {"version", "feature_names", "weights", "mean", "scale", "threshold",
               "train_groups", "validation_groups", "report_json"}
_APPEARANCE_KEYS = {"appearance_references", "appearance_groups", "appearance_limit"}


def ImageFeatures(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) < 16):
        raise ValueError("Invalid RGB image.")
    pixels = image.reshape(-1, 3).astype(np.float32) / 255
    fractions = pixels / np.maximum(pixels.sum(axis=1, keepdims=True), 1e-6)
    products = pixels[:, [0, 0, 1]] * pixels[:, [1, 2, 2]]
    return np.concatenate((pixels, fractions, pixels * pixels, products), axis=1)


def AppearanceFeatures(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) < 16):
        raise ValueError("Invalid RGB image.")
    # uses a central square without resampling so capture resolution does not change the bins
    height, width = image.shape[:2]
    side = round(0.8 * min(height, width))
    top, left = (height - side) // 2, (width - side) // 2
    square = image[top:top + side, left:left + side]
    histograms = [np.bincount((square[:, :, channel] // 16).reshape(-1), minlength=16)
                  for channel in range(3)]
    return np.concatenate(histograms).astype(float) / (side * side)


def FitAppearance(images, groups):
    try:
        references = []
        for image in images:
            if isinstance(image, (str, Path)):
                # reads capture paths one at a time instead of retaining full RGB frames
                with Image.open(image) as source:
                    references.append(AppearanceFeatures(np.asarray(source)))
            else:
                references.append(AppearanceFeatures(image))
        references = np.array(references)
        groups = np.asarray(groups)
        if (references.ndim != 2 or references.shape[1] != 48 or groups.shape != (len(references),)
                or groups.dtype.kind != "U" or len(np.unique(groups)) < 2
                or any(len(group) != 64 or any(char not in "0123456789abcdef" for char in group)
                       for group in groups)):
            raise ValueError("Invalid appearance groups.")
        # compares each reference only against other physical snapshot groups
        nearest = []
        for reference, group in zip(references, groups):
            distances = np.abs(references - reference).sum(axis=1) / 6
            distances[groups == group] = np.inf
            nearest.append(float(distances.min()))
        limit = float(np.quantile(nearest, 0.95))
    except (OSError, TypeError, ValueError, OverflowError):
        raise ValueError("Invalid appearance references.") from None
    return {"appearance_references": references, "appearance_groups": groups.copy(),
            "appearance_limit": np.array(limit)}


def AssessPrediction(image, model):
    appearance = AppearanceFeatures(image)
    height, width = image.shape[:2]
    # keeps tiny inputs large enough for the quality check
    vertical = min(height // 10, (height - 16) // 2)
    horizontal = min(width // 10, (width - 16) // 2)
    quality = AssessView(image[vertical:height - vertical, horizontal:width - horizontal])
    reasons = list(quality["issues"])
    distance = limit = None
    if (not isinstance(model, dict) or not _APPEARANCE_KEYS.issubset(model)
            or np.asarray(model.get("version")).shape != ()
            or np.asarray(model.get("version")).dtype.kind not in "iu"
            or int(model["version"]) != 2):
        reasons.append("profile_missing")
    else:
        try:
            references = np.asarray(model["appearance_references"])
            profile_groups = np.asarray(model["appearance_groups"])
            limit_array = np.asarray(model["appearance_limit"])
            if (references.ndim != 2 or references.shape[1] != 48 or len(references) < 2
                    or references.dtype.kind != "f" or not np.all(np.isfinite(references))
                    or np.any(references < 0)
                    or not np.allclose(references.reshape(-1, 3, 16).sum(axis=2), 1, rtol=0, atol=1e-6)
                    or profile_groups.shape != (len(references),) or profile_groups.dtype.kind != "U"
                    or len(np.unique(profile_groups)) < 2
                    or not set(profile_groups).issubset(set(model["train_groups"]))
                    or limit_array.shape != () or limit_array.dtype.kind != "f"
                    or not np.isfinite(limit_array) or not 0 <= limit_array <= 1):
                raise ValueError("Invalid appearance profile.")
            # measures appearance distance, not prediction confidence or surgical clearance
            distance = float((np.abs(references - appearance).sum(axis=1) / 6).min())
            limit = float(limit_array)
            if distance > limit:
                reasons.append("appearance_shift")
        except (TypeError, ValueError, KeyError, OverflowError):
            raise ValueError("Invalid appearance profile.") from None
    return {"supported": not reasons, "reasons": tuple(reasons), "distance": distance,
            "limit": limit, "quality": quality}


def PredictInstrument(image, model):
    features = ImageFeatures(image)
    try:
        if not isinstance(model, dict):
            raise ValueError("Invalid instrument model.")
        version = np.asarray(model.get("version"))
        if version.shape != () or version.dtype.kind not in "iu" or version not in (1, 2):
            raise ValueError("Invalid instrument model.")
        expected_keys = _MODEL_KEYS | _APPEARANCE_KEYS if version == 2 else _MODEL_KEYS
        if set(model) != expected_keys:
            raise ValueError("Invalid instrument model.")
        weights = np.asarray(model["weights"])
        mean = np.asarray(model["mean"])
        scale = np.asarray(model["scale"])
        threshold = np.asarray(model["threshold"])
        version = np.asarray(model["version"])
        if (weights.shape != (13,) or mean.shape != (12,) or scale.shape != (12,)
                or any(value.dtype.kind != "f" or not np.all(np.isfinite(value))
                       for value in (weights, mean, scale, threshold))
                or np.any(scale <= 0) or threshold.shape != () or not 0 < threshold < 1
                or version.shape != () or version.dtype.kind not in "iu" or version not in (1, 2)
                or not np.array_equal(model["feature_names"], _FEATURE_NAMES)):
            raise ValueError("Invalid instrument model.")
        # produces classifier scores; these are not calibrated safety probabilities
        logits = ((features - mean) / scale) @ weights[1:] + weights[0]
        scores = 1 / (1 + np.exp(-np.clip(logits, -35, 35)))
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError("Invalid instrument model.") from None
    return scores.reshape(image.shape[:2])


def EvaluateInstrument(image, mask, model):
    scores = PredictInstrument(image, model)
    if (not isinstance(mask, np.ndarray) or mask.dtype not in (np.dtype(bool), np.dtype(np.uint8))
            or mask.shape != scores.shape):
        raise ValueError("Invalid instrument mask.")
    predicted = scores >= float(model["threshold"])
    truth = mask == 1
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    tn = int(np.count_nonzero(~predicted & ~truth))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "false_positive_fraction": fp / (fp + tn) if fp + tn else None}


def LoadInstrument(path):
    try:
        with np.load(path, allow_pickle=False) as saved:
            model = {key: saved[key].copy() for key in saved.files}
        version = model["version"]
        if version.shape != () or version.dtype.kind not in "iu" or version not in (1, 2):
            raise ValueError("Invalid instrument model.")
        expected_keys = _MODEL_KEYS | _APPEARANCE_KEYS if version == 2 else _MODEL_KEYS
        if set(model) != expected_keys:
            raise ValueError("Invalid instrument model.")
        for key in ("weights", "mean", "scale", "threshold"):
            if model[key].dtype.kind != "f" or not np.all(np.isfinite(model[key])):
                raise ValueError("Invalid instrument model.")
        if (model["weights"].shape != (13,) or model["mean"].shape != (12,)
                or model["scale"].shape != (12,) or np.any(model["scale"] <= 0)
                or model["threshold"].shape != () or not 0 < model["threshold"] < 1
                or model["version"].shape != () or model["version"].dtype.kind not in "iu"
                or model["version"] not in (1, 2) or model["feature_names"].dtype.kind != "U"
                or not np.array_equal(model["feature_names"], _FEATURE_NAMES)
                or model["report_json"].shape != () or model["report_json"].dtype.kind != "U"):
            raise ValueError("Invalid instrument model.")
        train = model["train_groups"]
        validation = model["validation_groups"]
        if (train.ndim != 1 or validation.ndim != 1 or train.dtype.kind != "U"
                or validation.dtype.kind != "U" or not len(train) or not len(validation)):
            raise ValueError("Invalid instrument model.")
        groups = list(train) + list(validation)
        if (len(groups) < 8 or len(set(groups)) != len(groups)
                or any(len(group) != 64 or any(char not in "0123456789abcdef" for char in group)
                       for group in groups)):
            raise ValueError("Invalid instrument model.")
        if version == 2:
            references = model["appearance_references"]
            profile_groups = model["appearance_groups"]
            appearance_limit = model["appearance_limit"]
            if (references.ndim != 2 or references.shape[1] != 48 or len(references) < 2
                    or references.dtype.kind != "f" or not np.all(np.isfinite(references))
                    or np.any(references < 0)
                    or not np.allclose(references.reshape(-1, 3, 16).sum(axis=2), 1, rtol=0, atol=1e-6)
                    or profile_groups.shape != (len(references),) or profile_groups.dtype.kind != "U"
                    or len(np.unique(profile_groups)) < 2 or not set(profile_groups).issubset(set(train))
                    or appearance_limit.shape != () or appearance_limit.dtype.kind != "f"
                    or not np.isfinite(appearance_limit) or not 0 <= appearance_limit <= 1):
                raise ValueError("Invalid instrument model.")
        report = json.loads(str(model["report_json"]))
        json.dumps(report, allow_nan=False)
        if (not isinstance(report, dict) or type(report.get("version")) is not int or report["version"] != 1
                or report.get("training_groups") != len(train)
                or report.get("validation_groups") != len(validation)
                or report.get("features") != _FEATURE_NAMES.tolist()
                or type(report.get("threshold")) not in (int, float)
                or report["threshold"] != float(model["threshold"])
                or type(report.get("training_pixels")) is not int
                or not 0 < report["training_pixels"] <= 2000 * len(train)
                or type(report.get("seed")) is not int or report["seed"] < 0
                or not isinstance(report.get("frames"), list)):
            raise ValueError("Invalid instrument model.")
        totals = {category: {key: 0 for key in ("frames", "tp", "fp", "fn", "tn")}
                  for category in ("clean", "degraded")}
        support_totals = {category: {key: 0 for key in (
            "frames", "accepted", "withheld", "tp", "fp", "fn", "tn")}
            for category in ("clean", "degraded")}
        assignments = {group: "train" for group in train}
        assignments.update({group: "validation" for group in validation})
        represented = set()
        frame_counts = {"train": 0, "validation": 0}
        negative_fp = negative_pixels = 0
        for frame in report["frames"]:
            if (not isinstance(frame, dict) or not isinstance(frame.get("frame_id"), str)
                    or frame.get("group_id") not in assignments
                    or frame.get("split") != assignments[frame["group_id"]]
                    or frame.get("variant") not in totals):
                raise ValueError("Invalid instrument model.")
            represented.add(frame["group_id"])
            frame_counts[frame["split"]] += 1
            for key in ("tp", "fp", "fn", "tn"):
                if type(frame[key]) is not int or frame[key] < 0:
                    raise ValueError("Invalid instrument model.")
            tp, fp, fn, tn = (frame[key] for key in ("tp", "fp", "fn", "tn"))
            expected = {"precision": tp / (tp + fp) if tp + fp else None,
                        "recall": tp / (tp + fn) if tp + fn else None,
                        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
                        "false_positive_fraction": fp / (fp + tn) if fp + tn else None}
            for key, value in expected.items():
                measured = frame.get(key)
                if measured is not None and type(measured) not in (int, float):
                    raise ValueError("Invalid instrument model.")
                if measured != value:
                    raise ValueError("Invalid instrument model.")
            if version == 2:
                reasons = frame.get("reasons")
                distance = frame.get("distance")
                if (type(frame.get("supported")) is not bool or not isinstance(reasons, list)
                        or any(reason not in ("dark", "bright", "low_detail", "appearance_shift")
                               for reason in reasons)
                        or len(set(reasons)) != len(reasons)
                        or frame["supported"] != (not reasons)
                        or type(distance) not in (int, float) or not 0 <= distance <= 1
                        or ("appearance_shift" in reasons) != (distance > float(appearance_limit))):
                    raise ValueError("Invalid instrument model.")
            if frame["split"] == "validation":
                total = totals[frame["variant"]]
                total["frames"] += 1
                for key in ("tp", "fp", "fn", "tn"):
                    total[key] += frame[key]
                if tp + fn == 0:
                    negative_fp += fp
                    negative_pixels += fp + tn
                if version == 2:
                    accepted = support_totals[frame["variant"]]
                    accepted["frames"] += 1
                    accepted["accepted" if frame["supported"] else "withheld"] += 1
                    if frame["supported"]:
                        for key in ("tp", "fp", "fn", "tn"):
                            accepted[key] += frame[key]
        if (represented != set(groups) or report.get("training_frames") != frame_counts["train"]
                or report.get("validation_frames") != frame_counts["validation"]):
            raise ValueError("Invalid instrument model.")
        for category in ("clean", "degraded"):
            metrics = report[category]
            if not isinstance(metrics, dict):
                raise ValueError("Invalid instrument model.")
            for key in ("frames", "tp", "fp", "fn", "tn"):
                if type(metrics[key]) is not int or metrics[key] < 0:
                    raise ValueError("Invalid instrument model.")
                if metrics[key] != totals[category][key]:
                    raise ValueError("Invalid instrument model.")
            tp, fp, fn, tn = (metrics[key] for key in ("tp", "fp", "fn", "tn"))
            expected = {"precision": tp / (tp + fp) if tp + fp else None,
                        "recall": tp / (tp + fn) if tp + fn else None,
                        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
                        "false_positive_fraction": fp / (fp + tn) if fp + tn else None}
            for key in ("precision", "recall", "iou", "false_positive_fraction"):
                value = metrics[key]
                if value is not None and (type(value) not in (int, float) or not 0 <= value <= 1):
                    raise ValueError("Invalid instrument model.")
                if value != expected[key]:
                    raise ValueError("Invalid instrument model.")
        negative_rate = report.get("negative_false_positive_fraction")
        if negative_rate is not None and (type(negative_rate) not in (int, float)
                                          or not 0 <= negative_rate <= 1):
            raise ValueError("Invalid instrument model.")
        if negative_rate != (
                negative_fp / negative_pixels if negative_pixels else None):
            raise ValueError("Invalid instrument model.")
        if version == 2:
            support = report["support"]
            if (not isinstance(support, dict) or not isinstance(support.get("method"), str)
                    or not support["method"] or type(support.get("limit")) not in (int, float)
                    or support["limit"] != float(appearance_limit)
                    or type(support.get("training_groups")) is not int
                    or support["training_groups"] != len(np.unique(profile_groups))):
                raise ValueError("Invalid instrument model.")
            for category in ("clean", "degraded"):
                accepted = support[category]
                for key, value in support_totals[category].items():
                    if type(accepted[key]) is not int or accepted[key] != value:
                        raise ValueError("Invalid instrument model.")
                tp, fp, fn, tn = (accepted[key] for key in ("tp", "fp", "fn", "tn"))
                expected = {"precision": tp / (tp + fp) if tp + fp else None,
                            "recall": tp / (tp + fn) if tp + fn else None,
                            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
                            "false_positive_fraction": fp / (fp + tn) if fp + tn else None}
                for key, value in expected.items():
                    measured = accepted.get(key)
                    if measured is not None and type(measured) not in (int, float):
                        raise ValueError("Invalid instrument model.")
                    if measured != value:
                        raise ValueError("Invalid instrument model.")
    except (OSError, EOFError, BadZipFile, AttributeError, TypeError, ValueError, KeyError, OverflowError):
        raise ValueError("Invalid instrument model.") from None
    return model


def TrainInstrument(folder, output=None, seed=7, cancel=None):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Invalid training seed.")
    try:
        folder = Path(folder)
    except (TypeError, ValueError):
        raise ValueError("Invalid capture folder.") from None
    if not (folder / "images").is_dir() or not (folder / "labels").is_dir():
        raise ValueError("Invalid capture folder.")
    groups = {}
    seen_images = {}
    # groups matching physical snapshots before selecting any pixels or fitting a model
    for label_path in sorted((folder / "labels").glob("frame_*.npz")):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Training cancelled.")
        try:
            with np.load(label_path, allow_pickle=False) as labels:
                scenario = labels["scenario"]
                camera = labels["camera_configuration"]
                tool = labels["tool_configuration"]
                breathing = labels["breathing_displacement_mm"]
                mask = labels["semantic_mask"]
                class_ids = labels["class_ids"]
                class_names = labels["class_names"]
                if (scenario.shape != () or scenario.dtype.kind != "U" or not str(scenario)
                        or camera.shape != (3,) or tool.shape != (3,) or breathing.shape != ()
                        or any(value.dtype.kind not in "fiu" or not np.all(np.isfinite(value))
                               for value in (camera, tool, breathing))
                        or mask.dtype != np.uint8 or mask.ndim != 2 or min(mask.shape) < 16
                        or class_ids.ndim != 1 or class_ids.dtype.kind not in "iu"
                        or class_names.shape != class_ids.shape or class_names.dtype.kind != "U"
                        or len(np.unique(class_ids)) != len(class_ids)
                        or len(np.unique(class_names)) != len(class_names)
                        or np.any(class_ids < 0) or np.any(class_ids > 255)
                        or list(class_names).count("instrument") != 1
                        or int(class_ids[np.flatnonzero(class_names == "instrument")[0]]) != 1
                        or not np.all(np.isin(mask, class_ids))):
                    raise ValueError("Invalid training frame.")
                camera = camera.astype(float)
                tool = tool.astype(float)
                camera[camera == 0] = 0.0
                tool[tool == 0] = 0.0
                snapshot = [str(scenario), camera.tolist(), tool.tolist(),
                            float(breathing) if breathing != 0 else 0.0]
            group_id = hashlib.sha256(json.dumps(snapshot, separators=(",", ":")).encode()).hexdigest()
            group = groups.setdefault(group_id, [])
            seen = seen_images.setdefault(group_id, set())
            for variant, image_folder in (("clean", "images"), ("degraded", "degraded")):
                image_path = folder / image_folder / (label_path.stem + ".png")
                if variant == "degraded" and not image_path.exists():
                    continue
                if variant == "clean" and any(frame["variant"] == "clean" for frame in group):
                    continue
                with Image.open(image_path) as source:
                    if source.mode != "RGB" or source.size != (mask.shape[1], mask.shape[0]):
                        raise ValueError("Invalid training frame.")
                    pixels = np.asarray(source)
                    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                group.append({"frame_id": label_path.stem, "group_id": group_id,
                              "variant": variant, "image_path": image_path, "label_path": label_path,
                              "image_digest": digest})
        except (OSError, EOFError, BadZipFile, AttributeError, ValueError, TypeError, KeyError, IndexError, OverflowError):
            raise ValueError("Invalid training frame.") from None
    # also merges identical clean images whose metadata names a different snapshot
    merged = {}
    clean_groups = {}
    merged_images = {}
    for group_id, frames in groups.items():
        clean_digest = next(frame["image_digest"] for frame in frames if frame["variant"] == "clean")
        canonical = clean_groups.setdefault(clean_digest, group_id)
        combined = merged.setdefault(canonical, [])
        seen = merged_images.setdefault(canonical, set())
        for frame in frames:
            if frame["image_digest"] not in seen:
                frame["group_id"] = canonical
                combined.append(frame)
                seen.add(frame["image_digest"])
    groups = merged
    group_ids = sorted(groups)
    if len(group_ids) < 8:
        raise ValueError("Need eight distinct snapshot groups.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(group_ids))
    validation_count = max(1, int(np.ceil(len(group_ids) * 0.25)))
    validation_groups = sorted(group_ids[index] for index in order[:validation_count])
    train_groups = sorted(group_ids[index] for index in order[validation_count:])
    features = []
    targets = []
    # bounds sampling per group so repeated or augmented frames cannot dominate training
    for group_id in train_groups:
        frames = groups[group_id]
        quota, remainder = divmod(1000, len(frames))
        for frame_index, frame in enumerate(frames):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Training cancelled.")
            with Image.open(frame["image_path"]) as source:
                image = np.asarray(source)
            with np.load(frame["label_path"], allow_pickle=False) as labels:
                truth = labels["semantic_mask"].reshape(-1) == 1
            frame_features = ImageFeatures(image)
            for label in (False, True):
                positions = np.flatnonzero(truth == label)
                limit = quota + int(frame_index < remainder)
                selected = rng.choice(positions, min(limit, len(positions)), replace=False)
                if len(selected):
                    features.append(frame_features[selected])
                    targets.append(np.full(len(selected), label, dtype=float))
    features = np.concatenate(features).astype(float)
    targets = np.concatenate(targets)
    if not np.any(targets == 1) or not np.any(targets == 0):
        raise ValueError("Need instrument and background training pixels.")
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 0.01)
    features = np.column_stack((np.ones(len(features)), (features - mean) / scale))
    weights = np.zeros(13)
    first_moment = np.zeros(13)
    second_moment = np.zeros(13)
    # fits regularized logistic regression using Adam on training pixels only
    for iteration in range(1, 401):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Training cancelled.")
        scores = 1 / (1 + np.exp(-np.clip(features @ weights, -35, 35)))
        gradient = features.T @ (scores - targets) / len(targets)
        gradient[1:] += 0.001 * weights[1:]
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
        weights -= 0.03 * (first_moment / (1 - 0.9 ** iteration)) / (
            np.sqrt(second_moment / (1 - 0.999 ** iteration)) + 1e-8)
    model = {"version": np.array(1, dtype=np.int64), "feature_names": _FEATURE_NAMES.copy(),
             "weights": weights, "mean": mean, "scale": scale, "threshold": np.array(0.5),
             "train_groups": np.array(train_groups), "validation_groups": np.array(validation_groups),
             "report_json": np.array("{}")}
    # selects a threshold on complete training frames at their actual pixel prevalence
    thresholds = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995])
    threshold_counts = np.zeros((len(thresholds), 3), dtype=np.int64)
    for group_id in train_groups:
        for frame in groups[group_id]:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Training cancelled.")
            with Image.open(frame["image_path"]) as source:
                image = np.asarray(source)
            with np.load(frame["label_path"], allow_pickle=False) as labels:
                truth = (labels["semantic_mask"].reshape(-1) == 1)[:, None]
            predicted = PredictInstrument(image, model).reshape(-1, 1) >= thresholds
            threshold_counts[:, 0] += np.count_nonzero(predicted & truth, axis=0)
            threshold_counts[:, 1] += np.count_nonzero(predicted & ~truth, axis=0)
            threshold_counts[:, 2] += np.count_nonzero(~predicted & truth, axis=0)
    training_iou = threshold_counts[:, 0] / np.maximum(threshold_counts.sum(axis=1), 1)
    model["threshold"] = np.array(thresholds[np.argmax(training_iou)])
    appearance_images = []
    appearance_groups = []
    for group_id in train_groups:
        for frame in groups[group_id]:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Training cancelled.")
            appearance_images.append(frame["image_path"])
            appearance_groups.append(group_id)
    model.update(FitAppearance(appearance_images, appearance_groups))
    model["version"] = np.array(2, dtype=np.int64)
    report = {"version": 1, "seed": int(seed), "training_groups": len(train_groups),
              "validation_groups": len(validation_groups), "training_pixels": len(targets),
              "training_frames": sum(len(groups[group]) for group in train_groups),
              "validation_frames": sum(len(groups[group]) for group in validation_groups),
              "features": _FEATURE_NAMES.tolist(), "threshold": float(model["threshold"]),
              "threshold_rule": "maximum aggregate training-frame IoU among eight fixed thresholds",
              "frames": [],
              "note": "Experimental image-only pixel baseline; scores are not calibrated safety probabilities."}
    for category in ("clean", "degraded"):
        report[category] = {"frames": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    report["support"] = {"method": "95th percentile leave-one-group-out RGB histogram distance",
                         "limit": float(model["appearance_limit"]),
                         "training_groups": len(np.unique(model["appearance_groups"]))}
    for category in ("clean", "degraded"):
        report["support"][category] = {"frames": 0, "accepted": 0, "withheld": 0,
                                       "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    negative_fp = negative_pixels = 0
    for group_id in group_ids:
        split = "validation" if group_id in validation_groups else "train"
        for frame in groups[group_id]:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Training cancelled.")
            with Image.open(frame["image_path"]) as source:
                image = np.asarray(source)
            with np.load(frame["label_path"], allow_pickle=False) as labels:
                mask = labels["semantic_mask"]
            metrics = EvaluateInstrument(image, mask, model)
            assessment = AssessPrediction(image, model)
            report["frames"].append({"frame_id": frame["frame_id"], "group_id": group_id,
                                     "split": split, "variant": frame["variant"], **metrics,
                                     "supported": assessment["supported"],
                                     "reasons": list(assessment["reasons"]),
                                     "distance": assessment["distance"]})
            if split == "validation":
                aggregate = report[frame["variant"]]
                aggregate["frames"] += 1
                for key in ("tp", "fp", "fn", "tn"):
                    aggregate[key] += metrics[key]
                support = report["support"][frame["variant"]]
                support["frames"] += 1
                if assessment["supported"]:
                    support["accepted"] += 1
                    for key in ("tp", "fp", "fn", "tn"):
                        support[key] += metrics[key]
                else:
                    support["withheld"] += 1
                if metrics["tp"] + metrics["fn"] == 0:
                    negative_fp += metrics["fp"]
                    negative_pixels += metrics["fp"] + metrics["tn"]
    for category in ("clean", "degraded"):
        aggregate = report[category]
        tp, fp, fn, tn = (aggregate[key] for key in ("tp", "fp", "fn", "tn"))
        aggregate.update({"precision": tp / (tp + fp) if tp + fp else None,
                          "recall": tp / (tp + fn) if tp + fn else None,
                          "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
                          "false_positive_fraction": fp / (fp + tn) if fp + tn else None})
        support = report["support"][category]
        tp, fp, fn, tn = (support[key] for key in ("tp", "fp", "fn", "tn"))
        support.update({"precision": tp / (tp + fp) if tp + fp else None,
                        "recall": tp / (tp + fn) if tp + fn else None,
                        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
                        "false_positive_fraction": fp / (fp + tn) if fp + tn else None})
    report["negative_false_positive_fraction"] = negative_fp / negative_pixels if negative_pixels else None
    model["report_json"] = np.array(json.dumps(report, allow_nan=False, separators=(",", ":")))
    if output is not None:
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Training cancelled.")
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            # replaces the model only after a complete numeric archive has been validated
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix="." + output.name,
                                             suffix=".tmp", delete=False) as saved:
                temporary = Path(saved.name)
                np.savez_compressed(saved, **model)
            LoadInstrument(temporary)
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Training cancelled.")
            temporary.replace(output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return model, report
