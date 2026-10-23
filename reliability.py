import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from PIL import Image

from perception import AssessPrediction, PredictInstrument
from safesight import AssessView, DegradeView


def ReliabilityMetrics(rows, policy="appearance", cutoff=None):
    if policy not in ("always", "quality", "appearance"):
        raise ValueError("Invalid reliability policy.")
    if cutoff is not None and (isinstance(cutoff, (bool, np.bool_))
            or not isinstance(cutoff, (int, float, np.integer, np.floating))
            or not np.isfinite(cutoff) or not 0 <= cutoff <= 1):
        raise ValueError("Invalid appearance cutoff.")
    totals = {key: 0 for key in ("frames", "accepted", "withheld", "tp", "fp", "fn", "tn",
              "positive_frames", "positive_accepted", "negative_frames", "negative_accepted",
              "negative_false_detections", "withheld_positive_frames")}
    error_sum = 0.0
    try:
        for row in rows:
            counts = [row[key] for key in ("tp", "fp", "fn", "tn")]
            if (any(isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < 0 for value in counts)
                    or sum(counts) == 0 or not isinstance(row["quality_passed"], (bool, np.bool_))
                    or any(isinstance(row[key], (bool, np.bool_))
                           or not isinstance(row[key], (int, float, np.integer, np.floating))
                           or not np.isfinite(row[key]) or not 0 <= row[key] <= 1
                           for key in ("distance", "appearance_limit"))):
                raise ValueError("Invalid reliability rows.")
            tp, fp, fn, tn = map(int, counts)
            shown = (policy == "always" or bool(row["quality_passed"])
                     and (policy == "quality" or row["distance"] <= (
                         row["appearance_limit"] if cutoff is None else cutoff)))
            positive = tp + fn > 0
            totals["frames"] += 1
            totals["positive_frames" if positive else "negative_frames"] += 1
            totals["accepted" if shown else "withheld"] += 1
            if shown:
                totals["positive_accepted" if positive else "negative_accepted"] += 1
                for key, count in zip(("tp", "fp", "fn", "tn"), counts):
                    totals[key] += int(count)
                totals["negative_false_detections"] += int(not positive and fp > 0)
                # treats an empty prediction on a genuinely empty view as correct
                union = tp + fp + fn
                error_sum += (fp + fn) / union if union else 0.0
            elif positive:
                totals["withheld_positive_frames"] += 1
    except (KeyError, TypeError, OverflowError):
        raise ValueError("Invalid reliability rows.") from None
    tp, fp, fn, tn = (totals[key] for key in ("tp", "fp", "fn", "tn"))
    totals.update(
        coverage=totals["accepted"] / totals["frames"] if totals["frames"] else None,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        iou=tp / (tp + fp + fn) if tp + fp + fn else None,
        missed_tool_fraction=fn / (tp + fn) if tp + fn else None,
        false_positive_fraction=fp / (fp + tn) if fp + tn else None,
        mean_mask_error=error_sum / totals["accepted"] if totals["accepted"] else None,
        negative_false_detection_fraction=(totals["negative_false_detections"] / totals["negative_accepted"]
                                           if totals["negative_accepted"] else None))
    return totals


def EvaluateReliability(folder, model, cancel=None):
    try:
        folder = Path(folder)
        version = np.asarray(model["version"])
        if version.shape != () or version.dtype.kind not in "iu" or int(version) not in (2, 3):
            raise ValueError("Need an appearance-reference model.")
        # validates the frozen model before reading any labels or generating effects
        PredictInstrument(np.zeros((16, 16, 3), dtype=np.uint8), model)
        AssessPrediction(np.zeros((16, 16, 3), dtype=np.uint8), model)
        training = set(map(str, model["train_groups"]))
        heldout = set(map(str, model["validation_groups"]))
        calibration = set(map(str, model.get("calibration_groups", [])))
        expected_groups = training | heldout | calibration
        if (not training or not heldout or training & heldout or training & calibration or heldout & calibration
                or any(len(group) != 64 or any(char not in "0123456789abcdef" for char in group)
                       for group in expected_groups)):
            raise ValueError("Invalid model provenance.")
        fitted_report = json.loads(str(model["report_json"]))
        recorded = fitted_report["frames"]
        if not isinstance(recorded, list) or not recorded:
            raise ValueError("Invalid model provenance.")
    except (KeyError, TypeError, OverflowError, json.JSONDecodeError):
        raise ValueError("Invalid reliability inputs.") from None
    if cancel is not None and cancel.is_set():
        raise InterruptedError("Reliability evaluation cancelled.")
    sources = [("pool", folder)]
    if (folder / "calibration").exists():
        sources.append(("calibration", folder / "calibration"))
    if (folder / "evaluation").exists():
        sources.append(("evaluation", folder / "evaluation"))
    references = {}
    fresh = []
    reference_digests = set()
    reference_physical = set()
    # retains only paths and hashes; full images and masks are read one at a time
    for source_name, source_folder in sources:
        if not (source_folder / "images").is_dir() or not (source_folder / "labels").is_dir():
            raise ValueError("Invalid capture folder.")
        paths = sorted((source_folder / "labels").glob("frame_*.npz"))
        if not paths or len(paths) > 2000:
            raise ValueError("Need one to 2000 captures per folder.")
        for label_path in paths:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Reliability evaluation cancelled.")
            try:
                with np.load(label_path, allow_pickle=False) as labels:
                    scenario, camera, tool, breathing, mask, class_ids, class_names = (
                        labels[key] for key in ("scenario", "camera_configuration", "tool_configuration",
                                               "breathing_displacement_mm", "semantic_mask", "class_ids", "class_names"))
                    if (scenario.shape != () or scenario.dtype.kind != "U" or not str(scenario)
                            or camera.shape != (3,) or tool.shape != (3,) or breathing.shape != ()
                            or any(value.dtype.kind not in "fiu" or not np.all(np.isfinite(value))
                                   for value in (camera, tool, breathing))
                            or mask.dtype != np.uint8 or mask.ndim != 2 or min(mask.shape) < 16
                            or max(mask.shape) > 4096 or class_ids.ndim != 1 or class_ids.dtype.kind not in "iu"
                            or class_names.shape != class_ids.shape or class_names.dtype.kind != "U"
                            or len(np.unique(class_ids)) != len(class_ids)
                            or len(np.unique(class_names)) != len(class_names)
                            or np.any(class_ids < 0) or np.any(class_ids > 255)
                            or list(class_names).count("instrument") != 1
                            or int(class_ids[np.flatnonzero(class_names == "instrument")[0]]) != 1
                            or not np.all(np.isin(mask, class_ids))):
                        raise ValueError("Invalid evaluation capture.")
                    camera, tool = camera.astype(float), tool.astype(float)
                    camera[camera == 0], tool[tool == 0] = 0.0, 0.0
                    snapshot = [str(scenario), camera.tolist(), tool.tolist(),
                                float(breathing) if breathing != 0 else 0.0]
                    physical = hashlib.sha256(json.dumps(snapshot, separators=(",", ":")).encode()).hexdigest()
                    projected = occluded = None
                    if "tool_projected_pixels" in labels or "tool_occluded_pixels" in labels:
                        projected, occluded = (labels[key] for key in ("tool_projected_pixels", "tool_occluded_pixels"))
                        if any(value.shape != () or value.dtype.kind not in "iu" for value in (projected, occluded)):
                            raise ValueError("Invalid tool visibility labels.")
                        projected, occluded = int(projected), int(occluded)
                        if not 0 <= occluded <= projected <= mask.size:
                            raise ValueError("Invalid tool visibility labels.")
                    size = (mask.shape[1], mask.shape[0])
                frame_id = ("" if source_name == "pool" else source_name + "/") + label_path.stem
                record = {"frame_id": frame_id, "physical": physical, "label_path": label_path,
                          "image_path": source_folder / "images" / (label_path.stem + ".png"),
                          "source": source_name, "digests": {}, "size": size,
                          "tool_projected_pixels": projected, "tool_occluded_pixels": occluded}
                for variant, image_folder in (("clean", "images"), ("degraded", "degraded")):
                    image_path = source_folder / image_folder / (label_path.stem + ".png")
                    if variant == "degraded" and not image_path.exists():
                        continue
                    with Image.open(image_path) as source:
                        if source.mode != "RGB" or source.size != size:
                            raise ValueError("Invalid evaluation capture.")
                        digest = hashlib.sha256(np.asarray(source).tobytes()).hexdigest()
                    record["digests"][variant] = digest
                if source_name == "evaluation":
                    fresh.append(record)
                else:
                    references[frame_id] = record
                    reference_digests.update(record["digests"].values())
                    reference_physical.add(physical)
            except (OSError, EOFError, BadZipFile, AttributeError, ValueError, TypeError,
                    KeyError, IndexError, OverflowError):
                raise ValueError("Invalid evaluation capture.") from None
    physical_clean, clean_physical = {}, {}
    for record in references.values():
        physical_clean.setdefault(record["physical"], set()).add(record["digests"]["clean"])
        clean_physical.setdefault(record["digests"]["clean"], set()).add(record["physical"])
    components = {}
    visited_clean = set()
    for physical in reference_physical:
        if physical in components:
            continue
        components[physical] = physical
        pending = [physical]
        # connects canonical configurations only through clean-RGB recapture aliases
        for alias in pending:
            for digest in physical_clean[alias]:
                if digest in visited_clean:
                    continue
                visited_clean.add(digest)
                for linked in clean_physical[digest]:
                    if linked not in components:
                        components[linked] = physical
                        pending.append(linked)
    physical_splits = {}
    digest_splits = {}
    selected = {}
    represented = set()
    # matches the model's recorded frames, then follows physical and exact-RGB aliases
    for entry in recorded:
        try:
            group, split, variant = entry["group_id"], entry["split"], entry["variant"]
            expected_split = ("train" if group in training else "validation" if group in heldout
                              else "calibration" if group in calibration else None)
            if expected_split is None or split != expected_split or variant not in ("clean", "degraded"):
                raise ValueError("Invalid model provenance.")
            record = references[entry["frame_id"]]
            digest = record["digests"][variant]
            if group not in components or components[record["physical"]] != components[group]:
                raise ValueError("Capture provenance differs from the model.")
            if (physical_splits.get(record["physical"], split) != split
                    or digest_splits.get(digest, split) != split):
                raise ValueError("Capture groups overlap.")
            physical_splits[record["physical"]] = split
            digest_splits[digest] = split
            represented.add(group)
            if group in heldout and variant == "clean":
                selected.setdefault(group, record)
        except (KeyError, TypeError):
            raise ValueError("Choose the complete original capture folder.") from None
    if represented != expected_groups or set(selected) != heldout:
        raise ValueError("Choose the complete original capture folder.")
    physical_digests, digest_physical = {}, {}
    for record in references.values():
        physical_digests.setdefault(record["physical"], set()).update(record["digests"].values())
        for digest in record["digests"].values():
            digest_physical.setdefault(digest, set()).add(record["physical"])
    # follows complete alias chains without rescanning every capture on each step
    pending = list(physical_splits)
    visited_digests = set()
    for physical in pending:
        split = physical_splits[physical]
        for digest in physical_digests[physical]:
            if digest_splits.get(digest, split) != split:
                raise ValueError("Capture groups overlap.")
            digest_splits[digest] = split
            if digest in visited_digests:
                continue
            visited_digests.add(digest)
            for alias in digest_physical[digest]:
                if physical_splits.get(alias, split) != split:
                    raise ValueError("Capture groups overlap.")
                if alias not in physical_splits:
                    physical_splits[alias] = split
                    pending.append(alias)
    fresh_groups = []
    for record in fresh:
        if (record["physical"] in reference_physical
                or set(record["digests"].values()) & reference_digests):
            raise ValueError("Evaluation captures overlap the original captures.")
        # joins all physical/clean-RGB aliases, including chains through recaptures
        matches = [index for index, group in enumerate(fresh_groups)
                   if record["physical"] in group["physical"] or record["digests"]["clean"] in group["digests"]]
        if not matches:
            fresh_groups.append({"physical": {record["physical"]}, "digests": {record["digests"]["clean"]},
                                 "record": record})
        else:
            group = fresh_groups[matches[0]]
            group["physical"].add(record["physical"])
            group["digests"].add(record["digests"]["clean"])
            for index in reversed(matches[1:]):
                other = fresh_groups.pop(index)
                group["physical"].update(other["physical"])
                group["digests"].update(other["digests"])
    for group in fresh_groups:
        record = group["record"]
        selected[record["physical"]] = record
    if len(selected) > 500:
        raise ValueError("Use at most 500 evaluation snapshots.")
    cases = [{"name": "clean", "family": "clean", "level": 0.0,
              "blur": 0.0, "exposure": 0.0, "haze": 0.0}]
    for family, levels in (("blur", (0.25, 0.5, 1)),
                           ("exposure", (-1, -2, -4, 1, 2, 4)),
                           ("haze", (0.25, 0.5, 0.75, 1))):
        for level in levels:
            cases.append({"name": f"{family}_{level:g}", "family": family, "level": float(level),
                          "blur": float(level) if family == "blur" else 0.0,
                          "exposure": float(level) if family == "exposure" else 0.0,
                          "haze": float(level) if family == "haze" else 0.0})
    for name, blur, exposure, haze in (("moderate", 0.35, -1.0, 0.2), ("severe", 0.75, -2.0, 0.5)):
        cases.append({"name": "combined_" + name, "family": "combined", "level": 1.0 if name == "moderate" else 2.0,
                      "blur": blur, "exposure": exposure, "haze": haze})
    rows = []
    appearance_limit = float(model["appearance_limit"])
    for group, record in sorted(selected.items()):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Reliability evaluation cancelled.")
        with Image.open(record["image_path"]) as source:
            image = np.array(source)
        with np.load(record["label_path"], allow_pickle=False) as labels:
            truth = labels["semantic_mask"] == 1
        for case in cases:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Reliability evaluation cancelled.")
            view, view_truth = image, truth
            if case["family"] != "clean":
                # matches the live image-test path before applying deterministic effects
                if image.shape[1] > 640:
                    size = (640, max(16, round(image.shape[0] * 640 / image.shape[1])))
                    view = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
                    view_truth = np.asarray(Image.fromarray(truth).resize(size, Image.Resampling.NEAREST))
                view = DegradeView(view, blur=case["blur"], exposure=case["exposure"], haze=case["haze"])
            height, width = view.shape[:2]
            vertical, horizontal = min(height // 10, (height - 16) // 2), min(width // 10, (width - 16) // 2)
            cropped = view[vertical:height - vertical, horizontal:width - horizontal]
            cropped_truth = view_truth[vertical:height - vertical, horizontal:width - horizontal]
            quality = AssessView(cropped)
            gate = AssessPrediction(view, model, quality=quality)
            height, width = cropped.shape[:2]
            sample = cropped
            if max(height, width) > 320:
                scale = 320 / max(height, width)
                sample = np.asarray(Image.fromarray(cropped).resize(
                    (max(16, round(width * scale)), max(16, round(height * scale))), Image.Resampling.LANCZOS))
            # scores every case once so all three policies share exactly the same candidate mask
            predicted = PredictInstrument(sample, model) >= float(model["threshold"])
            predicted = np.asarray(Image.fromarray(predicted).resize((width, height), Image.Resampling.NEAREST))
            projected, occluded = record["tool_projected_pixels"], record["tool_occluded_pixels"]
            visibility = ("unmeasured" if projected is None else "out_of_frame" if projected == 0
                          else "occluded" if occluded > 0 else "unoccluded")
            rows.append({"frame_id": record["frame_id"], "group_id": group,
                         "source": "evaluation" if record["source"] == "evaluation" else "heldout",
                         "case": case["name"], "quality_passed": not quality["issues"],
                         "distance": gate["distance"], "appearance_limit": appearance_limit,
                         "reasons": list(gate["reasons"]), "visibility": visibility,
                         "tool_projected_pixels": projected, "tool_occluded_pixels": occluded,
                         "tp": int(np.count_nonzero(predicted & cropped_truth)),
                         "fp": int(np.count_nonzero(predicted & ~cropped_truth)),
                         "fn": int(np.count_nonzero(~predicted & cropped_truth)),
                         "tn": int(np.count_nonzero(~predicted & ~cropped_truth))})
    policies = ("always", "quality", "appearance")
    report = {"version": 1, "groups": len(selected), "snapshots": len(selected),
              "heldout_groups": len(heldout), "evaluation_groups": len(fresh_groups),
              "training_groups": len(training), "calibration_groups": len(calibration),
              "cases": cases, "appearance_limit": appearance_limit, "threshold": float(model["threshold"]),
              "model_digest": hashlib.sha256(b"".join(key.encode() + np.asarray(model[key]).tobytes()
                                                     for key in sorted(model))).hexdigest(),
              "policies": {policy: ReliabilityMetrics(rows, policy) for policy in policies},
              "by_case": [], "by_source": {}, "by_visibility": {}, "curve": [], "frames": rows,
              "note": "Fixed simulator test cases; repeated effects are correlated within snapshots. "
                      "Labels score masks only. No model, live gate or threshold is changed. "
                      "Saved provenance identifies configurations and aliases, not original pixel hashes; "
                      "unchanged RGB at the same configuration cannot be verified. "
                      "The descriptive cutoff curve is not a threshold-selection procedure. "
                      "Error is frame-averaged (FP+FN)/(TP+FP+FN); correct empty masks have zero error. "
                      "Quality and appearance checks use RGB only; simulator labels do not establish clinical validity."}
    for case in cases:
        subset = [row for row in rows if row["case"] == case["name"]]
        report["by_case"].append({"name": case["name"], "policies": {
            policy: ReliabilityMetrics(subset, policy) for policy in policies}})
    for key, categories in (("source", ("heldout", "evaluation")),
                            ("visibility", ("unmeasured", "out_of_frame", "occluded", "unoccluded"))):
        for category in categories:
            subset = [row for row in rows if row[key] == category]
            if subset:
                report["by_" + key][category] = {policy: ReliabilityMetrics(subset, policy) for policy in policies}
    for multiplier in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, None):
        cutoff = min(1.0, appearance_limit * multiplier) if multiplier is not None else 1.0
        report["curve"].append({"multiplier": multiplier, "cutoff": cutoff,
                                "current": multiplier == 1.0,
                                **ReliabilityMetrics(rows, "appearance", cutoff=cutoff)})
    if cancel is not None and cancel.is_set():
        raise InterruptedError("Reliability evaluation cancelled.")
    return report
