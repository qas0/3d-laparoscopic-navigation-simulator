import numpy as np

from astar import find_path
from feedback import get_target_feedback
from geometry import movement_is_clear, shaft_clearance, tip_position
from kalman import KalmanFilter
from motion import GetRespiratoryMotion
from scenarios import SCENARIOS
from sensors import ApplyBias, ApplyDropout, measure_position


def GetUncertaintyCoverage(error, position_covariance):
    try:
        error = np.asarray(error, dtype=float)
        covariance = np.asarray(position_covariance, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid error or covariance.") from None
    if (error.shape != (3,) or covariance.shape != (3, 3)
            or not np.all(np.isfinite(error)) or not np.all(np.isfinite(covariance))):
        raise ValueError("Invalid error or covariance.")
    tolerance = 3 * np.finfo(float).eps * np.max(np.abs(covariance))
    if not np.allclose(covariance, covariance.T, rtol=0, atol=tolerance):
        raise ValueError("Expected symmetric covariance.")
    covariance = 0.5 * covariance + 0.5 * covariance.T
    eigenvalues, directions = np.linalg.eigh(covariance)
    tolerance = 3 * np.finfo(float).eps * np.max(np.abs(eigenvalues))
    if np.min(eigenvalues) < -tolerance:
        raise ValueError("Expected nonnegative covariance.")
    supported = eigenvalues > tolerance
    rank = int(np.count_nonzero(supported))
    thresholds = (3.841, 5.991, 7.815)
    if rank == 3:
        score = float(error @ np.linalg.solve(covariance, error))
    else:
        projected = directions.T @ error
        # allows 1e-9 mm rounding error in directions with zero reported uncertainty
        if np.linalg.norm(projected[~supported]) > 1e-9:
            return False
        if rank == 0:
            return True
        score = float(np.sum(projected[supported] ** 2 / eigenvalues[supported]))
    return bool(score <= thresholds[rank - 1])


def RunTrial(scenario="Direct insertion", mode="conventional", seed=0, *,
             tracking="filtered", tool_noise=1.0, target_noise=2.0,
             tool_dropout=0.0, target_dropout=0.0,
             tool_bias=(0.0, 0.0, 0.0), target_bias=(0.0, 0.0, 0.0),
             tool_drift=(0.0, 0.0, 0.0), target_drift=(0.0, 0.0, 0.0),
             motion_amplitude=0.0, motion_frequency=0.2, timestep=0.02,
             duration=20.0, warmup=1.0, proximity_weight=1.0,
             uncertainty_scale=1.0, max_expansions=10000, planning_time_limit=5.0,
             cancel_event=None):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario.")
    if mode not in ("conventional", "prox_aware", "uncert_aware"):
        raise ValueError("Unknown planning mode.")
    if tracking not in ("ideal", "raw", "filtered"):
        raise ValueError("Unknown tracking source.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Invalid seed.")
    if (isinstance(max_expansions, (bool, np.bool_))
            or not isinstance(max_expansions, (int, np.integer)) or max_expansions < 1):
        raise ValueError("Invalid search budget.")

    try:
        scalars = np.asarray([tool_noise, target_noise, tool_dropout, target_dropout,
                              motion_amplitude, motion_frequency, timestep, duration,
                              warmup, proximity_weight, uncertainty_scale,
                              planning_time_limit], dtype=float)
        vectors = np.asarray([tool_bias, target_bias, tool_drift, target_drift], dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid experiment settings.") from None
    if (scalars.shape != (12,) or not np.all(np.isfinite(scalars))
            or np.any(scalars < 0) or vectors.shape != (4, 3)
            or not np.all(np.isfinite(vectors))):
        raise ValueError("Invalid experiment settings.")
    (tool_noise, target_noise, tool_dropout, target_dropout, motion_amplitude,
     motion_frequency, timestep, duration, warmup, proximity_weight,
     uncertainty_scale, planning_time_limit) = map(float, scalars)
    tool_bias, target_bias, tool_drift, target_drift = vectors
    if (max(tool_dropout, target_dropout) > 1 or not 0 < timestep <= 0.1
            or duration <= warmup or planning_time_limit <= 0):
        raise ValueError("Invalid probability or timing.")
    sample_frames = round(0.1 / timestep)
    if (not np.isclose(sample_frames * timestep, 0.1, rtol=0, atol=1e-12)
            or not np.isclose(round(warmup / 0.1) * 0.1, warmup, rtol=0, atol=1e-12)):
        raise ValueError("Align steps + warmup with 10 Hz sampling.")

    settings = {
        "tracking": tracking, "tool_noise": tool_noise, "target_noise": target_noise,
        "tool_dropout": tool_dropout, "target_dropout": target_dropout,
        "tool_bias": tuple(tool_bias), "target_bias": tuple(target_bias),
        "tool_drift": tuple(tool_drift), "target_drift": tuple(target_drift),
        "motion_amplitude": motion_amplitude, "motion_frequency": motion_frequency,
        "timestep": timestep, "duration": duration, "warmup": warmup,
        "proximity_weight": proximity_weight, "uncertainty_scale": uncertainty_scale,
        "max_expansions": int(max_expansions), "planning_time_limit": planning_time_limit,
    }
    scene = SCENARIOS[scenario]
    configuration = scene["start"].copy()
    port = np.zeros(3)
    target_origin = tip_position(port, *scene["target_configuration"])
    structure_centre, structure_radius = scene["structure_centre"], scene["structure_radius"]
    tool_radius, required_clearance, target_tolerance = 1.5, 2.0, 2.0
    lower_limits = np.array([np.deg2rad(-45), np.deg2rad(-35), 10.0])
    upper_limits = np.array([np.deg2rad(45), np.deg2rad(35), 110.0])
    speeds = np.array([np.deg2rad(20), np.deg2rad(20), 15.0])

    # repeats Gaussian + dropout draws at the same sample times across methods
    noise_generator = np.random.default_rng(seed)
    dropout_generators = [np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])]
    filters = [None, None]
    noise = [tool_noise, target_noise]
    dropout = [tool_dropout, target_dropout]
    biases, drift_rates = [tool_bias, target_bias], [tool_drift, target_drift]
    acceleration = [20.0, 1.0]
    error_sums = dict.fromkeys(("tool_raw", "target_raw", "tool_filtered", "target_filtered"), 0.0)
    counts = dict.fromkeys(error_sums, 0)
    covariance_sums = {"tool": 0.0, "target": 0.0}
    covered = {"tool": 0, "target": 0}
    dropped = {"tool": 0, "target": 0}
    sensor_samples = 0
    plan, planned_target, planned_at = None, None, None
    route_index, route_finished_at, arrival_since = 1, None, None
    elapsed_time, previous_time, travel = 0.0, 0.0, 0.0
    minimum_clearance = shaft_clearance(port, tip_position(port, *configuration),
                                        tool_radius, structure_centre, structure_radius)
    status, reported_arrival = "timeout", False

    for frame in range(int(np.ceil(duration / timestep)) + 1):
        if cancel_event is not None and cancel_event.is_set():
            status = "cancelled"
            break
        elapsed_time = min(frame * timestep, duration)
        step = elapsed_time - previous_time
        previous_time = elapsed_time

        if plan is not None and route_index < len(plan["path"]):
            remaining = plan["path"][route_index] - configuration
            movement_time = float(np.max(np.abs(remaining) / speeds))
            fraction = min(1.0, step / max(movement_time, 1e-12))
            proposed = configuration + fraction * remaining
            clearances = []
            if not movement_is_clear(port, configuration, proposed, tool_radius,
                                     structure_centre, structure_radius,
                                     plan["required_clearance"], clearance_samples=clearances):
                status = "movement_blocked"
                break
            change = np.abs(proposed - configuration)
            # sums exact tip travel along each single-axis A* movement
            travel += float(change[2] + configuration[2] * (
                change[1] + np.cos(configuration[1]) * change[0]))
            if np.any(change[:2]):
                bound = change[2] + max(configuration[2], proposed[2]) * sum(change[:2])
                edge_clearance = min(clearances) - bound / (2 * len(clearances))
            else:
                longest_tip = tip_position(port, proposed[0], proposed[1],
                                           max(configuration[2], proposed[2]))
                edge_clearance = shaft_clearance(port, longest_tip, tool_radius,
                                                  structure_centre, structure_radius)
            minimum_clearance = min(minimum_clearance, edge_clearance)
            configuration = proposed
            if fraction == 1.0:
                route_index += 1
                if route_index == len(plan["path"]):
                    route_finished_at = elapsed_time

        tool_position = tip_position(port, *configuration)
        displacement, _ = GetRespiratoryMotion(motion_amplitude, motion_frequency, elapsed_time)
        target_position = target_origin + np.array([0.0, 0.0, displacement])
        # samples at fixed 10 Hz ticks, including the initial observation
        if frame % sample_frames == 0 and np.isclose(
                elapsed_time, frame * timestep, rtol=0, atol=1e-12):
            observations = []
            for index, (name, position) in enumerate(zip(("tool", "target"),
                                                        (tool_position, target_position))):
                biased = ApplyBias(position, biases[index], drift_rates[index], elapsed_time)
                measured = measure_position(biased, noise[index], noise_generator)
                observed = ApplyDropout(measured, dropout[index], dropout_generators[index])
                observations.append(observed)
                if filters[index] is not None:
                    filters[index].predict(0.1)
                    if observed is not None:
                        filters[index].update(observed)
                elif observed is not None:
                    filters[index] = KalmanFilter(observed, noise[index], acceleration[index])
                if observed is None:
                    dropped[name] += 1
                else:
                    error_sums[name + "_raw"] += float(np.sum((observed - position) ** 2))
                    counts[name + "_raw"] += 1
                if filters[index] is not None:
                    error = filters[index].state[:3] - position
                    covariance = filters[index].covariance[:3, :3]
                    error_sums[name + "_filtered"] += float(np.sum(error ** 2))
                    counts[name + "_filtered"] += 1
                    # matches coverage + predicted uncertainty to the filtered RMSE samples
                    covariance_sums[name] += float(np.trace(covariance))
                    covered[name] += GetUncertaintyCoverage(error, covariance)
            sensor_samples += 1

            available = (tracking == "ideal" and mode != "uncert_aware") or (
                observations[0] is not None and (tracking == "ideal" or observations[1] is not None))
            if elapsed_time >= warmup - 1e-9:
                if not available:
                    status = "tracking_unavailable" if plan is None else "tracking_lost"
                    break
                if tracking == "ideal":
                    selected_tool, selected_target = tool_position, target_position
                elif tracking == "raw":
                    selected_tool, selected_target = observations
                else:
                    selected_tool, selected_target = [tracker.state[:3] for tracker in filters]

                if plan is None:
                    planned_at, planned_target = elapsed_time, selected_target.copy()
                    tool_offset, tool_covariance = None, None
                    if mode == "uncert_aware":
                        tool_offset = filters[0].state[:3] - tool_position
                        tool_covariance = filters[0].covariance[:3, :3].copy()
                    # pauses the simulation clock during planning + keeps the route frozen
                    plan = find_path(configuration, planned_target, port, tool_radius,
                                     structure_centre, structure_radius, lower_limits, upper_limits,
                                     required_clearance, target_tolerance,
                                     max_expansions=max_expansions, time_limit=planning_time_limit,
                                     cancel_event=cancel_event,
                                     include_target=True, mode=mode, proximity_weight=proximity_weight,
                                     tool_offset=tool_offset, tool_covariance=tool_covariance,
                                     uncertainty_scale=uncertainty_scale)
                    if plan["status"] != "found":
                        status = plan["status"]
                        break
                    if len(plan["path"]) == 1:
                        route_finished_at = elapsed_time

                feedback = get_target_feedback(selected_tool, selected_target, target_tolerance)
                if feedback["arrived"]:
                    if arrival_since is None:
                        arrival_since = elapsed_time
                    if elapsed_time - arrival_since >= 0.5 - 1e-9:
                        reported_arrival = True
                        break
                else:
                    arrival_since = None

            if route_finished_at is not None and elapsed_time - route_finished_at >= 3.0 - 1e-9:
                status = "not_confirmed"
                break

    # evaluates the final true position independently of the selected feedback
    tool_position = tip_position(port, *configuration)
    displacement, _ = GetRespiratoryMotion(motion_amplitude, motion_frequency, elapsed_time)
    target_position = target_origin + np.array([0.0, 0.0, displacement])
    actual = get_target_feedback(tool_position, target_position, target_tolerance)
    success = reported_arrival and actual["arrived"]
    false_arrival = reported_arrival and not actual["arrived"]
    if reported_arrival:
        status = "confirmed_arrival" if success else "false_arrival"
    rmse = {name: float(np.sqrt(error_sums[name] / count)) if count else None
            for name, count in counts.items()}
    predicted_rms, coverage = {}, {}
    for name in ("tool", "target"):
        count = counts[name + "_filtered"]
        predicted_rms[name] = float(np.sqrt(covariance_sums[name] / count)) if count else None
        coverage[name] = 100 * covered[name] / count if count else None
    return {
        "scenario": scenario, "mode": mode, "seed": int(seed), "tracking": tracking,
        "settings": settings, "status": status, "success": bool(success),
        "false_arrival": bool(false_arrival), "reported_arrival": reported_arrival,
        "actual_arrived": actual["arrived"], "actual_error": actual["distance"],
        "elapsed_time": elapsed_time,
        "completion_time": elapsed_time - planned_at if success else None,
        "travel": travel, "minimum_clearance": float(minimum_clearance),
        "plan": plan, "planned_target": planned_target, "true_target": target_position,
        "configuration": configuration, "sensor_samples": sensor_samples,
        "dropped": dropped, "rmse": rmse, "samples": counts,
        "predicted_rms": predicted_rms, "coverage": coverage,
    }


def CompareMethods(scenario="Direct insertion", seeds=range(10), *,
                   cancel_event=None, progress=None, **settings):
    seeds = list(seeds)
    if (not seeds or any(isinstance(seed, (bool, np.bool_))
                         or not isinstance(seed, (int, np.integer)) or seed < 0 for seed in seeds)):
        raise ValueError("Expected nonnegative integer seeds.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Expected different seeds.")
    if progress is not None and not callable(progress):
        raise ValueError("Invalid progress callback.")
    trials, summary, resolved_settings = [], {}, None
    cancelled = False
    for mode in ("conventional", "prox_aware", "uncert_aware"):
        runs = []
        for seed in seeds:
            if cancelled:
                break
            trial = RunTrial(scenario, mode, seed, cancel_event=cancel_event, **settings)
            resolved_settings = trial["settings"]
            if trial["status"] == "cancelled":
                cancelled = True
                break
            runs.append(trial)
            trials.append(trial)
            if progress is not None:
                progress((len(trials), 3 * len(seeds), mode, int(seed)))
            if cancel_event is not None and cancel_event.is_set():
                cancelled = len(trials) < 3 * len(seeds)
                break
        successes = sum(run["success"] for run in runs)
        statuses = {}
        for run in runs:
            statuses[run["status"]] = statuses.get(run["status"], 0) + 1
        values = {
            "actual_error": [run["actual_error"] for run in runs],
            "minimum_clearance": [run["minimum_clearance"] for run in runs],
            "planning_time": [run["plan"]["time"] for run in runs if run["plan"] is not None],
            "successful_travel": [run["travel"] for run in runs if run["success"]],
            "successful_completion_time": [run["completion_time"] for run in runs if run["success"]],
        }
        for name in ("tool_raw", "target_raw", "tool_filtered", "target_filtered"):
            values[name + "_rmse"] = [run["rmse"][name] for run in runs
                                       if run["rmse"][name] is not None]
        # summarises independent trials rather than pooling dependent sensor samples
        for metric in ("predicted_rms", "coverage"):
            for name in ("tool", "target"):
                values[name + "_" + metric] = [run[metric][name] for run in runs
                                               if run[metric][name] is not None]
        metrics = {}
        for name, measurements in values.items():
            count = len(measurements)
            metrics[name] = {"count": count,
                             "mean": float(np.mean(measurements)) if count else None,
                             "sd": float(np.std(measurements, ddof=1)) if count > 1 else None}
        summary[mode] = {"trials": len(runs), "successes": successes,
                         "success_rate": successes / len(runs) if runs else None,
                         "false_arrivals": sum(run["false_arrival"] for run in runs),
                         "statuses": statuses, "metrics": metrics}
    return {"scenario": scenario, "seeds": [int(seed) for seed in seeds],
            "settings": resolved_settings, "trials": trials, "summary": summary,
            "cancelled": cancelled}


def RunSensitivityStudy(scenario="Direct insertion", noise_levels=(0.0, 1.0, 2.0, 4.0),
                        seeds=range(10), *, cancel_event=None, progress=None, **settings):
    try:
        levels = np.asarray(noise_levels, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid noise levels.") from None
    if (levels.ndim != 1 or not levels.size or not np.all(np.isfinite(levels))
            or np.any(levels < 0) or np.unique(levels).size != levels.size):
        raise ValueError("Expected different nonnegative noise levels.")
    if progress is not None and not callable(progress):
        raise ValueError("Invalid progress callback.")
    levels = np.sort(levels).tolist()
    seeds = list(seeds)
    comparisons = []
    completed, total = 0, 3 * len(seeds) * len(levels)
    cancelled = False
    for level in levels:
        level_settings = dict(settings, tool_noise=level)

        def ReportProgress(update):
            count, _, mode, seed = update
            progress((completed + count, total, mode, seed))

        comparison = CompareMethods(scenario, seeds, cancel_event=cancel_event,
                                    progress=ReportProgress if progress is not None else None,
                                    **level_settings)
        if comparison["trials"]:
            comparisons.append(comparison)
        completed += len(comparison["trials"])
        if comparison["cancelled"] or (cancel_event is not None and cancel_event.is_set()):
            cancelled = completed < total
            break
    return {"scenario": scenario, "noise_levels": levels, "seeds": [int(seed) for seed in seeds],
            "comparisons": comparisons, "cancelled": cancelled}
