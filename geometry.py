import numpy as np


def instrument_direction(yaw, pitch):
    # calculates unit direction from yaw about +Z & pitch above XY (radians)
    return np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
    ])


def tip_position(port, yaw, pitch, insertion_depth):
    if insertion_depth < 0:
        raise ValueError("Negative insertion depth.")
    return np.asarray(port) + insertion_depth * instrument_direction(yaw, pitch)


def shaft_clearance(port, tip, tool_radius, structure_centre, structure_radius):
    # calculates shaft clearance in mm; negative values mean overlap
    if tool_radius < 0 or structure_radius < 0:
        raise ValueError("Negative radius.")

    port = np.asarray(port)
    shaft = np.asarray(tip) - port
    offset = np.asarray(structure_centre) - port
    squared_length = np.dot(shaft, shaft)
    projection = 0.0 if squared_length == 0 else np.dot(offset, shaft) / squared_length
    closest_point = port + min(max(projection, 0.0), 1.0) * shaft
    return float(np.linalg.norm(np.asarray(structure_centre) - closest_point)
                 - structure_radius - tool_radius)


def movement_is_clear(port, start, end, tool_radius, structure_centre,
                      structure_radius, required_clearance=0.0, max_step=0.5,
                      clearance_samples=None):
    # checks linear movement between (yaw, pitch, depth) configurations
    # uses radians + mm; max_step bounds shaft displacement per interval
    port, start, end, structure_centre = (
        np.asarray(value, dtype=float) for value in (port, start, end, structure_centre))
    vectors = (port, start, end, structure_centre)
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in vectors):
        raise ValueError("Expected three finite values.")
    if not np.all(np.isfinite([tool_radius, structure_radius, required_clearance, max_step])):
        raise ValueError("Non-finite radius, clearance or step.")
    if tool_radius < 0 or structure_radius < 0:
        raise ValueError("Negative radius.")
    if start[2] < 0 or end[2] < 0 or max_step <= 0 or required_clearance < 0:
        raise ValueError("Invalid depth, clearance or step.")

    change = end - start
    rotating = np.any(change[:2])
    if not rotating:
        # checks the longest shaft because fixed-direction retraction stays inside it
        tip = tip_position(port, start[0], start[1], max(start[2], end[2]))
        clearance = shaft_clearance(port, tip, tool_radius,
                                    structure_centre, structure_radius)
        if not clearance >= required_clearance:
            return False
        if clearance_samples is None:
            return True

    # bounds the displacement of every shaft point during the movement
    movement_bound = abs(change[2]) + max(start[2], end[2]) * (
        abs(change[0]) + abs(change[1]))
    intervals = max(1, int(np.ceil(movement_bound / max_step)))
    # covers unsampled motion because clearance loss is bounded by shaft displacement
    guard = movement_bound / (2 * intervals) if rotating else 0.0

    # checks midpoint shafts together instead of repeating small NumPy operations
    fractions = (np.arange(intervals) + 0.5) / intervals
    configurations = start + fractions[:, None] * change
    directions = instrument_direction(configurations[:, 0], configurations[:, 1]).T
    tips = port + configurations[:, 2, None] * directions
    shafts = tips - port
    squared_lengths = np.sum(shafts * shafts, axis=1)
    projections = np.divide(
        np.sum(shafts * (structure_centre - port), axis=1), squared_lengths,
        out=np.zeros(intervals), where=squared_lengths != 0)
    closest_points = port + np.clip(projections, 0.0, 1.0)[:, None] * shafts
    clearances = np.linalg.norm(structure_centre - closest_points, axis=1)
    clearances = clearances - structure_radius - tool_radius
    blocked = np.flatnonzero(clearances < required_clearance + guard)
    if clearance_samples is not None:
        clearance_samples.extend(clearances[:blocked[0] if blocked.size else intervals].tolist())
    return not blocked.size


def CheckRemainingRoute(port, configuration, path, waypoint, tool_radius,
                        structure_centre, structure_radius, required_clearance=0.0):
    path = np.asarray(path, dtype=float)
    if (path.ndim != 2 or path.shape[0] == 0 or path.shape[1] != 3
            or not np.all(np.isfinite(path)) or np.any(path[:, 2] < 0)):
        raise ValueError("Invalid route configurations.")
    if (isinstance(waypoint, (bool, np.bool_))
            or not isinstance(waypoint, (int, np.integer))
            or not 0 <= waypoint <= len(path)):
        raise ValueError("Invalid waypoint.")
    # checks the current shaft even when all waypoints have been reached
    if not movement_is_clear(port, configuration, configuration, tool_radius,
                             structure_centre, structure_radius, required_clearance):
        return False
    for next_configuration in path[waypoint:]:
        if not movement_is_clear(port, configuration, next_configuration, tool_radius,
                                 structure_centre, structure_radius, required_clearance):
            return False
        configuration = next_configuration
    return True
