import numpy as np


def instrument_direction(yaw, pitch):
    # calculates unit direction from yaw about +Z & pitch above XY (radians)
    return np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
    ])


def InstrumentRotation(yaw, pitch):
    # rotates local +X onto the shaft direction + keeps camera up perpendicular
    return np.array([
        [np.cos(pitch) * np.cos(yaw), -np.sin(yaw), -np.sin(pitch) * np.cos(yaw)],
        [np.cos(pitch) * np.sin(yaw), np.cos(yaw), -np.sin(pitch) * np.sin(yaw)],
        [np.sin(pitch), 0, np.cos(pitch)],
    ])


def tip_position(port, yaw, pitch, insertion_depth):
    if insertion_depth < 0:
        raise ValueError("Negative insertion depth.")
    return np.asarray(port) + insertion_depth * instrument_direction(yaw, pitch)


def SegmentDistance(first_start, first_end, second_start, second_end):
    # calculates the closest distance between segments, including point segments
    points = tuple(np.asarray(value, dtype=float) for value in (
        first_start, first_end, second_start, second_end))
    if any(value.shape[-1:] != (3,) or not np.all(np.isfinite(value)) for value in points):
        raise ValueError("Expected finite segment points.")
    try:
        first_start, first_end, second_start, second_end = np.broadcast_arrays(*points)
    except ValueError as error:
        raise ValueError("Incompatible segment shapes.") from error

    first = first_end - first_start
    second = second_end - second_start
    offset = second_start - first_start
    # checks each endpoint against the opposite segment
    offsets = np.stack((offset, second_end - first_start, -offset,
                        first_end - second_start), axis=-2)
    directions = np.stack((first, first, second, second), axis=-2)
    squared_lengths = np.sum(directions * directions, axis=-1)
    fractions = np.divide(np.sum(offsets * directions, axis=-1), squared_lengths,
                          out=np.zeros_like(squared_lengths), where=squared_lengths != 0)
    distances = np.linalg.norm(offsets - np.clip(fractions, 0, 1)[..., None] * directions,
                               axis=-1)
    distance = np.min(distances, axis=-1)

    # also checks interior closest points; cross products handle nearly parallel lines
    normal = np.cross(first, second)
    squared_normal = np.sum(normal * normal, axis=-1)
    normal_offset = np.cross(normal, offset)
    first_fraction = np.divide(np.sum(normal_offset * second, axis=-1),
                               squared_normal, out=np.zeros_like(squared_normal),
                               where=squared_normal != 0)
    second_fraction = np.divide(np.sum(normal_offset * first, axis=-1),
                                squared_normal, out=np.zeros_like(squared_normal),
                                where=squared_normal != 0)
    interior = ((squared_normal != 0) & (first_fraction >= 0) & (first_fraction <= 1)
                & (second_fraction >= 0) & (second_fraction <= 1))
    line_distance = np.divide(np.abs(np.sum(offset * normal, axis=-1)), np.sqrt(squared_normal),
                              out=np.full_like(squared_normal, np.inf), where=squared_normal != 0)
    distance = np.minimum(distance, np.where(interior, line_distance, np.inf))
    return float(distance) if distance.ndim == 0 else distance


def shaft_clearance(port, tip, tool_radius, structure_points, structure_radius):
    # calculates shaft clearance in mm; negative values mean overlap
    if tool_radius < 0 or structure_radius < 0:
        raise ValueError("Negative radius.")

    port, tip, structure_points = (np.asarray(value, dtype=float)
                                  for value in (port, tip, structure_points))
    if (port.shape != (3,) or tip.shape[-1:] != (3,) or tip.ndim not in (1, 2)
            or (tip.ndim == 2 and len(tip) == 0)):
        raise ValueError("Invalid shaft points.")
    if not (structure_points.shape == (3,) or (structure_points.ndim == 2
            and structure_points.shape[1] == 3 and len(structure_points) >= 2)):
        raise ValueError("Invalid structure points.")
    if (not all(np.all(np.isfinite(value)) for value in (port, tip, structure_points))
            or not np.all(np.isfinite([tool_radius, structure_radius]))):
        raise ValueError("Non-finite geometry.")

    if structure_points.ndim == 1:
        # keeps the direct point projection for spherical regression scenarios
        shaft = tip - port
        squared_length = np.sum(shaft * shaft, axis=-1)
        projection = np.divide(np.sum((structure_points - port) * shaft, axis=-1),
                               squared_length, out=np.zeros_like(squared_length),
                               where=squared_length != 0)
        closest_point = port + np.clip(projection, 0.0, 1.0)[..., None] * shaft
        distance = np.linalg.norm(structure_points - closest_point, axis=-1)
    else:
        distance = np.min(SegmentDistance(port, tip[..., None, :],
                                         structure_points[:-1], structure_points[1:]), axis=-1)
    clearance = distance - structure_radius - tool_radius
    return float(clearance) if clearance.ndim == 0 else clearance


def movement_is_clear(port, start, end, tool_radius, structure_points,
                      structure_radius, required_clearance=0.0, max_step=0.5,
                      clearance_samples=None):
    # checks linear movement between (yaw, pitch, depth) configurations
    # uses radians + mm; max_step bounds shaft displacement per interval
    port, start, end, structure_points = (
        np.asarray(value, dtype=float) for value in (port, start, end, structure_points))
    vectors = (port, start, end)
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in vectors):
        raise ValueError("Expected three finite values.")
    if not (structure_points.shape == (3,) or (structure_points.ndim == 2
            and structure_points.shape[1] == 3 and len(structure_points) >= 2)):
        raise ValueError("Invalid structure points.")
    if not np.all(np.isfinite(structure_points)):
        raise ValueError("Non-finite structure points.")
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
                                    structure_points, structure_radius)
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
    clearances = shaft_clearance(port, tips, tool_radius, structure_points, structure_radius)
    blocked = np.flatnonzero(clearances < required_clearance + guard)
    if clearance_samples is not None:
        clearance_samples.extend(clearances[:blocked[0] if blocked.size else intervals].tolist())
    return not blocked.size


def CheckRemainingRoute(port, configuration, path, waypoint, tool_radius,
                        structure_points, structure_radius, required_clearance=0.0):
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
                             structure_points, structure_radius, required_clearance):
        return False
    for next_configuration in path[waypoint:]:
        if not movement_is_clear(port, configuration, next_configuration, tool_radius,
                                 structure_points, structure_radius, required_clearance):
            return False
        configuration = next_configuration
    return True
