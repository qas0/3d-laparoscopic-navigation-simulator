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

    for index in range(intervals):
        fraction = (index + 0.5) / intervals
        yaw, pitch, depth = start + fraction * change
        tip = tip_position(port, yaw, pitch, depth)
        clearance = shaft_clearance(port, tip, tool_radius,
                                    structure_centre, structure_radius)
        if clearance < required_clearance + guard:
            return False
        if clearance_samples is not None:
            clearance_samples.append(clearance)
    return True
