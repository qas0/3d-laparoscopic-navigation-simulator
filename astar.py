from heapq import heappop, heappush
from time import perf_counter

import numpy as np

from geometry import instrument_direction, movement_is_clear, shaft_clearance, tip_position


def get_clearance_penalty(clearances, travel, required_clearance=2.0, clearance_scale=5.0):
    # weights tip travel by proximity to the required shaft clearance
    clearances = np.asarray(clearances, dtype=float)
    if (clearances.ndim != 1 or not clearances.size or not np.all(np.isfinite(clearances))
            or not np.all(np.isfinite([travel, required_clearance, clearance_scale]))
            or travel < 0 or required_clearance < 0 or clearance_scale <= 0):
        raise ValueError("Invalid clearance penalty settings.")
    slack = np.maximum(0.0, clearances - required_clearance)
    return float(travel * np.mean(np.exp(-slack / clearance_scale)))


def get_neighbours(node, shape):
    for axis, size in enumerate(shape):
        for direction in (-1, 1):
            neighbour = list(node)
            neighbour[axis] += direction
            if 0 <= neighbour[axis] < size:
                yield tuple(neighbour)


def reconstruct_path(parents, node):
    path = [node]
    while node in parents:
        node = parents[node]
        path.append(node)
    return path[::-1]


def find_path(start, target, port, tool_radius, structure_centre, structure_radius,
              lower_limits, upper_limits, required_clearance=2.0, target_tolerance=2.0,
              grid_step=None, max_expansions=10000, time_limit=5.0, cancel_event=None,
              include_target=False, mode="conventional", proximity_weight=1.0,
              clearance_scale=5.0):
    # searches yaw, pitch + insertion; angles use radians, distances use mm
    started = perf_counter()
    expanded = 0
    start, target, port, structure_centre, lower_limits, upper_limits = (
        np.asarray(value, dtype=float) for value in
        (start, target, port, structure_centre, lower_limits, upper_limits))
    if grid_step is None:
        grid_step = [np.deg2rad(5), np.deg2rad(5), 5.0]
    grid_step = np.asarray(grid_step, dtype=float)
    vectors = (start, target, port, structure_centre, lower_limits, upper_limits, grid_step)
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in vectors):
        raise ValueError("Expected three finite values.")
    scalars = [tool_radius, structure_radius, required_clearance, target_tolerance, time_limit,
               proximity_weight, clearance_scale]
    if (not np.all(np.isfinite(scalars)) or min(scalars) < 0
            or time_limit == 0 or clearance_scale == 0):
        raise ValueError("Invalid planning settings.")
    if mode not in ("conventional", "prox_aware"):
        raise ValueError("Unknown planning mode.")
    if (np.any(grid_step <= 0) or np.any(lower_limits > upper_limits)
            or lower_limits[2] < 0 or max_expansions < 1):
        raise ValueError("Invalid grid, limits or search budget.")
    if (lower_limits[0] < -np.pi or upper_limits[0] > np.pi
            or lower_limits[1] < -np.pi / 2 or upper_limits[1] > np.pi / 2):
        raise ValueError("Yaw or pitch limits out of range.")

    def finish(status, path=None, cost=None):
        length, penalty, minimum_clearance = None, None, None
        if path is not None:
            change = np.abs(np.diff(path, axis=0))
            # sums exact insertion distances + rotational arc lengths
            edge_lengths = (change[:, 2] + path[:-1, 2] * (
                change[:, 1] + np.cos(path[:-1, 1]) * change[:, 0]))
            length, penalty, minimum_clearance = float(np.sum(edge_lengths)), 0.0, start_clearance
            for first, second, travel in zip(path, path[1:], edge_lengths):
                clearances = []
                movement_is_clear(port, first, second, tool_radius, structure_centre,
                                  structure_radius, required_clearance,
                                  clearance_samples=clearances)
                penalty += get_clearance_penalty(clearances, travel, required_clearance,
                                                  clearance_scale)
                if np.any(first[:2] != second[:2]):
                    # subtracts the same displacement guard used for swept clearance
                    bound = abs(second[2] - first[2]) + max(first[2], second[2]) * np.sum(
                        np.abs(second[:2] - first[:2]))
                    edge_clearance = min(clearances) - bound / (2 * len(clearances))
                else:
                    tip = tip_position(port, first[0], first[1], max(first[2], second[2]))
                    edge_clearance = shaft_clearance(port, tip, tool_radius,
                                                      structure_centre, structure_radius)
                endpoint_clearance = shaft_clearance(port, tip_position(port, *second),
                                                     tool_radius, structure_centre,
                                                     structure_radius)
                minimum_clearance = min(minimum_clearance, edge_clearance, endpoint_clearance)
        return {"status": status, "path": path, "cost": cost,
                "expanded": expanded, "time": perf_counter() - started,
                "length": length, "clearance_penalty": penalty,
                "minimum_clearance": minimum_clearance}

    if cancel_event is not None and cancel_event.is_set():
        return finish("cancelled")
    if np.any(start < lower_limits) or np.any(start > upper_limits):
        return finish("invalid_start")
    start_tip = tip_position(port, *start)
    start_clearance = shaft_clearance(port, start_tip, tool_radius, structure_centre,
                                      structure_radius)
    if start_clearance < required_clearance:
        return finish("invalid_start")
    if np.linalg.norm(start_tip - target) <= target_tolerance + 1e-9:
        return finish("found", np.array([start]), 0.0)

    # finds the closest tip allowed by the angle + insertion limits
    offset = target - port
    target_yaw = np.arctan2(offset[1], offset[0])
    best_projection = -np.inf
    best_yaw, best_pitch = lower_limits[:2]
    for yaw in (lower_limits[0], upper_limits[0],
                np.clip(target_yaw, lower_limits[0], upper_limits[0])):
        horizontal = offset[0] * np.cos(yaw) + offset[1] * np.sin(yaw)
        pitch = np.clip(np.arctan2(offset[2], horizontal), lower_limits[1], upper_limits[1])
        for angle in (lower_limits[1], upper_limits[1], pitch):
            projection = horizontal * np.cos(angle) + offset[2] * np.sin(angle)
            if projection > best_projection:
                best_projection = projection
                best_yaw, best_pitch = yaw, angle
    depth = np.clip(best_projection, lower_limits[2], upper_limits[2])
    closest_configuration = np.array([best_yaw, best_pitch, depth])
    closest_distance = np.linalg.norm(tip_position(port, *closest_configuration) - target)
    target_clearance = shaft_clearance(port, target, tool_radius,
                                       structure_centre, structure_radius)
    # rejects a target only when its entire tolerance region is infeasible
    if (closest_distance > target_tolerance + 1e-9
            or target_clearance + target_tolerance < required_clearance):
        return finish("invalid_target")

    axes = []
    start_node = []
    for axis in range(3):
        count = int(np.floor((upper_limits[axis] - lower_limits[axis]) / grid_step[axis] + 1e-9))
        values = lower_limits[axis] + np.arange(count + 1) * grid_step[axis]
        if np.isclose(values[-1], upper_limits[axis], atol=1e-12, rtol=0):
            values[-1] = upper_limits[axis]
        else:
            values = np.append(values, upper_limits[axis])
        # includes the actual start instead of snapping to an unchecked configuration
        matching = np.flatnonzero(np.isclose(values, start[axis], atol=1e-12, rtol=0))
        if matching.size:
            values[matching[0]] = start[axis]
        else:
            values = np.sort(np.append(values, start[axis]))
        # includes a supplied target configuration while preserving the actual start
        if include_target and not np.any(np.isclose(
                values, closest_configuration[axis], atol=1e-12, rtol=0)):
            values = np.sort(np.append(values, closest_configuration[axis]))
        axes.append(values)
        start_node.append(int(np.argmin(abs(values - start[axis]))))
    start_node = tuple(start_node)
    configurations = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    directions = np.moveaxis(instrument_direction(configurations[..., 0],
                                                 configurations[..., 1]), 0, -1)
    tips = port + configurations[..., 2, None] * directions
    distances = np.linalg.norm(tips - target, axis=-1)
    # uses straight-line distance to the target region as a lower bound on travel
    heuristic = np.maximum(0.0, distances - target_tolerance - 1e-9)
    valid_nodes = {}
    goals = set()
    for indices in np.argwhere(distances <= target_tolerance + 1e-9):
        node = tuple(indices)
        valid_nodes[node] = shaft_clearance(port, tips[node], tool_radius,
                                            structure_centre, structure_radius) >= required_clearance
        if valid_nodes[node]:
            goals.add(node)
    if not goals:
        return finish("no_grid_goal")

    frontier = [(float(heuristic[start_node]), float(heuristic[start_node]), 0.0, start_node)]
    costs = {start_node: 0.0}
    parents = {}
    visited = set()
    while frontier:
        if cancel_event is not None and cancel_event.is_set():
            return finish("cancelled")
        if perf_counter() - started >= time_limit:
            return finish("budget_exceeded")
        _, _, cost, node = heappop(frontier)
        if node in visited or cost > costs[node]:
            continue
        if node in goals:
            nodes = reconstruct_path(parents, node)
            return finish("found", np.array([configurations[index] for index in nodes]), cost)
        if expanded >= max_expansions:
            return finish("budget_exceeded")
        visited.add(node)
        expanded += 1
        configuration = configurations[node]
        for neighbour in get_neighbours(node, distances.shape):
            if neighbour in visited:
                continue
            if neighbour not in valid_nodes:
                valid_nodes[neighbour] = shaft_clearance(
                    port, tips[neighbour], tool_radius,
                    structure_centre, structure_radius) >= required_clearance
            if not valid_nodes[neighbour]:
                continue
            proposed = configurations[neighbour]
            axis = next(index for index in range(3) if node[index] != neighbour[index])
            travel = abs(proposed[axis] - configuration[axis])
            # calculates arc length for rotation + straight-line length for insertion
            if axis < 2:
                travel *= configuration[2]
                if axis == 0:
                    travel *= np.cos(configuration[1])
            if cost + travel >= costs.get(neighbour, np.inf):
                continue
            clearances = [] if mode == "prox_aware" and proximity_weight > 0 else None
            if not movement_is_clear(port, configuration, proposed, tool_radius,
                                     structure_centre, structure_radius, required_clearance,
                                     clearance_samples=clearances):
                continue
            candidate_cost = cost + travel
            if clearances is not None:
                candidate_cost += proximity_weight * get_clearance_penalty(
                    clearances, travel, required_clearance, clearance_scale)
            if candidate_cost >= costs.get(neighbour, np.inf):
                continue
            costs[neighbour] = candidate_cost
            parents[neighbour] = node
            remaining = float(heuristic[neighbour])
            heappush(frontier, (candidate_cost + remaining, remaining, candidate_cost, neighbour))
    return finish("no_path")
