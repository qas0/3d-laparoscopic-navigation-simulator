from heapq import heappop, heappush
from time import perf_counter

import numpy as np

from geometry import instrument_direction, movement_is_clear, shaft_clearance, tip_position


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
              grid_step=None, max_expansions=10000, time_limit=5.0, cancel_event=None):
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
    scalars = [tool_radius, structure_radius, required_clearance, target_tolerance, time_limit]
    if not np.all(np.isfinite(scalars)) or min(scalars) < 0 or time_limit == 0:
        raise ValueError("Invalid radius, tolerance or time limit.")
    if (np.any(grid_step <= 0) or np.any(lower_limits > upper_limits)
            or lower_limits[2] < 0 or max_expansions < 1):
        raise ValueError("Invalid grid, limits or search budget.")
    if (lower_limits[0] < -np.pi or upper_limits[0] > np.pi
            or lower_limits[1] < -np.pi / 2 or upper_limits[1] > np.pi / 2):
        raise ValueError("Yaw or pitch limits out of range.")

    def finish(status, path=None, cost=None):
        return {"status": status, "path": path, "cost": cost,
                "expanded": expanded, "time": perf_counter() - started}

    if cancel_event is not None and cancel_event.is_set():
        return finish("cancelled")
    if np.any(start < lower_limits) or np.any(start > upper_limits):
        return finish("invalid_start")
    start_tip = tip_position(port, *start)
    if shaft_clearance(port, start_tip, tool_radius, structure_centre,
                       structure_radius) < required_clearance:
        return finish("invalid_start")
    if np.linalg.norm(start_tip - target) <= target_tolerance + 1e-9:
        return finish("found", np.array([start]), 0.0)

    # finds the closest tip allowed by the angle + insertion limits
    offset = target - port
    target_yaw = np.arctan2(offset[1], offset[0])
    best_projection = -np.inf
    for yaw in (lower_limits[0], upper_limits[0],
                np.clip(target_yaw, lower_limits[0], upper_limits[0])):
        horizontal = offset[0] * np.cos(yaw) + offset[1] * np.sin(yaw)
        pitch = np.clip(np.arctan2(offset[2], horizontal), lower_limits[1], upper_limits[1])
        for angle in (lower_limits[1], upper_limits[1], pitch):
            projection = horizontal * np.cos(angle) + offset[2] * np.sin(angle)
            best_projection = max(best_projection, projection)
    depth = np.clip(best_projection, lower_limits[2], upper_limits[2])
    closest_distance = np.sqrt(max(0.0, np.dot(offset, offset)
                                   + depth * depth - 2 * depth * best_projection))
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
    valid_edges = {}
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
            candidate_cost = cost + travel
            if candidate_cost >= costs.get(neighbour, np.inf):
                continue
            edge = tuple(sorted((node, neighbour)))
            if edge not in valid_edges:
                valid_edges[edge] = movement_is_clear(
                    port, configuration, proposed, tool_radius,
                    structure_centre, structure_radius, required_clearance)
            if not valid_edges[edge]:
                continue
            costs[neighbour] = candidate_cost
            parents[neighbour] = node
            remaining = float(heuristic[neighbour])
            heappush(frontier, (candidate_cost + remaining, remaining, candidate_cost, neighbour))
    return finish("no_path")
