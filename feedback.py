import numpy as np


def get_target_feedback(tool_position, target_position, target_tolerance=2.0,
                        tool_covariance=None, target_covariance=None):
    try:
        tool_position = np.asarray(tool_position, dtype=float)
        target_position = np.asarray(target_position, dtype=float)
        target_tolerance = np.asarray(target_tolerance, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid feedback inputs.") from None

    if any(position.shape != (3,) or not np.all(np.isfinite(position))
           for position in (tool_position, target_position)):
        raise ValueError("Expected three finite values.")
    if (target_tolerance.shape != () or not np.isfinite(target_tolerance)
            or target_tolerance < 0):
        raise ValueError("Invalid target tolerance.")
    if (tool_covariance is None) != (target_covariance is None):
        raise ValueError("Expected both covariances.")

    uncertainty = None
    if tool_covariance is not None:
        try:
            covariances = [np.asarray(value, dtype=float)
                           for value in (tool_covariance, target_covariance)]
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Invalid covariance.") from None

        for covariance in covariances:
            if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
                raise ValueError("Expected a finite 3 x 3 covariance.")
            if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-9):
                raise ValueError("Covariance must be symmetric.")
            if np.linalg.eigvalsh(covariance).min() < -1e-9:
                raise ValueError("Covariance must be nonnegative.")

        # calculates RMS relative-position uncertainty for independent tool + target errors
        variance = sum(float(np.trace(covariance)) for covariance in covariances)
        uncertainty = float(np.sqrt(max(variance, 0.0)))

    # calculates arrival using only the supplied positions
    distance = float(np.linalg.norm(tool_position - target_position))
    return {"distance": distance,
            "arrived": bool(distance <= float(target_tolerance) + 1e-9),
            "uncertainty": uncertainty}


def GetRouteGuidance(configuration, path, waypoint=0,
                     angle_tolerance=np.deg2rad(0.5), depth_tolerance=0.5):
    try:
        configuration = np.asarray(configuration, dtype=float)
        path = np.asarray(path, dtype=float)
        tolerances = np.asarray((angle_tolerance, depth_tolerance), dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid route inputs.") from None

    if (configuration.shape != (3,) or path.ndim != 2
            or path.shape[0] == 0 or path.shape[1] != 3):
        raise ValueError("Expected three-value configurations.")
    if not np.all(np.isfinite(configuration)) or not np.all(np.isfinite(path)):
        raise ValueError("Configurations must be finite.")
    if configuration[2] < 0 or np.any(path[:, 2] < 0):
        raise ValueError("Depth cannot be negative.")
    if (tolerances.shape != (2,) or not np.all(np.isfinite(tolerances))
            or np.any(tolerances < 0)):
        raise ValueError("Invalid route tolerances.")
    if (isinstance(waypoint, (bool, np.bool_))
            or not isinstance(waypoint, (int, np.integer))
            or not 0 <= waypoint <= len(path)):
        raise ValueError("Invalid waypoint.")

    # advances through reached configurations in their planned order
    waypoint = int(waypoint)
    tolerances = tolerances[[0, 0, 1]]
    while waypoint < len(path):
        change = path[waypoint] - configuration
        if not np.all(np.abs(change) <= tolerances + 1e-9):
            return {"waypoint": waypoint, "complete": False, "change": change}
        waypoint += 1

    return {"waypoint": waypoint, "complete": True, "change": np.zeros(3)}
