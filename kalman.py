import numpy as np


def CheckMeasurement(innovation, covariance, thresholds=(6.635, 9.210, 11.345)):
    try:
        innovation = np.asarray(innovation, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        thresholds = np.asarray(thresholds, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid error or covariance.") from None
    if (innovation.shape != (3,) or covariance.shape != (3, 3)
            or not np.all(np.isfinite(innovation)) or not np.all(np.isfinite(covariance))):
        raise ValueError("Invalid error or covariance.")
    if thresholds.shape != (3,) or not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0):
        raise ValueError("Invalid thresholds.")
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
    if rank == 3:
        score = float(innovation @ np.linalg.solve(covariance, innovation))
    else:
        projected = directions.T @ innovation
        # allows 1e-9 mm rounding error in directions with zero reported uncertainty
        if np.linalg.norm(projected[~supported]) > 1e-9:
            return False
        if rank == 0:
            return True
        score = float(np.sum(projected[supported] ** 2 / eigenvalues[supported]))
    return bool(score <= thresholds[rank - 1])


class KalmanFilter:
    def __init__(self, position, noise_std, acceleration_std=10.0, velocity_std=25.0):
        try:
            position = np.asarray(position, dtype=float)
            deviations = np.asarray([noise_std, acceleration_std, velocity_std], dtype=float)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Invalid filter inputs.") from None

        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("Expected three finite values.")
        if deviations.shape != (3,) or not np.all(np.isfinite(deviations)) or np.any(deviations < 0):
            raise ValueError("Invalid standard deviation.")

        self.noise_std, self.acceleration_std, velocity_std = map(float, deviations)
        # initialises position from the first observation + assumes zero velocity
        self.state = np.concatenate((position, np.zeros(3)))
        self.covariance = np.diag([self.noise_std ** 2] * 3 + [velocity_std ** 2] * 3)

    def predict(self, dt):
        try:
            dt = np.asarray(dt, dtype=float)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Invalid time step.") from None
        if dt.shape != () or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid time step.")
        dt = float(dt)

        identity = np.eye(3)
        transition = np.block([[identity, dt * identity],
                               [np.zeros((3, 3)), identity]])
        # models an independent acceleration acting throughout each time interval
        acceleration = np.vstack((0.5 * dt ** 2 * identity, dt * identity))
        process_noise = self.acceleration_std ** 2 * acceleration @ acceleration.T

        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        self.covariance = (self.covariance + self.covariance.T) / 2

    def update(self, position, reject_outliers=False):
        try:
            position = np.asarray(position, dtype=float)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Invalid observation.") from None
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("Expected three finite values.")

        measurement = np.hstack((np.eye(3), np.zeros((3, 3))))
        measurement_noise = self.noise_std ** 2 * np.eye(3)
        innovation = position - measurement @ self.state
        innovation_covariance = measurement @ self.covariance @ measurement.T + measurement_noise
        if reject_outliers and not CheckMeasurement(innovation, innovation_covariance):
            return False
        cross_covariance = self.covariance @ measurement.T

        if self.noise_std == 0 and not np.any(innovation_covariance):
            # keeps velocity unchanged when exact observations leave no uncertainty
            gain = np.zeros((6, 3))
        else:
            gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T

        self.state = self.state + gain @ innovation
        correction = np.eye(6) - gain @ measurement
        # uses the Joseph form to preserve covariance under floating-point rounding
        self.covariance = (correction @ self.covariance @ correction.T
                           + gain @ measurement_noise @ gain.T)
        self.covariance = (self.covariance + self.covariance.T) / 2

        if self.noise_std == 0:
            # assigns exact observations + removes uncertainty in the observed position
            self.state[:3] = position
            self.covariance[:3, :] = 0
            self.covariance[:, :3] = 0
        return True
