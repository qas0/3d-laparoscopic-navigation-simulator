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
    def __init__(self, position, noise_std, acceleration_std=10.0, velocity_std=25.0,
                 confirm_startup=False):
        try:
            position = np.asarray(position, dtype=float)
            deviations = np.asarray([noise_std, acceleration_std, velocity_std], dtype=float)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Invalid filter inputs.") from None

        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("Expected three finite values.")
        if deviations.shape != (3,) or not np.all(np.isfinite(deviations)) or np.any(deviations < 0):
            raise ValueError("Invalid standard deviation.")
        if not isinstance(confirm_startup, (bool, np.bool_)):
            raise ValueError("Expected startup check on or off.")

        self.noise_std, self.acceleration_std, velocity_std = map(float, deviations)
        # initialises position from the first observation + assumes zero velocity
        self.state = np.concatenate((position, np.zeros(3)))
        self.covariance = np.diag([self.noise_std ** 2] * 3 + [velocity_std ** 2] * 3)
        self.ready = not confirm_startup
        self.startup_time = 0.0
        self.startup_readings = [(0.0, position.copy())] if confirm_startup else []

    def predict(self, dt):
        try:
            dt = np.asarray(dt, dtype=float)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Invalid time step.") from None
        if dt.shape != () or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid time step.")
        dt = float(dt)

        if not self.ready:
            self.startup_time += dt
            # prevents old readings from confirming startup after a tracking gap
            if self.startup_readings and self.startup_time - self.startup_readings[-1][0] > 0.25 + 1e-9:
                self.startup_readings.clear()
            return

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

        if not self.ready:
            # counts separate sample times rather than repeated updates of one reading
            if self.startup_readings and self.startup_time == self.startup_readings[-1][0]:
                self.startup_readings[-1] = (self.startup_time, position.copy())
                return True
            self.startup_readings.append((self.startup_time, position.copy()))
            self.startup_readings = self.startup_readings[-3:]
            if len(self.startup_readings) < 3:
                return True
            (first_time, first), (middle_time, middle), (last_time, last) = self.startup_readings
            fraction = (middle_time - first_time) / (last_time - first_time)
            # compares the middle reading with a straight motion between its neighbours
            error = middle - ((1 - fraction) * first + fraction * last)
            variance = self.noise_std ** 2 * (1 + (1 - fraction) ** 2 + fraction ** 2)
            acceleration_error = (0.5 * self.acceleration_std
                                  * (middle_time - first_time) * (last_time - middle_time))
            covariance = (variance + acceleration_error ** 2) * np.eye(3)
            if not CheckMeasurement(error, covariance, (3.841, 5.991, 7.815)):
                return False
            # starts from the latest consistent reading without inferring a spike-driven velocity
            self.state[:3] = position
            self.state[3:] = 0.0
            self.ready = True
            self.startup_readings.clear()
            return True

        measurement_noise = self.noise_std ** 2 * np.eye(3)
        innovation = position - self.state[:3]
        innovation_covariance = self.covariance[:3, :3] + measurement_noise
        if reject_outliers and not CheckMeasurement(innovation, innovation_covariance):
            return False
        cross_covariance = self.covariance[:, :3]

        if self.noise_std == 0 and not np.any(innovation_covariance):
            # keeps velocity unchanged when exact observations leave no uncertainty
            gain = np.zeros((6, 3))
        else:
            gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T

        if self.noise_std == 0:
            velocity_scale = np.max(np.abs(self.covariance[3:, 3:]))
        self.state = self.state + gain @ innovation
        correction = np.eye(6)
        correction[:, :3] -= gain
        # uses the Joseph form to preserve covariance under floating-point rounding
        self.covariance = (correction @ self.covariance @ correction.T
                           + gain @ measurement_noise @ gain.T)
        self.covariance = (self.covariance + self.covariance.T) / 2

        if self.noise_std == 0:
            # assigns exact observations + removes uncertainty in the observed position
            self.state[:3] = position
            self.covariance[:3, :] = 0
            self.covariance[:, :3] = 0
            eigenvalues, directions = np.linalg.eigh(self.covariance[3:, 3:])
            # removes small negative velocity variances caused by rounding
            if -6 * np.finfo(float).eps * velocity_scale <= eigenvalues[0] < 0:
                self.covariance[3:, 3:] = (directions * np.maximum(eigenvalues, 0)) @ directions.T
        return True
