import numpy as np


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

    def update(self, position):
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
