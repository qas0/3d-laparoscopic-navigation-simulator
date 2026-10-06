import numpy as np


def GetRespiratoryMotion(amplitude, frequency, elapsed_time):
    try:
        amplitude, frequency, elapsed_time = [
            np.asarray(value, dtype=float) for value in (amplitude, frequency, elapsed_time)
        ]
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid motion inputs.") from None

    if any(value.shape != () or not np.isfinite(value) or value < 0
           for value in (amplitude, frequency, elapsed_time)):
        raise ValueError("Expected finite nonnegative scalars.")
    if amplitude == 0 or frequency == 0:
        return 0.0, 0.0

    # calculates sinusoidal displacement in mm + its velocity in mm/s
    with np.errstate(over="ignore", invalid="ignore"):
        angular_frequency = 2 * np.pi * frequency
        phase = angular_frequency * elapsed_time
        displacement = amplitude * np.sin(phase)
        velocity = amplitude * angular_frequency * np.cos(phase)
    if not np.all(np.isfinite([displacement, velocity])):
        raise ValueError("Invalid motion result.")
    return float(displacement), float(velocity)
