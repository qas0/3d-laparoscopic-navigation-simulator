import numpy as np


def measure_position(position, noise_std, random_generator):
    try:
        position = np.asarray(position, dtype=float)
        noise_std = np.asarray(noise_std, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid measurement inputs.") from None

    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("Expected three finite values.")
    if noise_std.shape != () or not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("Invalid noise standard deviation.")

    # adds independent Gaussian noise to each position axis in mm
    # uses three draws even at zero noise to keep seeded comparisons consistent
    return position + random_generator.standard_normal(3) * float(noise_std)


def ApplyDropout(position, dropout_probability, random_generator):
    try:
        values = np.asarray(position, dtype=float)
        probability = np.asarray(dropout_probability, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid dropout inputs.") from None

    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("Expected three finite values.")
    if probability.shape != () or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid dropout probability.")

    # uses one draw even at 0 or 1 to keep seeded dropout comparisons consistent
    return None if random_generator.random() < float(probability) else position
