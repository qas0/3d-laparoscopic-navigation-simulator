import numpy as np


def FitCalibration(logits, labels, weights=None, cancel=None):
    try:
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=float)
        weights = np.ones(logits.shape) if weights is None else np.asarray(weights, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid calibration data.") from None
    if (logits.ndim != 1 or not len(logits) or labels.shape != logits.shape
            or weights.shape != logits.shape
            or any(not np.all(np.isfinite(value)) for value in (logits, labels, weights))
            or not np.all((labels == 0) | (labels == 1)) or np.any(weights < 0)
            or not np.any(weights[labels == 0] > 0) or not np.any(weights[labels == 1] > 0)):
        raise ValueError("Invalid calibration data.")
    if cancel is not None and cancel.is_set():
        raise InterruptedError("Calibration cancelled.")

    # restores the supplied pixel prevalence instead of balancing the two classes
    weights = weights / weights.max()
    weights = weights / weights.sum()
    coefficients = np.array([1.0, 0.0])
    lower = np.array([0.05, -20.0])
    upper = np.array([10.0, 20.0])
    ridge = 1e-6
    identity_loss = float(np.sum(weights * (np.logaddexp(0, logits) - labels * logits)))
    for iteration in range(60):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Calibration cancelled.")
        scaled = coefficients[0] * logits + coefficients[1]
        # evaluates the sigmoid without exponentiating positive values
        probabilities = np.empty_like(scaled)
        positive = scaled >= 0
        probabilities[positive] = 1 / (1 + np.exp(-scaled[positive]))
        exponent = np.exp(scaled[~positive])
        probabilities[~positive] = exponent / (1 + exponent)
        residual = weights * (probabilities - labels)
        variance = weights * probabilities * (1 - probabilities)
        penalty = coefficients - [1.0, 0.0]
        gradient = np.array([np.sum(residual * logits), residual.sum()]) + ridge * penalty
        hessian = np.array([[np.sum((variance * logits) * logits), np.sum(variance * logits)],
                            [np.sum(variance * logits), variance.sum()]]) + ridge * np.eye(2)
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                step = np.linalg.solve(hessian, gradient)
                # fits the remaining parameter when a Newton step points out of a bound
                free = ~(((coefficients <= lower + 1e-7) & (step > 0))
                         | ((coefficients >= upper - 1e-7) & (step < 0)))
                if not np.any(free):
                    break
                if not np.all(free):
                    step[:] = 0
                    step[free] = np.linalg.solve(hessian[np.ix_(free, free)], gradient[free])
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        loss = float(np.sum(weights * (np.logaddexp(0, scaled) - labels * scaled))
                     + 0.5 * ridge * np.sum(penalty * penalty))
        rate = 1.0
        accepted = False
        # keeps the map increasing and accepts only descending calibration-set steps
        for attempt in range(30):
            candidate = coefficients - rate * step
            if 0.05 <= candidate[0] <= 10 and -20 <= candidate[1] <= 20:
                with np.errstate(over="ignore", invalid="ignore"):
                    proposed = candidate[0] * logits + candidate[1]
                if np.all(np.isfinite(proposed)):
                    offset = candidate - [1.0, 0.0]
                    proposed_loss = float(np.sum(weights * (
                        np.logaddexp(0, proposed) - labels * proposed))
                        + 0.5 * ridge * np.sum(offset * offset))
                    if proposed_loss <= loss:
                        coefficients = candidate
                        accepted = True
                        break
            rate *= 0.5
        if not accepted or np.linalg.norm(rate * step) < 1e-8:
            break
    scaled = coefficients[0] * logits + coefficients[1]
    fitted_loss = float(np.sum(weights * (np.logaddexp(0, scaled) - labels * scaled)))
    # retains identity if the fitted map worsens the unpenalized calibration loss
    if not np.isfinite(fitted_loss) or fitted_loss > identity_loss:
        coefficients = np.array([1.0, 0.0])
    return {"temperature": float(1 / coefficients[0]), "bias": float(coefficients[1])}


def CalibrationMetrics(probabilities, labels, weights=None, bins=10):
    if (isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer))
            or not 1 <= bins <= 100):
        raise ValueError("Invalid calibration bins.")
    unweighted = weights is None
    try:
        probabilities = np.asarray(probabilities, dtype=float)
        labels = np.asarray(labels, dtype=float)
        weights = np.ones(probabilities.shape) if unweighted else np.asarray(weights, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid calibration data.") from None
    if (probabilities.ndim != 1 or not len(probabilities) or labels.shape != probabilities.shape
            or weights.shape != probabilities.shape
            or any(not np.all(np.isfinite(value)) for value in (probabilities, labels, weights))
            or np.any((probabilities < 0) | (probabilities > 1))
            or not np.all((labels == 0) | (labels == 1)) or np.any(weights < 0)):
        raise ValueError("Invalid calibration data.")
    with np.errstate(over="ignore"):
        total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid calibration weights.")
    normalized = weights / total
    positive = labels == 1
    positive_weight = float(weights[positive].sum())
    negative_weight = float(weights[~positive].sum())
    squared_error = (probabilities - labels) ** 2
    safe = np.clip(probabilities, 1e-12, 1 - 1e-12)
    # uses a binary instrument probability, never maximum score or pixel accuracy
    log_loss = -labels * np.log(safe) - (1 - labels) * np.log1p(-safe)
    indices = np.minimum((probabilities * bins).astype(int), bins - 1)
    counts = np.bincount(indices, weights=weights, minlength=bins)
    positive_counts = np.bincount(indices, weights=weights * labels, minlength=bins)
    # normalizes bin sums before multiplication to avoid underflow for tiny weights
    normalized_counts = np.bincount(indices, weights=normalized, minlength=bins)
    normalized_positive = np.bincount(indices, weights=normalized * labels, minlength=bins)
    probability_sums = np.bincount(indices, weights=normalized * probabilities, minlength=bins)
    mean_probability = [float(value / count) if count > 0 else None
                        for value, count in zip(probability_sums, normalized_counts)]
    positive_fraction = [float(value / count) if count > 0 else None
                         for value, count in zip(normalized_positive, normalized_counts)]
    return {"pixels": len(probabilities), "positives": int(np.count_nonzero(positive)),
            "weight": total, "positive_weight": positive_weight,
            "prevalence": positive_weight / total,
            "brier": float(np.sum(normalized * squared_error)),
            "nll": float(np.sum(normalized * log_loss)),
            "ece": float(np.sum(np.abs(normalized_positive - probability_sums))),
            "positive_brier": float(np.sum(weights[positive] / positive_weight
                                            * squared_error[positive])) if positive_weight > 0 else None,
            "negative_brier": float(np.sum(weights[~positive] / negative_weight
                                            * squared_error[~positive])) if negative_weight > 0 else None,
            "positive_nll": float(np.sum(weights[positive] / positive_weight
                                          * log_loss[positive])) if positive_weight > 0 else None,
            "negative_nll": float(np.sum(weights[~positive] / negative_weight
                                          * log_loss[~positive])) if negative_weight > 0 else None,
            "bins": int(bins), "bin_edges": np.linspace(0, 1, bins + 1).tolist(),
            "bin_counts": counts.astype(int).tolist() if unweighted else counts.tolist(),
            "bin_positive_counts": positive_counts.astype(int).tolist() if unweighted else positive_counts.tolist(),
            "bin_mean_probability": mean_probability, "bin_positive_fraction": positive_fraction}
