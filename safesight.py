import numpy as np
from PIL import Image, ImageFilter


def DegradeView(image, blur=0.0, exposure=0.0, haze=0.0):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) < 16):
        raise ValueError("Invalid RGB image.")
    try:
        settings = np.asarray((blur, exposure, haze), dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid image effects.") from None
    if (settings.shape != (3,) or not np.all(np.isfinite(settings))
            or any(isinstance(value, (bool, np.bool_)) for value in (blur, exposure, haze))):
        raise ValueError("Invalid image effects.")
    blur, exposure, haze = map(float, settings)
    if not (0 <= blur <= 1 and -4 <= exposure <= 4 and 0 <= haze <= 1):
        raise ValueError("Invalid image effects.")
    if blur == 0 and exposure == 0 and haze == 0:
        return image.copy()

    # blurs first, using the same radius relative to image width at each resolution
    if blur > 0:
        radius = 12 * blur * image.shape[1] / 640
        image = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius)))
    # multiplies linear-light sRGB by 2**EV before encoding and clipping
    if exposure != 0:
        # builds one curve for the 256 possible byte values
        levels = np.arange(256, dtype=np.float32) / 255
        linear = np.where(levels <= 0.04045, levels / 12.92,
                          ((levels + 0.055) / 1.055) ** 2.4)
        linear *= 2 ** exposure
        curve = np.where(linear <= 0.0031308, 12.92 * linear,
                         1.055 * linear ** (1 / 2.4) - 0.055)
        pixels = np.clip(curve, 0, 1)[image]
    else:
        pixels = image.astype(np.float32) / 255

    # blends neutral haze last in display RGB; this approximates a veil, not physical smoke
    if haze > 0:
        neutral = np.array([185, 190, 195], dtype=np.float32) / 255
        pixels = (1 - haze) * pixels + haze * neutral
    return np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)


def AssessView(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) < 16):
        raise ValueError("Invalid RGB image.")

    # reduces resolution differences without enlarging small frames
    height, width = image.shape[:2]
    if max(height, width) > 320:
        scale = 320 / max(height, width)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))

    luminance = image.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722],
                                                  dtype=np.float32) / 255
    mean_luminance = float(luminance.mean())
    dark_fraction = float(np.mean(luminance < 0.10))
    bright_fraction = float(np.mean(luminance > 0.95))

    # measures local detail; a low value does not establish the cause
    if min(luminance.shape) >= 3:
        laplacian = (luminance[1:-1, :-2] + luminance[1:-1, 2:]
                     + luminance[:-2, 1:-1] + luminance[2:, 1:-1]
                     - 4 * luminance[1:-1, 1:-1])
        detail = float(np.abs(laplacian).mean())
    else:
        # supports very narrow frames after resizing
        axis = int(luminance.shape[1] > luminance.shape[0])
        detail = float(np.abs(np.diff(luminance, n=2, axis=axis)).mean())

    # applies empirical simulator thresholds, not clinical safety limits
    issues = []
    if mean_luminance < 0.15 or dark_fraction > 0.60:
        issues.append("dark")
    if mean_luminance > 0.85 or bright_fraction > 0.50:
        issues.append("bright")
    if detail < 0.006:
        issues.append("low_detail")

    return {"status": "limited" if issues else "no_issue",
            "issues": tuple(issues),
            "mean_luminance": mean_luminance,
            "dark_fraction": dark_fraction,
            "bright_fraction": bright_fraction,
            "detail": detail}
