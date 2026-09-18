import numpy as np


def smoothstep(start, end, fraction: float):
    """Minimum-jerk-like cubic interpolation for synchronized targets."""
    u = float(np.clip(fraction, 0.0, 1.0))
    u = u * u * (3.0 - 2.0 * u)
    return np.asarray(start) + u * (np.asarray(end) - np.asarray(start))

