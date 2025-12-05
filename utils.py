import numpy as np
import scipy

def max_normalise(x, max_dB = 5):
    """
    Normalize an input vector by clipping its maximum value and scaling to the range [0, 1].

    Args:
        x: Input array of values to be normalized.
        max_dB: Maximum allowed value before clipping. All elements greater than this
                threshold are clipped to `max_dB`.

    Returns:
        x_norm: The normalized array, scaled by its maximum value (after clipping) so
                that the output lies within [0, 1].
    """
    
    x = np.clip(x, 0, max_dB)
    x_norm = x / (1e-6 + np.max(x))
    
    return x_norm

def lower_envelope(x, area = 10):
    """
    Compute the lower envelope of a 1D signal by extracting local minima over sliding windows.

        Args:
            x: Input 1D array representing the signal.
            area: Window size used to search for local minima.

        Returns:
            idx: Array of indices where the lower envelope has support points.
            hull: Array of envelope values corresponding to the indices in `idx`.
    """
    
    idx = []
    hull = []
    for i in range(len(x)-area+1):
        patch = x[i:i+area]
        rel_idx = np.argmin(patch)
        abs_idx = rel_idx + i
        if abs_idx not in idx:
            idx.append(abs_idx)
            hull.append(patch[rel_idx])

    if idx[0] != 0:
        idx.insert(0, 0)
        hull.insert(0, x[0])
    if idx[-1] != len(x)-1:
        idx.append(len(x)-1)
        hull.append(x[-1])

    return np.array(idx), np.array(hull)


def curve_profile(x, c, f_range = [5000, 16000], min_dB = -45):
    """
    Compute the profile of a curve above its lower envelope within a given frequency range.

    This function extracts a frequency-limited segment of the curve `c(x)`, estimates its
    lower envelope using local minima, and computes how much the curve rises above this
    envelope. The result highlights the spectral structure relative to a smooth baseline.

    Args:
        x: Frequency axis.
        c: Curve values (e.g., magnitude in dB) corresponding to `x`.
        f_range: Two-element list [f_min, f_max] selecting the frequency range of interest.
        min_dB: Minimum allowed value for the lower envelope to avoid excessively low baselines. 

    Returns:
        x_: Frequencies within the selected range.
        profile: Curve values above the lower envelope, clipped to be non-negative.
    """
    
    cutoff_idx = np.where( (f_range[0] < x) & (x < f_range[1] ))
    x_ = x[cutoff_idx]
    c_ = c[cutoff_idx]
    lower_x, lower_c = lower_envelope(c_, area=10)

    low_hull_curve = scipy.interpolate.interp1d(x_[ lower_x ], lower_c, kind="quadratic")(x_)
    low_hull_curve = np.clip( low_hull_curve, min_dB, None)
    
    profile = np.clip( c_ - low_hull_curve, 0, None )

    return x_, profile