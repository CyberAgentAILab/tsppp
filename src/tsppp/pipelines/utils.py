from typing import NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.signal import correlate, find_peaks


class PipeplineOutput(NamedTuple):
    samples: torch.Tensor
    images: torch.Tensor
    destinations: torch.Tensor


def get_trimmed_path(samples: NDArray, height: float = 0) -> NDArray:
    """
    Trims the path samples based on the peaks in the correlation of the x and y coordinates.

    Args:
        samples (NDArray): 3D array of shape (N, 2, T) representing the path samples.
        height (float): The minimum height of peaks in the correlation.

    Returns:
        NDArray: Trimmed path samples with shape (N, 2, T_trimmed).
    """
    assert (samples.ndim == 3) and (
        samples.shape[1] == 2
    ), "Input must be 3D array of shape (N, 2, T)"

    T = samples.shape[2]
    peak_max = -np.inf

    def get_corr(s1, s2):
        s1_ = (s1 - s1.mean()) / s1.std()
        s2_ = (s2 - s2.mean()) / s2.std()
        return correlate(s1_, s2_, "full")

    # get alignment
    # for i in range(len(samples)):
    #     corr = get_corr(samples[0][0], samples[i][0]) + get_corr(
    #         samples[0][1], samples[i][1]
    #     )
    #     peaks = find_peaks(corr, height=height)[0] - T + 1
    #     peaks = peaks[peaks >= 0]
    #     print(peaks)
    #     peak = peaks.min()
    #     samples[i] = np.hstack((samples[i, :, peak:], samples[i, :, :peak]))

    for s in samples:
        corr = get_corr(s[0], s[0]) + get_corr(s[1], s[1])
        peaks = find_peaks(corr, height=height)[0] - T + 1
        if np.any(peaks > 0):
            peak_max = np.maximum(peaks[peaks > 0][0], peak_max).astype(int)
    peak_max = peak_max if peak_max > 0 else T

    return samples[:, :, :peak_max]
