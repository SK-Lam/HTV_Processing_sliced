"""
tilt.py — the tilt branch, protocol Step 5.

    theta = (lambda / 2p) * f0                      Eq. (9)

Depends only on where the carrier sits in Fourier space, never on its
phase.  That is why tilt survives frames whose displacement phase is
corrupted: the two quantities come from independent stages.
"""

import numpy as np

from .config import LAM, PIXEL_PITCH, M_OPT


def tilt_scale():
    """lambda / (2 p M) — radians per cycle/pixel.  Eq. (10)."""
    return LAM / (2.0 * PIXEL_PITCH * M_OPT)


def absolute_tilt(fx, fy):
    """Absolute tilt angle from a carrier frequency.  Returns (θx, θy, |θ|)."""
    k = tilt_scale()
    tx, ty = fx * k, fy * k
    return tx, ty, float(np.hypot(tx, ty))


def working_point(cal):
    """Mean tilt over the calibration block, from the calibrated carrier."""
    return absolute_tilt(cal.f0x, cal.f0y)[2]


def tilt_deviation(ux, uy, cal):
    """
    Per-frame tilt referenced to the calibration working point.

    ux, uy may be scalars or arrays.  Returns (dtheta_x, dtheta_y).
    """
    k = tilt_scale()
    return (ux - cal.f0x) * k, (uy - cal.f0y) * k
