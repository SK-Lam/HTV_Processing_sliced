"""
displacement.py — the piston branch, protocol Steps 6-11.

    Step 6    phi_tilt = 2 pi f0 . r
    Step 7    B = A exp(-i phi_tilt) = (1/2) I_env V exp(i phi_0)
    Step 10   phi_dis = arg( SUM a(x,y) B(x,y) )        Eq. (18)
    Step 11   dz = (lambda / 4 pi) phi_dis              Eq. (19)

Steps 8-9 (the V and phi_0 maps) are diagnostics of B and are not needed
for the scalar estimate, so they are not computed here.
"""

import numpy as np

from .config import LAM


# ─── Steps 6-7 — de-ramp to the complex field B ──────────────────────────────

def deramp(proc, ux, uy, cache):
    """
    Demodulate the carrier and isolate the sideband.

    Multiplying the spatial frame by exp(-i phi_tilt) translates the +f0
    sideband to the origin; the DC-centred window then isolates it.  This
    combines Steps 3, 6 and 7 into a single pass and yields B directly.
    """
    ramp    = np.exp(-2j * np.pi * (ux * cache.XX32 + uy * cache.YY32)).astype(np.complex64)
    shifted = proc.astype(np.complex64) * ramp

    F = np.fft.fft2(shifted) * cache.dc_window
    return np.fft.ifft2(F)


# ─── Step 10 — weighted estimator ────────────────────────────────────────────

def estimate_phase(B, a):
    """
    Phasor-weighted scalar phase, Eq. (18).

    Weighting the complex field before taking the argument avoids the 2D
    spatial unwrap that the angle-weighted form, Eq. (17), would require.
    """
    return float(np.angle(np.sum(a * B)))


# ─── frame quality ───────────────────────────────────────────────────────────

def detect_bad_frames(amplitude, cfg):
    """
    Flag frames whose fringe amplitude collapses, plus a guard band.

    Dropped or dark frames show near-zero sideband amplitude.  Guarding
    either side keeps partially-corrupted neighbours out of the analysis.
    """
    bad = amplitude < cfg.amp_slip_frac * np.median(amplitude)

    if cfg.guard > 0:
        padded = bad.copy()
        for shift in range(1, cfg.guard + 1):
            padded[shift:]  |= bad[:-shift]
            padded[:-shift] |= bad[shift:]
        bad = padded

    return bad


# ─── Step 11 — unwrap and convert ────────────────────────────────────────────

def displacement_scale():
    """lambda / (4 pi) — metres per radian."""
    return LAM / (4.0 * np.pi)


def unwrap_phase(phi, bad_mask):
    """
    Unwrap the per-frame phase, bridging flagged frames.

    np.unwrap integrates along the record, so a single corrupted sample
    offsets everything after it.  Interpolating across flagged frames first
    gives the unwrap a continuous trajectory; the flags are then restored
    as NaN so downstream spectral estimation still excludes them.
    """
    phi_clean = phi.copy()
    good = np.where(~bad_mask)[0]
    bad  = np.where(bad_mask)[0]

    if len(bad) and len(good) >= 2:
        phi_clean[bad] = np.interp(bad, good, phi[good])

    unwrapped = np.unwrap(phi_clean)
    unwrapped[bad_mask] = np.nan
    return unwrapped


def phase_to_displacement(phi_unwrapped):
    """dz = (lambda / 4 pi) phi.  Eq. (19)."""
    return displacement_scale() * phi_unwrapped
