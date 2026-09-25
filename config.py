"""
config.py — all pipeline constants in one place.

Physical constants are module-level (they describe the apparatus).
Everything tunable lives in the Config dataclass so it can be swept.
"""

from dataclasses import dataclass

# ─── physical constants of the setup ─────────────────────────────────────────
LAM         = 688e-9      # probe wavelength [m]
PIXEL_PITCH = 4e-6        # sensor pixel pitch [m]
M_OPT       = 1.0         # magnification between mirror and sensor


@dataclass(frozen=True)
class Config:
    # Step 2 — envelope extraction
    n_cal:       int   = 300     # calibration frames
    r_lp:        float = 0.007   # DC low-pass radius [cycles/px]
    i_min_frac:  float = 0.15    # beam-mask threshold, fraction of envelope peak

    # Steps 3-4 — carrier isolation and refinement
    annulus_lo:  float = 0.007   # f0 search annulus inner radius [cycles/px]
    annulus_hi:  float = 0.040   # f0 search annulus outer radius [cycles/px]
    search_frac: float = 0.10    # per-frame search disc radius, fraction of |f0|
    r_win_frac:  float = 0.10    # sideband filter radius, fraction of |f0|

    # Steps 10-11 — frame quality
    amp_slip_frac: float = 0.25  # amplitude below this fraction of median -> bad
    guard:         int   = 2     # guard frames either side of a bad frame

    # spectral analysis
    nperseg_hz:  int   = 791     # Welch nperseg -> df = 1 Hz at 791 fps
    int_lo:      float = 1.0     # band-integration lower limit [Hz]

    # execution
    chunk:       int   = 2000    # frames per processing chunk

    # targets
    target_theta_nrad: float = 10.0
    target_v_um_per_s: float = 1.0
