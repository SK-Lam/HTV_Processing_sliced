"""
htv — Fourier-domain fringe demodulation for Michelson interferograms.

Module layout follows the protocol's own stage boundaries:

    common        Steps 1-4    envelope, mask, weights, carrier f0
    tilt          Step 5       f0 -> theta
    displacement  Steps 6-11   de-ramp, weighted estimator, unwrap, dz

    htv_io        capture reading
    spectra       Welch PSDs and RMS integrals
    report        CSV / JSON / figures
    pipeline      orchestration
"""

from .config import Config
from .pipeline import run

__all__ = ["Config", "run"]
