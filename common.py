"""
common.py — the stage shared by both branches.

Covers the protocol's Steps 1-4:

    Step 1   2D FFT
    Step 2   envelope extraction I_env, beam mask, weight map
    Step 3   restrict attention to the +f0 sideband
    Step 4   sub-bin parabolic refinement of the carrier f0

Output of this module is everything both the tilt branch (Step 5) and the
displacement branch (Steps 6-11) need, and nothing either one owns alone.
"""

from dataclasses import dataclass

import numpy as np


# ─── Fourier and spatial grids ───────────────────────────────────────────────

@dataclass(frozen=True)
class Grids:
    """Precomputed frequency and coordinate grids for an (H, W) frame."""
    FX: np.ndarray      # (H, W) x spatial frequency [cycles/px]
    FY: np.ndarray      # (H, W) y spatial frequency [cycles/px]
    FR: np.ndarray      # (H, W) radial frequency [cycles/px]
    fx_arr: np.ndarray  # (W,)   fftfreq axis
    fy_arr: np.ndarray  # (H,)   fftfreq axis
    XX: np.ndarray      # (H, W) pixel x coordinate
    YY: np.ndarray      # (H, W) pixel y coordinate
    hann: np.ndarray    # (H, W) separable 2D Hann window

    @property
    def dfx(self):
        return self.fx_arr[1] - self.fx_arr[0]

    @property
    def dfy(self):
        return self.fy_arr[1] - self.fy_arr[0]


def make_grids(H, W):
    fy_arr = np.fft.fftfreq(H)
    fx_arr = np.fft.fftfreq(W)
    FX, FY = np.meshgrid(fx_arr, fy_arr, indexing="xy")
    FR     = np.hypot(FX, FY)

    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    hann = np.outer(np.hanning(H), np.hanning(W))

    return Grids(FX=FX, FY=FY, FR=FR, fx_arr=fx_arr, fy_arr=fy_arr,
                 XX=XX, YY=YY, hann=hann)


# ─── Step 4 — sub-bin parabolic interpolation ────────────────────────────────

def parabolic_sub(arr, r, c):
    """
    Separable three-point parabolic refinement around bin (r, c).

    Returns (dr, dc), the sub-bin offsets in row and column, each in units
    of one FFT bin.  Protocol Eq. (8).
    """
    H, W = arr.shape
    r = max(1, min(H - 2, r))
    c = max(1, min(W - 2, c))

    dm, d0, dp = arr[r - 1, c], arr[r, c], arr[r + 1, c]
    den_r = 2 * dm - 4 * d0 + 2 * dp
    dr = (dm - dp) / den_r if abs(den_r) > 1e-12 else 0.0

    lm, lp = arr[r, c - 1], arr[r, c + 1]
    den_c = 2 * lm - 4 * d0 + 2 * lp
    dc = (lm - lp) / den_c if abs(den_c) > 1e-12 else 0.0

    return dr, dc


def locate_peak(amp, region_mask, grids):
    """
    Argmax within region_mask, then sub-bin refine.

    Returns (fx, fy, r_pk, c_pk) with fx, fy in cycles/px.
    """
    masked = amp * region_mask
    r_pk, c_pk = np.unravel_index(np.argmax(masked), masked.shape)
    dr, dc = parabolic_sub(masked, r_pk, c_pk)

    fy = grids.fy_arr[r_pk] + dr * grids.dfy
    fx = grids.fx_arr[c_pk] + dc * grids.dfx

    return fx, fy, r_pk, c_pk


# ─── Steps 1-4 — calibration ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Calibration:
    """Frame-independent quantities derived from the calibration block."""
    I_env: np.ndarray        # (H, W) intensity envelope
    mask:  np.ndarray        # (H, W) bool beam mask
    w:     np.ndarray        # (H, W) weight map, I_env inside mask
    a:     np.ndarray        # (H, W) normalised weights, sum = 1   (Eq. 16)
    f0x:   float             # carrier, cycles/px
    f0y:   float
    f0_mag: float
    n_eff: float             # effective pixel count
    n_cal: int


def extract_envelope(mean_frame, grids, cfg):
    """
    Steps 1-2.  Low-pass the DC neighbourhood and inverse-transform.

    Returns (I_env, mask, w, a, n_eff).
    """
    F = np.fft.fft2(mean_frame)
    F[grids.FR > cfg.r_lp] = 0.0
    I_env = np.real(np.fft.ifft2(F))

    mask = I_env > cfg.i_min_frac * np.max(I_env)
    w = I_env * mask
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    a = w / w.sum()

    return I_env, mask, w, a, n_eff


def find_carrier(mean_amp, grids, cfg):
    """
    Steps 3-4.  Locate the carrier inside the search annulus.

    mean_amp is an incoherently averaged |FFT| over the calibration block:
    averaging magnitudes keeps the sideband even when the fringes partially
    cancel in the coherent frame average.
    """
    annulus = (grids.FR >= cfg.annulus_lo) & (grids.FR <= cfg.annulus_hi)
    f0x, f0y, _, _ = locate_peak(mean_amp, annulus, grids)
    return f0x, f0y, float(np.hypot(f0x, f0y))


def calibrate(capture, grids, cfg):
    """Run Steps 1-4 over the first n_cal frames."""
    n_cal = min(cfg.n_cal, capture.n_frames // 4)
    H, W = capture.height, capture.width

    acc     = np.zeros((H, W), dtype=np.float64)
    amp_acc = np.zeros((H, W), dtype=np.float64)

    for i in range(n_cal):
        frame = capture.frames[i].astype(np.float64)
        acc     += frame
        amp_acc += np.abs(np.fft.fft2(frame))

    I_env, mask, w, a, n_eff = extract_envelope(acc / n_cal, grids, cfg)
    f0x, f0y, f0_mag = find_carrier(amp_acc / n_cal, grids, cfg)

    return Calibration(I_env=I_env, mask=mask, w=w, a=a,
                       f0x=f0x, f0y=f0y, f0_mag=f0_mag,
                       n_eff=n_eff, n_cal=n_cal)


# ─── per-frame shared work ───────────────────────────────────────────────────

@dataclass(frozen=True)
class FrameCache:
    """Per-frame constants precomputed once from the calibration."""
    search_mask: np.ndarray   # (H, W) float32, disc around f0
    dc_window:   np.ndarray   # (H, W) float32, low-pass centred at DC
    mask_f32:    np.ndarray
    hann_f32:    np.ndarray
    XX32:        np.ndarray
    YY32:        np.ndarray


def build_frame_cache(cal, grids, cfg):
    r_search = cfg.search_frac * cal.f0_mag
    dist_f0  = np.hypot(grids.FX - cal.f0x, grids.FY - cal.f0y)
    search_mask = (dist_f0 < r_search).astype(np.float32)

    # After demodulation the +f0 sideband sits at the origin, so the
    # isolating window is centred on DC rather than on f0.
    r_win = cfg.r_win_frac * cal.f0_mag
    dc_window = (grids.FR < r_win).astype(np.float32)

    return FrameCache(search_mask=search_mask,
                      dc_window=dc_window,
                      mask_f32=cal.mask.astype(np.float32),
                      hann_f32=grids.hann.astype(np.float32),
                      XX32=grids.XX.astype(np.float32),
                      YY32=grids.YY.astype(np.float32))


def prepare_frame(frame, cache):
    """Mask, mean-subtract, window.  Returns float32 (H, W)."""
    inside = cache.mask_f32 > 0
    return (frame - np.mean(frame[inside])) * cache.mask_f32 * cache.hann_f32


def measure_carrier(proc, cache, grids):
    """
    Steps 3-4 applied per frame.

    Returns (ux, uy, amplitude): the refined carrier for this frame and the
    sideband peak magnitude, normalised by the frame size.
    """
    F   = np.fft.fft2(proc)
    amp = np.abs(F)

    ux, uy, r_pk, c_pk = locate_peak(amp, cache.search_mask, grids)
    amplitude = amp[r_pk, c_pk] / amp.size

    return ux, uy, amplitude
