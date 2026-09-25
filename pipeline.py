"""
pipeline.py — orchestration.

Owns the loop over frames and the order of operations; contains no physics
of its own.  Each frame is read once and fed to both branches: the tilt
branch needs only the carrier location, the displacement branch needs the
de-ramped field.

Usage:  python3 -m htv.pipeline capture.htv
"""

import os
import sys

import numpy as np

from . import common, displacement, htv_io, report, spectra, tilt
from .config import Config


def run(input_path, cfg=None):
    cfg = cfg or Config()

    # ── ingest ───────────────────────────────────────────────────────────
    capture = htv_io.open_capture(input_path)
    print(f"[1] {capture.width}×{capture.height}, {capture.n_frames} frames, "
          f"{capture.fps:.2f} fps")

    grids = common.make_grids(capture.height, capture.width)

    # ── Steps 1-4: shared stage ──────────────────────────────────────────
    print("[2] Calibrate")
    cal = common.calibrate(capture, grids, cfg)
    cache = common.build_frame_cache(cal, grids, cfg)
    theta_wp = tilt.working_point(cal)
    print(f"    f0 = ({cal.f0x:.5f}, {cal.f0y:.5f}) cycles/px")
    print(f"    working point θ = {theta_wp * 1e6:.2f} µrad, "
          f"N_eff = {cal.n_eff:.0f} px")

    # ── per-frame loop ───────────────────────────────────────────────────
    N = capture.n_frames
    phi_all = np.empty(N, dtype=np.float64)
    ux_all  = np.empty(N, dtype=np.float64)
    uy_all  = np.empty(N, dtype=np.float64)
    amp_all = np.empty(N, dtype=np.float64)

    a_weights = cal.a

    print(f"[3] Process {N} frames")
    for start, end, block in htv_io.iter_chunks(capture, cfg.chunk):
        for k in range(end - start):
            proc = common.prepare_frame(block[k], cache)

            # shared: Steps 3-4
            ux, uy, amplitude = common.measure_carrier(proc, cache, grids)

            # displacement branch: Steps 6-7, 10
            B = displacement.deramp(proc, ux, uy, cache)
            phi = displacement.estimate_phase(B, a_weights)

            i = start + k
            ux_all[i]  = ux
            uy_all[i]  = uy
            amp_all[i] = amplitude
            phi_all[i] = phi

        print(f"    {end}/{N} ({100 * end / N:.0f}%)", flush=True)

    # ── Step 5: tilt branch ──────────────────────────────────────────────
    print("[4] Tilt")
    thx, thy = tilt.tilt_deviation(ux_all, uy_all, cal)

    # ── Step 11: displacement branch ─────────────────────────────────────
    print("[5] Displacement")
    bad_mask = displacement.detect_bad_frames(amp_all, cfg)
    phi_uw   = displacement.unwrap_phase(phi_all, bad_mask)
    dz       = displacement.phase_to_displacement(phi_uw)
    if bad_mask.any():
        print(f"    {bad_mask.sum()} bad/guarded frames "
              f"({100 * bad_mask.mean():.1f}%)")

    # ── spectra ──────────────────────────────────────────────────────────
    print("[6] Spectra")
    f, Szz, Svv, Stt, Sxx, Syy, n_seg, n_total = spectra.compute_psds(
        dz, thx, thy, capture.fps, bad_mask, cfg)
    print(f"    Welch: {n_seg}/{n_total} clean segments")

    f_nyq     = capture.f_nyquist
    theta_rms = spectra.band_rms(f, Stt, cfg.int_lo, f_nyq)
    v_rms     = spectra.band_rms(f, Svv, cfg.int_lo, f_nyq)

    # ── outputs ──────────────────────────────────────────────────────────
    print("[7] Write")
    paths = report.output_paths(input_path)
    report.write_frames_csv(paths["frames_csv"], capture.fps,
                            phi_uw, dz, thx, thy, bad_mask)
    report.write_psd_csv(paths["psd_csv"], f, Szz, Svv, Stt, Sxx, Syy, f_nyq)
    report.plot_psds(paths["psd_png"], os.path.basename(input_path),
                     capture.fps, f, Szz, Svv, Stt, Sxx, Syy, f_nyq)
    report.write_summary_json(paths["summary"], report.build_summary(
        input_path, capture, cal, theta_wp, bad_mask,
        f, theta_rms, v_rms, cfg, paths))
    for p in paths.values():
        print(f"    → {p}")

    print("─" * 60)
    print(f"  θ_RMS = {theta_rms * 1e9:.2f} nrad  "
          f"(target < {cfg.target_theta_nrad:.0f} nrad)")
    print(f"  v_RMS = {v_rms * 1e6:.3f} µm/s  "
          f"(target < {cfg.target_v_um_per_s:.0f} µm/s)")
    print("─" * 60)

    return {"f": f, "Szz": Szz, "Svv": Svv, "Stt": Stt,
            "theta_rms": theta_rms, "v_rms": v_rms,
            "phi": phi_uw, "dz": dz, "thx": thx, "thy": thy,
            "bad_mask": bad_mask, "calibration": cal}


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 -m htv.pipeline capture.htv")
        sys.exit(1)
    run(sys.argv[1])


if __name__ == "__main__":
    main()
