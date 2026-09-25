"""
report.py — writing results out.  No physics here.
"""

import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .spectra import cumulative_rms


def write_frames_csv(path, fps, phi, dz, thx, thy, bad_mask):
    t = np.arange(len(phi)) / fps
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t_s", "phi_rad", "delta_z_m",
                     "theta_x_rad", "theta_y_rad", "bad"])
        for i in range(len(phi)):
            wr.writerow([f"{t[i]:.6f}", f"{phi[i]:.8e}", f"{dz[i]:.8e}",
                         f"{thx[i]:.8e}", f"{thy[i]:.8e}", int(bad_mask[i])])
    return path


def write_psd_csv(path, f, Szz, Svv, Stt, Sxx, Syy, f_nyq):
    f_c, cum_t = cumulative_rms(f, Stt, f_nyq)
    _,   cum_v = cumulative_rms(f, Svv, f_nyq)
    cum_t_full = np.interp(f, f_c, cum_t)
    cum_v_full = np.interp(f, f_c, cum_v)

    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["freq_hz",
                     "Szz_m2_per_hz", "Svv_ms2_per_hz",
                     "Stt_rad2_per_hz", "Sxx_rad2_per_hz", "Syy_rad2_per_hz",
                     "ASD_tilt_nrad_per_sqrthz", "ASD_vel_um_per_sqrthz",
                     "cum_rms_tilt_nrad", "cum_rms_vel_um_per_s"])
        for i in range(len(f)):
            wr.writerow([
                f"{f[i]:.6f}",
                f"{Szz[i]:.6e}", f"{Svv[i]:.6e}",
                f"{Stt[i]:.6e}", f"{Sxx[i]:.6e}", f"{Syy[i]:.6e}",
                f"{np.sqrt(Stt[i]) * 1e9:.6e}", f"{np.sqrt(Svv[i]) * 1e6:.6e}",
                f"{cum_t_full[i] * 1e9:.6e}", f"{cum_v_full[i] * 1e6:.6e}",
            ])
    return path


def plot_psds(path, label, fps, f, Szz, Svv, Stt, Sxx, Syy, f_nyq):
    f_c, cum_t = cumulative_rms(f, Stt, f_nyq)
    _,   cum_v = cumulative_rms(f, Svv, f_nyq)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.loglog(f[1:], Stt[1:], label="Sxx + Syy")
    ax.loglog(f[1:], Sxx[1:], lw=0.8, alpha=0.6, label=r"$\theta_x$")
    ax.loglog(f[1:], Syy[1:], lw=0.8, alpha=0.6, label=r"$\theta_y$")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel(r"PSD [rad$^2$/Hz]")
    ax.set_title("Tilt PSD"); ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[0, 1]
    ax.loglog(f[1:], Svv[1:], color="C1")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel(r"PSD [(m/s)$^2$/Hz]")
    ax.set_title("Mirror velocity PSD")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1, 0]
    ax.loglog(f[1:], np.sqrt(Stt[1:]) * 1e9)
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel(r"ASD [nrad/$\sqrt{Hz}$]")
    ax.set_title("Tilt ASD")
    ax.grid(True, which="both", alpha=0.3)

    ax  = axes[1, 1]
    ax2 = ax.twinx()
    ax.semilogy(f_c[1:],  cum_t[1:] * 1e9, color="C0", label=r"$\theta_{RMS}$ [nrad]")
    ax2.semilogy(f_c[1:], cum_v[1:] * 1e6, color="C1", label=r"$v_{RMS}$ [µm/s]")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Cumulative RMS tilt [nrad]", color="C0")
    ax2.set_ylabel("Cumulative RMS velocity [µm/s]", color="C1")
    ax.set_title("Cumulative RMS (high → low)")
    ax.grid(True, which="both", alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)

    plt.suptitle(f"{label}   λ = 688 nm   fps = {fps:.1f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_summary_json(path, payload):
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


def build_summary(input_path, capture, cal, theta_wp, bad_mask,
                  f, theta_rms, v_rms, cfg, outputs):
    n_bad = int(bad_mask.sum())
    return {
        "input_file": input_path,
        "capture": {
            "width_px":   capture.width,
            "height_px":  capture.height,
            "n_frames":   capture.n_frames,
            "fps":        capture.fps,
            "duration_s": capture.duration_s,
        },
        "calibration": {
            "n_cal_frames":         cal.n_cal,
            "f0x_cycles_per_px":    cal.f0x,
            "f0y_cycles_per_px":    cal.f0y,
            "f0_mag_cycles_per_px": cal.f0_mag,
            "theta_wp_urad":        theta_wp * 1e6,
            "N_eff_px":             cal.n_eff,
        },
        "bad_frames": {
            "n_bad":    n_bad,
            "fraction": n_bad / capture.n_frames,
        },
        "spectral": {
            "f_min_hz": float(f[1]),
            "f_max_hz": float(f[-1]),
            "df_hz":    float(f[1] - f[0]),
            "integration_band_lo_hz": cfg.int_lo,
            "integration_band_hi_hz": capture.f_nyquist,
        },
        "results": {
            "theta_rms_nrad":    theta_rms * 1e9,
            "v_rms_um_per_s":    v_rms * 1e6,
            "target_theta_nrad": cfg.target_theta_nrad,
            "target_v_um_per_s": cfg.target_v_um_per_s,
            "theta_target_met":  bool(theta_rms * 1e9 < cfg.target_theta_nrad),
            "v_target_met":      bool(v_rms * 1e6 < cfg.target_v_um_per_s),
        },
        "outputs": outputs,
    }


def output_paths(input_path):
    base = os.path.splitext(input_path)[0]
    return {
        "frames_csv": base + "_frames.csv",
        "psd_csv":    base + "_psd.csv",
        "psd_png":    base + "_psd.png",
        "summary":    base + "_summary.json",
    }
