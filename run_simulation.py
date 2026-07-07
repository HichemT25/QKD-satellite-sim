import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry import PassGeometry
from channel import ChannelParams, eta_total as eta_tot_comp, alpha_zenith
from qkd_bb84 import SourceParams, detection_probabilities, qber, secure_key_rate, qber_max
# qkd_decoy_finite_key depends on the optional tno.quantum.communication.qkd_key_rate
# package (see requirements.txt); imported lazily so that --skip-decoy runs (and the
# new wavelength/low-elevation/ensemble analyses, none of which need it) work without
# that dependency installed.

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FIGDIR = os.path.join(OUTDIR, "figures")
TABDIR = os.path.join(OUTDIR, "tables")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7.5, "font.family": "serif",
})


def run_pass(nt=4000, emin=5.0, emax=80.0, alt=550.0, cp=None, sp=None):
    cp = cp or ChannelParams()
    sp = sp or SourceParams()
    geo = PassGeometry(alt=alt, emin=emin, emax=emax)
    th = geo.half_dur()
    t = np.linspace(-th, th, nt)
    e = geo.elev(t)
    l = geo.slant(t)
    comp = eta_tot_comp(l, e, cp)
    eta = comp["eta_total"]
    psig, pdc, pbg, pclick = detection_probabilities(eta, sp)
    q = qber(psig, pdc, pbg, pclick, sp)
    r = secure_key_rate(pclick, q, sp)
    return {
        "t": t, "e_deg": np.degrees(e), "l_km": l,
        "geo": geo, "cp": cp, "sp": sp,
        **comp,
        "psig": psig, "pdc": pdc, "pbg": pbg, "pclick": pclick,
        "qber": q, "skr": r,
    }


def fig_transmittance(d, fn="fig_transmittance.pdf"):
    t = d["t"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.semilogy(t, d["eta_total"], lw=2.0, color="#1a5fb4", label=r"$\eta_{\mathrm{total}}$")
    ax.semilogy(t, d["eta_fs"], lw=1.2, ls="--", color="#26a269", label=r"$\eta_{\mathrm{fs}}$")
    ax.semilogy(t, d["eta_atm"], lw=1.1, ls=":", color="#c01c28", label=r"$\eta_{\mathrm{atm}}$")
    ax.semilogy(t, d["eta_turb"], lw=1.1, ls="-.", color="#9a9996", label=r"$\eta_{\mathrm{turb}}$")
    ax.semilogy(t, d["eta_point"], lw=1.0, ls=(0, (3, 1, 1, 1)), color="#e5a50a", label=r"$\eta_{\mathrm{point}}$")
    ax.set_xlabel("Time during satellite pass (s)")
    ax.set_ylabel("Channel transmittance")
    ax.set_xlim(t.min(), t.max())
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fn))
    plt.close(fig)


def fig_detection(d, fn="fig_detection.pdf"):
    t = d["t"]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.semilogy(t, d["pclick"], color="k", lw=1.8, label=r"$P_{\mathrm{click}}(t)$ (total)")
    ax.semilogy(t, d["psig"], color="#1a5fb4", lw=1.3, label=r"$P_{\mathrm{sig}}(t)$ (signal)")
    ax.semilogy(t, d["pdc"], color="#c01c28", lw=1.0, ls="--", label=r"$P_{\mathrm{dc}}$ (dark counts)")
    ax.semilogy(t, d["pbg"], color="#9a9996", lw=1.0, ls=":", label=r"$P_{\mathrm{bg}}$ (background)")
    ax.set_xlabel("Time during satellite pass (s)")
    ax.set_ylabel("Detection probability (per pulse)")
    ax.set_xlim(t.min(), t.max())
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)

    ai = fig.add_axes([0.22, 0.68, 0.25, 0.18])
    ai.plot(t, d["e_deg"], color="#1a5fb4", lw=1.0)
    ai.set_xticks([])
    ai.set_title("Elevation (deg)", fontsize=6)
    ai.tick_params(labelsize=5.5)

    fig.savefig(os.path.join(FIGDIR, fn))
    plt.close(fig)


def fig_qber(d, fn="fig_qber.pdf", d_low=None, low_label="Low peak-elevation pass (30 deg)"):
    t = d["t"]
    m = t <= 0
    e = d["e_deg"][m]
    q = d["qber"][m]
    en = d["eta_total"][m] / np.nanmax(d["eta_total"])
    o = np.argsort(e)
    es, qs, ens = e[o], q[o], en[o]

    fig, ax1 = plt.subplots(figsize=(5.0, 3.4))
    l1, = ax1.plot(es, qs * 100, color="#1a5fb4", lw=2.0, label="QBER, baseline pass (80 deg peak)")
    handles = [l1]

    # reviewer comment 3: overlay a low-peak-elevation pass on the same axes
    # so the reader can see directly how much of that pass's elevation range
    # ever crosses below the QBER security threshold (baseline pass alone
    # cannot show this, since it always reaches deep into the signal-dominated
    # regime near zenith).
    if d_low is not None:
        tl = d_low["t"]
        ml = tl <= 0
        el = d_low["e_deg"][ml]
        ql = d_low["qber"][ml]
        ol = np.argsort(el)
        l3, = ax1.plot(el[ol], ql[ol] * 100, color="#c01c28", lw=1.8, ls="--", label=f"QBER, {low_label}")
        handles.append(l3)

    qm = qber_max(d["sp"].f_EC) * 100
    l4 = ax1.axhline(qm, color="#5e5c64", lw=1.0, ls=":", label=r"QBER$_{\max}$ security threshold")
    handles.append(l4)

    ax1.set_xlabel("Satellite elevation angle (degrees)")
    ax1.set_ylabel("Quantum bit error rate, QBER (%)")
    ax1.set_xlim(min(es.min(), (d_low["e_deg"].min() if d_low is not None else es.min())), es.max())
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    l2, = ax2.plot(es, ens, color="#77767b", lw=1.3, ls="--", label=r"Normalized $\eta_{\mathrm{total}}$ (baseline)")
    ax2.set_ylabel("Normalized channel transmittance")
    ax2.set_ylim(0, 1.05)
    handles.append(l2)

    ax1.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fn))
    plt.close(fig)


def fig_skr(d, fn="fig_skr.pdf"):
    t = d["t"]
    r = d["skr"]
    en = d["eta_total"] / np.nanmax(d["eta_total"])

    fig, ax1 = plt.subplots(figsize=(5.0, 3.4))
    m = r > 0
    if m.any():
        ax1.semilogy(t[m], np.clip(r[m], 1, None), color="#1a5fb4", lw=2.0, label="Secure key rate (SKR)")
        ax1.axvspan(t[m].min(), t[m].max(), color="#26a269", alpha=0.12)

    ax2 = ax1.twinx()
    ax2.plot(t, en, color="#9a9996", lw=1.2, ls="--", label=r"Normalized $\eta_{\mathrm{total}}$")
    ax2.set_ylabel("Normalized channel transmittance")
    ax2.set_ylim(0, 1.05)

    ax1.set_xlabel("Time during satellite pass (s)")
    ax1.set_ylabel("Secure key rate (bits/s)")
    ax1.set_xlim(t.min(), t.max())
    ax1.grid(True, which="both", alpha=0.25)

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fn))
    plt.close(fig)


def fig_skr_decoy(d, fn="fig_skr_decoy_comparison.pdf", npulse=1e10, nd=2,
                   na=8, nf=4, tmax_a=90, tmax_f=120):
    from qkd_decoy_finite_key import (
        DecoyFiniteKeyParams,
        decoy_skr_asymptotic,
        decoy_skr_finite_key,
    )
    t = d["t"]
    ec = d["eta_total"]
    dp = DecoyFiniteKeyParams(f_rep=d["sp"].f_rep, e_opt=d["sp"].e_opt,
                               dc=d["sp"].D, eta_det=d["cp"].eta_det, n=npulse, nd=nd)

    mp = d["skr"] > 0
    if mp.sum() >= na:
        tp = t[mp]
        fr = np.linspace(0, 1, na // 2)
        th = tp.max() * 0.97 * fr ** 1.5
        ta = np.unique(np.concatenate([-th, th]))
    else:
        ta = np.linspace(t.min() * 0.5, t.max() * 0.5, na)
    ea = np.interp(ta, t, ec)

    print(f"  evaluating {len(ta)} multi-decoy asymptotic points (tmax={tmax_a}s each)...")
    sa = decoy_skr_asymptotic(ea, dp, multi_decoy=True, verbose=True, tmax=tmax_a)

    o = np.argsort(ta)
    tg, sg = ta[o], sa[o]
    sfull = np.interp(t, tg, sg, left=0.0, right=0.0)

    if mp.sum() >= nf:
        tmaxp = t[mp].max()
        fr = np.linspace(0.0, 0.75, nf)
        ts = tmaxp * fr
    else:
        ts = np.linspace(t.min() * 0.3, t.max() * 0.3, nf)
    es = np.interp(ts, t, ec)

    print(f"  evaluating {nf} rigorous finite-key points (tmax={tmax_f}s each)...")
    sf, params_f = decoy_skr_finite_key(es, dp, verbose=True, tmax=tmax_f)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if mp.any():
        ax.semilogy(t[mp], np.clip(d["skr"][mp], 1, None), color="#9a9996", lw=1.6, ls="--",
                    label="Simplified asymptotic (Eq. 12)")
    m2 = sfull > 0
    if m2.any():
        ax.semilogy(t[m2], np.clip(sfull[m2], 1, None), color="#1a5fb4", lw=2.0,
                    label="Multi-decoy asymptotic\n(BB84AsymptoticKeyRateEstimate)")
    pp = sf > 0
    if pp.any():
        ax.semilogy(ts[pp], np.clip(sf[pp], 1, None), "o", color="#c01c28", markersize=6,
                    zorder=5, label=f"Rigorous finite-key\n($N={npulse:.0e}$ pulses, 2-decoy)")

    ax.set_xlabel("Time during satellite pass (s)")
    ax.set_ylabel("Secure key rate (bits/s)")
    ax.set_xlim(t.min(), t.max())
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fn))
    plt.close(fig)

    return dict(t_sample=ts, eta_sample=es, skr_finite_pts=sf,
                t_asymp_grid=tg, eta_asymp_grid=ea[o], skr_asymp_grid=sg,
                finite_key_params=params_f, number_of_decoy=nd, n_pulses=npulse)


def fig_availability(d, fn="fig_availability.pdf"):
    t = d["t"]
    r = d["skr"]
    m = r > 0
    tt = t.max() - t.min()
    tk = (t[m].max() - t[m].min()) if m.any() else 0.0
    av = tk / tt

    fig = plt.figure(figsize=(5.2, 3.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.0], hspace=0.05)

    a0 = fig.add_subplot(gs[0])
    a0.plot(t, d["e_deg"], color="#1a5fb4", lw=1.3)
    a0.set_xticks([])
    a0.set_ylabel("Elevation\n(deg)", fontsize=7)
    a0.set_xlim(t.min(), t.max())
    a0.tick_params(labelsize=6.5)

    a1 = fig.add_subplot(gs[1])
    a1.hlines(1, t.min(), t.max(), color="k", lw=1.3)
    a1.text(t.min(), 1.08, "AOS", fontsize=7.5, ha="left")
    a1.text(t.max(), 1.08, "LOS", fontsize=7.5, ha="right")

    if m.any():
        a1.axvspan(t[m].min(), t[m].max(), ymin=0.05, ymax=0.55, color="#26a269", alpha=0.35)
        a1.text(0, 0.3, f"Positive SKR region\n({av*100:.0f}% of pass)", ha="center", va="center", fontsize=7.5)

    a1.set_xlim(t.min(), t.max())
    a1.set_ylim(0, 1.2)
    a1.set_yticks([])
    a1.set_xlabel("Time during satellite pass (s)")
    for sp in ["top", "right", "left"]:
        a1.spines[sp].set_visible(False)

    fig.savefig(os.path.join(FIGDIR, fn), bbox_inches="tight")
    plt.close(fig)
    return av


def stats(d):
    eta = d["eta_total"]
    q = d["qber"]
    r = d["skr"]
    t = d["t"]

    emax, emin = np.nanmax(eta), np.nanmin(eta)

    mr = t <= 0
    er, qr = d["e_deg"][mr], q[mr]
    o = np.argsort(er)
    qlo, qhi = qr[o][0] * 100, qr[o][-1] * 100

    rpk = np.nanmax(r)
    mp = r > 0
    tt = t.max() - t.min()
    tk = (t[mp].max() - t[mp].min()) if mp.any() else 0.0
    av = tk / tt
    qm = qber_max(d["sp"].f_EC) * 100

    return {
        "eta_max": emax, "eta_min": emin,
        "qber_lo": qlo, "qber_hi": qhi, "qber_max": qm,
        "skr_pk": rpk, "avail": av * 100, "t_tot": tt, "t_key": tk,
    }


def write_summary(s, fn="table_baseline_summary.csv"):
    p = os.path.join(TABDIR, fn)
    with open(p, "w") as f:
        f.write("metric,value,unit\n")
        f.write(f"eta_total_max,{s['eta_max']:.4e},dimensionless\n")
        f.write(f"eta_total_min,{s['eta_min']:.4e},dimensionless\n")
        f.write(f"QBER_low_elevation,{s['qber_lo']:.2f},percent\n")
        f.write(f"QBER_high_elevation,{s['qber_hi']:.2f},percent\n")
        f.write(f"QBER_max_threshold,{s['qber_max']:.2f},percent\n")
        f.write(f"SKR_peak,{s['skr_pk']:.4e},bits_per_second\n")
        f.write(f"link_availability,{s['avail']:.1f},percent\n")
        f.write(f"total_visibility_duration,{s['t_tot']:.1f},seconds\n")
        f.write(f"effective_key_window,{s['t_key']:.1f},seconds\n")
    print(f"[written] {p}")


def write_decoy_param_appendix(ts, es, params, number_of_decoy, npulse,
                                fn="table_decoy_state_parameters_appendix.csv"):
    """Reviewer comment 4: write the exact, labeled decoy-state/finite-key
    optimization parameters (signal + decoy intensities mu, and the
    per-basis probabilities of choosing each intensity) underlying each
    rigorous finite-key point plotted in fig_skr_decoy_comparison.pdf, so
    the paper's appendix can cite concrete numbers instead of only naming
    the external library. See qkd_decoy_finite_key.label_decoy_params for
    the parameter-vector convention (Attema et al. 2021)."""
    from qkd_decoy_finite_key import label_decoy_params
    p = os.path.join(TABDIR, fn)
    n = number_of_decoy + 1
    with open(p, "w") as f:
        f.write(f"# number_of_decoy={number_of_decoy}, n_pulses={npulse:.3e}\n")
        f.write("t_s,eta_total," + ",".join(f"mu_{i}" for i in range(n)) + ","
                + ",".join(f"pX_{i}" for i in range(n)) + ","
                + ",".join(f"pZ_{i}" for i in range(n)) + "\n")
        for ti, ei, xi in zip(ts, es, params):
            lab = label_decoy_params(xi, number_of_decoy)
            # lab may be None, a "raw" fallback dict, an unrecognised dict with
            # arbitrary keys, or the expected {"mu": [...], "p_X": [...], "p_Z": [...]}
            if lab is None or "raw" in lab:
                f.write(f"{ti:.2f},{ei:.4e}," + ",".join(["nan"] * (3 * n)) + "\n")
                continue
            if "mu" in lab and "p_X" in lab and "p_Z" in lab:
                vals = lab["mu"] + lab["p_X"] + lab["p_Z"]
                f.write(f"{ti:.2f},{ei:.4e}," + ",".join(f"{v:.6f}" for v in vals) + "\n")
            else:
                # unrecognised dict structure (e.g. future TNO API change):
                # write the raw key=value pairs as a single comment-style entry
                raw_str = ";".join(f"{k}={v}" for k, v in lab.items())
                f.write(f"{ti:.2f},{ei:.4e},{raw_str}" + ",nan" * max(0, 3 * n - 1) + "\n")
    print(f"[written] {p}")


def write_decoy_table(dr, fn="table_decoy_comparison.csv"):
    p = os.path.join(TABDIR, fn)
    with open(p, "w") as f:
        f.write("series,t_s,eta_total,SKR_bps\n")
        for ti, ei, si in zip(dr["t_sample"], dr["eta_sample"], dr["skr_finite_pts"]):
            f.write(f"finite_key_2decoy,{ti:.2f},{ei:.4e},{si:.4e}\n")
        for ti, ei, si in zip(dr["t_asymp_grid"], dr["eta_asymp_grid"], dr["skr_asymp_grid"]):
            f.write(f"multi_decoy_asymptotic,{ti:.2f},{ei:.4e},{si:.4e}\n")
    print(f"[written] {p}")


def run_wavelength_comparison(alt=550.0, emin=5.0, emax=80.0):
    """Reviewer comment 2: parametric comparison of the baseline 785 nm link
    against a 1550 nm (telecom-band) configuration, using the same
    trajectory. The wavelength enters the model in two competing ways:
      (i) beam divergence increases with lambda (theta ~ lambda/(pi w0)),
          which INCREASES free-space geometric loss eta_fs at fixed range;
      (ii) both atmospheric extinction (Rayleigh term ~ lambda^-4.09) and
          daytime sky background (~lambda^-4 empirical scaling) DECREASE
          substantially at 1550 nm.
    This function runs both configurations and reports the net effect on
    eta_total, QBER, and SKR so the trade-off is quantified rather than
    asserted."""
    cp785 = ChannelParams(lam_nm=785.0)
    sp785 = SourceParams(lam_nm=785.0)
    d785 = run_pass(alt=alt, emin=emin, emax=emax, cp=cp785, sp=sp785)

    cp1550 = ChannelParams(lam_nm=1550.0)
    sp1550 = SourceParams(lam_nm=1550.0)
    d1550 = run_pass(alt=alt, emin=emin, emax=emax, cp=cp1550, sp=sp1550)

    s785, s1550 = stats(d785), stats(d1550)
    rows = [
        {"wavelength_nm": 785.0, "a0": cp785.a0, "bg_rate_Hz": sp785.bg_rate,
         "eta_fs_peak": np.nanmax(d785["eta_fs"]), **s785},
        {"wavelength_nm": 1550.0, "a0": cp1550.a0, "bg_rate_Hz": sp1550.bg_rate,
         "eta_fs_peak": np.nanmax(d1550["eta_fs"]), **s1550},
    ]
    return d785, d1550, rows


def fig_wavelength_comparison(d785, d1550, fn="fig_wavelength_comparison.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))

    ax = axes[0]
    ax.semilogy(d785["t"], d785["eta_total"], color="#1a5fb4", lw=1.8, label="785 nm")
    ax.semilogy(d1550["t"], d1550["eta_total"], color="#c01c28", lw=1.8, ls="--", label="1550 nm")
    ax.set_xlabel("Time during satellite pass (s)")
    ax.set_ylabel(r"Channel transmittance $\eta_{\mathrm{total}}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9)

    ax = axes[1]
    m785, m1550 = d785["skr"] > 0, d1550["skr"] > 0
    if m785.any():
        ax.semilogy(d785["t"][m785], np.clip(d785["skr"][m785], 1, None), color="#1a5fb4", lw=1.8, label="785 nm")
    if m1550.any():
        ax.semilogy(d1550["t"][m1550], np.clip(d1550["skr"][m1550], 1, None), color="#c01c28", lw=1.8, ls="--", label="1550 nm")
    ax.set_xlabel("Time during satellite pass (s)")
    ax.set_ylabel("Secure key rate (bits/s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fn))
    plt.close(fig)


def write_wavelength_table(rows, fn="table_wavelength_comparison.csv"):
    p = os.path.join(TABDIR, fn)
    with open(p, "w") as f:
        f.write("wavelength_nm,a0,bg_rate_Hz,eta_fs_peak,eta_max,qber_lo,qber_hi,skr_pk,avail\n")
        for r in rows:
            f.write(f"{r['wavelength_nm']:.0f},{r['a0']:.4f},{r['bg_rate_Hz']:.2f},"
                    f"{r['eta_fs_peak']:.4e},{r['eta_max']:.4e},{r['qber_lo']:.2f},"
                    f"{r['qber_hi']:.2f},{r['skr_pk']:.4e},{r['avail']:.1f}\n")
    print(f"[written] {p}")


def run_pass_ensemble(emax_bins_deg=(10.0, 15.0, 20.0, 30.0, 40.0, 55.0, 70.0, 80.0),
                       n_passes_per_day=5, alt=550.0, emin=5.0, cp=None, sp=None):
    """Reviewer comment 3: automate the propagation/key-rate calculation over
    an ENSEMBLE of representative passes with different peak elevations,
    rather than the single near-zenith pass used for the main figures, to
    support the framework's claim of being a scheduling/planning tool.

    Passes are weighted by an illustrative occurrence distribution
    w(emax) ~ cos(emax): for a satellite whose ground track offset from the
    station is approximately uniformly distributed, larger offsets (which
    are geometrically more numerous, since the annulus of possible offsets
    grows with distance) produce LOWER peak elevations, so most real passes
    over a fixed station are low-elevation ones and only a small minority
    approach zenith. This weighting is a simplified, illustrative
    approximation for system-level planning, not a rigorously derived
    orbital-mechanics result; a full implementation would replace it with
    actual ephemeris propagation (e.g. SGP4) for a specific constellation
    and ground station.
    """
    emax_bins = np.asarray(emax_bins_deg, dtype=float)
    emax_bins = emax_bins[emax_bins > emin]
    weights = np.cos(np.radians(emax_bins))
    weights = weights / weights.sum()

    rows = []
    for em, w in zip(emax_bins, weights):
        d = run_pass(alt=alt, emin=emin, emax=em, cp=cp, sp=sp)
        r = d["skr"]
        t = d["t"]
        # trapezoidal integral of SKR(t) over the pass -> bits per pass
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        bits_per_pass = _trapz(np.clip(r, 0.0, None), t)
        s = stats(d)
        rows.append({
            "emax_deg": em, "weight": w,
            "bits_per_pass": bits_per_pass,
            "avail_pct": s["avail"],
            "skr_pk": s["skr_pk"],
        })

    expected_bits_per_pass = sum(r["weight"] * r["bits_per_pass"] for r in rows)
    expected_bits_per_day = n_passes_per_day * expected_bits_per_pass
    return rows, expected_bits_per_pass, expected_bits_per_day


def write_ensemble_table(rows, expected_bits_per_pass, expected_bits_per_day,
                          n_passes_per_day, fn="table_multi_pass_ensemble.csv"):
    p = os.path.join(TABDIR, fn)
    with open(p, "w") as f:
        f.write("emax_deg,weight,bits_per_pass,avail_pct,skr_pk_bps\n")
        for r in rows:
            f.write(f"{r['emax_deg']:.1f},{r['weight']:.4f},{r['bits_per_pass']:.4e},"
                    f"{r['avail_pct']:.1f},{r['skr_pk']:.4e}\n")
        f.write(f"\n# n_passes_per_day assumed,{n_passes_per_day}\n")
        f.write(f"# expected_bits_per_pass (weighted),{expected_bits_per_pass:.4e}\n")
        f.write(f"# expected_bits_per_day,{expected_bits_per_day:.4e}\n")
    print(f"[written] {p}")


def run_sensitivity():
    rows = []

    # Atmospheric scenarios now vary BOTH the zenith extinction coefficient
    # a0 (Beer-Lambert term) and the ground-level turbulence strength
    # Cn2(0) that feeds the Hufnagel-Valley profile of channel.py (reviewer
    # comment 1); sc0 is retained only for the legacy 'simple' turbulence
    # model and is no longer the active turbulence parameter by default.
    atmo = {
        "Clear": dict(a0=0.10, sc0=0.15, cn2_ground=5.0e-15),
        "Moderate": dict(a0=0.15, sc0=0.30, cn2_ground=1.7e-14),
        "Turbulent": dict(a0=0.25, sc0=0.55, cn2_ground=5.0e-14),
    }
    for name, kw in atmo.items():
        cp = ChannelParams(**kw)
        d = run_pass(cp=cp)
        rows.append({"category": "Atmosphere", "scenario": name, **stats(d)})

    det = {
        "High-performance": dict(eta_det=0.8, dc=50.0),
        "Medium-performance": dict(eta_det=0.6, dc=200.0),
        "Low-performance": dict(eta_det=0.4, dc=2000.0),
    }
    for name, kw in det.items():
        cp = ChannelParams(eta_det=kw["eta_det"])
        sp = SourceParams(dc=kw["dc"])
        d = run_pass(cp=cp, sp=sp)
        rows.append({"category": "Detector", "scenario": name, **stats(d)})

    alt = {"Low LEO (500 km)": 500.0, "Mid LEO (550 km)": 550.0, "High LEO (600 km)": 600.0}
    for name, a in alt.items():
        d = run_pass(alt=a)
        rows.append({"category": "Orbital altitude", "scenario": name, **stats(d)})

    # reviewer comment 3: most real passes over a given ground station peak
    # at low elevation (20-40 deg), not the near-zenith 80 deg pass used as
    # the main worked example; a very-low-elevation case is added here to
    # show that the effective key-generation window can shrink to zero,
    # not just narrow, once the pass never climbs high enough to push QBER
    # below QBER_max.
    pe = {"Very-low peak-elevation pass (12 deg)": 12.0,
          "Low peak-elevation pass (30 deg)": 30.0,
          "Medium peak-elevation pass (55 deg)": 55.0,
          "High peak-elevation pass (80 deg)": 80.0}
    for name, em in pe.items():
        d = run_pass(emax=em)
        rows.append({"category": "Peak elevation", "scenario": name, **stats(d)})

    return rows


def write_sensitivity(rows, fn="table_sensitivity.csv"):
    p = os.path.join(TABDIR, fn)
    with open(p, "w") as f:
        f.write("category,scenario,eta_max,qber_lo,qber_hi,skr_pk,avail\n")
        for r in rows:
            f.write(f"{r['category']},{r['scenario']},{r['eta_max']:.3e},"
                    f"{r['qber_lo']:.2f},{r['qber_hi']:.2f},{r['skr_pk']:.3e},{r['avail']:.1f}\n")
    print(f"[written] {p}")


def report(s, rows):
    print("\n" + "=" * 72)
    print("BASELINE PASS SUMMARY")
    print("=" * 72)
    print(f"  eta range     : {s['eta_min']:.3e} to {s['eta_max']:.3e}")
    print(f"  qber lo/hi    : {s['qber_lo']:.2f}% / {s['qber_hi']:.2f}%")
    print(f"  qber_max      : {s['qber_max']:.2f}%")
    print(f"  peak skr      : {s['skr_pk']:.3e} bits/s")
    print(f"  visibility    : {s['t_tot']:.1f} s")
    print(f"  key window    : {s['t_key']:.1f} s")
    print(f"  availability  : {s['avail']:.1f}%")

    print("\n" + "=" * 72)
    print("SENSITIVITY SCENARIOS")
    print("=" * 72)
    for r in rows:
        print(f"{r['category']:<18}{r['scenario']:<32}{r['eta_max']:>10.2e}"
              f"{r['qber_lo']:>8.2f}%{r['qber_hi']:>8.2f}%{r['skr_pk']:>12.2e}{r['avail']:>7.1f}%")


def main(decoy=True, nf=4, tmax_f=120, tmax_a=90):
    print("running baseline pass...")
    d = run_pass()

    print("running low peak-elevation pass (30 deg) for Fig. 5 overlay...")
    d_low = run_pass(emax=30.0)

    print("generating figures...")
    fig_transmittance(d)
    fig_detection(d)
    fig_qber(d, d_low=d_low, low_label="Low peak-elevation pass (30 deg)")
    fig_skr(d)
    fig_availability(d)

    print("running wavelength comparison (785 nm vs 1550 nm)...")
    d785, d1550, wl_rows = run_wavelength_comparison()
    fig_wavelength_comparison(d785, d1550)
    write_wavelength_table(wl_rows)

    print("running multi-pass ensemble (planning-tool automation)...")
    ens_rows, exp_bits_pass, exp_bits_day = run_pass_ensemble()
    write_ensemble_table(ens_rows, exp_bits_pass, exp_bits_day, n_passes_per_day=5)

    if decoy:
        print("generating decoy-state finite-key comparison (slow)...")
        dr = fig_skr_decoy(d, nf=nf, tmax_f=tmax_f, tmax_a=tmax_a)
        write_decoy_table(dr)
        write_decoy_param_appendix(
            dr["t_sample"], dr["eta_sample"], dr["finite_key_params"],
            dr["number_of_decoy"], dr["n_pulses"],
        )
    else:
        print("skipping decoy comparison (decoy=False)")

    s = stats(d)
    write_summary(s)

    rows = run_sensitivity()
    write_sensitivity(rows)

    report(s, rows)
    print("\n" + "=" * 72)
    print("WAVELENGTH COMPARISON (785 nm vs 1550 nm)")
    print("=" * 72)
    for r in wl_rows:
        print(f"  {r['wavelength_nm']:.0f} nm: a0={r['a0']:.3f}  bg={r['bg_rate_Hz']:.1f} Hz  "
              f"eta_fs_peak={r['eta_fs_peak']:.2e}  skr_pk={r['skr_pk']:.3e} bits/s  avail={r['avail']:.1f}%")

    print("\n" + "=" * 72)
    print("MULTI-PASS ENSEMBLE (illustrative daily key volume)")
    print("=" * 72)
    for r in ens_rows:
        print(f"  emax={r['emax_deg']:5.1f} deg  weight={r['weight']:.3f}  "
              f"bits/pass={r['bits_per_pass']:.3e}  avail={r['avail_pct']:.1f}%")
    print(f"  expected bits/pass (weighted): {exp_bits_pass:.3e}")
    print(f"  expected bits/day (5 passes/day): {exp_bits_day:.3e}")

    print(f"\nfigures: {FIGDIR}\ntables:  {TABDIR}")


if __name__ == "__main__":
    main(decoy="--skip-decoy" not in sys.argv)
