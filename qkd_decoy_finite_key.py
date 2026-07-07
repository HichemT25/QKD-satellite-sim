# Uses TNO's qkd_key_rate package (pip install tno.quantum.communication.qkd_key_rate)
# rather than a hand-rolled decoy-state/finite-key formula: two independent from-scratch
# transcriptions of the Lim 2014 / Rusca 2018 equations silently mis-estimated the
# single-photon yield by up to 7x, so this module defers to the published, tested
# implementation instead.
#
# Timeout note: the underlying SLSQP optimizer has no built-in timeout and can take
# anywhere from <1s to several minutes per point. signal.alarm/SIGALRM would work on
# Unix but doesn't exist on Windows and only works from the main thread; we instead run
# each call in a daemon thread and join() with a timeout, which is portable across
# platforms and doesn't require any particular signal.

import threading
import time

import numpy as np

from tno.quantum.communication.qkd_key_rate.quantum import standard_detector
from tno.quantum.communication.qkd_key_rate.quantum.bb84 import (
    BB84AsymptoticKeyRateEstimate,
    BB84FiniteKeyRateEstimate,
    BB84FullyAsymptoticKeyRateEstimate,
)
from tno.quantum.communication.qkd_key_rate.quantum._config import OptimizationError


class TO(Exception):
    pass


def run_to(fn, tmax, *a, **kw):
    box = {}

    def tgt():
        try:
            box["r"] = fn(*a, **kw)
        except BaseException as e:
            box["e"] = e

    th = threading.Thread(target=tgt, daemon=True)
    th.start()
    th.join(tmax)
    if th.is_alive():
        raise TO(f"exceeded {tmax}s")
    if "e" in box:
        raise box["e"]
    return box["r"]


class DecoyFiniteKeyParams:
    def __init__(self, f_rep=100e6, e_opt=0.02, dc=5000.0, eta_det=0.6,
                 dt=45e-9, jit=50e-12, n=1e10, nd=2):
        self.f_rep = f_rep
        self.e_opt = e_opt
        self.dark_count_rate_Hz = dc
        self.eta_det = eta_det
        self.dead_time = dt
        self.jitter_detector = jit
        self.n_pulses = n
        self.number_of_decoy = nd
        dc_gate = dc / f_rep
        self.detector = standard_detector.customise(
            name="sat_detector",
            dark_count_rate=dc_gate,
            dark_count_frequency=None,
            detection_frequency=f_rep,
            interval=None,
            polarization_drift=0.0,
            error_detector=e_opt,
            efficiency_party=eta_det,
            efficiency_detector=None,
            dead_time=dt,
            jitter_detector=jit,
            jitter_source=0.0,
            detection_window=5,
        )

    def asymptotic_estimator(self):
        return BB84FullyAsymptoticKeyRateEstimate(detector=self.detector)

    def asymptotic_decoy_estimator(self):
        return BB84AsymptoticKeyRateEstimate(detector=self.detector, number_of_decoy=self.number_of_decoy)

    def finite_key_estimator(self):
        return BB84FiniteKeyRateEstimate(detector=self.detector, number_of_pulses=int(self.n_pulses),
                                          number_of_decoy=self.number_of_decoy)


def eta_to_attenuation_dB(eta):
    e = np.clip(np.asarray(eta, dtype=float), 1e-300, 1.0)
    return -10.0 * np.log10(e)


def decoy_skr_asymptotic(eta_arr, p, multi_decoy=True, verbose=False, tmax=90):
    ea = np.atleast_1d(np.asarray(eta_arr, dtype=float))
    est = p.asymptotic_decoy_estimator() if multi_decoy else p.asymptotic_estimator()
    skr = np.zeros_like(ea)
    for i, e in enumerate(ea):
        if not np.isfinite(e) or e <= 0:
            continue
        db = eta_to_attenuation_dB(e)
        t0 = time.time()
        try:
            _, rpp = run_to(est.optimize_rate, tmax, attenuation=float(db))
            skr[i] = max(rpp, 0.0) * p.f_rep
            st = "ok"
        except (ValueError, OptimizationError):
            skr[i] = 0.0
            st = "no positive rate"
        except TO:
            skr[i] = 0.0
            st = f"timeout after {tmax}s"
        if verbose:
            print(f"  pt {i+1}/{len(ea)}: eta={e:.2e} ({db:.1f} dB) -> skr={skr[i]:.3e} [{time.time()-t0:.1f}s, {st}]")
    return skr


def label_decoy_params(x, number_of_decoy):
    """Reviewer comment 4: the underlying TNO qkd_key_rate optimizer
    (BB84FiniteKeyRateEstimate.optimize_rate) returns a raw parameter vector
    x with no built-in field names, which is why the paper's Section 5.5
    text could point only to "the library" rather than to explicit
    protocol variables. Following the parameterization of Attema et al.,
    "Optimizing the decoy-state BB84 QKD protocol parameters" (2021, the
    paper this package implements), for number_of_decoy = nd the optimizer
    exposes 3*(nd+1) free parameters, grouped as:
      - mu[0..nd]   : the nd+1 pulse intensities (index 0 = signal
                       intensity mu_s; indices 1..nd = decoy intensities,
                       conventionally including a vacuum/near-vacuum decoy)
      - p_X[0..nd]  : probability of choosing each intensity when Alice
                       prepares in the X (rectilinear) basis
      - p_Z[0..nd]  : probability of choosing each intensity when Alice
                       prepares in the Z (diagonal) basis
    (p_X and p_Z are each simplex-normalized by the optimizer; the overall
    X/Z basis-choice probability is a separate, additional protocol
    parameter not part of this vector.) This function slices x into that
    labeled structure so it can be written out explicitly (see
    write_decoy_param_appendix in run_simulation.py) instead of remaining
    opaque optimizer internals.

    The TNO library's optimize_rate return type has changed across versions:
    older releases return a plain numpy array; newer releases return a dict
    keyed by parameter name, or a scipy OptimizeResult object whose .x
    attribute holds the raw array. All three cases are handled below.
    """
    if x is None:
        return None

    # --- case 1: TNO returns a dict keyed by parameter name ---------------
    # Newer TNO versions (>=0.3) return something like:
    #   {"mu": [...], "probability_basis_X": [...], "probability_basis_Z": [...]}
    # or with keys "intensity", "p_x", "p_z" etc.  We try the known key names.
    if isinstance(x, dict):
        mu_keys = ("mu", "intensity", "intensities")
        px_keys = ("probability_basis_X", "p_x", "p_X", "px")
        pz_keys = ("probability_basis_Z", "p_z", "p_Z", "pz")

        def _get(d, keys):
            for k in keys:
                if k in d:
                    v = d[k]
                    return list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else [v]
            return None

        mu = _get(x, mu_keys)
        px = _get(x, px_keys)
        pz = _get(x, pz_keys)
        if mu is not None:
            return {"mu": mu, "p_X": px or [], "p_Z": pz or []}
        # unrecognised dict structure — dump key/value pairs as-is
        return {str(k): (list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v)
                for k, v in x.items()}

    # --- case 2: scipy OptimizeResult or any object with a .x attribute ---
    if hasattr(x, "x"):
        x = x.x

    # --- case 3: plain numpy array or list ---------------------------------
    try:
        arr = np.asarray(x, dtype=float).ravel()
    except (TypeError, ValueError):
        # give up gracefully; the appendix table will show NaNs for this point
        return None

    n = number_of_decoy + 1
    if arr.size < 3 * n:
        return {"raw": arr.tolist()}
    return {
        "mu": arr[0:n].tolist(),
        "p_X": arr[n:2 * n].tolist(),
        "p_Z": arr[2 * n:3 * n].tolist(),
    }


def decoy_skr_finite_key(eta_arr, p, x0=None, verbose=False, tmax=120):
    ea = np.atleast_1d(np.asarray(eta_arr, dtype=float))
    order = np.argsort(-ea)
    est = p.finite_key_estimator()
    skr = np.zeros_like(ea)
    params = [None] * len(ea)
    dx0 = None if x0 is None else np.array(x0, copy=True)

    for i in order:
        e = ea[i]
        if not np.isfinite(e) or e <= 0:
            continue
        db = float(eta_to_attenuation_dB(e))
        ax0 = dx0
        ok = False
        t0 = time.time()
        for _ in range(2):
            try:
                x, rpp = run_to(est.optimize_rate, tmax, attenuation=db, x0=ax0)
                skr[i] = max(rpp, 0.0) * p.f_rep
                params[i] = x
                ok = True
                break
            except (ValueError, OptimizationError) as err:
                if verbose:
                    print(f"  eta={e:.2e} ({db:.1f} dB): retry after {err} [{time.time()-t0:.1f}s]")
                if ax0 is None:
                    nd = p.number_of_decoy
                    base = np.concatenate([np.full(nd + 1, 1.0 / (nd + 1))] * 3)
                else:
                    base = ax0
                ax0 = np.clip(base * (1 + 0.1 * np.random.randn(len(base))), 1e-6, 1.0)
            except TO:
                if verbose:
                    print(f"  eta={e:.2e} ({db:.1f} dB): timeout after {tmax}s")
                break
        if not ok:
            skr[i] = 0.0
            if verbose:
                print(f"  eta={e:.2e} ({db:.1f} dB): no positive key rate [{time.time()-t0:.1f}s]")
        elif verbose:
            print(f"  eta={e:.2e} ({db:.1f} dB): skr={skr[i]:.3e} [{time.time()-t0:.1f}s]")

    return skr, params


if __name__ == "__main__":
    p = DecoyFiniteKeyParams(n=1e10, nd=2)

    print("=== single-intensity asymptotic ===")
    eg = np.array([1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    s1 = decoy_skr_asymptotic(eg, p, multi_decoy=False)
    for e, s in zip(eg, s1):
        print(f"  eta={e:9.2e} ({eta_to_attenuation_dB(e):5.1f} dB)  skr={s:10.3e}")

    print("\n=== rigorous finite-key (small grid) ===")
    eg2 = np.array([1e-2, 5e-3, 1e-3])
    s2, _ = decoy_skr_finite_key(eg2, p, verbose=True)
    for e, s in zip(eg2, s2):
        print(f"  eta={e:9.2e} ({eta_to_attenuation_dB(e):5.1f} dB)  skr={s:10.3e}")
