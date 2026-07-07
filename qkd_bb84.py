import numpy as np

from channel import background_wavelength_scale, BG_LAM_REF_NM


class SourceParams:
    def __init__(self, mu=0.3, f_rep=100e6, e_opt=0.02, dc=5000.0,
                 gate=1e-9, bg=2000.0, f_ec=1.15, lam_nm=None):
        self.mu = mu
        self.f_rep = f_rep
        self.e_opt = e_opt
        self.D = dc
        self.gate = gate
        # bg (kwarg default 2000 Hz) is calibrated at BG_LAM_REF_NM = 785 nm.
        # If lam_nm is given and differs from the reference, the background
        # rate is rescaled via the wavelength-dependent sky-radiance model
        # in channel.py (reviewer comment 2): NIR/telecom wavelengths
        # (~1550 nm) see substantially reduced daylight background relative
        # to 785 nm. Pass lam_nm=None (default) to keep bg as given verbatim.
        if lam_nm is not None and lam_nm != BG_LAM_REF_NM:
            bg = bg * background_wavelength_scale(lam_nm)
        self.bg_rate = bg
        self.lam_nm = lam_nm if lam_nm is not None else BG_LAM_REF_NM
        self.f_EC = f_ec

    def pdc(self):
        return 1.0 - np.exp(-self.D * self.gate)

    def pbg(self):
        return 1.0 - np.exp(-self.bg_rate * self.gate)

    # kept for backward compatibility with older call sites
    P_dc_per_pulse = pdc
    P_bg_per_pulse = pbg


def h2(x):
    x = np.clip(np.atleast_1d(np.asarray(x, dtype=float)), 0.0, 1.0)
    r = np.zeros_like(x)
    m = (x > 0) & (x < 1)
    r[m] = -x[m] * np.log2(x[m]) - (1 - x[m]) * np.log2(1 - x[m])
    return r


def detection_probabilities(eta, src):
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    me = src.mu * eta
    psig = 1.0 - np.exp(-me)
    pdc = src.pdc() * np.ones_like(eta)
    pbg = src.pbg() * np.ones_like(eta)
    pclick = 1.0 - (1.0 - psig) * (1.0 - pdc) * (1.0 - pbg)
    return psig, pdc, pbg, pclick


def qber(psig, pdc, pbg, pclick, src):
    perr = src.e_opt * psig + 0.5 * (pdc + pbg)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(pclick > 0, perr / pclick, np.nan)
    return np.clip(q, 0.0, 1.0)


def secure_key_rate(pclick, q, src):
    h = h2(q)
    b = np.clip(1.0 - src.f_EC * h - h, 0.0, None)
    r = src.f_rep * pclick * 0.5 * b
    return np.where(np.isnan(q), 0.0, r)


def qber_max(f_ec):
    tgt = 1.0 / (1.0 + f_ec)
    lo, hi = 1e-9, 0.5
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if h2(mid)[0] < tgt:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    src = SourceParams()
    print(f"pdc: {src.pdc():.3e}")
    print(f"pbg: {src.pbg():.3e}")
    qm = qber_max(src.f_EC)
    print(f"qber_max (f_EC={src.f_EC}): {qm*100:.2f}%")

    ev = np.array([1e-5, 1e-4, 1e-3, 1e-2])
    psig, pdc, pbg, pclick = detection_probabilities(ev, src)
    q = qber(psig, pdc, pbg, pclick, src)
    r = secure_key_rate(pclick, q, src)
    print(f"\n{'eta':>10} {'psig':>12} {'pclick':>12} {'qber':>8} {'skr':>12}")
    for i in range(len(ev)):
        print(f"{ev[i]:10.1e} {psig[i]:12.3e} {pclick[i]:12.3e} "
              f"{q[i]*100:7.2f}% {r[i]:12.3e}")
