import numpy as np

# ---------------------------------------------------------------------------
# Wavelength-dependent atmospheric extinction (reviewer comment 2)
# ---------------------------------------------------------------------------
# The zenith extinction coefficient alpha0 is decomposed into a molecular
# (Rayleigh) term and an aerosol (Mie) term, each with its own power-law
# wavelength scaling (Angstrom-type law): alpha(lambda) = A_ray*(lambda/lambda_ref)^-p_ray
# + A_mie*(lambda/lambda_ref)^-p_mie. Rayleigh scattering follows the classic
# lambda^-4.09 law; aerosol (Mie) scattering follows a much shallower,
# site-dependent Angstrom exponent, taken here as 1.3 (typical continental/
# rural aerosol, Giggenbach & Shrestha 2022). The two reference coefficients
# at lambda_ref = 785 nm are chosen so that their sum reproduces the paper's
# original baseline alpha0 = 0.15 (split ~60% Rayleigh / 40% aerosol at
# 785 nm, consistent with the relative contribution of molecular scattering
# at NIR wavelengths). This lets alpha0 fall out *from wavelength* rather
# than being an independent free parameter, so that a change of lambda (e.g.
# 785 -> 1550 nm) automatically produces a physically consistent, reduced
# extinction coefficient.
LAM_REF_NM = 785.0
ALPHA_RAYLEIGH_REF = 0.09
ALPHA_AEROSOL_REF = 0.06
P_RAYLEIGH = 4.09
P_AEROSOL = 1.3


def alpha_zenith(lam_nm, a_ray_ref=ALPHA_RAYLEIGH_REF, a_mie_ref=ALPHA_AEROSOL_REF,
                  lam_ref_nm=LAM_REF_NM, p_ray=P_RAYLEIGH, p_mie=P_AEROSOL):
    """Zenith extinction coefficient alpha0(lambda), decomposed into a Rayleigh
    (molecular) term scaling as lambda^-p_ray and an aerosol (Mie) term scaling
    as lambda^-p_mie. Both reference values are calibrated at lam_ref_nm."""
    r = np.asarray(lam_nm, dtype=float) / lam_ref_nm
    return a_ray_ref * r ** (-p_ray) + a_mie_ref * r ** (-p_mie)


# ---------------------------------------------------------------------------
# Wavelength-dependent sky background scaling (used by qkd_bb84.SourceParams)
# ---------------------------------------------------------------------------
# Daytime sky radiance is dominated by scattered sunlight; the same Rayleigh
# lambda^-4 trend that sets the molecular extinction term above also governs
# (to first order) how much sunlight is scattered into the receiver's field
# of view, which is why NIR/telecom-band links (~1550 nm) see substantially
# lower background than 785 nm under daylight conditions. This is an
# approximate, single-exponent empirical scaling (not a full radiative-
# transfer calculation) intended to capture the qualitative and
# order-of-magnitude effect for system-level trade studies.
BG_LAM_REF_NM = 785.0
BG_SCALING_EXPONENT = 4.0


def background_wavelength_scale(lam_nm, lam_ref_nm=BG_LAM_REF_NM, p=BG_SCALING_EXPONENT):
    """Multiplicative scale factor applied to a reference (785 nm) background
    count rate to obtain the background rate at another wavelength."""
    return (lam_ref_nm / np.asarray(lam_nm, dtype=float)) ** p


# ---------------------------------------------------------------------------
# Turbulence: Hufnagel-Valley Cn2(h) profile and slant-path Rytov integration
# (reviewer comment 1)
# ---------------------------------------------------------------------------
# Section 2.4 of the paper originally aggregated turbulence into a single
# elevation-independent scintillation index sigma_scint combined with a bare
# 1/sin(E) airmass scaling (eta_turb_simple / 'sc0' model below). That
# collapses the altitude-dependence of Cn2(h) into one free number and does
# not reflect how the *integrated* turbulence along a rapidly-changing slant
# path actually evolves with elevation. This module adds an explicit,
# altitude-resolved alternative using the standard Hufnagel-Valley (HV5/7)
# refractive-index structure profile:
#
#   Cn2(h) = 0.00594 (v/27)^2 (1e-5 h)^10 exp(-h/1000)
#            + 2.7e-16 exp(-h/1500)
#            + A exp(-h/100)
#
# with h in meters, v the rms high-altitude wind speed (m/s, Bufton wind
# model default ~21 m/s), and A = Cn2(0) the ground-level turbulence
# strength (m^-2/3, default 1.7e-14 for a typical mid-latitude site).
#
# For a satellite-to-ground downlink the transmitter is effectively a point
# source far above the turbulent layer, so the appropriate weighting is the
# *spherical-wave* Rytov-variance integral (Andrews & Phillips, "Laser Beam
# Propagation through Random Media"):
#
#   sigma_R^2(eps) = 2.25 k^(7/6) sec(zeta)^(11/6)
#                    * integral_{h0}^{H} Cn2(h) [(h-h0)(H-h)/(H-h0)]^(5/6) dh
#
# where zeta = 90 deg - eps is the zenith angle, sec(zeta) = 1/sin(eps) is
# the airmass factor (the SAME factor used in the simplified model, but now
# multiplying a physically integrated Cn2 profile instead of a constant),
# H is the satellite altitude, and h0 the ground-station altitude. Because
# Cn2(h) decays by orders of magnitude above ~20-25 km, the integral is
# truncated at h_top (default 25 km) with negligible error; turbulence above
# this altitude is assumed negligible, which is the one simplifying
# assumption of this profile (stated here explicitly, as requested).
#
# The resulting Rytov variance is mapped to a (possibly saturated)
# scintillation index sigma_I^2 using the standard unbounded-plane-wave
# interpolation formula that correctly saturates for sigma_R^2 >> 1 (strong
# turbulence), and the mean/median turbulence-induced transmittance loss is
# then taken as exp(-sigma_I^2/2), i.e. the median of a unit-mean log-normal
# fading distribution with variance sigma_I^2 -- the same convention as the
# original simplified model, so the two are directly comparable.
CN2_GROUND_DEFAULT = 1.7e-14   # m^-2/3, Cn2(0)
WIND_RMS_DEFAULT = 21.0        # m/s, Bufton wind model rms
HV_TOP_ALT_M = 25e3            # m, altitude above which Cn2 is neglected
HV_N_PTS = 400


def cn2_hufnagel_valley(h_m, A=CN2_GROUND_DEFAULT, v_rms=WIND_RMS_DEFAULT):
    """Hufnagel-Valley (HV5/7-form) refractive-index structure parameter
    profile Cn2(h), h in meters above the ground station."""
    h = np.clip(np.atleast_1d(np.asarray(h_m, dtype=float)), 0.0, None)
    term_high = 0.00594 * (v_rms / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
    term_mid = 2.7e-16 * np.exp(-h / 1500.0)
    term_ground = A * np.exp(-h / 100.0)
    return term_high + term_mid + term_ground


def rytov_variance_downlink(e_rad, lam_m, h0_m=0.0, h_sat_m=550e3,
                             cn2_ground=CN2_GROUND_DEFAULT, wind_rms=WIND_RMS_DEFAULT,
                             h_top_m=HV_TOP_ALT_M, n_pts=HV_N_PTS):
    """Spherical-wave Rytov variance for a satellite-to-ground downlink,
    integrating the HV Cn2(h) profile along the slant path as a function of
    elevation angle e_rad. Returns sigma_R^2, elementwise over e_rad."""
    e = np.atleast_1d(np.asarray(e_rad, dtype=float))
    sec_zeta = 1.0 / np.clip(np.sin(e), 1e-6, None)
    h_top = min(h_top_m, h_sat_m - 1.0)
    hgrid = np.linspace(h0_m, h_top, n_pts)
    cn2 = cn2_hufnagel_valley(hgrid, cn2_ground, wind_rms)
    weight = ((hgrid - h0_m) * (h_sat_m - hgrid) / (h_sat_m - h0_m)) ** (5.0 / 6.0)
    # np.trapz was removed in numpy>=2.0 in favor of np.trapezoid; support both.
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    integral = _trapz(cn2 * weight, hgrid)
    k = 2.0 * np.pi / lam_m
    return 2.25 * k ** (7.0 / 6.0) * sec_zeta ** (11.0 / 6.0) * integral


def scintillation_index(sigma_r2):
    """Unbounded plane-wave scintillation-index interpolation formula
    (Andrews & Phillips), which reduces to sigma_I^2 ~= 4*sigma_R^2 in the
    weak-fluctuation limit and correctly saturates as sigma_R^2 -> infinity."""
    s = np.clip(np.atleast_1d(np.asarray(sigma_r2, dtype=float)), 0.0, None)
    a = 0.49 * s / (1.0 + 1.11 * s ** (6.0 / 5.0)) ** (7.0 / 6.0)
    b = 0.51 * s / (1.0 + 0.69 * s ** (6.0 / 5.0)) ** (5.0 / 6.0)
    return np.exp(a + b) - 1.0


class ChannelParams:
    def __init__(self, lam_nm=785.0, w0=0.05, dr=0.5, eta_rx=0.5, eta_det=0.6,
                 a0=None, sp_urad=2.0, cn2=1e-15, sc0=0.3,
                 turb_model="hv", h_sat_km=550.0, h0_km=0.0,
                 cn2_ground=CN2_GROUND_DEFAULT, wind_rms=WIND_RMS_DEFAULT):
        self.lam_nm = lam_nm
        self.lam = lam_nm * 1e-9
        self.w0 = w0
        self.dr = dr
        self.eta_rx = eta_rx
        self.eta_det = eta_det
        # a0=None (default): derive the zenith extinction coefficient from
        # wavelength via the Rayleigh+aerosol model above, so switching
        # wavelength (e.g. 785 -> 1550 nm) self-consistently reduces
        # atmospheric extinction. Pass an explicit numeric a0 to override.
        self.a0 = alpha_zenith(lam_nm) if a0 is None else a0
        self.sp = sp_urad * 1e-6
        self.cn2 = cn2
        self.sc0 = sc0
        # turb_model: "hv" uses the altitude-resolved Hufnagel-Valley +
        # spherical-wave Rytov integration (default, see module docstring);
        # "simple" reproduces the original single-parameter sc0/sin(E)
        # aggregate model, kept only for backward-compatible comparison.
        self.turb_model = turb_model
        self.h_sat_m = h_sat_km * 1e3
        self.h0_m = h0_km * 1e3
        self.cn2_ground = cn2_ground
        self.wind_rms = wind_rms


def beam_r(lm, p):
    zr = np.pi * p.w0 ** 2 / p.lam
    return p.w0 * np.sqrt(1.0 + (lm / zr) ** 2)


def eta_fs(lkm, p):
    lm = lkm * 1e3
    w = beam_r(lm, p)
    e = (p.dr / 2.0) ** 2 / w ** 2
    return np.clip(e, 0.0, 1.0)


def eta_atm(e_rad, p):
    s = np.sin(e_rad)
    am = 1.0 / np.clip(s, 1e-6, None)
    return np.exp(-p.a0 * am)


def eta_turb_simple(e_rad, p):
    """Original aggregate model: a single scintillation index sc0 scaled by
    the 1/sin(E) airmass factor. Elevation-dependence enters only through
    the airmass; the vertical Cn2(h) profile is not resolved. Kept for
    backward-compatible comparison against the HV-profile model below."""
    s = np.sin(e_rad)
    sg2 = (p.sc0 ** 2) / np.clip(s, 1e-6, None)
    return np.exp(-0.5 * sg2)


def eta_turb_hv(e_rad, p):
    """Altitude-resolved model: Hufnagel-Valley Cn2(h) integrated along the
    slant path via the spherical-wave Rytov variance, converted to a
    (possibly saturated) scintillation index, and mapped to a median
    turbulence-fading transmittance exp(-sigma_I^2/2). See module docstring
    for the explicit assumptions (HV profile, integration truncated at
    h_top_m, median- rather than mean-fading convention)."""
    sig_r2 = rytov_variance_downlink(
        e_rad, p.lam, h0_m=p.h0_m, h_sat_m=p.h_sat_m,
        cn2_ground=p.cn2_ground, wind_rms=p.wind_rms,
    )
    sig_i2 = scintillation_index(sig_r2)
    return np.exp(-0.5 * sig_i2)


def eta_turb(e_rad, p):
    e_rad = np.atleast_1d(np.asarray(e_rad, dtype=float))
    if p.turb_model == "simple":
        return eta_turb_simple(e_rad, p)
    return eta_turb_hv(e_rad, p)


def eta_point(p):
    td = p.lam / (np.pi * p.w0)
    r2 = (p.sp / td) ** 2
    return 1.0 / (1.0 + 2.0 * r2)


def eta_total(lkm, e_rad, p):
    fs = eta_fs(lkm, p)
    atm = eta_atm(e_rad, p)
    turb = eta_turb(e_rad, p)
    pt = eta_point(p) * np.ones_like(np.atleast_1d(e_rad), dtype=float)
    tot = fs * atm * turb * pt * p.eta_rx * p.eta_det
    return {
        "eta_fs": fs,
        "eta_atm": atm,
        "eta_turb": turb,
        "eta_point": pt,
        "eta_rx_det": p.eta_rx * p.eta_det * np.ones_like(np.atleast_1d(e_rad), dtype=float),
        "eta_total": tot,
    }


if __name__ == "__main__":
    from geometry import PassGeometry

    p = ChannelParams()
    g = PassGeometry(alt=550, emin=10, emax=80)
    th = g.half_dur()
    t = np.linspace(-th, th, 7)
    e = g.elev(t)
    l = g.slant(t)
    r = eta_total(l, e, p)
    print(f"a0(785 nm) = {p.a0:.4f}  (Rayleigh+aerosol model)")
    print(f"a0(1550 nm) = {alpha_zenith(1550.0):.4f}")
    print(f"{'t(s)':>8} {'e(deg)':>7} {'l(km)':>9} {'fs':>10} {'atm':>9} "
          f"{'turb':>9} {'pt':>10} {'tot':>11}")
    for i in range(len(t)):
        print(f"{t[i]:8.1f} {np.degrees(e[i]):7.2f} {l[i]:9.2f} "
              f"{r['eta_fs'][i]:10.3e} {r['eta_atm'][i]:9.4f} "
              f"{r['eta_turb'][i]:9.4f} {r['eta_point'][i]:10.4f} "
              f"{r['eta_total'][i]:11.3e}")

    print("\nturbulence model comparison at fixed elevation grid:")
    p_simple = ChannelParams(turb_model="simple")
    th_s = eta_turb_simple(e, p_simple)
    th_hv = eta_turb_hv(e, p)
    print(f"{'e(deg)':>7} {'turb_simple':>12} {'turb_hv':>12}")
    for i in range(len(t)):
        print(f"{np.degrees(e[i]):7.2f} {th_s[i]:12.4f} {th_hv[i]:12.4f}")
