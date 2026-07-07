import numpy as np

MU_E = 398600.4418
R_E = 6371.0


class PassGeometry:
    def __init__(self, alt=550.0, emin=10.0, emax=80.0):
        self.h = alt
        self.re = R_E
        self.rs = R_E + alt
        self.emin = np.radians(emin)
        self.emax = np.radians(emax)
        self.w = np.sqrt(MU_E / self.rs ** 3)
        self._build()

    def _eta(self, e):
        return np.arccos((self.re / self.rs) * np.cos(e)) - e

    def _build(self):
        eg = np.linspace(self.emin, self.emax, 4000)
        ng = self._eta(eg)
        self.eg, self.ng = eg, ng

    def elev(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        psi = np.abs(self.w * t)
        o = np.argsort(self.ng)
        ns, es = self.ng[o], self.eg[o]
        lo, hi = ns[0], ns[-1]
        pc = np.clip(psi, lo, hi)
        e = np.interp(pc, ns, es)
        return np.where(psi > hi + 1e-12, np.nan, e)

    def slant(self, t):
        return self._slant_from_e(self.elev(t))

    def _slant_from_e(self, e):
        n = self._eta(e)
        return np.sqrt(self.re ** 2 + self.rs ** 2 - 2 * self.re * self.rs * np.cos(n))

    def half_dur(self):
        return self.ng.max() / self.w

    def dur(self):
        return 2.0 * self.half_dur()


if __name__ == "__main__":
    g = PassGeometry(alt=550, emin=10, emax=80)
    th = g.half_dur()
    print(f"half duration: {th:.1f} s")
    print(f"total duration: {g.dur():.1f} s")
    t = np.linspace(-th, th, 9)
    e = np.degrees(g.elev(t))
    l = g.slant(t)
    for ti, ei, li in zip(t, e, l):
        print(f"t={ti:7.1f}s  e={ei:6.2f} deg  l={li:8.2f} km")
