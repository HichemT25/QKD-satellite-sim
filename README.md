# Satellite-to-Ground BB84 QKD Link Simulation

Reference implementation of the physics-aware satellite-to-ground photonic
QKD link model described in the paper *"Physics-Aware Performance
Evaluation of Satellite-to-Ground Photonic Quantum Communication Links in
Low Earth Orbit."* This code generates every figure and every quantitative
result (Tables 1–3, comparison statistics) reported in the paper directly
from the model equations — nothing in the paper's Results section is
hand-typed or assumed.

## Files

| File | Contents | Paper section |
|---|---|---|
| `geometry.py` | LEO orbital geometry: elevation angle and slant range vs. time during a pass | §2.5 |
| `channel.py` | Free-space (Gaussian beam) loss, atmospheric (Beer–Lambert) extinction, turbulence fading, pointing loss, total transmittance | §2.2–§2.7, Eqs. (1)–(5) |
| `qkd_bb84.py` | WCP detection statistics, QBER, asymptotic BB84 secure key rate, QBER security threshold (simplified model used for the main pass figures) | §3, Eqs. (6)–(12) |
| `qkd_decoy_finite_key.py` | Rigorous decoy-state finite-key BB84 secure key rate (validation/extension model), via the verified TNO Quantum library | §3 (extension) |
| `run_simulation.py` | Main driver: runs the baseline pass + sensitivity scenarios + wavelength comparison + multi-pass ensemble, generates all figures (PDF) and tables (CSV) | §4–§5 |

## Requirements

```
python >= 3.9
numpy
matplotlib
tno.quantum.communication.qkd_key_rate   (for qkd_decoy_finite_key.py only)
```

Install with:
```bash
pip install numpy matplotlib
pip install tno.quantum.communication.qkd_key_rate
```

## Usage

```bash
python3 run_simulation.py                # full run, including decoy-state validation figure
python3 run_simulation.py --skip-decoy    # skip the slow decoy-state step (seconds instead of minutes)
```

This creates:
```
outputs/
├── figures/
│   ├── fig_transmittance.pdf          (Fig. "transmittance": eta_total(t) and components)
│   ├── fig_detection.pdf              (Fig. "detection": P_sig, P_dc, P_bg, P_click vs t)
│   ├── fig_qber.pdf                   (Fig. "qber": QBER vs elevation angle, now overlaying a
│   │                                    low peak-elevation pass + QBER_max threshold, comment 3)
│   ├── fig_skr.pdf                    (Fig. "skr": SKR(t) and key-generation window, simplified model)
│   ├── fig_skr_decoy_comparison.pdf   (simplified vs. rigorous decoy-state/finite-key SKR)
│   ├── fig_availability.pdf           (Fig. "availability": visibility vs. key window)
│   └── fig_wavelength_comparison.pdf  (785 nm vs. 1550 nm transmittance and SKR, comment 2)
└── tables/
    ├── table_baseline_summary.csv               (single-pass scalar statistics, Table 1-adjacent)
    ├── table_sensitivity.csv                    (atmosphere / detector / altitude / peak-elevation
    │                                              scenarios, Table 2/3; now includes a 12 deg
    │                                              zero-availability case, comment 3)
    ├── table_decoy_comparison.csv               (multi-decoy asymptotic + rigorous finite-key SKR points)
    ├── table_decoy_state_parameters_appendix.csv (labeled mu/p_X/p_Z per finite-key point, comment 4)
    ├── table_wavelength_comparison.csv          (785 nm vs. 1550 nm summary statistics, comment 2)
    └── table_multi_pass_ensemble.csv            (weighted multi-pass daily key-volume estimate, comment 3)
```

## Model summary and default parameters

The simulation follows the causal chain

```
Orbital geometry -> Optical channel loss -> Photon detection statistics -> QBER -> Secure key rate
```

**Baseline link** (`ChannelParams()` / `SourceParams()` defaults):

- Orbital altitude: 550 km, minimum elevation 5°-10°, peak elevation 80°
  (near-zenith pass)
- Wavelength: 785 nm; transmit beam waist 5 cm; receiver aperture 0.5 m
- Mean photon number per pulse mu = 0.3; pulse rate 100 MHz
- Detector efficiency 0.6; dark-count rate 5 kHz (free-running Si-SPAD,
  representative of Hadfield, *Nat. Photonics* 2009); background rate 2 kHz
  (filtered sky background)
- Error-correction inefficiency f_EC = 1.15

These are documented inline in `channel.py` (`ChannelParams`) and
`qkd_bb84.py` (`SourceParams`) with literature justification for each
choice; change them there (or pass keyword overrides to `run_baseline_pass`)
to explore other link configurations.
