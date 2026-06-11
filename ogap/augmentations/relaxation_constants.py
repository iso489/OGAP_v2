"""Field-dependent brain-tissue relaxation constants for the low-field Bloch simulator.

This module is the *physically-consistent* data layer for ``bloch_lowfield.py``: it
provides in-vivo T1/T2 (ms) for the tissues OGAP must render, anchored at the source
fields of the training data — **1.5 T (BraTS-Africa)** and **3.0 T (BraTS-2023)** — and
at the **~64 mT** deployment target, plus a field-extrapolation that respects the two
robustly-established facts of brain relaxometry:

1. **T1 is strongly field-dependent**, rising with B0 as a power law
   ``T1(B0) = C * B0**beta`` with ``beta ~ 0.31-0.38`` (tissue-dependent). Because the
   exponent differs between GM and WM, the **GM/WM T1 contrast collapses at both very
   high and very low field** and peaks at intermediate field — the single most important
   reason low-field brain segmentation is intrinsically hard.
2. **T2 is approximately field-independent** at clinical field (1.5-3 T) and lengthens
   only mildly toward very low field; **CSF is field-independent** in both T1 and T2.

A single power law from 3 T *over-extrapolates* at 64 mT (it ignores the low-field
plateau), so this module interpolates **in log-B0 space between measured anchors**
(1.5 T, 3 T, and the in-vivo 50-64 mT measurements) rather than extrapolating one law
across three orders of magnitude — which is both more accurate and more honest.

References (read for this module)
--------------------------------
* Wansapura 1999, JMRI 9:531 — in-vivo 3 T: WM T1 832/T2 80; GM T1 1331/T2 110; plus the
  Bottomley ``T1 = a*f**b`` and Fischer 4-parameter dispersion fits.
* Stanisz 2005, MRM 54:507 — 3 T vs 1.5 T: **T2 ~field-independent** (WM modelled flat; GM
  has a mild rise within large in-vivo scatter); T1 WM +22 %, GM +62 %, blood +34 % (1.5->3 T).
  NOTE: those percentages are Stanisz's own (Stanisz 3 T WM/GM T1 ~1084/1820 ms). This module
  anchors WM/GM T1 to **Wansapura 1999** 3 T (832/1331) plus chosen 1.5 T mid-range values, so
  its modelled 1.5->3 T rise is ~+28 %/+33 % — a consistent but different source, not a mismatch.
* Rooney 2007, MRM 57:308 — ``T1 = C*f**beta`` with f = gamma*B0/(2*pi) the Larmor frequency
  in MHz: WM beta 0.382, GM 0.376, blood 0.340; **CSF T1 is B0-independent (~4.3 s)**.
* Pohmann 2016, MRM 75:801 — WM ``T1 = 659 * B0**0.35`` ms/T; T2* decreases with field.
* Fischer 1990, MRM 16:317 — field-cycling dispersion; GM/WM T1 contrast peaks at medium
  field, collapses at low/high field.
* O'Reilly 2022, MRM 87:884 — in-vivo 50 mT: WM T1 ~275, GM T1 ~327 (small contrast gap).
* Tumour values (3 T MR-fingerprinting / SyMRI, author-curated and literature):
  GBM solid T1 1700-2100 / T2 110-140; lower-grade glioma T1 1500-1800 / T2 160-172;
  vasogenic peritumoural edema T1 ~1400 (1066-1578) / T2 elevated (~150). Large biological
  scatter and overlap: these drive *plausible contrast*, NOT histology classification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math


# Tesla anchors we trust most for each regime.
_B0_LOW = 0.064   # Hyperfine Swoop / deployment target
_B0_15 = 1.5      # BraTS-Africa
_B0_3 = 3.0       # BraTS-2023


@dataclass(frozen=True)
class TissueRelaxation:
    """In-vivo T1/T2 (ms) anchored at 64 mT, 1.5 T and 3 T, with a field model.

    ``t1_anchors`` maps B0 (Tesla) -> T1 (ms) at measured anchors; ``t2_anchors``
    likewise for T2. ``proton_density`` is the relative equilibrium magnetisation
    (water/PD weight) used by the partial-volume Bloch sum.
    """

    name: str
    t1_anchors: Dict[float, float]
    t2_anchors: Dict[float, float]
    proton_density: float = 1.0

    def t1(self, b0_t: float) -> float:
        return _interp_logfield(self.t1_anchors, b0_t)

    def t2(self, b0_t: float) -> float:
        return _interp_logfield(self.t2_anchors, b0_t)


def _interp_logfield(anchors: Dict[float, float], b0_t: float) -> float:
    """Log-B0 piecewise-linear interpolation between measured anchors.

    Outside the anchor range we hold the nearest anchor (no wild extrapolation):
    the low-field plateau and the CSF B0-independence are then both respected. A
    single anchor (e.g. CSF) is field-independent by construction.
    """
    items = sorted(anchors.items())
    if len(items) == 1:
        return items[0][1]
    b0 = max(min(b0_t, items[-1][0]), items[0][0])  # clamp into measured range
    for (b_a, v_a), (b_b, v_b) in zip(items[:-1], items[1:]):
        if b_a <= b0 <= b_b:
            la, lb, lx = math.log(b_a), math.log(b_b), math.log(b0)
            w = 0.0 if lb == la else (lx - la) / (lb - la)
            return v_a + w * (v_b - v_a)
    return items[-1][1]


# ── Normal tissue (in vivo) ──────────────────────────────────────────────────
# WM/GM anchored at 1.5 T (literature mid-range), 3 T (Wansapura), 64 mT (O'Reilly 50 mT
# in-vivo, the closest measured low-field point). T2 ~ flat (Stanisz) with mild low-field
# lengthening. CSF B0-independent (Rooney).
WHITE_MATTER = TissueRelaxation(
    "white_matter",
    t1_anchors={_B0_LOW: 275.0, _B0_15: 650.0, _B0_3: 832.0},
    # ~flat at clinical field (Stanisz 2005 / Wansapura 1999 WM T2 ~80 ms); mild
    # low-field lengthening toward 64 mT.
    t2_anchors={_B0_LOW: 85.0, _B0_15: 80.0, _B0_3: 80.0},
    proton_density=0.70,   # WM ~70 % water (myelin lipids invisible to liquid signal)
)
GRAY_MATTER = TissueRelaxation(
    "gray_matter",
    t1_anchors={_B0_LOW: 327.0, _B0_15: 1000.0, _B0_3: 1331.0},
    # ~field-independent within large in-vivo scatter (Stanisz/Wansapura GM T2 ~99-110 ms);
    # monotone non-increasing with B0 (mild low-field lengthening), matching the WM model
    # and this module's stated field-dependence (no T2 rise from 1.5 T -> 3 T).
    t2_anchors={_B0_LOW: 110.0, _B0_15: 105.0, _B0_3: 100.0},
    proton_density=0.82,   # GM ~82 % water
)
CSF = TissueRelaxation(
    "csf",
    t1_anchors={_B0_3: 4300.0},   # single anchor -> B0-independent (Rooney)
    t2_anchors={_B0_3: 2000.0},
    proton_density=1.00,
)

# ── Tumour sub-regions (BraTS label convention) ──────────────────────────────
# Prolonged T1/T2 vs normal brain. 3 T values from MR-fingerprinting/SyMRI studies;
# low-field anchors via the WM-like exponent (clamped to the 64 mT plateau).
# NOTE: these are *plausible-contrast* generators, not histology — large overlap exists.
ENHANCING_TUMOUR = TissueRelaxation(   # BraTS ET (label 3) — solid, contrast-enhancing
    "enhancing_tumour",
    t1_anchors={_B0_LOW: 560.0, _B0_15: 1550.0, _B0_3: 1900.0},
    # Oh 2005 (in-vivo 1.5T): glioma tumour T2 ~160 ms; 3T MR-fingerprinting ~125 ms.
    t2_anchors={_B0_LOW: 165.0, _B0_15: 160.0, _B0_3: 135.0},
    proton_density=0.85,
)
TUMOUR_CORE = TissueRelaxation(        # BraTS TC (necrotic + non-enhancing, label 1)
    "tumour_core",
    t1_anchors={_B0_LOW: 620.0, _B0_15: 1700.0, _B0_3: 2050.0},
    t2_anchors={_B0_LOW: 170.0, _B0_15: 160.0, _B0_3: 150.0},
    proton_density=0.88,
)
EDEMA = TissueRelaxation(              # BraTS ED (peritumoural vasogenic edema, label 2)
    "edema",
    t1_anchors={_B0_LOW: 470.0, _B0_15: 1200.0, _B0_3: 1400.0},
    # Oh 2005 (in-vivo 1.5T): peritumoural (vasogenic) edema T2 203-226 ms — much
    # higher than tumour core; roughly field-independent.
    t2_anchors={_B0_LOW: 215.0, _B0_15: 210.0, _B0_3: 200.0},
    proton_density=0.85,
)

NORMAL_TISSUES: Tuple[TissueRelaxation, ...] = (WHITE_MATTER, GRAY_MATTER, CSF)
# BraTS integer label -> tumour relaxation (label 0 = healthy, handled by tissue priors).
TUMOUR_BY_LABEL: Dict[int, TissueRelaxation] = {
    1: TUMOUR_CORE,       # NCR/NET
    2: EDEMA,             # ED
    3: ENHANCING_TUMOUR,  # ET
}

# Low-field validation anchors (for the physics-consistency unit test): at 64 mT the
# GM/WM T1 ratio must be small (contrast collapse), unlike at 3 T.
LOWFIELD_VALIDATION = {
    "gm_wm_t1_ratio_64mT": GRAY_MATTER.t1(_B0_LOW) / WHITE_MATTER.t1(_B0_LOW),   # ~1.19
    "gm_wm_t1_ratio_3T": GRAY_MATTER.t1(_B0_3) / WHITE_MATTER.t1(_B0_3),         # ~1.60
}


# ── Gadolinium enhancement (post-contrast T1c) ────────────────────────────────
# Solomon–Bloembergen–Morgan: a paramagnetic agent adds a *relaxation rate*, so
#   1/T1_post = 1/T1_pre + r1*[Gd]   (NOT a fixed multiplicative factor).
# The increment R1_Gd = r1*[Gd] (s^-1) is ~field-flat over 64 mT–3 T (mild Gd-DTPA
# NMRD dispersion), so one R1_Gd reproduces the physics a fixed 0.45x multiplier
# missed: because the rate (not T1) is shortened, the *relative* T1 drop is larger
# where T1_pre is long (high field) — matching real post-contrast enhancement, which
# is much brighter at 3 T than a 0.45x model gives. Default 2.5 s^-1 ≈ a clinical
# Gd-DTPA dose (r1 ~ 4.5 mM^-1 s^-1, [Gd] ~ 0.5 mM in enhancing tissue).
_GD_R1_PER_S: float = 2.5


def gd_shortened_t1(t1_pre_ms: float, r1_gd_per_s: float = _GD_R1_PER_S) -> float:
    """Post-Gd T1 (ms) via SBM: ``1/T1_post = 1/T1_pre + r1*[Gd]``.

    ``t1_pre_ms`` is the pre-contrast T1 in ms; ``r1_gd_per_s`` the effective
    Gd relaxation-rate increment in s^-1. Returns the shortened T1 in ms.
    """
    t1 = float(t1_pre_ms)
    return t1 / (1.0 + float(r1_gd_per_s) * t1 / 1000.0)


def bloch_signal(t1_ms: float, t2_ms: float, pd: float,
                 tr_ms: float, te_ms: float, ti_ms: float = 0.0) -> float:
    """Bloch steady-state magnitude signal for a (PD,T1,T2) tissue.

    Spin-echo / saturation-recovery: ``S = PD * (1 - exp(-TR/T1)) * exp(-TE/T2)``.
    Inversion recovery (``ti_ms > 0``): the longitudinal term becomes
    ``|1 - 2 exp(-TI/T1) + exp(-TR/T1)|`` (Olman/Haacke Bloch IR result). Pure
    scalar form used by the per-voxel partial-volume render in bloch_lowfield.py.
    """
    if ti_ms and ti_ms > 0.0:
        lon = abs(1.0 - 2.0 * math.exp(-ti_ms / t1_ms) + math.exp(-tr_ms / t1_ms))
    else:
        lon = 1.0 - math.exp(-tr_ms / t1_ms)
    return pd * lon * math.exp(-te_ms / t2_ms)
