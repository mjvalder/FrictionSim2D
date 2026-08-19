"""Versioned QC status classifier (spec §7).

Consumes the logged ingredients — slide distance, penetration, contact count,
the F2 drift detectors, achieved-vs-requested load, ramp convergence — and
assigns one status. Post-hoc and versioned: the taxonomy can be revised and
re-applied without re-simulating (the 2026 70/28/2 split was itself inferred
post-hoc, which is why the version is recorded).

This is better-instrumented inference, not ground truth.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

QC_VERSION = '1.0.0'

STATUSES = ('converged', 'seized', 'detached', 'punched_through', 'drifted')


@dataclass(frozen=True)
class QCResult:
    status: str
    reasons: List[str]
    version: str


@dataclass(frozen=True)
class QCThresholds:
    """Defaults calibrated against KEY_RESULTS §3's diagnostics; the pe-drift
    and thermostat-linearity bands are re-pinned by pilot arm G."""
    slide_min_ang: float = 2.0          # below -> seized (KEY_RESULTS §3)
    nf_cv_max: float = 0.30             # above -> detached
    pe_drift_max_ev_ps: float = 0.5     # above -> drifted
    therm_work_r2_min: float = 0.95     # thermostat-work linearity (F2 #1)
    load_tol_frac: float = 0.10


def classify(df: pd.DataFrame, provenance: dict,
             thresholds: QCThresholds = QCThresholds()) -> QCResult:
    """First matching status wins. AFM-shaped input (tipx/tipy/tipz/nf)."""
    reasons: List[str] = []

    # punched_through — prefer min-tip-z (registry channel tip_min_z);
    # the COM fallback is weaker: the tip centroid sits well above its apex
    sheet_ref = df['comz_L1'] if 'comz_L1' in df else df['comz']
    if 'tip_min_z' in df:
        pen = (df['tip_min_z'] - sheet_ref).min()
    else:
        pen = (df['tipz'] - sheet_ref).min()
        reasons.append('punch-through check on tip COM (weaker; no tip_min_z)')
    if pen < 0:
        return QCResult('punched_through',
                        reasons + [f'tip below bottom layer, pen={pen:.2f}'],
                        QC_VERSION)

    # detached — normal force not maintained
    nf_cv = df['nf'].std() / max(abs(df['nf'].mean()), 1e-9)
    if nf_cv > thresholds.nf_cv_max:
        return QCResult('detached', [f'nf CV {nf_cv:.2f}'], QC_VERSION)

    # seized — no lateral travel despite driving
    slide = float(np.hypot(df['tipx'].iloc[-1] - df['tipx'].iloc[0],
                           df['tipy'].iloc[-1] - df['tipy'].iloc[0]))
    if slide < thresholds.slide_min_ang:
        return QCResult('seized', [f'slid {slide:.2f} A'], QC_VERSION)

    # drifted — two detectors (spec §5 F2): pe trend AND thermostat-work
    # linearity (etotal is not conserved by design: langevin + setforce +
    # aveforce + driving spring all inject/remove energy)
    if 'pe' in df and len(df) > 2:
        drift = abs(np.polyfit(df['time'], df['pe'], 1)[0]) * 1000.0
        if drift > thresholds.pe_drift_max_ev_ps:
            return QCResult('drifted', [f'pe drift {drift:.3f} eV/ps'],
                            QC_VERSION)
    for wcol in ('w_thermostat_tip', 'w_thermostat_sub'):
        if wcol in df and len(df) > 10:
            r2 = float(np.corrcoef(df['time'], df[wcol])[0, 1] ** 2)
            if r2 < thresholds.therm_work_r2_min:
                return QCResult('drifted',
                                [f'{wcol} nonlinear (R2={r2:.3f})'],
                                QC_VERSION)

    # load achieved vs requested (diagnostic-only when unavailable)
    load_req = provenance.get('load_requested')
    if load_req:
        rel = abs(abs(df['nf'].mean()) - abs(load_req)) / abs(load_req)
        if rel > thresholds.load_tol_frac:
            reasons.append(f'load off by {rel:.0%} (diagnostic only)')
    if not provenance.get('ramp_converged', True):
        reasons.append('ramp unconverged (diagnostic only)')

    return QCResult('converged', reasons, QC_VERSION)
