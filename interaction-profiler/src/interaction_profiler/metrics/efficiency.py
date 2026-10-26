"""Ligand efficiency metrics — LE, LLE, and BEI.

Implements a priority-based affinity selection strategy:

    1. ``experimental`` — highest priority; any unit/type accepted.
    2. ``boltz``        — Boltz-2 predicted affinity, gated by quality
                         checks (confidence_score, ligand_iptm, PoseBusters).
    3. ``model``        — generic ML model output.
    4. ``docking``      — docking score with explicit conversion record.

All affinity values are converted to pIC50 before computing efficiency
metrics.  Every conversion step is logged to ``conversion_record`` so
results are fully auditable.

Formulas (Reynolds et al., Hopkins et al.)
------------------------------------------
    RT  = 0.593 kcal/mol  (298 K, R = 1.987 cal/mol·K)
    LE  = (RT × ln10 × pIC50) / N_heavy      [kcal/mol per heavy atom]
    LLE = pIC50 − clogP                       [dimensionless]
    BEI = pIC50 × 1000 / MW_Da               [pIC50 per kDa, ≡ pIC50/MW_kDa]

clogP and MW are computed from the ligand SMILES using RDKit.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# kcal/mol at 298 K  (R = 8.314 J/mol·K → 1.9872 cal/mol·K)
_RT_KCAL = 0.593
_LN10 = math.log(10)

# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

# Supported concentration units → multiplier to convert to molar
_UNIT_TO_MOLAR: dict[str, float] = {
    "nm": 1e-9,
    "nm ": 1e-9,
    "nanomolar": 1e-9,
    "um": 1e-6,
    "µm": 1e-6,
    "micromolar": 1e-6,
    "mm": 1e-3,
    "millimolar": 1e-3,
    "m": 1.0,
    "molar": 1.0,
    "pm": 1e-12,
    "picomolar": 1e-12,
}

# Affinity types that are already on the pKi / pIC50 / pKd scale
_ALREADY_P_SCALE = {"pic50", "pki", "pkd"}

# Affinity types expressed as concentrations (IC50 / Ki / Kd)
_CONCENTRATION_TYPES = {"ic50", "ki", "kd"}


def _concentration_to_pic50(value: float, unit: str, affinity_type: str) -> tuple[float, str]:
    """Convert a concentration-based affinity to pIC50.

    Parameters
    ----------
    value:
        Numeric affinity value.
    unit:
        Concentration unit string (case-insensitive), e.g. ``"nM"``, ``"µM"``.
    affinity_type:
        Affinity measurement type (case-insensitive), e.g. ``"IC50"``, ``"Ki"``.

    Returns
    -------
    pic50:
        Converted pIC50 value.
    record:
        Human-readable description of the conversion applied.

    Raises
    ------
    ValueError
        If the unit is unrecognised or the value is non-positive.
    """
    if value <= 0:
        raise ValueError(f"Affinity value must be positive; got {value!r}")

    unit_key = unit.strip().lower()
    multiplier = _UNIT_TO_MOLAR.get(unit_key)
    if multiplier is None:
        raise ValueError(
            f"Unrecognised concentration unit {unit!r}. "
            f"Supported: {sorted(_UNIT_TO_MOLAR.keys())}"
        )

    conc_molar = value * multiplier
    pic50 = -math.log10(conc_molar)
    record = (
        f"{affinity_type}={value} {unit} → {conc_molar:.3e} M → "
        f"pIC50 = -log10({conc_molar:.3e}) = {pic50:.3f}"
    )
    return pic50, record


def _affinity_to_pic50(
    value: float,
    unit: str,
    affinity_type: str,
) -> tuple[float, str]:
    """Dispatch to the correct pIC50 conversion based on affinity type.

    Parameters
    ----------
    value:
        Numeric potency value.
    unit:
        Unit string (used for concentration types; ignored for p-scale types).
    affinity_type:
        Measurement type string (case-insensitive).

    Returns
    -------
    pic50, record
        See :func:`_concentration_to_pic50`.
    """
    atype_key = affinity_type.strip().lower()

    if atype_key in _ALREADY_P_SCALE:
        record = f"{affinity_type}={value:.3f} (already on pIC50 scale; used directly)"
        return float(value), record

    if atype_key in _CONCENTRATION_TYPES:
        return _concentration_to_pic50(value, unit, affinity_type)

    raise ValueError(
        f"Unrecognised affinity type {affinity_type!r}. "
        f"Supported: {sorted(_ALREADY_P_SCALE | _CONCENTRATION_TYPES)}"
    )


# ---------------------------------------------------------------------------
# Boltz quality gate
# ---------------------------------------------------------------------------

def _boltz_passes_gate(boltz_entry: dict[str, Any]) -> tuple[bool, str]:
    """Check whether a Boltz affinity prediction meets the quality threshold.

    Gate criteria (all must hold):
        * ``confidence_score > 0.6``
        * ``ligand_iptm > 0.5``
        * ``posebusters_all_pass == True``

    Parameters
    ----------
    boltz_entry:
        The ``potency["boltz"]`` sub-dictionary.

    Returns
    -------
    passed:
        ``True`` if all criteria are satisfied.
    reason:
        Human-readable gate result string.
    """
    checks: list[str] = []
    passed = True

    conf = boltz_entry.get("confidence_score")
    if conf is None or conf <= 0.6:
        passed = False
        checks.append(f"confidence_score={conf!r} (need >0.6)")
    else:
        checks.append(f"confidence_score={conf:.3f} ✓")

    iptm = boltz_entry.get("ligand_iptm")
    if iptm is None or iptm <= 0.5:
        passed = False
        checks.append(f"ligand_iptm={iptm!r} (need >0.5)")
    else:
        checks.append(f"ligand_iptm={iptm:.3f} ✓")

    pb = boltz_entry.get("posebusters_all_pass")
    if pb is not True:
        passed = False
        checks.append(f"posebusters_all_pass={pb!r} (need True)")
    else:
        checks.append("posebusters_all_pass=True ✓")

    status = "PASS" if passed else "FAIL"
    reason = f"Boltz gate [{status}]: {'; '.join(checks)}"
    return passed, reason


# ---------------------------------------------------------------------------
# Potency selection with priority ordering
# ---------------------------------------------------------------------------

def _select_potency(
    potency: dict[str, Any],
) -> tuple[float, str, str, str] | None:
    """Return ``(pic50, source_label, type_original, conversion_record)`` or None.

    Tries each potency source in descending priority order and returns the
    first that can be converted to a valid pIC50.
    """
    records: list[str] = []

    # --- 1. Experimental ---
    exp = potency.get("experimental")
    if exp is not None:
        try:
            val = float(exp["value"])
            unit = str(exp.get("unit", ""))
            atype = str(exp.get("type", ""))
            pic50, rec = _affinity_to_pic50(val, unit, atype)
            records.append(f"[experimental] {rec}")
            return pic50, "experimental", atype, "\n".join(records)
        except Exception as exc:  # noqa: BLE001
            records.append(f"[experimental] skipped — {exc}")

    # --- 2. Boltz (gate-checked) ---
    boltz = potency.get("boltz")
    if boltz is not None:
        gate_ok, gate_msg = _boltz_passes_gate(boltz)
        records.append(f"[boltz] {gate_msg}")
        if gate_ok:
            try:
                raw = float(boltz["affinity_pred_value"])
                # Boltz reports log10(IC50/µM); convert to pIC50:
                # pIC50 = -log10(IC50_M) = -log10(10^raw × 1e-6) = -raw + 6
                pic50 = -raw + 6.0
                rec = (
                    f"affinity_pred_value={raw:.4f} (log10(IC50/µM)) → "
                    f"pIC50 = -{raw:.4f} + 6 = {pic50:.4f}"
                )
                records.append(f"[boltz] {rec}")
                return pic50, "boltz", "log10(IC50/µM)", "\n".join(records)
            except Exception as exc:  # noqa: BLE001
                records.append(f"[boltz] skipped — {exc}")
        else:
            records.append("[boltz] gate not satisfied; trying next source.")

    # --- 3. Model ---
    model = potency.get("model")
    if model is not None:
        try:
            val = float(model["value"])
            unit = str(model.get("unit", ""))
            atype = str(model.get("type", "IC50"))
            pic50, rec = _affinity_to_pic50(val, unit, atype)
            records.append(f"[model] {rec}")
            return pic50, "model", atype, "\n".join(records)
        except Exception as exc:  # noqa: BLE001
            records.append(f"[model] skipped — {exc}")

    # --- 4. Docking ---
    docking = potency.get("docking")
    if docking is not None:
        try:
            val = float(docking["score"])
            unit = str(docking.get("unit", "")).strip().lower()
            dock_rec = str(docking.get("conversion_record", "no conversion record provided"))
            atype = str(docking.get("type", "IC50"))

            if unit == "kcal/mol":
                # ΔG → pKd: ΔG = −RT ln(Ka) = −RT × ln(10) × pKd
                # pKd = −ΔG / (RT × ln(10))  where RT×ln(10) = 1.364 kcal/mol at 298 K
                # Note: more negative ΔG = tighter binding = higher pKd
                RT_LN10 = 0.593 * 2.3026  # ≈ 1.365 kcal/mol
                pic50 = -val / RT_LN10
                rec = (
                    f"docking_score={val} kcal/mol → pKd = -{val:.3f} / {RT_LN10:.3f} = {pic50:.3f}"
                )
            else:
                pic50, rec = _affinity_to_pic50(val, unit, atype)

            records.append(f"[docking] {rec} | conversion_record: {dock_rec}")
            return pic50, "docking", atype, "\n".join(records)
        except Exception as exc:  # noqa: BLE001
            records.append(f"[docking] skipped — {exc}")

    logger.debug("No valid potency source found. Attempt log:\n%s", "\n".join(records))
    return None


# ---------------------------------------------------------------------------
# RDKit property helpers
# ---------------------------------------------------------------------------

def _rdkit_properties(smiles: str) -> tuple[float | None, float | None, int]:
    """Return (clogP, MW_Da, n_heavy_atoms) from a SMILES string.

    Returns ``(None, None, 0)`` if RDKit is unavailable or the SMILES is
    unparseable.
    """
    try:
        from rdkit.Chem import Crippen, Descriptors, MolFromSmiles  # noqa: PLC0415

        mol = MolFromSmiles(smiles)
        if mol is None:
            return None, None, 0
        clogp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        n_heavy = mol.GetNumHeavyAtoms()
        return clogp, mw, n_heavy
    except Exception as exc:  # noqa: BLE001
        logger.debug("RDKit property calculation failed for SMILES %r: %s", smiles, exc)
        return None, None, 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_efficiency(
    smiles: str,
    potency: dict[str, Any] | None,
    parent_le: float | None = None,
    parent_n_heavy: int | None = None,
) -> "EfficiencyResult":
    """Compute ligand efficiency metrics (LE, LLE, BEI, GE, LELP) for a molecule.

    Parameters
    ----------
    smiles:
        Canonical SMILES of the ligand.  Used to derive clogP, MW, and
        heavy-atom count via RDKit.
    potency:
        Dictionary with zero or more of the following keys:

        ``"experimental"``
            Highest priority.  Expected sub-keys: ``value`` (float),
            ``unit`` (str, e.g. ``"nM"``), ``type`` (str, e.g. ``"IC50"``).

        ``"boltz"``
            Boltz-2 predicted affinity.  Required sub-keys:
            ``affinity_pred_value`` (float, log10(IC50/µM)),
            ``confidence_score`` (float), ``ligand_iptm`` (float),
            ``posebusters_all_pass`` (bool).  All quality gates must pass.

        ``"model"``
            Generic ML model affinity.  Required sub-keys: ``value``
            (float), ``unit`` (str), ``type`` (str).

        ``"docking"``
            Docking score.  Required sub-keys: ``score`` (float),
            ``unit`` (str), ``conversion_record`` (str).

        Pass ``None`` (or an empty dict) to get ``error="no_valid_potency"``.

    Returns
    -------
    EfficiencyResult
        ``error`` is ``None`` on success.  If no potency source passes
        validation, ``error`` is set to ``"no_valid_potency"``.

    Notes
    -----
    * If RDKit is unavailable, LE / LLE / BEI cannot be computed and
      ``error`` will report the missing dependency.
    * The ``conversion_record`` field contains a human-readable log of
      every selection decision and unit-conversion step applied.

    Examples
    --------
    >>> result = compute_efficiency(
    ...     smiles="CCc1cc(C(=O)Nc2ccc(S(N)(=O)=O)cc2)no1",
    ...     potency={"experimental": {"value": 12.0, "unit": "nM", "type": "IC50"}},
    ... )
    >>> round(result.potency_value_pic50, 2)
    7.92
    """
    from interaction_profiler.schemas import EfficiencyResult  # noqa: PLC0415

    # ------------------------------------------------------------------ #
    # 1.  Potency selection
    # ------------------------------------------------------------------ #
    if not potency:
        return EfficiencyResult(error="no_valid_potency")

    selection = _select_potency(potency)
    if selection is None:
        return EfficiencyResult(error="no_valid_potency")

    pic50, source_label, type_original, conv_record = selection

    # ------------------------------------------------------------------ #
    # 2.  Molecular properties from RDKit
    # ------------------------------------------------------------------ #
    try:
        clogp, mw_da, n_heavy = _rdkit_properties(smiles)
    except Exception as exc:  # noqa: BLE001
        logger.exception("RDKit property extraction failed for SMILES %r", smiles)
        return EfficiencyResult(
            error=f"RDKit property extraction failed: {exc}",
            potency_source=source_label,
            potency_value_pic50=pic50,
            potency_type_original=type_original,
            conversion_record=conv_record,
        )

    if n_heavy == 0:
        return EfficiencyResult(
            error=f"SMILES could not be parsed by RDKit: {smiles!r}",
            potency_source=source_label,
            potency_value_pic50=pic50,
            potency_type_original=type_original,
            conversion_record=conv_record,
        )

    # ------------------------------------------------------------------ #
    # 3.  Efficiency metrics
    # ------------------------------------------------------------------ #
    try:
        # LE = (RT × ln10 × pIC50) / N_heavy
        le = (_RT_KCAL * _LN10 * pic50) / n_heavy if n_heavy > 0 else None

        # LLE = pIC50 − clogP
        lle = (pic50 - clogp) if clogp is not None else None

        # BEI = pIC50 / MW_kDa  =  pIC50 × 1000 / MW_Da
        bei = (pic50 * 1000.0 / mw_da) if mw_da is not None and mw_da > 0 else None

        # G1: Group Efficiency GE = delta(LE) * parent_n_heavy / delta_n_heavy
        ge: float | None = None
        if le is not None and parent_le is not None and parent_n_heavy is not None:
            delta_le = le - parent_le
            delta_hac = n_heavy - parent_n_heavy
            if delta_hac > 0:
                ge = round(delta_le * parent_n_heavy / delta_hac, 4)

        # G2: LELP = LE / clogP
        lelp: float | None = None
        if le is not None and clogp is not None and clogp > 0:
            lelp = round(le / clogp, 4)

        return EfficiencyResult(
            le=le,
            lle=lle,
            bei=bei,
            ge=ge,
            lelp=lelp,
            potency_source=source_label,
            potency_value_pic50=pic50,
            potency_type_original=type_original,
            conversion_record=conv_record,
            clogp=clogp,
            mw_da=mw_da,
            n_heavy_atoms=n_heavy,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Efficiency metric computation failed")
        return EfficiencyResult(
            error=str(exc),
            potency_source=source_label,
            potency_value_pic50=pic50,
            potency_type_original=type_original,
            conversion_record=conv_record,
            clogp=clogp,
            mw_da=mw_da,
            n_heavy_atoms=n_heavy,
        )


__all__ = [
    "compute_efficiency",
]
