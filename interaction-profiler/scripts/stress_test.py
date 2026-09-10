"""Stress / robustness test for the interaction profiler + REST service.

Throws adversarial SMILES and dirty / big / complex structures at the
pipeline and checks that:

* nothing crashes the process (every failure is a per-item error, never a 500);
* fallbacks (template -> geometry perception -> sanitise-only) are *surfaced*
  as warnings rather than degrading silently;
* result invariants hold (counts consistent, errored items carry no
  interactions, InChIKey/mode/bond-assignment agree, etc.).

Writes ``examples/stress_report.json`` with per-case outcomes plus an error /
warning taxonomy and a list of detected inconsistencies.

Run on the SLURM allocation (structure profiling needs OpenMM/RDKit)::

    python scripts/stress_test.py --out examples/stress_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from pathlib import Path

from interaction_profiler.core import profile_molecule_all
from interaction_profiler.io.batch_input import MoleculeInput
from interaction_profiler.schemas import MoleculeResult

REPO = Path(__file__).resolve().parent.parent
EXAMPLE_PDB = REPO / "examples" / "1OIU_boltz.pdb"
N76_SMILES = "c1cc(cc(c1)S(=O)(=O)N)Nc2nc3c(c(n2)OCC4CCCCC4)nc[nH]3"

# --- Adversarial SMILES (tested SMILES-only -> taxonomy path) --------------
ADVERSARIAL_SMILES: list[tuple[str, str]] = [
    ("empty", ""),
    ("whitespace", "   "),
    ("unclosed_ring", "C1CC"),
    ("unbalanced_paren", "C(C"),
    ("garbage", "not_a_smiles"),
    ("lone_bracket", "["),
    ("bad_valence_xenon", "F[Xe]F"),
    ("radical", "[CH3]"),
    ("isotope", "[13CH4]"),
    ("single_atom", "C"),
    ("water", "O"),
    ("ion_only", "[Cl-]"),
    ("salt_mixture", "CC(=O)[O-].[Na+]"),
    ("multi_component", "c1ccccc1.Cl.O"),
    ("cisplatin", "[Pt](Cl)(Cl)([NH3])[NH3]"),
    ("wildcard_rgroup", "*c1ccccc1"),
    ("dummy_atom", "[*]CC[*]"),
    ("stereo_defined", "N[C@@H](C)C(=O)O"),
    ("tautomer_prone", "O=c1cc[nH]cn1"),
    ("charged_zwitterion", "C(C(=O)[O-])[NH3+]"),
    ("embedded_newline", "CCO\nCCO"),
    ("embedded_space", "CC O"),
    ("very_long_chain", "C" * 300),
    ("big_peptide", "N[C@@H](C)C(=O)" + "N[C@@H](Cc1ccccc1)C(=O)" * 25 + "O"),
    ("cyclosporine_like", "C1CCCCCCCCCCCCCCCCCCCC1"),
    ("highly_charged", "C(=O)([O-])" + "C(=O)([O-])" * 8),
    ("fullerene_ish", "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61"),
]

def _norm_warn(w: str) -> str:
    w = re.sub(r"\d+", "N", w)
    w = re.sub(r"/\S+", "<path>", w)
    return w.strip()[:80]


def check_invariants(label: str, r: MoleculeResult) -> list[str]:
    """Structural invariants that must hold for every result."""
    issues: list[str] = []
    md, prep = r.metadata, r.preparation
    if md.n_interactions != len(r.interactions):
        issues.append(f"{label}: n_interactions={md.n_interactions} != len(interactions)={len(r.interactions)}")
    if md.n_triples != len(r.triples):
        issues.append(f"{label}: n_triples={md.n_triples} != len(triples)={len(r.triples)}")
    if r.error and (r.interactions or r.triples):
        issues.append(f"{label}: error set but interactions/triples not cleared")
    if md.n_interactions_detected and md.n_interactions > md.n_interactions_detected:
        issues.append(f"{label}: kept ({md.n_interactions}) > detected ({md.n_interactions_detected})")
    sum_conf = sum(t.n_confident + t.n_uncertain for t in r.triples)
    total_by_triples = sum(t.count for t in r.triples)
    if sum_conf != total_by_triples:
        issues.append(f"{label}: triple confidence counts ({sum_conf}) != total count ({total_by_triples})")
    if prep:
        if prep.bond_assignment == "template" and prep.inchikey_match is False:
            issues.append(f"{label}: template bond assignment but InChIKey mismatch (silent wrong template?)")
        if md.input_mode == "structure_only" and not any("no SMILES template" in w for w in prep.warnings):
            issues.append(f"{label}: structure_only but no 'no SMILES template' warning (silent perception)")
        if prep.inchikey_match is False and not any("does not match" in w for w in prep.warnings):
            issues.append(f"{label}: InChIKey mismatch not surfaced as a warning")
    return issues


def run_smiles_case(label: str, smiles: str) -> dict:
    t0 = time.time()
    try:
        rs = profile_molecule_all(MoleculeInput(name=label, smiles=smiles))
    except Exception:
        return {"label": label, "kind": "smiles", "smiles": smiles[:40],
                "crash": traceback.format_exc().splitlines()[-1], "issues": [f"{label}: CRASH"]}
    r = rs[0]
    prep = r.preparation
    issues = check_invariants(label, r)
    return {
        "label": label, "kind": "smiles", "smiles": (smiles[:40] + ("…" if len(smiles) > 40 else "")),
        "mode": r.metadata.input_mode, "error": r.error,
        "n_fragments": len(r.composition.fragments),
        "bond": getattr(prep, "bond_assignment", None),
        "inchikey_match": getattr(prep, "inchikey_match", None),
        "warnings": (getattr(prep, "warnings", []) or [])[:3],
        "seconds": round(time.time() - t0, 3), "issues": issues,
    }


def run_structure_case(label: str, structure: str, *, resname: str | None = "LIG",
                       smiles: str | None = None, all_ligands: bool = False,
                       strict: bool = False) -> dict:
    name = label
    inp = MoleculeInput(name=name, structure=structure, ligand_resname=resname, smiles=smiles)
    t0 = time.time()
    try:
        rs = profile_molecule_all(inp, all_ligands=all_ligands, strict_ligand_match=strict)
    except Exception:
        return {"label": label, "kind": "structure", "name": name,
                "crash": traceback.format_exc().splitlines()[-1], "issues": [f"{label}: CRASH"]}
    out_cases = []
    all_issues: list[str] = []
    for r in rs:
        prep = r.preparation
        iss = check_invariants(f"{label}/{r.metadata.name}", r)
        all_issues += iss
        out_cases.append({
            "name": r.metadata.name, "mode": r.metadata.input_mode, "receptor": r.metadata.receptor,
            "n_int": r.metadata.n_interactions, "n_detected": r.metadata.n_interactions_detected,
            "n_triples": r.metadata.n_triples, "error": r.error,
            "bond": getattr(prep, "bond_assignment", None),
            "inchikey_match": getattr(prep, "inchikey_match", None),
            "inchikey_exact": getattr(prep, "inchikey_exact_match", None),
            "n_warnings": len(getattr(prep, "warnings", []) or []),
            "warnings": (getattr(prep, "warnings", []) or [])[:4],
        })
    return {"label": label, "kind": "structure", "name": name, "category": label,
            "seconds": round(time.time() - t0, 3), "results": out_cases, "issues": all_issues}


def make_dirty_files(tmp: Path) -> dict[str, str]:
    tmp.mkdir(parents=True, exist_ok=True)
    files = {}
    (tmp / "garbage.pdb").write_text("this is not a pdb file at all\nrandom text\n")
    files["garbage_text"] = str(tmp / "garbage.pdb")
    (tmp / "empty.pdb").write_text("")
    files["empty_file"] = str(tmp / "empty.pdb")
    if EXAMPLE_PDB.exists():
        lines = EXAMPLE_PDB.read_text().splitlines()[:25]
        (tmp / "truncated.pdb").write_text("\n".join(lines) + "\n")
        files["truncated_pdb"] = str(tmp / "truncated.pdb")
    return files


def run_service_cases() -> list[dict]:
    out: list[dict] = []
    try:
        import importlib.util

        from fastapi.testclient import TestClient

        # App now lives in the deployment wrapper (single app per boltz2/posebusters convention).
        app_path = REPO.parent / "services" / "interaction-profiler" / "app.py"
        spec = importlib.util.spec_from_file_location("ip_service_app", app_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        create_app = mod.create_app
    except Exception as exc:  # noqa: BLE001
        return [{"label": "service_import", "error": str(exc)}]
    os.environ["IP_SHARED_DIR"] = str(REPO / "examples")
    client = TestClient(create_app())
    checks = [
        ("path_traversal", {"molecules": [{"name": "x", "structure": "../../etc/passwd"}]}, "escapes"),
        ("invalid_smiles", {"molecules": [{"name": "x", "smiles": "C1CC"}]}, "invalid SMILES"),
        ("neither", {"molecules": [{"name": "x"}]}, "neither"),
        ("valid_smiles", {"molecules": [{"name": "x", "smiles": "CCO"}]}, None),
    ]
    for label, body, expect in checks:
        r = client.post("/analyze", json=body)
        err = (r.json().get("molecules", [{}])[0].get("error") if r.status_code == 200 else f"HTTP {r.status_code}")
        ok = (expect is None and not err) or (expect is not None and err and expect in err)
        out.append({"label": label, "http": r.status_code, "error": err, "expected": expect, "ok": bool(ok)})
    # malformed body -> 422
    r = client.post("/analyze", json={"batch_id": "x"})
    out.append({"label": "malformed_body", "http": r.status_code, "ok": r.status_code == 422})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/stress_report.json"))
    args = ap.parse_args()
    report: dict = {"smiles_cases": [], "structure_cases": [], "service_cases": []}

    print("== SMILES battery ==")
    for label, smi in ADVERSARIAL_SMILES:
        c = run_smiles_case(label, smi)
        report["smiles_cases"].append(c)
        print(f"  {label:22} mode={c.get('mode')} err={c.get('error')} issues={len(c['issues'])}")

    print("== dirty structure files ==")
    dirty = make_dirty_files(Path("/tmp/ip_stress"))
    for label, path in dirty.items():
        c = run_structure_case(label, path, resname=None, smiles=None)
        report["structure_cases"].append(c)
        errs = [r.get("error") for r in c.get("results", [])]
        print(f"  {label:22} err={errs} issues={len(c['issues'])}")

    print("== example structure (Boltz-2 1OIU) ==")
    pdb = str(EXAMPLE_PDB)
    example_cases = [
        ("template", dict(smiles=N76_SMILES)),
        ("structure_only", dict(smiles=None)),
        ("mismatch_template", dict(smiles="CC(=O)Oc1ccccc1C(=O)O")),
        ("mismatch_strict", dict(smiles="CC(=O)Oc1ccccc1C(=O)O", strict=True)),
        ("all_ligands", dict(smiles=N76_SMILES, all_ligands=True)),
    ]
    if not EXAMPLE_PDB.exists():
        print(f"  SKIP: {EXAMPLE_PDB} not found")
    else:
        for label, kw in example_cases:
            c = run_structure_case(label, pdb, **kw)
            report["structure_cases"].append(c)
            rr = c.get("results", [{}])
            print(f"  {label:18} err={c.get('error')} "
                  f"n_int={rr[0].get('n_int') if rr else '-'} "
                  f"bond={rr[0].get('bond') if rr else '-'} issues={len(c['issues'])}")

    print("== service edge cases ==")
    report["service_cases"] = run_service_cases()
    for c in report["service_cases"]:
        print(f"  {c['label']:16} ok={c.get('ok')} http={c.get('http')} err={c.get('error')}")

    # --- aggregate taxonomy + inconsistencies ---
    err_tax: dict[str, int] = {}
    warn_tax: dict[str, int] = {}
    inconsistencies: list[str] = []
    crashes: list[str] = []
    n_cases = 0
    for c in report["smiles_cases"]:
        n_cases += 1
        if c.get("crash"):
            crashes.append(f"{c['label']}: {c['crash']}")
        if c.get("error"):
            err_tax[c["error"].split(":")[0]] = err_tax.get(c["error"].split(":")[0], 0) + 1
        for w in c.get("warnings", []):
            warn_tax[_norm_warn(w)] = warn_tax.get(_norm_warn(w), 0) + 1
        inconsistencies += c.get("issues", [])
    for c in report["structure_cases"]:
        n_cases += 1
        if c.get("crash"):
            crashes.append(f"{c['label']}: {c['crash']}")
        for r in c.get("results", []):
            if r.get("error"):
                err_tax[r["error"].split(":")[0]] = err_tax.get(r["error"].split(":")[0], 0) + 1
            for w in r.get("warnings", []):
                warn_tax[_norm_warn(w)] = warn_tax.get(_norm_warn(w), 0) + 1
        inconsistencies += c.get("issues", [])

    report["summary"] = {
        "n_cases": n_cases,
        "n_smiles": len(report["smiles_cases"]),
        "n_structure_cases": len(report["structure_cases"]),
        "crashes": crashes,
        "n_inconsistencies": len(inconsistencies),
        "service_all_ok": all(c.get("ok") for c in report["service_cases"]),
    }
    report["error_taxonomy"] = dict(sorted(err_tax.items(), key=lambda kv: -kv[1]))
    report["warning_taxonomy"] = dict(sorted(warn_tax.items(), key=lambda kv: -kv[1]))
    report["inconsistencies"] = inconsistencies

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n== SUMMARY == crashes={len(crashes)} inconsistencies={len(inconsistencies)} "
          f"service_ok={report['summary']['service_all_ok']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
