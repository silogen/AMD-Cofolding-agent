"""Custom geometric interaction detector.

Given a prepared ligand Mol (explicit Hs, correct bonds) and an RDKit Mol
of the cropped pocket (carrying PDB residue info), detect a rich set of
typed interactions with geometry-derived confidence. No ProLIF.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from interaction_profiler.interactions import features as F
from interaction_profiler.interactions import rules as R
from interaction_profiler.interactions.geometry import (
    angle_deg,
    pairwise_distances,
    plane_angle_deg,
    ring_offset,
)
from interaction_profiler.schemas import Interaction

logger = logging.getLogger(__name__)


class _Collector:
    """De-duplicates interactions, keeping the best (closest / most linear)."""

    def __init__(self) -> None:
        self._best: dict[tuple, Interaction] = {}
        self._score: dict[tuple, float] = {}

    def add(self, key: tuple, inter: Interaction | None, score: float) -> None:
        if inter is None:
            return
        if key not in self._best or score < self._score.get(key, float("inf")):
            self._best[key] = inter
            self._score[key] = score

    def list(self) -> list[Interaction]:
        return list(self._best.values())


def _rec_fields(rf: F.ReceptorFeatures, idx: int) -> dict:
    m = rf.meta[idx]
    return {
        "receptor_chain": m["chain"],
        "receptor_resname": m["resname"],
        "receptor_resnum": m["resnum"],
        "receptor_reskind": m["reskind"],
        "receptor_atom_name": m["name"],
        "receptor_moiety": m["moiety"],
    }


def _lig_fields(lf: F.LigandFeatures, names: list[str], idx: int) -> dict:
    return {
        "ligand_atom_idx": int(idx),
        "ligand_atom_name": names[idx] if 0 <= idx < len(names) else None,
        "ligand_element": lf.elements[idx] if 0 <= idx < len(lf.elements) else None,
    }


def detect_interactions(
    lig_mol: Any,
    lig_atom_names: list[str],
    receptor_ag: Any,
    *,
    metals: list | None = None,
    waters: list | None = None,
    covalent_links: set | None = None,
    use_gpu: bool = False,
) -> list[Interaction]:
    lf = F.perceive_ligand(lig_mol)
    rf = F.perceive_receptor(receptor_ag)
    rf.metals = list(metals or [])
    rf.waters = list(waters or [])
    col = _Collector()

    residues_with_typed: set[tuple] = set()

    def note_typed_key(chain: str, resnum) -> None:
        residues_with_typed.add((chain, resnum))

    def note_typed(recv_idx: int) -> None:
        m = rf.meta[recv_idx]
        note_typed_key(m["chain"], m["resnum"])

    _covalent(lf, rf, lig_atom_names, col, note_typed, use_gpu, links=covalent_links)
    _hbonds(lf, rf, lig_atom_names, col, note_typed, use_gpu)
    _halogen_bonds(lf, rf, lig_atom_names, col, note_typed)
    _ionic(lf, rf, lig_atom_names, col, note_typed)
    _pi(lf, rf, lig_atom_names, col, note_typed)
    _chalcogen_bond(lf, rf, lig_atom_names, col, note_typed)
    _sulfur_arene(lf, rf, lig_atom_names, col, note_typed)
    _amide_pi(lf, rf, lig_atom_names, col, note_typed)
    _metal(lf, rf, lig_atom_names, col, note_typed_key)
    _water_bridge(lf, rf, lig_atom_names, col, note_typed_key, use_gpu)
    _hydrophobic(lf, rf, lig_atom_names, col, note_typed, use_gpu)
    _chpi(lf, rf, lig_atom_names, col, note_typed)
    _vdw(lf, rf, lig_atom_names, col, residues_with_typed, use_gpu)

    return col.list()


# Nucleophilic sidechain atoms that form covalent bonds to warheads.
REACTIVE_ATOMS = {
    "CYS": {"SG"}, "SER": {"OG"}, "THR": {"OG1"}, "LYS": {"NZ"},
    "TYR": {"OH"}, "HIS": {"ND1", "NE2"}, "HID": {"ND1", "NE2"},
    "HIE": {"ND1", "NE2"}, "HIP": {"ND1", "NE2"}, "SEC": {"SE"},
}
# Electrophilic ligand warhead atoms.
WARHEAD_ELEMENTS = {"C", "S", "B", "P"}


def _covalent(lf, rf, names, col, note_typed, use_gpu, links=None):
    """Flag covalent bonds. A ligand heavy atom within bonding distance
    (~2.1 A) of a receptor heavy atom, restricted to plausible warheads
    (ligand C/S/B/P) and nucleophilic receptor atoms (Cys SG, Ser/Thr OG,
    Lys NZ, Tyr OH, His N). PDB LINK records, when present, are trusted
    directly. Non-bonded contacts never get this close, but the residue
    restriction avoids mislabelling an incidental short contact (e.g. to a
    Phe ring) as covalent."""
    rec_heavy = [i for i, m in enumerate(rf.meta) if m["element"].upper() != "H"]
    lig_heavy = [i for i, e in enumerate(lf.elements) if e.upper() != "H"]
    if not rec_heavy or not lig_heavy:
        return
    links = links or set()
    dm = pairwise_distances(lf.coords[lig_heavy], rf.coords[rec_heavy], use_gpu)
    for i, laidx in enumerate(lig_heavy):
        for j, ridx in enumerate(rec_heavy):
            dist = float(dm[i, j])
            if not (0.5 < dist <= 2.1):
                continue
            m = rf.meta[ridx]
            resname = (m["resname"] or "").upper()
            linked = (resname, m["resnum"], m["chain"]) in links or (resname, m["resnum"]) in links
            reactive = (
                lf.elements[laidx].upper() in WARHEAD_ELEMENTS
                and (m["name"] or "").upper() in REACTIVE_ATOMS.get(resname, set())
            )
            if not (linked or reactive):
                continue
            inter = Interaction(
                interaction_type="covalent", subtype="covalent_bond",
                distance=round(dist, 2), confidence="confident",
                **_lig_fields(lf, names, laidx), **_rec_fields(rf, ridx),
            )
            col.add(("covalent", laidx, ridx), inter, dist)
            note_typed(ridx)


def _hbonds(lf, rf, names, col, note_typed, use_gpu):
    # ligand donor -> receptor acceptor
    if lf.donors and rf.acceptors:
        rec_xyz = rf.coords[rf.acceptors]
        d_xyz = np.array([d.heavy_xyz for d in lf.donors])
        dm = pairwise_distances(d_xyz, rec_xyz, use_gpu)
        for i, donor in enumerate(lf.donors):
            for j, ridx in enumerate(rf.acceptors):
                dist = float(dm[i, j])
                if dist > R.HBOND_DIST.ext_max:
                    continue
                ang = _best_hbond_angle(donor.h_xyz, donor.heavy_xyz, rf.coords[ridx])
                conf = R.combine(R.HBOND_DIST.classify_max(dist), R.HBOND_ANGLE.classify_min(ang))
                if conf is None:
                    continue
                inter = Interaction(
                    interaction_type="hbond", subtype="hbond_donor",
                    distance=round(dist, 2), angle=(round(ang, 1) if ang == ang else None),
                    direction="ligand_donor", confidence=conf,
                    **_lig_fields(lf, names, donor.heavy_idx), **_rec_fields(rf, ridx),
                )
                col.add(("hbond", "ligand_donor", donor.heavy_idx, ridx), inter, dist)
                note_typed(ridx)
    # receptor donor -> ligand acceptor
    if rf.donors and lf.acceptors:
        lig_xyz = lf.coords[lf.acceptors]
        d_xyz = np.array([d.heavy_xyz for d in rf.donors])
        dm = pairwise_distances(d_xyz, lig_xyz, use_gpu)
        for i, donor in enumerate(rf.donors):
            for j, laidx in enumerate(lf.acceptors):
                dist = float(dm[i, j])
                if dist > R.HBOND_DIST.ext_max:
                    continue
                ang = _best_hbond_angle(donor.h_xyz, donor.heavy_xyz, lf.coords[laidx])
                conf = R.combine(R.HBOND_DIST.classify_max(dist), R.HBOND_ANGLE.classify_min(ang))
                if conf is None:
                    continue
                inter = Interaction(
                    interaction_type="hbond", subtype="hbond_acceptor",
                    distance=round(dist, 2), angle=(round(ang, 1) if ang == ang else None),
                    direction="ligand_acceptor", confidence=conf,
                    **_lig_fields(lf, names, laidx), **_rec_fields(rf, donor.heavy_idx),
                )
                col.add(("hbond", "ligand_acceptor", laidx, donor.heavy_idx), inter, dist)
                note_typed(donor.heavy_idx)


def _best_hbond_angle(h_list, heavy, acceptor) -> float:
    if not h_list:
        return float("nan")
    return max(angle_deg(heavy, h, acceptor) for h in h_list)


def _rec_heavy_neighbor(rf, ridx: int) -> np.ndarray | None:
    """Coords of the closest covalently-bonded heavy atom to receptor atom
    ``ridx`` (nearest heavy atom within 1.9 A), used for acceptor-side angles."""
    a = rf.coords[ridx]
    best = None
    best_d = 1.9
    for j, m in enumerate(rf.meta):
        if j == ridx or m["element"].upper() == "H":
            continue
        d = float(np.linalg.norm(rf.coords[j] - a))
        if 0.4 < d < best_d:
            best_d = d
            best = rf.coords[j]
    return best


def _halogen_bonds(lf, rf, names, col, note_typed):
    for hal in lf.halogens:
        for ridx in rf.acceptors:
            dist = float(np.linalg.norm(hal.x_xyz - rf.coords[ridx]))
            if dist > R.HALOGEN_DIST.ext_max:
                continue
            ang = angle_deg(hal.c_xyz, hal.x_xyz, rf.coords[ridx])
            bands = [R.HALOGEN_DIST.classify_max(dist), R.HALOGEN_ANGLE.classify_min(ang)]
            # acceptor-side directionality (X..A-Y), if a neighbour is found
            neigh = _rec_heavy_neighbor(rf, ridx)
            if neigh is not None:
                acc_ang = angle_deg(hal.x_xyz, rf.coords[ridx], neigh)
                bands.append(R.HALOGEN_ACCEPTOR_ANGLE.classify_min(acc_ang))
            conf = R.combine(*bands)
            if conf is None:
                continue
            inter = Interaction(
                interaction_type="halogen_bond", subtype="halogen_donor",
                distance=round(dist, 2), angle=round(ang, 1), direction="ligand_donor",
                confidence=conf, **_lig_fields(lf, names, hal.x_idx), **_rec_fields(rf, ridx),
            )
            col.add(("halogen", hal.x_idx, ridx), inter, dist)
            note_typed(ridx)


def _ionic(lf, rf, names, col, note_typed):
    for lig_group, subtype in ((lf.cations, "ionic_pos"), (lf.anions, "ionic_neg")):
        rec_groups = rf.anions if subtype == "ionic_pos" else rf.cations
        for lg in lig_group:
            for rg in rec_groups:
                dist = float(np.linalg.norm(lg.center - rg.center))
                band = R.IONIC_DIST.classify_max(dist)
                conf = R.combine(band)
                if conf is None:
                    continue
                ridx = rg.atom_idxs[0]
                inter = Interaction(
                    interaction_type="ionic", subtype=subtype, distance=round(dist, 2),
                    confidence=conf, **_lig_fields(lf, names, lg.atom_idxs[0]),
                    **_rec_fields(rf, ridx),
                )
                col.add(("ionic", subtype, lg.atom_idxs[0], ridx), inter, dist)
                note_typed(ridx)


def _pi(lf, rf, names, col, note_typed):
    # pi-stacking: ligand ring vs receptor ring
    for lr in lf.aromatics:
        for rr in rf.aromatics:
            dist = float(np.linalg.norm(lr.centroid - rr.centroid))
            band = R.PISTACK_CENTROID.classify_max(dist)
            if band is None:
                continue
            # lateral offset: a ring must sit over the partner ring
            offset = min(
                ring_offset(rr.centroid, lr.centroid, lr.normal),
                ring_offset(lr.centroid, rr.centroid, rr.normal),
            )
            if offset > R.PISTACK_OFFSET_EXT:
                continue
            pangle = plane_angle_deg(lr.normal, rr.normal)
            if pangle <= R.PISTACK_FACE_ANGLE_MAX:
                subtype = "pi_face"
            elif pangle >= R.PISTACK_EDGE_ANGLE_MIN:
                subtype = "pi_edge"
            else:
                subtype = "pi_tilted"
            good = band == "core" and subtype != "pi_tilted" and offset <= R.PISTACK_OFFSET_CORE
            conf = "confident" if good else "uncertain"
            lig_atom = _closest_atom(lr.atom_idxs, lf.coords, rr.centroid)
            ridx = _closest_atom(rr.atom_idxs, rf.coords, lr.centroid)
            inter = Interaction(
                interaction_type="pi_stacking", subtype=subtype, distance=round(dist, 2),
                angle=round(pangle, 1), confidence=conf,
                **_lig_fields(lf, names, lig_atom), **_rec_fields(rf, ridx),
            )
            col.add(("pistack", tuple(sorted(lr.atom_idxs)), tuple(sorted(rr.atom_idxs))), inter, dist)
            note_typed(ridx)
    # pi-cation: ligand ring vs receptor cation
    for lr in lf.aromatics:
        for rg in rf.cations:
            dist = float(np.linalg.norm(lr.centroid - rg.center))
            band = R.PICATION_DIST.classify_max(dist)
            if band is None:
                continue
            offset = ring_offset(rg.center, lr.centroid, lr.normal)
            if offset > R.PICATION_OFFSET_EXT:
                continue
            conf = "confident" if (band == "core" and offset <= R.PICATION_OFFSET_CORE) else "uncertain"
            ridx = rg.atom_idxs[0]
            lig_atom = _closest_atom(lr.atom_idxs, lf.coords, rg.center)
            inter = Interaction(
                interaction_type="pi_cation", subtype=None, ligand_role="pi",
                distance=round(dist, 2), confidence=conf,
                **_lig_fields(lf, names, lig_atom), **_rec_fields(rf, ridx),
            )
            col.add(("picat", tuple(sorted(lr.atom_idxs)), ridx), inter, dist)
            note_typed(ridx)
    # cation-pi: ligand cation vs receptor ring
    for lg in lf.cations:
        for rr in rf.aromatics:
            dist = float(np.linalg.norm(lg.center - rr.centroid))
            band = R.PICATION_DIST.classify_max(dist)
            if band is None:
                continue
            offset = ring_offset(lg.center, rr.centroid, rr.normal)
            if offset > R.PICATION_OFFSET_EXT:
                continue
            conf = "confident" if (band == "core" and offset <= R.PICATION_OFFSET_CORE) else "uncertain"
            ridx = _closest_atom(rr.atom_idxs, rf.coords, lg.center)
            inter = Interaction(
                interaction_type="cation_pi", subtype=None, ligand_role="cation",
                distance=round(dist, 2), confidence=conf,
                **_lig_fields(lf, names, lg.atom_idxs[0]), **_rec_fields(rf, ridx),
            )
            col.add(("catpi", lg.atom_idxs[0], tuple(sorted(rr.atom_idxs))), inter, dist)
            note_typed(ridx)


def _chalcogen_bond(lf, rf, names, col, note_typed):
    """Ligand C-S/Se sigma-hole to a receptor acceptor (halogen-bond analogue)."""
    for ch in lf.chalcogens:
        for ridx in rf.acceptors:
            dist = float(np.linalg.norm(ch.x_xyz - rf.coords[ridx]))
            if dist > R.CHALCOGEN_DIST.ext_max:
                continue
            ang = angle_deg(ch.c_xyz, ch.x_xyz, rf.coords[ridx])
            conf = R.combine(R.CHALCOGEN_DIST.classify_max(dist), R.CHALCOGEN_ANGLE.classify_min(ang))
            if conf is None:
                continue
            inter = Interaction(
                interaction_type="chalcogen_bond", subtype="chalcogen_donor",
                distance=round(dist, 2), angle=round(ang, 1), direction="ligand_donor",
                confidence=conf, **_lig_fields(lf, names, ch.x_idx), **_rec_fields(rf, ridx),
            )
            col.add(("chalcogen", ch.x_idx, ridx), inter, dist)
            note_typed(ridx)


def _sulfur_arene(lf, rf, names, col, note_typed):
    """Sulfur/selenium over an aromatic ring (both directions)."""
    # ligand chalcogen -> receptor ring
    for ch in lf.chalcogens:
        for rr in rf.aromatics:
            dist = float(np.linalg.norm(ch.x_xyz - rr.centroid))
            band = R.SARENE_DIST.classify_max(dist)
            if band is None:
                continue
            if ring_offset(ch.x_xyz, rr.centroid, rr.normal) > R.SARENE_OFFSET + 1.0:
                continue
            conf = "confident" if band == "core" else "uncertain"
            ridx = _closest_atom(rr.atom_idxs, rf.coords, ch.x_xyz)
            inter = Interaction(
                interaction_type="chalcogen_pi", subtype=None, ligand_role="chalcogen",
                distance=round(dist, 2), confidence=conf,
                **_lig_fields(lf, names, ch.x_idx), **_rec_fields(rf, ridx),
            )
            col.add(("sarene_l", ch.x_idx, tuple(sorted(rr.atom_idxs))), inter, dist)
            note_typed(ridx)
    # receptor sulfur (Met/Cys) -> ligand ring
    for sidx in rf.sulfurs:
        for lr in lf.aromatics:
            dist = float(np.linalg.norm(rf.coords[sidx] - lr.centroid))
            band = R.SARENE_DIST.classify_max(dist)
            if band is None:
                continue
            if ring_offset(rf.coords[sidx], lr.centroid, lr.normal) > R.SARENE_OFFSET + 1.0:
                continue
            conf = "confident" if band == "core" else "uncertain"
            lig_atom = _closest_atom(lr.atom_idxs, lf.coords, rf.coords[sidx])
            inter = Interaction(
                interaction_type="chalcogen_pi", subtype=None, ligand_role="pi",
                distance=round(dist, 2), confidence=conf,
                **_lig_fields(lf, names, lig_atom), **_rec_fields(rf, sidx),
            )
            col.add(("sarene_r", sidx, tuple(sorted(lr.atom_idxs))), inter, dist)
            note_typed(sidx)


def _amide_pi(lf, rf, names, col, note_typed):
    """Ligand planar amide stacked over a receptor aromatic ring."""
    for am in lf.amides:
        for rr in rf.aromatics:
            dist = float(np.linalg.norm(am.centroid - rr.centroid))
            band = R.AMIDE_PI_DIST.classify_max(dist)
            if band is None:
                continue
            pangle = plane_angle_deg(am.normal, rr.normal)
            if pangle > R.AMIDE_PI_ANGLE_MAX:
                continue
            offset = ring_offset(am.centroid, rr.centroid, rr.normal)
            if offset > R.AMIDE_PI_OFFSET + 1.0:
                continue
            conf = "confident" if (band == "core" and offset <= R.AMIDE_PI_OFFSET) else "uncertain"
            lig_atom = am.atom_idxs[0]
            ridx = _closest_atom(rr.atom_idxs, rf.coords, am.centroid)
            inter = Interaction(
                interaction_type="amide_pi", subtype=None, distance=round(dist, 2),
                angle=round(pangle, 1), confidence=conf,
                **_lig_fields(lf, names, lig_atom), **_rec_fields(rf, ridx),
            )
            col.add(("amidepi", tuple(sorted(am.atom_idxs)), tuple(sorted(rr.atom_idxs))), inter, dist)
            note_typed(ridx)


def _chpi(lf, rf, names, col, note_typed):
    """C-H over an aromatic ring (both directions), weak."""
    lig_c = sorted(set(lf.hydrophobes))
    # ligand C-H -> receptor ring
    for rr in rf.aromatics:
        for laidx in lig_c:
            dist = float(np.linalg.norm(lf.coords[laidx] - rr.centroid))
            band = R.CHPI_DIST.classify_max(dist)
            if band is None or ring_offset(lf.coords[laidx], rr.centroid, rr.normal) > R.CHPI_OFFSET:
                continue
            ridx = _closest_atom(rr.atom_idxs, rf.coords, lf.coords[laidx])
            inter = Interaction(
                interaction_type="ch_pi", subtype=None, ligand_role="ch",
                distance=round(dist, 2), confidence=("confident" if band == "core" else "uncertain"),
                **_lig_fields(lf, names, laidx), **_rec_fields(rf, ridx),
            )
            col.add(("chpi_l", laidx, tuple(sorted(rr.atom_idxs))), inter, dist)
            note_typed(ridx)
    # receptor C-H -> ligand ring
    rec_c = sorted(set(rf.hydrophobes))
    for lr in lf.aromatics:
        for ridx in rec_c:
            dist = float(np.linalg.norm(rf.coords[ridx] - lr.centroid))
            band = R.CHPI_DIST.classify_max(dist)
            if band is None or ring_offset(rf.coords[ridx], lr.centroid, lr.normal) > R.CHPI_OFFSET:
                continue
            lig_atom = _closest_atom(lr.atom_idxs, lf.coords, rf.coords[ridx])
            inter = Interaction(
                interaction_type="ch_pi", subtype=None, ligand_role="pi",
                distance=round(dist, 2), confidence=("confident" if band == "core" else "uncertain"),
                **_lig_fields(lf, names, lig_atom), **_rec_fields(rf, ridx),
            )
            col.add(("chpi_r", ridx, tuple(sorted(lr.atom_idxs))), inter, dist)
            note_typed(ridx)


def _metal_fields(site: F.MetalSite) -> dict:
    m = site.meta
    return {
        "receptor_chain": m.get("chain", ""),
        "receptor_resname": m.get("resname", ""),
        "receptor_resnum": m.get("resnum"),
        "receptor_reskind": "ion",
        "receptor_atom_name": m.get("name"),
        "receptor_moiety": "ion",
    }


def _water_fields(site: F.WaterSite) -> dict:
    m = site.meta
    return {
        "receptor_chain": m.get("chain", ""),
        "receptor_resname": m.get("resname", "HOH"),
        "receptor_resnum": m.get("resnum"),
        "receptor_reskind": "water",
        "receptor_atom_name": m.get("name", "O"),
        "receptor_moiety": "water",
    }


def _metal(lf, rf, names, col, note_typed_key):
    if not rf.metals:
        return
    coordinating = list(lf.acceptors) + [a for g in lf.anions for a in g.atom_idxs]
    for si, site in enumerate(rf.metals):
        for laidx in set(coordinating):
            dist = float(np.linalg.norm(site.xyz - lf.coords[laidx]))
            band = R.METAL_DIST.classify_max(dist)
            conf = R.combine(band)
            if conf is None:
                continue
            inter = Interaction(
                interaction_type="metal", subtype="metal_coordination",
                distance=round(dist, 2), direction="ligand_donor", confidence=conf,
                **_lig_fields(lf, names, laidx), **_metal_fields(site),
            )
            col.add(("metal", laidx, si), inter, dist)
            note_typed_key(site.meta.get("chain", ""), site.meta.get("resnum"))


def _water_bridge(lf, rf, names, col, note_typed_key, use_gpu):
    if not rf.waters:
        return
    water_xyz = np.array([w.xyz for w in rf.waters])
    lig_polar = sorted(set(list(lf.acceptors) + [d.heavy_idx for d in lf.donors]))
    if not lig_polar:
        return
    dm_lig = pairwise_distances(water_xyz, lf.coords[lig_polar], use_gpu)
    rec_polar = [i for i in sorted(set(list(rf.acceptors) + [d.heavy_idx for d in rf.donors]))
                 if rf.meta[i]["reskind"] != "water"]
    if not rec_polar:
        return
    dm_rec = pairwise_distances(water_xyz, rf.coords[rec_polar], use_gpu)
    # Collect candidate bridges, dedupe by (ligand atom, bridged receptor
    # residue) keeping the shortest, then keep only the few shortest overall
    # -- crystallographic waters are numerous and mostly redundant.
    candidates: dict[tuple, tuple] = {}
    for wi, site in enumerate(rf.waters):
        lig_hits = [(lig_polar[j], dm_lig[wi, j]) for j in range(len(lig_polar))
                    if dm_lig[wi, j] <= R.WATER_HBOND_DIST.ext_max]
        rec_hits = [(rec_polar[j], dm_rec[wi, j]) for j in range(len(rec_polar))
                    if dm_rec[wi, j] <= R.WATER_HBOND_DIST.ext_max]
        if not lig_hits or not rec_hits:
            continue
        laidx, dl = min(lig_hits, key=lambda x: x[1])
        ridx, _ = min(rec_hits, key=lambda x: x[1])
        bridged = (rf.meta[ridx]["chain"], rf.meta[ridx]["resnum"])
        key = (laidx, bridged)
        if key not in candidates or dl < candidates[key][0]:
            candidates[key] = (float(dl), wi, site)

    kept = sorted(candidates.items(), key=lambda kv: kv[1][0])[:R.MAX_WATER_BRIDGES]
    for (laidx, _bridged), (dl, wi, site) in kept:
        conf = "confident" if (dl <= R.WATER_HBOND_DIST.core_max) else "uncertain"
        inter = Interaction(
            interaction_type="water_bridge", subtype=None, ligand_role="water",
            distance=round(float(dl), 2), confidence=conf,
            **_lig_fields(lf, names, laidx), **_water_fields(site),
        )
        col.add(("water", laidx, wi), inter, float(dl))
        note_typed_key(site.meta.get("chain", ""), site.meta.get("resnum"))


def _hydrophobic(lf, rf, names, col, note_typed, use_gpu):
    if not lf.hydrophobes or not rf.hydrophobes:
        return
    lig_h = sorted(set(lf.hydrophobes))
    rec_h = sorted(set(rf.hydrophobes))
    dm = pairwise_distances(lf.coords[lig_h], rf.coords[rec_h], use_gpu)
    for i, laidx in enumerate(lig_h):
        for j, ridx in enumerate(rec_h):
            dist = float(dm[i, j])
            band = R.HYDROPHOBIC_DIST.classify_max(dist)
            conf = R.combine(band)
            if conf is None:
                continue
            m = rf.meta[ridx]
            inter = Interaction(
                interaction_type="hydrophobic", subtype=None, distance=round(dist, 2),
                confidence=conf, **_lig_fields(lf, names, laidx), **_rec_fields(rf, ridx),
            )
            # dedupe to closest per (ligand atom, residue)
            col.add(("hydrophobic", laidx, m["chain"], m["resnum"]), inter, dist)
            note_typed(ridx)


_VDW_RADII = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8,
              "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98}


def _vdw(lf, rf, names, col, residues_with_typed, use_gpu):
    # heavy-atom vdW contacts, but only representative closest per residue
    # that has NO typed interaction (keeps output typed-interaction-centric).
    lig_heavy = [i for i, e in enumerate(lf.elements) if e.upper() != "H"]
    rec_heavy = [i for i, m in enumerate(rf.meta) if m["element"].upper() != "H"]
    if not lig_heavy or not rec_heavy:
        return
    dm = pairwise_distances(lf.coords[lig_heavy], rf.coords[rec_heavy], use_gpu)
    best_per_res: dict[tuple, tuple] = {}
    for i, laidx in enumerate(lig_heavy):
        le = lf.elements[laidx].upper()
        lr = _VDW_RADII.get(le, 1.7)
        for j, ridx in enumerate(rec_heavy):
            m = rf.meta[ridx]
            reskey = (m["chain"], m["resnum"])
            if reskey in residues_with_typed:
                continue
            dist = float(dm[i, j])
            rr = _VDW_RADII.get(m["element"].upper(), 1.7)
            if dist <= lr + rr + R.VDW_TOLERANCE:
                cur = best_per_res.get(reskey)
                if cur is None or dist < cur[0]:
                    best_per_res[reskey] = (dist, laidx, ridx)
    for reskey, (dist, laidx, ridx) in best_per_res.items():
        inter = Interaction(
            interaction_type="vdw_contact", subtype=None, distance=round(dist, 2),
            confidence="uncertain", **_lig_fields(lf, names, laidx), **_rec_fields(rf, ridx),
        )
        col.add(("vdw",) + reskey, inter, dist)


def _closest_atom(atom_idxs: list[int], coords: np.ndarray, point: np.ndarray) -> int:
    best = atom_idxs[0]
    best_d = float("inf")
    for i in atom_idxs:
        d = float(np.linalg.norm(coords[i] - point))
        if d < best_d:
            best_d = d
            best = i
    return best


__all__ = ["detect_interactions"]
