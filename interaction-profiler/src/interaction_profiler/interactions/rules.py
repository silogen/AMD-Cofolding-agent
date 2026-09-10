"""Geometric cutoffs and confidence bands for each interaction class.

Cutoffs follow common structural-interaction conventions (PLIP / ProLIF
defaults). Each rule has a ``core`` band (confident) and an ``extended``
band (uncertain) for distance and, where relevant, angle / offset.

Per-bond requirements (distance is heavy-atom..heavy-atom unless noted):

    interaction     atoms (ligand / receptor)              distance(core/ext)  angle(core/ext)         extra
    ------------    -------------------------------------  ------------------  ----------------------  -------------------------
    covalent        C/S/B/P  /  nucleophile (Cys SG,...)   <=2.1                -                      LINK record or reactive atom
    hbond           donor(D-H)+acceptor either side        D..A <=3.5/4.0       D-H..A >=120/90        needs an explicit H
    halogen_bond     C-X (X=Cl/Br/I)  /  O/N/S acceptor    X..A <=3.5/4.0       C-X..A >=140/120       -
    ionic           charged group  /  charged group        centre <=4.5/5.5     -                      opposite formal charge
    metal           O/N/S (acceptor/anion)  /  metal ion   <=2.8/3.2            -                      -
    pi_stacking     aromatic ring  /  aromatic ring        centroid <=5.5/6.0   face<=30 / edge>=50     lateral offset <=2.0/3.0
    pi_cation       aromatic ring  /  cation               centroid <=6.0/6.5   -                      offset <=2.0/2.5
    cation_pi       cation  /  aromatic ring               centroid <=6.0/6.5   -                      offset <=2.0/2.5
    water_bridge    donor/acceptor  /  bridged via HOH     each leg <=3.5/4.0   -                      dedup, <=3 per ligand
    hydrophobic     C  /  C                                 <=4.5/5.0            -                      -
    vdw_contact     heavy  /  heavy                          <= sum vdW + 0.5    -                      only if no typed contact
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Band:
    core_max: float
    ext_max: float

    def classify_max(self, value: float) -> str | None:
        """For upper-bounded quantities (distances)."""
        if value <= self.core_max:
            return "core"
        if value <= self.ext_max:
            return "ext"
        return None


@dataclass(frozen=True)
class AngleBand:
    core_min: float
    ext_min: float

    def classify_min(self, value: float) -> str | None:
        """For lower-bounded quantities (H-bond / halogen angles)."""
        if value != value:  # nan -> treat as passing but uncertain
            return "ext"
        if value >= self.core_min:
            return "core"
        if value >= self.ext_min:
            return "ext"
        return None


# Distance bands (angstrom). Core = confident, extended = uncertain.
# Cutoffs follow common structural-interaction conventions (PLIP / ProLIF).
# A donor-acceptor H-bond is accepted to 3.5 A (core) with the D-H..A angle
# >= 120 deg counted as confident -- the textbook lower bound for a linear
# H-bond -- so ordinary 3.3 A / 130 deg bonds are not demoted to uncertain.
HBOND_DIST = Band(core_max=3.5, ext_max=4.0)
HBOND_ANGLE = AngleBand(core_min=120.0, ext_min=90.0)  # D-H..A

HALOGEN_DIST = Band(core_max=3.5, ext_max=4.0)
HALOGEN_ANGLE = AngleBand(core_min=140.0, ext_min=120.0)  # C-X..A (sigma hole)
# acceptor-side directionality: X..A-Y angle should point at a lone pair
HALOGEN_ACCEPTOR_ANGLE = AngleBand(core_min=90.0, ext_min=70.0)

IONIC_DIST = Band(core_max=4.5, ext_max=5.5)
# Hydrophobic C..C contacts: confident to 4.5 A (van der Waals contact of two
# carbons ~3.5-4.0 A + tolerance), soft out to 5.0 A.
HYDROPHOBIC_DIST = Band(core_max=4.5, ext_max=5.0)
# pi-stacking: ring-centroid distance, inter-plane angle (face <=30, edge >=50),
# and lateral offset of one centroid over the other ring (a ring must sit over
# the partner, not merely be nearby). Offset > ext -> not stacking.
PISTACK_CENTROID = Band(core_max=5.5, ext_max=6.0)
PISTACK_FACE_ANGLE_MAX = 30.0
PISTACK_EDGE_ANGLE_MIN = 50.0
PISTACK_OFFSET_CORE = 2.0
PISTACK_OFFSET_EXT = 3.0
# pi-cation: cation-to-centroid distance + the cation must project near the
# ring centre (offset), else it is not a cation-pi geometry.
PICATION_DIST = Band(core_max=6.0, ext_max=6.5)
PICATION_OFFSET_CORE = 2.0
PICATION_OFFSET_EXT = 2.5
# CH-pi: a C-H carbon over an aromatic ring (weak).
CHPI_DIST = Band(core_max=4.5, ext_max=5.0)
CHPI_OFFSET = 2.0
# sulfur/selenium-arene: chalcogen over an aromatic ring.
SARENE_DIST = Band(core_max=5.5, ext_max=6.0)
SARENE_OFFSET = 2.0
# chalcogen bond: C-S/Se sigma-hole to an acceptor (halogen-bond analogue).
CHALCOGEN_DIST = Band(core_max=3.6, ext_max=4.0)
CHALCOGEN_ANGLE = AngleBand(core_min=140.0, ext_min=120.0)
# amide-pi: planar amide stacked (parallel) over an aromatic ring.
AMIDE_PI_DIST = Band(core_max=4.5, ext_max=5.5)
AMIDE_PI_ANGLE_MAX = 30.0
AMIDE_PI_OFFSET = 2.0
METAL_DIST = Band(core_max=2.8, ext_max=3.2)
WATER_HBOND_DIST = Band(core_max=3.5, ext_max=4.0)
# Crystallographic waters are plentiful; keep only the few shortest bridges.
MAX_WATER_BRIDGES = 3
VDW_TOLERANCE = 0.5  # added to sum of vdW radii


def combine(*bands: str | None) -> str | None:
    """Combine per-criterion band labels into a confidence verdict.

    Returns ``None`` if any criterion is out of range, else 'confident'
    when all are core, otherwise 'uncertain'.
    """
    labels = list(bands)
    if any(b is None for b in labels):
        return None
    return "confident" if all(b == "core" for b in labels) else "uncertain"


__all__ = [
    "Band",
    "AngleBand",
    "HBOND_DIST",
    "HBOND_ANGLE",
    "HALOGEN_DIST",
    "HALOGEN_ANGLE",
    "HALOGEN_ACCEPTOR_ANGLE",
    "IONIC_DIST",
    "HYDROPHOBIC_DIST",
    "PISTACK_CENTROID",
    "PISTACK_FACE_ANGLE_MAX",
    "PISTACK_EDGE_ANGLE_MIN",
    "PISTACK_OFFSET_CORE",
    "PISTACK_OFFSET_EXT",
    "PICATION_DIST",
    "PICATION_OFFSET_CORE",
    "PICATION_OFFSET_EXT",
    "CHPI_DIST",
    "CHPI_OFFSET",
    "SARENE_DIST",
    "SARENE_OFFSET",
    "CHALCOGEN_DIST",
    "CHALCOGEN_ANGLE",
    "AMIDE_PI_DIST",
    "AMIDE_PI_ANGLE_MAX",
    "AMIDE_PI_OFFSET",
    "METAL_DIST",
    "WATER_HBOND_DIST",
    "MAX_WATER_BRIDGES",
    "VDW_TOLERANCE",
    "combine",
]
