"""Structure-conditioning helpers for the Boltz2 service.

Isolated from FastAPI / Boltz imports so these functions are independently
testable. They validate input templates, map chain IDs to Boltz template IDs,
and construct the template section of a Boltz input YAML file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_SHARED_ROOT = Path("/shared")
_DEFAULT_SUFFIX = "1"


def validate_template_input(
    structure_path: str,
    expected_sha256: str | None,
    chain_ids: list[str],
) -> tuple[Path, str, list[str]]:
    """Validate a prepared PDB/mmCIF template under the shared volume."""
    p = Path(structure_path)
    if not p.is_absolute():
        p = _SHARED_ROOT / p

    try:
        p = p.resolve(strict=False)
    except OSError:
        pass

    try:
        p.relative_to(_SHARED_ROOT.resolve())
    except ValueError:
        raise ValueError(
            f"structure_path {structure_path!r} is outside the shared directory. "
            f"All prepared structures must be under {_SHARED_ROOT}."
        )

    if not p.is_file():
        raise FileNotFoundError(
            f"Structure file not found: {p}. "
            "Use list_shared_assets to discover available prepared structures."
        )

    actual_sha256 = _compute_sha256(p)
    if (
        expected_sha256 is not None
        and actual_sha256.lower() != expected_sha256.strip().lower()
    ):
        raise ValueError(
            f"SHA-256 mismatch for {p.name}: "
            f"expected {expected_sha256!r}, got {actual_sha256!r}. "
            "The file may have been modified since it was prepared."
        )

    suffix = p.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        chains_in_file = _parse_mmcif_chains(p)
    else:
        chains_in_file = _parse_pdb_chains(p)

    missing = [chain for chain in chain_ids if chain not in chains_in_file]
    if missing:
        raise ValueError(
            f"Chain(s) {missing!r} not found in {p.name}. "
            f"Available chains: {sorted(chains_in_file)!r}."
        )

    return p, actual_sha256, sorted(chains_in_file)


def auto_assign_template_ids(chain_ids: list[str]) -> list[str]:
    """Map PDB chain IDs to Boltz's A1/B1 template-ID convention."""
    return [f"{chain}{_DEFAULT_SUFFIX}" for chain in chain_ids]


def infer_template_ids(
    structure_path: Path,
    chain_ids: list[str],
) -> list[str]:
    """Infer format-appropriate template IDs.

    Boltz preserves mmCIF chain names, while its PDB importer assigns
    subchains using the A1/B1 convention.
    """
    if structure_path.suffix.lower() in (".cif", ".mmcif"):
        return list(chain_ids)
    return auto_assign_template_ids(chain_ids)


def prepare_template_for_boltz(
    structure_path: Path,
    chain_ids: list[str],
    input_sha256: str,
) -> Path:
    """Return a Boltz-compatible template without modifying the input.

    Canonical mmCIF files already carry polymer sequence metadata. Prepared
    PDB files often omit SEQRES records, which makes Boltz's PDB converter
    create an empty polymer sequence and fail during parsing. For PDB inputs,
    create a deterministic mmCIF sidecar containing only the selected protein
    chains, normalized subchain IDs, and explicit polymer sequences.
    """
    if structure_path.suffix.lower() in (".cif", ".mmcif"):
        return structure_path

    try:
        import gemmi  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "PDB template normalization requires gemmi in the Boltz service"
        ) from exc

    selected = "-".join(chain_ids)
    output_dir = _SHARED_ROOT / "boltz_templates"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{structure_path.stem}-{input_sha256[:12]}-{selected}.cif"
    )
    if output_path.is_file():
        return output_path

    structure = gemmi.read_structure(str(structure_path))
    if len(structure) == 0:
        raise ValueError(f"No coordinate models found in {structure_path.name}")

    selected_set = set(chain_ids)
    for model in structure:
        for chain in list(model):
            if chain.name not in selected_set:
                model.remove_chain(chain.name)
        model.remove_hydrogens()

    present = {chain.name for chain in structure[0]}
    missing = [chain for chain in chain_ids if chain not in present]
    if missing:
        raise ValueError(
            f"Chain(s) {missing!r} disappeared while normalizing "
            f"{structure_path.name}"
        )

    structure.setup_entities()
    for chain in structure[0]:
        old_subchains = {residue.subchain for residue in chain}
        for model in structure:
            model_chain = model.find_chain(chain.name)
            if model_chain is not None:
                for residue in model_chain:
                    residue.subchain = chain.name
        for entity in structure.entities:
            if any(subchain in entity.subchains for subchain in old_subchains):
                entity.subchains = [chain.name]
                entity.full_sequence = [residue.name for residue in chain]

    structure.assign_label_seq_id()
    temporary_path = output_path.with_suffix(".tmp.cif")
    structure.make_mmcif_document().write_file(str(temporary_path))
    temporary_path.replace(output_path)
    return output_path


def build_template_dict(
    structure_path: Path,
    chain_ids: list[str],
    template_ids: list[str],
    force: bool = False,
    threshold: float = 2.0,
) -> dict[str, Any]:
    """Build one entry for the Boltz YAML ``templates`` list."""
    suffix = structure_path.suffix.lower()
    format_key = "cif" if suffix in (".cif", ".mmcif") else "pdb"
    entry: dict[str, Any] = {
        format_key: str(structure_path),
        "chain_id": chain_ids,
        "template_id": template_ids,
    }
    if force:
        entry["force"] = True
        entry["threshold"] = float(threshold)
    return entry


def _compute_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_pdb_chains(path: Path) -> set[str]:
    """Extract unique chain IDs from PDB ATOM/HETATM records."""
    chains: set[str] = set()
    with open(path, errors="replace") as handle:
        for line in handle:
            if line[:6] in ("ATOM  ", "HETATM"):
                chain = line[21]
                if chain.strip():
                    chains.add(chain)
    return chains


def _parse_mmcif_chains(path: Path) -> set[str]:
    """Extract ``_atom_site.auth_asym_id`` values from an mmCIF file."""
    chains: set[str] = set()
    with open(path, errors="replace") as handle:
        lines = handle.readlines()

    in_loop = False
    column_names: list[str] = []
    auth_asym_column: int | None = None
    reading_data = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if reading_data:
                break
            continue

        if stripped == "loop_":
            if reading_data:
                break
            in_loop = True
            column_names = []
            auth_asym_column = None
            reading_data = False
            continue

        if in_loop:
            if stripped.startswith("_atom_site."):
                column_name = stripped.split(".")[1].lower()
                column_names.append(column_name)
                if column_name == "auth_asym_id":
                    auth_asym_column = len(column_names) - 1
                continue

            if column_names and not stripped.startswith("_"):
                if auth_asym_column is not None:
                    reading_data = True
                    parts = stripped.split()
                    if len(parts) > auth_asym_column:
                        chain = parts[auth_asym_column].strip("'\"")
                        if chain and chain != ".":
                            chains.add(chain)
                    continue
                in_loop = False
                column_names = []
                continue

        elif reading_data:
            if (
                stripped.startswith("_")
                or stripped == "loop_"
                or stripped.startswith("#")
            ):
                break
            parts = stripped.split()
            if auth_asym_column is not None and len(parts) > auth_asym_column:
                chain = parts[auth_asym_column].strip("'\"")
                if chain and chain != ".":
                    chains.add(chain)

    if not chains:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("ATOM", "HETATM")):
                parts = stripped.split()
                if len(parts) >= 7:
                    chain = parts[6].strip("'\"")
                    if chain and chain.isalpha():
                        chains.add(chain)

    return chains
