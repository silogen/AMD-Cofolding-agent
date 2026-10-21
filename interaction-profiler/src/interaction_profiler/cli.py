# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``interaction-profile`` CLI.

Batch mode (primary)::

    interaction-profile examples/example_batch.json --ph 7.4 --out batch.result.json --report

Single-shot convenience::

    interaction-profile --structure examples/1OIU_boltz.pdb --ligand-resname LIG \
        --smiles "c1cc..." --name 1OIU --out r.json --report

One batch in -> one JSON out.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from interaction_profiler.io.batch_input import MoleculeInput
from interaction_profiler.schemas import BatchResult

app = typer.Typer(
    name="interaction-profile",
    help="Batch medchem taxonomy + structural interaction profiling. One batch in, one JSON out.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def main(
    batch: Path | None = typer.Argument(None, help="Batch file (.json/.jsonl/.csv)."),
    structure: Path | None = typer.Option(None, "--structure", "-s", help="Single structure (PDB/mmCIF)."),
    smiles: str | None = typer.Option(None, "--smiles", help="Ligand SMILES (single-shot / template)."),
    name: str | None = typer.Option(None, "--name", "-n", help="Name for single-shot mode."),
    ligand_resname: str | None = typer.Option(None, "--ligand-resname", "-r"),
    ligand_chain: str | None = typer.Option(None, "--ligand-chain", "-c"),
    ligand_keyword: str | None = typer.Option(
        None, "--ligand-keyword", "-k", help="Case-insensitive substring match on the ligand residue id."
    ),
    peptide_ligand_chain: str | None = typer.Option(
        None, "--peptide-ligand-chain",
        help="Treat this whole chain as the ligand (e.g. a cyclic-peptide macrocycle).",
    ),
    source: str | None = typer.Option(None, "--source", help="crystal | cofold."),
    ph: float = typer.Option(7.4, "--ph", help="pH for ligand + receptor protonation."),
    radius: float = typer.Option(8.0, "--radius", help="Pocket crop radius (A)."),
    max_interactions: int | None = typer.Option(
        None, "--max-interactions", "-m",
        help="Max interactions kept per input, ranked by energy then distance. "
             "Default: adaptive (~1.2x ligand heavy atoms). 0 = no cap.",
    ),
    keep_uncertain: bool = typer.Option(
        False, "--keep-uncertain",
        help="Keep uncertain interactions too. Default: only confident are kept "
             "(uncertain removed AFTER the cap).",
    ),
    all_ligands: bool = typer.Option(
        False, "--all-ligands", help="Profile every qualifying ligand in each structure."
    ),
    strict_ligand_match: bool = typer.Option(
        False, "--strict-ligand-match",
        help="Error a structure+SMILES molecule when the prepared ligand's InChIKey "
             "does not match the SMILES template.",
    ),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel worker processes for the batch."),
    gpu: bool = typer.Option(False, "--gpu", help="Use Torch GPU for distance matrices if available."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write batch JSON here. Default stdout."),
    csv_out: Path | None = typer.Option(
        None, "--csv", help="Also write a flattened triples CSV (one row per triple)."
    ),
    readable: Path | None = typer.Option(
        None, "--readable", "-R", help="Also write a human-readable report to this file."
    ),
    report: bool = typer.Option(False, "--report", help="Print the human-readable summary to stderr."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    _configure_logging(verbose)
    from interaction_profiler.core import profile_batch, profile_molecule_all
    from interaction_profiler.schemas import BatchProvenance

    drop_uncertain = not keep_uncertain

    if batch is not None:
        result = profile_batch(batch, ph=ph, radius=radius, use_gpu=gpu,
                               max_interactions=max_interactions, drop_uncertain=drop_uncertain,
                               all_ligands=all_ligands, strict_ligand_match=strict_ligand_match,
                               workers=workers, out_path=out)
    elif structure is not None or smiles is not None:
        # Fail loudly if a structure was requested but the file is absent,
        # instead of silently falling back to SMILES-only taxonomy.
        if structure is not None and not structure.exists():
            typer.echo(
                f"Structure file not found: {structure}\n"
                f"(cwd: {Path.cwd()}) — check the path, or cd into the repo so "
                "relative paths like examples/1OIU_boltz.pdb resolve.",
                err=True,
            )
            raise typer.Exit(code=2)
        inp = MoleculeInput(
            name=(name or (structure.stem if structure else "molecule")),
            smiles=smiles,
            structure=str(structure) if structure else None,
            ligand_resname=ligand_resname,
            ligand_chain=ligand_chain,
            ligand_keyword=ligand_keyword,
            peptide_ligand_chain=peptide_ligand_chain,
            source=source,
        )
        mols = profile_molecule_all(inp, ph_default=ph, radius=radius, use_gpu=gpu,
                                    max_interactions=max_interactions, drop_uncertain=drop_uncertain,
                                    all_ligands=all_ligands, strict_ligand_match=strict_ligand_match)
        result = BatchResult(
            batch_id=inp.name, n_molecules=len(mols),
            provenance=BatchProvenance(interaction_radius=radius,
                                       max_interactions=max_interactions,
                                       drop_uncertain=drop_uncertain, gpu=gpu),
            molecules=mols,
        )
        js = result.model_dump_json(indent=2)
        if out:
            out.write_text(js)
        else:
            sys.stdout.write(js + "\n")
    else:
        typer.echo("Provide a batch file, or --structure/--smiles.", err=True)
        raise typer.Exit(code=2)

    if csv_out:
        _write_csv(result, csv_out)

    text = render_report(result)
    if readable:
        readable.write_text(text)
    if report:
        sys.stderr.write(text)


def _write_csv(result: BatchResult, path: Path) -> None:
    """Write flattened triples to CSV (one row per triple)."""
    import csv

    rows = result.triple_rows()
    cols = ["name", "ligand_smiles", "receptor", "ph", "source", "fragment", "residue",
            "resname", "resnum", "chain", "moiety", "interaction_type", "subtype",
            "direction", "count", "distance_min", "distance_max", "angle_min", "angle_max",
            "n_confident", "n_uncertain"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in cols})


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )


def render_report(result: BatchResult) -> str:
    """Render a human-readable text report of a batch result."""
    lines: list[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    for mol in result.molecules:
        md = mol.metadata
        w()
        com = ",".join(f"{x}" for x in (md.ligand_com or [])) if md.ligand_com else "?"
        if md.chain_types:
            chains_str = ",".join(f"{c}:{md.chain_types.get(c, 'other')}" for c in md.chains)
        else:
            chains_str = ",".join(md.chains)
        dropped = f" (+{md.n_dropped} dropped)" if md.n_dropped else ""
        w(f"# target={md.name} · ligand={md.ligand_smiles} · com={com} · "
          f"receptor={md.receptor} · chains={chains_str} · pH={md.ph} · "
          f"source={md.source} · {md.n_interactions} interactions{dropped} · {md.n_triples} triples "
          f"({md.n_confident} confident · {md.n_uncertain} uncertain)")
        if mol.error:
            w(f"  ERROR: {mol.error}")
            continue
        if not md.has_structure:
            w(f"  taxonomy: {len(mol.composition.fragments)} fragments, "
              f"coverage={mol.composition.coverage:.0%}")
            for f in mol.composition.fragments:
                if f.level == "functional_group":
                    w(f"    {f.canonical_name:<28} atoms={f.atom_indices}  smiles={f.smiles}")
            continue
        for t in mol.triples:
            d = ""
            if t.distance_min is not None:
                d = f"d={t.distance_min}A" if t.distance_min == t.distance_max else f"d={t.distance_min}-{t.distance_max}A"
            ang = ""
            if t.angle_min is not None:
                ang = f" · θ={t.angle_min}°" if t.angle_min == t.angle_max else f" · θ={t.angle_min}-{t.angle_max}°"
            label = t.subtype or t.interaction_type
            mult = f"×{t.count}" if t.count > 1 else ""
            conf = "confident" if t.n_uncertain == 0 else (
                "uncertain" if t.n_confident == 0 else f"{t.n_confident} confident,{t.n_uncertain} uncertain")
            dirs = f" · dir={t.direction}" if t.direction else ""
            moiety = f" · {t.moiety}" if t.moiety else ""
            w(f"{t.residue_label}{moiety} · {label}{mult} · {d}{ang}{dirs} · frag={t.fragment} · {conf}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    app()
