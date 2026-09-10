# interaction-profiler

A batch tool that profiles small molecules and their complexes and emits,
per molecule, a JSON record that flattens cleanly to a table:

- **Hierarchical medchem taxonomy** — named functional-group / heterocycle /
  ring-system / scaffold fragments (AccFG + a curated SMARTS library) with
  atom-index mappings. Always produced, even without a structure.
- **Typed interactions** (when a structure is supplied) — H-bonds (with
  D-H..A angle and donor/acceptor direction), hydrophobic, pi-stacking
  (face/edge), pi-cation / cation-pi, ionic (salt bridge), halogen bonds
  (with acceptor-side geometry), metal coordination, bridging waters, CH–π,
  sulfur/selenium–arene, chalcogen–π, amide–π, and covalent bonds. Each
  contact is annotated with residue name / number / chain,
  sidechain-vs-backbone, ligand + receptor atom, distance, angle, fragment,
  and a confidence flag.
- **Triples** — grouped `(fragment, residue, interaction)` rows.

Receptors may be **protein or RNA**. The ligand of a triple is always a
small molecule. One batch in → one JSON out.

> **Role: out-of-band analysis service — NOT a REINVENT RL scorer.** It emits rich
> descriptors (medchem taxonomy + typed interaction geometry) for the agent to reason about,
> and deliberately does **not** implement the `POST /score` contract that `ExternalRESTScorer`
> expects (`{smiles, params}` → `{scores, metadata, artifacts}`). The agent consumes
> `/analyze` / `/profile` out-of-band (e.g. to see what a ligand is doing in the pocket); it is
> not wired into the RL scoring loop. A `/score` route + a defensible scalar objective could be
> added later if it should participate in scoring. Follows the boltz2/posebusters convention: a
> single FastAPI app in [../services/interaction-profiler/app.py](../services/interaction-profiler/app.py).

## Install

Normal use needs no manual install: dependencies are provided by the Docker image and the
service is brought up via `docker compose` (see [SERVICE.md](SERVICE.md)).

<details><summary>Local dev only (no Docker)</summary>

```bash
cd interaction-profiler
pip install -e ".[dev,service]"    # library + test/lint + FastAPI/uvicorn
```

Depends on RDKit, MDAnalysis, AccFG, dimorphite-dl, OpenMM + PDBFixer, pydantic, typer,
pandas, numpy, scipy. Optional extras: `[gpu]` adds Torch for GPU-accelerated distance
matrices; `[service]` adds the HTTP server deps.
</details>

## Quickstart

The repo ships a tiny Boltz-2 example (`examples/1OIU_boltz.pdb` — CDK2 + a
sulfonamide inhibitor) so everything runs out of the box:

```bash
# 1) a single SMILES -> hierarchical medchem taxonomy
interaction-profile --smiles "CC(=O)Oc1ccccc1C(=O)O" --name aspirin --report

# 2) a structure + its ligand SMILES -> full interaction profile
interaction-profile \
    --structure examples/1OIU_boltz.pdb --ligand-resname LIG \
    --smiles "c1cc(cc(c1)S(=O)(=O)N)Nc2nc3c(c(n2)OCC4CCCCC4)nc[nH]3" \
    --name 1OIU --report

# 3) the bundled example batch (three copies of the structure + a SMILES-only entry)
interaction-profile examples/example_batch.json --out result.json --report
```

## Tests

```bash
pip install -e ".[dev,service]"
pytest                              # unit + integration (integration uses the bundled example;
                                    # it auto-skips if the heavy stack is missing)
ruff check src tests scripts        # lint
```

## Batch input

A JSON list (or `{"batch_id", "molecules": [...]}`, or CSV). Each item:

```json
[
  {"name": "1OIU", "structure": "1OIU_boltz.pdb", "ligand_resname": "LIG",
   "smiles": "c1cc(cc(c1)S(=O)(=O)N)Nc2nc3c(c(n2)OCC4CCCCC4)nc[nH]3", "source": "cofold"},
  {"name": "imatinib", "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"}
]
```

Structure paths are resolved relative to the batch file's directory.

- `smiles` only → hierarchical taxonomy for that molecule.
- `smiles` + `structure` → full pipeline; the SMILES is used as the bond-order
  template so the ligand is chemically correct.
- `structure` only → the ligand is taken from the structure (bond orders
  perceived from geometry; supplying SMILES is recommended).

## CLI

```bash
# batch (see examples/example_batch.json for the format)
interaction-profile examples/example_batch.json --ph 7.4 --radius 8.0 \
    --out batch.result.json --report

# single-shot convenience
interaction-profile --structure examples/1OIU_boltz.pdb --ligand-resname LIG \
    --smiles "c1cc(cc(c1)S(=O)(=O)N)Nc2nc3c(c(n2)OCC4CCCCC4)nc[nH]3" \
    --name 1OIU --out r.json --report
```

Key options: `--ph` (ligand + receptor protonation), `--radius` (pocket crop,
Å), `--ligand-resname` / `--ligand-chain` / `--ligand-keyword` (pin the ligand),
`--all-ligands` (profile every qualifying ligand), `--peptide-ligand-chain`
(treat a chain as the ligand), `--strict-ligand-match` (error when the ligand's
InChIKey ≠ the SMILES template), `--keep-uncertain`, `--workers N` (parallel
batch), `--gpu`, `--out`, `--csv`, `--readable`, `--report`.

## Library

```python
from interaction_profiler import profile_batch, profile_molecule, to_table

result = profile_batch("examples/example_batch.json", ph=7.4, radius=8.0)
df = to_table(result)          # pandas: one row per interaction
result.model_dump_json(indent=2)
```

## Pipeline

```
batch -> per molecule:
  no structure -> prepare ligand from SMILES (pH) -> hierarchical taxonomy
  structure    -> prepare receptor (PDBFixer, pH, keep metals/waters)
               -> prepare ligand (template bond orders + explicit H, pH, stereo)
               -> fragment (AccFG + SMARTS + rings + scaffold)
               -> crop pocket (radius) -> geometric interaction detector
               -> attach fragments -> group into triples
```

### Preparation

- **Ligand:** `AssignBondOrdersFromTemplate` from the protonated reference
  SMILES, then `AddHs(addCoords=True)`. Protonation state chosen at the
  requested pH via `dimorphite_dl`. Stereochemistry is perceived from 3D and
  compared to the input SMILES; findings are reported.
- **Receptor:** PDBFixer adds missing heavy atoms and explicit hydrogens at
  the requested pH for protein and nucleic residues. Metals and waters are
  preserved for metal-coordination and water-bridge detection.

### Interaction detection

A custom geometric engine. Pharmacophore features are perceived
on the ligand (RDKit feature factory on the prepared Mol) and on the receptor
(residue templates + the structure's explicit hydrogens). Distances use a
vectorized numpy core (optionally Torch on GPU); per-class distance/angle
cutoffs follow common structural conventions, and a core/extended band sets
the `confident`/`uncertain` flag.

## Output

A `BatchResult` (`schemas.py`): `batch_id`, `provenance`, and `molecules`,
each with `metadata` (name, `input_mode`, ligand SMILES, center of mass,
receptor type, chains, pH, source, counts, timing), `composition` (fragments +
coverage), `interactions`, `triples`, and a `preparation` report (bond
assignment, `template_inchikey` / `ligand_inchikey` / `inchikey_match` /
`inchikey_exact_match`, warnings). `BatchResult.to_rows()` (or `to_table`)
flattens to one row per interaction. A published JSON schema is in
[`schemas/batch_result.schema.json`](schemas/batch_result.schema.json).

## REST service

For pipelines / scoring code that call the profiler over HTTP instead of
importing it, there is a FastAPI wrapper (`interaction_profiler.service`).
It mirrors the `dd_agentic` service contract: `GET /health`, `POST /analyze`
(a list of molecules → one `BatchResult`), `POST /profile` (single molecule →
per-ligand results), and `GET /schema`. Structures are exchanged through a
shared volume (`IP_SHARED_DIR`, default `/shared`); SMILES-only inputs need
no volume.

```bash
# dev, from interaction-profiler/ : the app lives in the deployment wrapper
uvicorn app:app --app-dir ../services/interaction-profiler --host 0.0.0.0 --port 8080
# curl localhost:8080/health
```

Docker (build from the repo root, `dd_agentic/`, like `services/boltz2`):

```bash
docker build -f services/interaction-profiler/Dockerfile -t interaction-profiler:latest .
docker run --rm -p 8080:8080 -v "$PWD/examples:/shared" interaction-profiler:latest
```

`POST /analyze` body:

```json
{
  "batch_id": "round7",
  "molecules": [
    {"name": "1OIU", "structure": "1OIU_boltz.pdb", "ligand_resname": "LIG",
     "smiles": "c1cc(cc(c1)S(=O)(=O)N)Nc2nc3c(c(n2)OCC4CCCCC4)nc[nH]3"},
    {"name": "cand_042", "structure": "boltz/cand_042.cif", "smiles": "Cc1ccc(cc1)S(N)(=O)=O"},
    {"name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}
  ],
  "options": {"ph": 7.4, "radius": 8.0, "drop_uncertain": true}
}
```

### Worked example

Start the server (point `IP_SHARED_DIR` at a directory of structures; here the
bundled `examples/`), then call it from another shell:

```bash
IP_SHARED_DIR="$PWD/examples" uvicorn app:app --app-dir ../services/interaction-profiler --port 8090 &
```

**Single SMILES → medchem taxonomy** (no structure, no shared volume needed):

```bash
curl -s localhost:8090/profile -H 'content-type: application/json' -d '{
  "molecule": {"name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}
}' | python3 -m json.tool
```

Trimmed response (one result per molecule):

```json
{
  "provenance": {"tool": "interaction-profiler", "version": "0.3.0",
                 "tool_versions": {"rdkit": "2025.09.6", "openmm": "8.5.2", "...": "..."}},
  "results": [{
    "metadata": {"name": "aspirin", "input_mode": "smiles_only",
                 "ligand_smiles": "CC(=O)Oc1ccccc1C(=O)[O-]", "n_interactions": 0,
                 "timing_seconds": 0.014},
    "composition": {"n_heavy_atoms": 13, "coverage": 1.0, "fragments": [
      {"canonical_name": "carboxylic_acid", "level": "functional_group", "atom_indices": [10,11,12], "smiles": "O=C[O-]"},
      {"canonical_name": "benzene",         "level": "functional_group", "atom_indices": [4,5,6,7,8,9], "smiles": "c1ccccc1"},
      {"canonical_name": "ester",           "level": "functional_group", "smarts": "[#6][CX3](=O)[OX2H0][#6]"},
      {"canonical_name": "murcko_scaffold", "level": "scaffold", "smiles": "c1ccccc1"}
    ]},
    "interactions": [], "triples": [],
    "preparation": {"bond_assignment": "smiles_only",
                    "ligand_smiles_protonated": "CC(=O)Oc1ccccc1C(=O)[O-]",
                    "template_inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                    "ligand_inchikey":   "BSYNRYMUTXBXSQ-UHFFFAOYSA-M",
                    "inchikey_match": true, "inchikey_exact_match": false},
    "error": null
  }]
}
```

Reading it: `input_mode=smiles_only` (no structure) so `interactions`/`triples`
are empty and only the **medchem taxonomy** is returned — 8 fragments at 100%
coverage. The carboxylic acid is deprotonated to the **carboxylate** at pH 7.4
(`ligand_smiles_protonated`). `inchikey_match=true` (connectivity matches) with
`inchikey_exact_match=false` reflects exactly that protonation change (the
trailing `-N` vs `-M` on the InChIKey).

**A structure** (the bundled Boltz-2 example, resolved from the shared dir):

```bash
curl -s localhost:8090/analyze -H 'content-type: application/json' -d '{
  "molecules": [{"name": "1OIU", "structure": "1OIU_boltz.pdb", "ligand_resname": "LIG",
                 "smiles": "c1cc(cc(c1)S(=O)(=O)N)Nc2nc3c(c(n2)OCC4CCCCC4)nc[nH]3"}]
}' | python3 -m json.tool
```

Trimmed response:

```json
{
  "n_molecules": 1,
  "molecules": [{
    "metadata": {"name": "1OIU", "input_mode": "structure+smiles", "receptor": "protein",
                 "chains": ["A", "B"], "chain_types": {"A": "protein", "B": "ligand"},
                 "ligand_smiles": "NS(=O)(=O)c1cccc(Nc2nc(OCC3CCCCC3)c3nc[nH]c3n2)c1",
                 "n_interactions": 22, "n_triples": 12, "interaction_cap": 34,
                 "n_dropped": 16, "timing_seconds": 0.695},
    "composition": {"n_heavy_atoms": 28, "coverage": 1.0, "fragments": [
      {"canonical_name": "sulfonamide", "smiles": "NS(=O)=O"},
      {"canonical_name": "purine",      "smiles": "c1ncc2ncnc2n1"},
      {"canonical_name": "imidazole"}, {"canonical_name": "pyrimidine"}, {"...": "..."}
    ]},
    "triples": [
      {"fragment": "amine",       "residue_label": "LEU83.A", "moiety": "backbone",
       "interaction_type": "hbond", "subtype": "hbond_donor",    "distance_min": 2.32},
      {"fragment": "pyrimidine",  "residue_label": "LEU83.A", "moiety": "backbone",
       "interaction_type": "hbond", "subtype": "hbond_acceptor", "distance_min": 3.33},
      {"fragment": "imidazole",   "residue_label": "GLU81.A", "moiety": "backbone",
       "interaction_type": "hbond", "subtype": "hbond_donor",    "distance_min": 2.87},
      {"fragment": "sulfonamide", "residue_label": "ASP86.A", "moiety": "sidechain",
       "interaction_type": "hbond", "subtype": "hbond_donor",    "distance_min": 2.73},
      {"fragment": "benzene",     "residue_label": "ILE13.A", "moiety": "sidechain",
       "interaction_type": "hydrophobic", "count": 6, "distance_min": 3.79, "distance_max": 4.09},
      {"fragment": "imidazole",   "residue_label": "LEU134.A", "moiety": "sidechain",
       "interaction_type": "ch_pi", "count": 2, "distance_min": 3.27, "distance_max": 3.59}
    ],
    "preparation": {"bond_assignment": "template",
                    "template_inchikey": "BKDUVKJYBJDZQW-UHFFFAOYSA-N",
                    "ligand_inchikey":   "BKDUVKJYBJDZQW-UHFFFAOYSA-N",
                    "inchikey_match": true, "inchikey_exact_match": true,
                    "receptor_prepared": true,
                    "prepared_receptor_path": "/tmp/interaction_profiler_receptors/1OIU_boltz_8a976a75b63c_prepared.pdb"},
    "error": null
  }]
}
```

Reading it: `input_mode=structure+smiles` and `bond_assignment=template` mean the
SMILES applied cleanly to Boltz's predicted coordinates (sulfonamide and the
aromatic purine round-trip intact), and `inchikey_match`/`inchikey_exact_match`
both `true` confirm the modeled ligand *is* the input molecule. The ligand is
auto-labeled chain **B**; 38 raw contacts were capped to 34 then filtered to 22
confident ones, grouped into 12 `(fragment, residue, interaction)` triples —
including the canonical **CDK2 hinge** H-bonds to Leu83 (donor 2.32 Å + acceptor
3.33 Å) and Glu81, the sulfonamide→Asp86 H-bond, and a hydrophobic/CH–π cage
(Ile13 ×6, Phe82, Leu134). The prepared receptor is cached under `IP_CACHE_DIR`
(here the default temp dir), so repeat calls on the same target are fast.

If you get `"error": "structure not found: /shared/1OIU_boltz.pdb"`, the server
wasn't started with `IP_SHARED_DIR` pointing at the folder that contains the
file (the default is `/shared`).

The service also validates SMILES up front, classifies each input
(`smiles_only` / `structure_only` / `structure+smiles`), matches the SMILES to
the structure's ligand by formula + InChIKey (with an optional
`strict_ligand_match`), confines structure paths to the shared dir, and caches
prepared receptors under `IP_CACHE_DIR` (outside `/shared`). It is CPU-bound,
so scale with worker **processes** (`IP_WORKERS`), not client threads.
Benchmark (60 crystal complexes, warm receptor cache): ~471 complexes/min
single-worker serial; ~934/min at 4 workers with 8-way client concurrency
(1.93×).

**See [`SERVICE.md`](SERVICE.md) for the full install / run / Docker / endpoint
reference.** Harnesses: `scripts/service_benchmark.py` (speed),
`scripts/stress_test.py` (robustness).

## Validation & robustness

Validated on **130 systems** (110 crystal + 20 Boltz-2 cofolded)
spanning covalent, PROTAC, molecular-glue, fragment, metalloenzyme, RNA and DNA
modalities — all profiled with 0 errors. Those bulk structure sets are kept out
of the repo to keep it lightweight, plus two harnesses:

- [`scripts/service_benchmark.py`](scripts/service_benchmark.py) — REST-service
  throughput / latency.
- [`scripts/stress_test.py`](scripts/stress_test.py) — adversarial SMILES +
  dirty / edge structures with result-invariant checks (0 crashes,
  0 inconsistencies).

## License

MIT. See [`LICENSE`](LICENSE).
