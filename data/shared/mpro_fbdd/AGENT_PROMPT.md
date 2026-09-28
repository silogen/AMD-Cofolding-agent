# Fragment-Based Drug Design — SARS-CoV-2 Mpro

## Target

SARS-CoV-2 main protease (nsp5 / 3CLpro), UniProt P0DTD1, processed chain
PRO_0000449633 (polyprotein residues 3264–3569 = Mpro 1–306). Obligate
homodimer; each protomer carries a Cys145–His41 catalytic dyad. The substrate
groove is an open, well-defined pocket of roughly 350 Å³ spanning subsites
S1, S2, S4 and S1'. Residues that matter:

- **S1** — His163 (essential for substrate recognition), Phe140, Glu166
- **S2** — hydrophobic, Met49 and Met165
- **S4** — small and shallow, tolerates substitution poorly
- **Oxyanion hole** — backbone NH of Gly143, Ser144, Cys145

There is no close human homolog with comparable activity, so the selectivity
window is intrinsically wide. Do not spend budget on selectivity modelling.

## Starting material

A crystallographic fragment screen, already triaged to x-series only (the 2020
XChem primary screen — elaborated Moonshot compounds have been removed):

```
/shared/mpro_fbdd/                          ← container path (mounted as /shared)
  fragments.csv                  366 rows / 357 unique SMILES, 206 fragment-like
  structures/<ID>/protein_apo.pdb
  structures/<ID>/ligand.sdf     bound pose, crystal coordinates
  structures/<ID>/complex.pdb
  README.md                      composition, caveats, provenance
```

Read the CSV with `read_shared_file('mpro_fbdd/fragments.csv')`. Paths in the
`protein_apo`, `ligand_sdf` and `complex_pdb` columns are absolute container
paths (`/shared/mpro_fbdd/...`) — use them verbatim in tool calls.

`fragments.csv` already carries MW, heavy-atom count, cLogP, TPSA, HBD/HBA,
rotatable bonds, ring count and Murcko scaffold, so you do not need to
recompute them. The `fragment_like` column flags MW < 300 and heavy atoms < 22.
The `duplicate_of` column is blank for the 357 primary entries.

Two things this set does **not** tell you, which you must establish yourself:

1. **Site occupancy is not annotated.** Many x-series hits bind at surface or
   crystal-packing sites rather than the substrate groove.
2. **There are no activity values.** A crystallographic screen gives you
   binding poses, not potencies. Ligand efficiency must be reasoned from the
   structures and your scoring stack, not read off a column.

## Scope

Design **non-covalent** inhibitors only. Do not propose Cys145 warheads,
acrylamides, chloroacetamides, vinyl sulfones or other reactive electrophiles.
This matches the second-generation inhibitor rationale — nirmatrelvir
resistance mutations (A173V, L167F) are driving interest in non-covalent
chemotypes. Encode this as an explicit substructure constraint in the TTP
rather than relying on judgement at review time.

## Tool use and provenance

You have live ChEMBL and RCSB access and you may use them. Be aware that both
contain post-2020 Mpro data, including COVID Moonshot and DNDI-6510, so they
can tell you the answer rather than help you derive it. That is acceptable
here, but it must be traceable. **Tag every proposed compound with how it
arose:**

- `structure-derived` — from the fragment poses and your scoring stack
- `retrieved` — informed by a ChEMBL/RCSB/literature lookup
- `recalled` — from your own knowledge of known Mpro inhibitors

If a design converges on something you recognise (nirmatrelvir, ensitrelvir,
S-217622, a Moonshot lead), say so explicitly and tag it. A structure-derived
rediscovery of a known chemotype is the strongest possible result; an untagged
lookup is worthless. Keep the counts per tag in your final report.

## Workflow

### 1. Triage

Load the x-series fragments. Keep those bound in the substrate groove — near
the dyad or spanning S1/S2 — and discard surface and crystal-packing binders.
Use `characterize_site_profile` on the apo structures with
`analysis_profile='full'` and `uniprot_accession='P0DTD1'` — this activates
UniProt active-site overlay (Cys145, His41, His163, Glu166) and subpocket
labelling (S1/S2/S1'/S4). Without these arguments the call returns raw pocket
geometry only (volume, druggability) with no subpocket IDs or residue roles,
which is insufficient for triage. Use `profile_complex_interaction` on the
bound complexes to decide, rather than guessing from SMILES.

Report, per retained fragment: which subsite it occupies, the specific contacts
it makes (flag His163 and oxyanion-hole hydrogen bonds), heavy-atom count, and
an estimated ligand efficiency. Target a shortlist of 10–20. Prefer small,
well-anchored hits with open growth vectors over larger hits that already fill
the pocket — the smallest entries in the set run down to 5–11 heavy atoms and
are the most informative starting points.

### 2. Plan the three FBDD moves

From the shortlist, choose and justify one of each:

| Move | When | Tool path |
|---|---|---|
| **Grow** | one anchored fragment with a clear vector into an adjacent empty subsite | LibInvent — scaffold = the fragment, attachment points on that vector |
| **Link** | two fragments in adjacent subsites with a plausible span | LinkInvent — warheads = the two fragments |
| **Merge** | overlapping fragments sharing a pharmacophore | Mol2Mol seeded on the merged hypothesis (`.m2m_scaffold` or `.m2m_similarity`) |

State the structural evidence for each choice — which fragments, which
subsites, measured distance for the link — **before** building anything.

### 3. Build the TTP

`create_ttp` with:

- **target** — the apo PDB of the parent fragment complex; set
  `binding_site_residues: [["A",41],["A",163],["A",166],["A",145]]` and
  `binding_site_force: true` so Boltz always places the ligand in the S1/S2
  groove rather than drifting to a surface pocket.
- **scoring** — keep to ≤ 4 objectives in `arithmetic_mean` (geometric mean
  with many competing objectives kills the RL signal): BoltzStructure confidence
  (primary structure quality gate) + one property filter (MW or QED) + one
  interaction term rewarding the Glu166 backbone H-bond. Add a PoseBusters
  gate automatically via `_objectives_with_pose_gate`. Add RetroDMTA potency
  only in a second-round TTP after the first round narrows the scaffold.
- **constraints** — no PAINS; no reactive electrophiles (see Scope)

Growth discipline: keep ligand efficiency at or above the parent fragment's.
A larger molecule that is only proportionally more potent is a failed
elaboration, and you should report it as such rather than ranking it highly.

### 4. Run

Call `validate_generation_campaign` first, then `create_generation_campaign`
and `run_generation_campaign`. Jobs are asynchronous and serialise on a single
GPU, so launch one branch at a time and poll `get_optimization_status`. Pull
winners with `get_top_candidates`.

### 5. Confirm

For the top ~20 across all three branches:

- Cofold with `predict_structure_conditioned` against the parent apo structure.
  This produces a new 3D pose for each candidate. **Use this pose — not any
  pre-existing crystal complex PDB — as the input to the next step.** Passing a
  crystal LIG complex as the structure will profile the crystal ligand's
  coordinates, not the candidate's.
- Re-profile the Boltz output PDB with `profile_complex_interaction` (pass the
  candidate SMILES explicitly) and **discard anything that loses the anchoring
  interaction its parent fragment made** — this is the single most important
  filter in the whole workflow. For Mpro the mandatory anchor is the Glu166
  backbone H-bond (≤ 3.2 Å, angle ≥ 120°); discard any candidate that loses it.
- `predict_admet` (report applicability-domain pass/fail per endpoint, not
  just the point estimate)
- `plan_synthesis` and `check_building_block_availability`

## Deliverable

A ranked table of 10–15 proposed compounds. Per compound: SMILES; parent
fragment(s); move (grow / link / merge); provenance tag; predicted affinity;
interactions retained versus parent; ligand efficiency versus parent; ADMET
flags with AD status; synthetic accessibility and building-block availability.

Then a short narrative covering: what the fragment map told you about the
S1/S2 groove, which of the three moves you would push first and why, which
fragments you rejected and on what evidence, and an honest account of where
the lookup tools did versus did not change your designs.
