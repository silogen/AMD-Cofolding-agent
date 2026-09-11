# ROADMAP — interaction-profiler

This file carries the long-term direction for `interaction-profiler`.
v1 is the standalone block (composition + tripartite per complex). The
items below describe where this tool fits in the broader chemistry-aware
agentic stack. They are explicit hooks the v1 design preserves so we can
reach them later without rework — and explicit cuts where we are not
seeding capability into v1.

## In long-term scope (v1 leaves seams open)

1. **The unit of memory is the pharmacophore triple, not the molecule.**
   v1 outputs are keyed on a stable `ligand_id` so a future store can be
   triples-first (`fragment × residue × interaction_type → frequency,
   provenance`), with molecules as instantiations. The `Triple` pydantic
   model is already the natural row type for that store.

2. **Canonical chemistry vocabulary that survives tool swaps.** v1 routes
   every fragment label and interaction type through `vocabulary.py`
   (near-identity in v1) so SmartChemist patterns and PLIP can be added
   later without invalidating prior outputs.

3. **Reference-anchored campaigns.** Reference IFPs are a future feature;
   v1 does not compute IFP-distance-to-reference but the output schema
   reserves `Provenance.reference_id` so future runs can attach one.

4. **Active steering in pharmacophore-space.** The agent eventually
   expresses chemistry intent as triple constraints. That requires a
   stable vocabulary and stable atom-idx provenance, both of which v1
   enforces. No actuator logic in v1.

5. **SmartChemist union as the eventual fragmenter.** AccFG is the v1
   default; the `Fragmenter` ABC and `get_fragmenter(method=...)`
   factory make SmartChemist and EFG addable as alternate or unioned
   backends later.

6. **Complex-centric, not protein-ligand-centric.** v1 input and output
   models treat a complex as a set of biomolecules with chain roles —
   not a `(protein, ligand)` pair. PROTAC, peptide-protein, cyclic
   peptide, MHC-peptide, and PPI mimetic cases are first-class inputs
   from day one (covered in the integration test set).

7. **Per-molecule artifact bundles.** v1 produces one JSON artifact per
   `(complex, ligand_id)` pair on disk with provenance, suitable for
   later extension to a 3D-viewable bundle (SDF + PDB + IFP + fragment
   JSON + provenance).

8. **(New) Score–composition join layer.** A future, separate effort —
   not v1 — will statistically join this tool's outputs (composition +
   triples) to the scoring track's outputs (Boltz affinity, QED, ADMET,
   custom alerts, ...). It will surface relations like *"fragment X
   enriches in the high-affinity quintile,"* *"triple (carboxylate →
   LYS86 → ionic) co-occurs with ADMET warning Y,"* *"fragment Y trades
   affinity for permeability."* This is not just `triple → activity`: it
   generalises over any score the scorer emits, including ADMET.

   v1 enables this later by preserving stable `ligand_id` and
   `provenance` on every output. v1 ships zero correlation logic.

## Explicitly out of scope (no v1 hooks)

- **Persistent cross-campaign chemistry knowledge base.** Not seeded.
- **Evaluation / benchmarking pipelines.** Not seeded.
- **Fragment-aware generative-model integration.** Not seeded.

These are deliberate cuts. They do not influence the v1 design.

## Touchpoints with `dd_agentic` (later, not v1)

For reference so v1 doesn't paint corners:

- A future `services/interaction-profiler/app.py` in
  `dd_agentic/services/` wraps this library in a FastAPI service
  mirroring the existing `services/posebusters/app.py` (uniform
  `POST /analyze` + `GET /health`, reads structures from `/shared`,
  `depends_on` Boltz in the artifact-passing protocol).
- Future MCP tools (`enumerate_substructures`, `analyze_interactions`,
  `map_substructures_to_interactions`) in `reinvent4_mcp/tools/` call
  this service.
- `"interaction_profiler"` is now listed in `KNOWN_EXTERNAL_SERVICES`
  in `reinvent4_mcp/ttp/schema.py`: beyond the out-of-band annotation
  role, the service exposes a `POST /score` contract that counts
  interactions of a chosen type, so it can be used as an RL signal.
