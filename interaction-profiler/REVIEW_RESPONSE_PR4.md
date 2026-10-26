# Review response — PR #4 (02-structural-intelligence-annotations)

One entry per review comment: the concern, the **original** behaviour, the **new**
behaviour, and why.  All changes are in `interaction-profiler`; the full
`interaction-profiler` suite passes (113 passed; the 7 pre-existing `test_service.py`
failures are an unrelated local pydantic-version issue).  New regression tests live in
`tests/unit/test_review_fixes.py`.

---

## 1. `metrics/strain.py` — MMFF94 strain compared two different atomic systems

> "bound_energy is an MMFF94 energy over the molecule including its explicit hydrogens.
> opt_energy is an MMFF94 energy over a copy with hydrogens removed … strain =
> E(heavy+H) − E(heavy only) … dominated by the sum of all hydrogen-involving energy
> terms … essentially meaningless, and nothing errors."

Correct and load-bearing: MMFF94 is all-atom, so subtracting an H-free energy from an
H-containing one leaves a large H-count offset, not conformational strain.

Original — reference energy stripped hydrogens:
```python
opt_mol = copy.deepcopy(mol)
opt_mol = Chem.RemoveHs(opt_mol, sanitize=False)   # <-- different atom set
Chem.SanitizeMol(opt_mol)
...                                                 # minimise, CalcEnergy
# strain = bound_energy(with H) − opt_energy(without H)
```

New — both energies use the identical all-atom system:
```python
mol, _added_h = _ensure_explicit_hs(mol)            # add H (with coords) if absent

# bound pose: heavy atoms fixed, only hydrogens relaxed
ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
for atom in mol.GetAtoms():
    if atom.GetAtomicNum() != 1:
        ff.AddFixedPoint(atom.GetIdx())
ff.Minimize(maxIts=500)
bound_energy = ff.CalcEnergy()

# reference: deep copy, full minimisation, hydrogens kept
opt_energy = _mmff94_minimized_energy(mol)          # same atoms as bound
strain = bound_energy - opt_energy
```

Why: identical atom set on both sides, so the difference isolates heavy-atom
conformational strain.  Relaxing only hydrogens on the bound pose (heavy atoms fixed)
removes artefacts from poorly determined crystallographic H placement — the original
motivation for stripping H — without changing the atom count.

Regression test: a fully relaxed conformer now returns strain ≈ 0 (the old code
returned tens of kcal/mol); a perturbed torsion returns a larger value.

---

## 2. `entity/resolver.py` — phantom entities, contract, case, vocabulary drift, dead code

> "unvalidated chain IDs → silent phantom entities … `ct.get(ch, 'protein')` … you
> still emit a fully-formed Entity for a chain that isn't there … the worst failure
> mode because it's invisible."

Original:
```python
tgt_types = [ct.get(ch, "protein") for ch in target_chains]   # typo chain → 'protein'
```
New:
```python
valid_target_chains = _validate_chains(list(target_chains), ct,
                                       role_label="target", warnings=warnings)
# chains absent from the structure are dropped with a warning; never defaulted
tgt_types = [ct[ch] for ch in valid_target_chains]
```

> "The 'does not raise' contract isn't fully defended … the Entity(...) constructors
> aren't wrapped … sel.chain/resname/resnum reads are not."

New: every `Entity(...)` construction and the post-`detect_ligands` `sel.*` reads are
wrapped; failures append a warning and omit that entity instead of propagating.

> "Case sensitivity … if chain_types() ever returns 'Protein', every lookup misses and
> silently defaults to protein."

New: `chain_types()` output is lower-cased once at the boundary in `resolve_entities`.

> "Unknown types default to protein with no warning" and "'buffer' … has no entry in
> `_infer_entity_role_from_chain_type`'s mapping."

New: `_infer_entity_role_from_chain_type(chain_type, warnings, context)` maps the full
`chain_types()` vocabulary (adds `buffer` and `hetero` → `small_molecule`) and appends a
warning for any label outside the vocabulary instead of silently returning `protein`.

> "Dead code: `_chain_length` … and `_classify_protein_by_length` … leftovers."

New: both removed; replaced by `_validate_chains`.

Regression tests: bogus chain → no entity + warning; mixed-case types resolve; a
buffer-dominant target is `small_molecule`, not `protein`; unknown type warns.

---

## 3. `interactions/enricher.py` — resnum-only lookups collide across chains

> "ss_map is keyed by (chain_id, resnum) and you look it up chain-aware … but
> feature_map and interpro_map are keyed by resnum alone … looked up with just
> feature_map.get(target_resnum)."

Original:
```python
uniprot_entry = feature_map.get(target_resnum)      # chain-blind
interpro_entry = interpro_map.get(target_resnum)
```
New — a shared helper, chain-aware with resnum fallback (keeps single-chain callers
working, fixes multi-chain collisions):
```python
def _lookup_annotation(amap, chain, resnum):
    if amap is None or resnum is None:
        return None
    return amap.get((chain, resnum)) or amap.get(resnum)

uniprot_entry = _lookup_annotation(feature_map, target_chain, target_resnum)
interpro_entry = _lookup_annotation(interpro_map, target_chain, target_resnum)
```

> "ix.ligand_atom_idx >= 0 assumes the field is an int."

New:
```python
lig_idx = ix.ligand_atom_idx
... if isinstance(lig_idx, int) and lig_idx >= 0 else None
```

> "UniProt requires a truthy `mapped` flag … InterPro attaches on mere presence, no
> flag check."

New: both use the same rule — attach when the entry is present and
`entry.get("mapped", True)` is truthy (InterPro entries carry no flag, so they default
to True, preserving behaviour while removing the inconsistency).

Also removed the now-unused `from typing import Any` import.

---

## 4. `metrics/interface.py` — fixed-column PDB round-trip corrupts big assemblies

> "every intermediate is written back out as PDB … fixed-column PDB format silently
> overflows … atom serials wrap past 99,999, residue numbers past 9,999, multi-character
> chain IDs get truncated … Outputs might get generated but they won't make sense."

MDAnalysis 2.x cannot write mmCIF, so the intermediates must stay PDB.  The fix makes the
risk explicit rather than silent.

New — detection helper and a `warnings` field on `InterfaceGeometry`:
```python
def _detect_pdb_overflow(atom_group, label) -> list[str]:
    # flags >99,999 atoms, residue numbers outside the 4-column range,
    # and multi-character chain IDs

overflow_warnings = _detect_pdb_overflow(binder_ag, "binder") + \
                    _detect_pdb_overflow(target_ag, "target")
...
return InterfaceGeometry(..., warnings=overflow_warnings)
```

Why: BSA (total SASA) is robust to numbering, but per-residue interface assignments are
not; callers now receive an explicit warning when the selection exceeds PDB limits
instead of a silently wrong residue list.

Regression tests: multi-character chains and >99,999-atom / >9,999-resid selections each
produce a warning; a clean selection produces none.

---

## 5. `site/characterizer.py` — feature_map vs idx_to_label index-scheme mismatch

> "Differences in indexing through how feature_map is built vs how idx_to_label is built
> in _fetch_uniprot_overlay."

`map_uniprot_features_to_residues()` returns keys that are 0-based indices into *that
chain's* residue list.

Original — merged with `update()` (chain A res 5 and chain B res 5 collide) while
`idx_to_label` used a global running offset, so the two schemes disagreed for every chain
after the first:
```python
for ch in chains:
    ...
    feature_map.update(ch_map)              # per-chain 0-based keys, collide
# separate loop:
idx_to_label[offset + i] = label           # global 0-based keys
```

New — a single pass that shifts each chain's feature keys by the same running offset used
to build `idx_to_label`, so keys never collide and every key resolves to its label:
```python
offset = 0
for ch in chains:
    residues = struct.u.select_atoms(...).residues
    for i, r in enumerate(residues):
        idx_to_label[offset + i] = f"{r.resname}:{r.resid}.{chain_id}"
    for local_idx, feat in map_uniprot_features_to_residues(...).items():
        feature_map[offset + local_idx] = feat
    offset += len(residues)
```

Why: one global index space shared by both maps.  For single-chain inputs (the common
case) `offset == 0`, so behaviour is unchanged; multi-chain inputs are now correct.
