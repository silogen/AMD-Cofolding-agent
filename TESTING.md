# Testing Guide

This document lists all tests that can be run against the current codebase (Phases 1, 2, 3, 4, and 5a). Tests are organized into three tiers: unit tests that need no external dependencies, integration tests that require REINVENT4, and end-to-end tests that verify the full agent stack.

## Quick Start

```bash
cd examples/dd_agentic
pip install pytest numpy pydantic PyYAML pandas fastapi httpx requests
python -m pytest tests/ -v
```

All unit tests mock REINVENT4 imports and external services, so they run anywhere with Python 3.10+ and the packages above.

---

## Tier 1: Unit Tests (no REINVENT4 needed)

These tests validate internal logic with mocked REINVENT4. Run them on any machine.

### TTP Schema Validation (`tests/test_ttp.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 1 | Transform config types | All 8 transform types accepted (`sigmoid`, `reverse_sigmoid`, `double_sigmoid`, `right_step`, `left_step`, `step`, `value_mapping`, `exponential_decay`) | `pytest tests/test_ttp.py::TestTransformConfig -v` |
| 2 | Transform extra params | Extra params like `low`, `high`, `k` pass through | same class |
| 3 | Invalid transform rejected | Unknown transform type raises `ValidationError` | same class |
| 4 | Target protein sequence | Valid amino-acid sequences accepted, short (<10 aa) and invalid characters rejected, auto-uppercased | `pytest tests/test_ttp.py::TestTargetConfig -v` |
| 5 | Pre-scoring components | Valid components (`QED`, `MolecularWeight`, etc.) accepted; unknown components and zero/negative weights rejected | `pytest tests/test_ttp.py::TestPreScoringComponent -v` |
| 6 | Pre-scoring duplicates | Duplicate component names within pre-scoring rejected | `pytest tests/test_ttp.py::TestPreScoringSection -v` |
| 7 | Scoring objectives | Native requires `component`, external requires `service` + `service_url`; valid entries accepted | `pytest tests/test_ttp.py::TestScoringObjective -v` |
| 8 | Scoring section | `depends_on` referencing nonexistent objective rejected; self-dependency rejected; duplicate names rejected | `pytest tests/test_ttp.py::TestScoringSection -v` |
| 9 | Constraint config | `custom_alerts` requires non-empty `smarts` list | `pytest tests/test_ttp.py::TestConstraintConfig -v` |
| 10 | Full TTP validation | Minimal and full TTPs parse correctly; invalid ID rejected; Boltz without target rejected; cross-section name collision rejected | `pytest tests/test_ttp.py::TestTTP -v` |
| 11 | TTP properties | `has_external_objectives`, `native_objective_names`, `external_objective_names` return correct values | same class |
| 12 | TTP JSON roundtrip | `model_dump` -> `model_validate` preserves all fields | same class |

### TTP Store (`tests/test_ttp.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 13 | Create and get | TTP persisted as JSON, retrievable by ID | `pytest tests/test_ttp.py::TestTTPStore -v` |
| 14 | Duplicate create rejected | `FileExistsError` on duplicate ID | same class |
| 15 | Get nonexistent raises | `FileNotFoundError` for unknown ID | same class |
| 16 | Update increments version | Version auto-incremented, fields merged | same class |
| 17 | Update preserves ID | Cannot change ID via update | same class |
| 18 | Version history | All versions retrievable by number; latest returned by default | same class |
| 19 | List all | Returns summary metadata for all TTPs | same class |
| 20 | Delete | Removes TTP and all version files | same class |
| 21 | Delete nonexistent raises | `FileNotFoundError` for unknown ID | same class |
| 22 | Exists check | `exists()` returns correct boolean | same class |

### TTP Mapper (`tests/test_ttp.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 23 | Extract native components | Correct flat list from full TTP (6 pre-scoring + 1 native scoring); external excluded | `pytest tests/test_ttp.py::TestMapper -v` |
| 24 | Transform preservation | Transforms pass through to extracted components | same class |
| 25 | Full scorer config structure | `ttp_to_scorer_config` produces correct shape (type + component list), including externals | `pytest tests/test_ttp.py::TestTTPToScorerConfig -v` |
| 26 | Native component format | Endpoint has `name`, `weight`, `transform`, `params` | same class |
| 27 | External component format | `ExternalRESTScorer` entries for Boltz/PoseBusters with service params | same class |
| 28 | Constraint format | `CustomAlerts` appended with SMARTS params | same class |
| 29 | Ordering | Pre-scoring first, scoring middle, constraints last | same class |
| 30 | Aggregation from TTP | TTP's `aggregation` field used as scorer `type` | same class |
| 31 | Boltz target injection | Protein sequence, MSA path, PDB path injected into external params | same class |
| 32 | PoseBusters depends_on | Pipeline dependency preserved in config | same class |
| 33 | Service params forwarded | `affinity_mode`, `recycling_steps`, etc. passed through | same class |
| 34 | External transforms | Transform preserved on external objectives | same class |
| 35 | Null transform omitted | Objectives with `transform: null` have no transform key | same class |
| 36 | Native-only scorer config | `ttp_to_native_scorer_config` excludes all externals | `pytest tests/test_ttp.py::TestTTPToNativeScorerConfig -v` |
| 37 | Native-only includes constraints | Constraints included even in native-only config | same class |
| 38 | All-external returns empty | Native-only config has empty component list for all-external TTP | same class |

### Topological Sort (`tests/test_ttp.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 39 | Boltz before PoseBusters | Dependency ordering respected in output | `pytest tests/test_ttp.py::TestTopologicalSort -v` |
| 40 | Reversed input reordered | Even if PoseBusters listed before Boltz in TTP, output is correct | same class |
| 41 | No dependencies preserves order | Original order kept when no `depends_on` | same class |
| 42 | Chain A -> B -> C | Multi-hop chains correctly ordered | same class |

### Mapper Summary (`tests/test_ttp.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 43 | Full TTP summary | Correct counts for pre-scoring, native, external, constraints, total | `pytest tests/test_ttp.py::TestGetMapperSummary -v` |
| 44 | Component names | Native and external names listed correctly | same class |
| 45 | Pipeline dependencies listed | `from`/`to` pairs in summary | same class |
| 46 | Minimal TTP summary | Zeros for absent sections | same class |

### Score Molecules Routing (`tests/test_scoring.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 47 | Empty SMILES | Returns empty result, no error | `pytest tests/test_scoring.py::TestScoreMoleculesRouting -v` |
| 48 | No mode specified | Returns error with guidance | same class |
| 49 | Both ttp_id and inline TTP | Returns error (ambiguous) | same class |

### TTP Resolution in Scoring (`tests/test_scoring.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 50 | Nonexistent ttp_id | Returns "not found" error | `pytest tests/test_scoring.py::TestTTPResolution -v` |
| 51 | Invalid inline TTP | Returns "Invalid" error | same class |
| 52 | All-external TTP | Returns error about external-only TTPs | same class |

### TTP-Based Scoring (`tests/test_scoring.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 53 | Minimal inline TTP | Mock scorer called, results formatted correctly | `pytest tests/test_scoring.py::TestScoreWithInlineTTP -v` |
| 54 | TTP metadata in response | `ttp_id`, `ttp_version` present | same class |
| 55 | Mapper summary in response | `mapper_summary` with correct counts | same class |
| 56 | External objectives noted | `note` field lists skipped external objectives | same class |
| 57 | No external = no note | Native-only TTP has no `note` | same class |
| 58 | Pre-scoring included | Pre-scoring components counted in mapper summary | same class |
| 59 | Scorer config structure | Config passed to Scorer has correct shape | same class |
| 60 | Constraints in config | `CustomAlerts` present in scorer config | same class |

### Stored TTP Scoring (`tests/test_scoring.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 61 | Score by ttp_id | Stored TTP loaded and scored | `pytest tests/test_scoring.py::TestScoreWithStoredTTP -v` |
| 62 | Updated version reflected | After `update`, scoring uses new version | same class |

### Summary Statistics (`tests/test_scoring.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 63 | Summary fields | `n_molecules`, `n_valid`, `mean_total_score`, `best_total_score`, `per_component` all correct | `pytest tests/test_scoring.py::TestSummaryStatistics -v` |

### Scoring Error Handling (`tests/test_scoring.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 64 | Scorer creation failure | Catches `RuntimeError`, returns `"Failed to create scorer"` | `pytest tests/test_scoring.py::TestScoringErrors -v` |
| 65 | Compute results failure | Catches `RuntimeError`, returns `"Scoring failed"` | same class |

### Boltz2 Wrapper Service (`tests/test_boltz_wrapper.py`)

Tests for the Boltz2 FastAPI wrapper (`services/boltz2/app.py`). Validates YAML generation, CLI command construction, output parsing, and endpoint behavior. No GPU or real Boltz installation required -- subprocess calls and file I/O are mocked.

**Dependencies**: `fastapi`, `httpx`, `pyyaml`

#### YAML Generation

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 66 | Basic structure | YAML has `version: 1`, protein + ligand entries, affinity properties | `pytest tests/test_boltz_wrapper.py::TestYAMLGeneration::test_basic_structure -v` |
| 67 | MSA path included | `msa` field present in protein entry when `msa_path` is provided | `pytest tests/test_boltz_wrapper.py::TestYAMLGeneration::test_msa_path_included_when_provided -v` |
| 68 | MSA path absent | `msa` field absent when `msa_path` is `None` | `pytest tests/test_boltz_wrapper.py::TestYAMLGeneration::test_msa_path_absent_when_none -v` |
| 69 | SMILES preserved | Stereochemistry-rich SMILES passed through without modification | `pytest tests/test_boltz_wrapper.py::TestYAMLGeneration::test_smiles_preserved_exactly -v` |
| 70 | YAML roundtrip | Dump and reload preserves all fields | `pytest tests/test_boltz_wrapper.py::TestYAMLGeneration::test_yaml_roundtrip -v` |

#### Batch YAML Writing

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 71 | One YAML per SMILES | `mol_0.yaml`, `mol_1.yaml`, ... created for each molecule | `pytest tests/test_boltz_wrapper.py::TestBatchYAMLWriting::test_creates_one_yaml_per_smiles -v` |
| 72 | Content matches SMILES | Each YAML file contains the correct SMILES and MSA path | `pytest tests/test_boltz_wrapper.py::TestBatchYAMLWriting::test_yaml_content_matches_smiles -v` |
| 73 | Directory creation | Batch directory created automatically, including nested parents | `pytest tests/test_boltz_wrapper.py::TestBatchYAMLWriting::test_creates_batch_directory -v` |

#### Predict Command Construction

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 74 | Basic command | `boltz predict <dir> --out_dir --override --recycling_steps` present | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_basic_command_structure -v` |
| 75 | MSA server flag | `--use_msa_server` included when `use_msa_server=True` | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_msa_server_flag -v` |
| 76 | No MSA server default | Flag absent when using pre-computed MSA | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_no_msa_server_flag_by_default -v` |
| 77 | MW correction flag | `--affinity_mw_correction` included when enabled | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_affinity_mw_correction_flag -v` |
| 78 | Custom sampling steps | `--sampling_steps` and `--sampling_steps_affinity` with custom values | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_custom_sampling_steps -v` |
| 79 | Output format | `--output_format mmcif` when specified | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_output_format -v` |
| 253 | Optimized kernels off | `--no_kernels` present while `BOLTZ_NO_KERNELS` keeps them off | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_no_kernels_flag_present_when_kernels_disabled -v` |
| 254 | Optimized kernels on | `--no_kernels` absent once the kernels are enabled | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_no_kernels_flag_absent_when_kernels_enabled -v` |

#### Kernel Flag Parsing

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 255 | `BOLTZ_NO_KERNELS` | Unset or blank keeps today's behaviour; `0`, `false`, `no`, and `off` enable the optimized kernels | `pytest tests/test_boltz_wrapper.py::TestKernelFlagEnv -v` |

#### Output Parsing

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 80 | Complete outputs | Affinity score, confidence metadata, and structure path all parsed | `pytest tests/test_boltz_wrapper.py::TestOutputParsing::test_parse_complete_outputs -v` |
| 81 | Probability binary mode | `affinity_probability_binary` selected when `affinity_mode="probability_binary"` | `pytest tests/test_boltz_wrapper.py::TestOutputParsing::test_parse_probability_binary_mode -v` |
| 82 | Missing affinity JSON | Score is `None`, structure path still returned | `pytest tests/test_boltz_wrapper.py::TestOutputParsing::test_missing_affinity_returns_none_score -v` |
| 83 | Missing mol directory | All outputs `None` when molecule directory absent | `pytest tests/test_boltz_wrapper.py::TestOutputParsing::test_missing_mol_dir_returns_none -v` |
| 84 | Missing structure file | Score still parsed, structure path is `None` | `pytest tests/test_boltz_wrapper.py::TestOutputParsing::test_missing_structure_file -v` |
| 85 | mmCIF output format | Structure path ends in `.mmcif` when format is `mmcif` | `pytest tests/test_boltz_wrapper.py::TestOutputParsing::test_mmcif_output_format -v` |

#### Results Directory Discovery

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 86 | Finds expected dir | Locates `boltz_results_<batch>/predictions` under output dir | `pytest tests/test_boltz_wrapper.py::TestFindResultsDir::test_finds_expected_dir -v` |
| 87 | Missing dir returns None | Returns `None` when no results directory exists | `pytest tests/test_boltz_wrapper.py::TestFindResultsDir::test_returns_none_when_missing -v` |

#### Response Assembly

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 88 | Multi-molecule response | Scores, metadata lists, and structure paths aligned across molecules | `pytest tests/test_boltz_wrapper.py::TestResponseAssembly::test_assembles_multi_molecule_response -v` |
| 89 | Partial failure | First molecule scored, second returns `None` when its directory is absent | `pytest tests/test_boltz_wrapper.py::TestResponseAssembly::test_handles_partial_failure -v` |
| 90 | Metadata alignment | Metadata keys padded with `None` for molecules missing that field | `pytest tests/test_boltz_wrapper.py::TestResponseAssembly::test_metadata_aligned_with_molecules -v` |

#### FastAPI Endpoints

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 91 | Health endpoint | `GET /health` reports `status: ok` | `pytest tests/test_boltz_wrapper.py::TestHealthEndpoint::test_health_returns_ok -v` |
| 256 | Health reports kernels | `GET /health` carries `no_kernels`, so an A/B run can confirm the live arm | `pytest tests/test_boltz_wrapper.py::TestHealthEndpoint::test_health_reports_the_kernel_flag -v` |
| 92 | Missing MSA config → 400 | No `msa_path` and `use_msa_server=False` returns HTTP 400 | `pytest tests/test_boltz_wrapper.py::TestScoreEndpoint::test_missing_msa_config_returns_400 -v` |
| 93 | Subprocess failure → 500 | Non-zero exit code from `boltz predict` returns HTTP 500 | `pytest tests/test_boltz_wrapper.py::TestScoreEndpoint::test_subprocess_failure_returns_500 -v` |
| 94 | Successful scoring | Mocked subprocess + output → HTTP 200 with scores and artifacts | `pytest tests/test_boltz_wrapper.py::TestScoreEndpoint::test_successful_scoring -v` |
| 95 | Timeout → 500 | `subprocess.TimeoutExpired` returns HTTP 500 with "timed out" | `pytest tests/test_boltz_wrapper.py::TestScoreEndpoint::test_timeout_returns_500 -v` |
| 96 | Missing results dir → 500 | Subprocess succeeds but no output directory → HTTP 500 | `pytest tests/test_boltz_wrapper.py::TestScoreEndpoint::test_missing_results_dir_returns_500 -v` |

### PoseBusters Wrapper Service (`tests/test_posebusters_wrapper.py`)

Tests for the PoseBusters FastAPI wrapper (`services/posebusters/app.py`). Validates fraction-of-checks scoring, metadata alignment, mode handling, and endpoint validation. PoseBusters library is mocked.

**Dependencies**: `fastapi`, `httpx`, `pandas` (the `TestBoltzPdbToSdf` class additionally requires `rdkit` and is skipped automatically when it is not installed)

#### Single-Molecule Scoring

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 97 | All checks pass | Score is `1.0`, all checks `True`, counts correct | `pytest tests/test_posebusters_wrapper.py::TestScoreSingleMolecule::test_all_checks_pass -v` |
| 98 | Partial checks pass | Score is `n_passed / n_total` (e.g., 3/4) | `pytest tests/test_posebusters_wrapper.py::TestScoreSingleMolecule::test_partial_checks_pass -v` |
| 99 | Empty DataFrame → None | Unloadable structure returns `None` score | `pytest tests/test_posebusters_wrapper.py::TestScoreSingleMolecule::test_empty_dataframe_returns_none -v` |
| 100 | Exception → None | RDKit parse failure returns `None` score | `pytest tests/test_posebusters_wrapper.py::TestScoreSingleMolecule::test_exception_returns_none -v` |
| 101 | Passes mol_cond/mol_true | Protein and reference paths forwarded to PoseBusters.bust() | `pytest tests/test_posebusters_wrapper.py::TestScoreSingleMolecule::test_passes_mol_cond_and_mol_true -v` |
| 102 | No bool columns → None | DataFrame with only non-bool columns returns `None` | `pytest tests/test_posebusters_wrapper.py::TestScoreSingleMolecule::test_no_bool_columns_returns_none -v` |

#### Multi-Molecule Scoring

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 103 | Multiple molecules | Scores computed independently per molecule | `pytest tests/test_posebusters_wrapper.py::TestScoreMolecules::test_scores_multiple_molecules -v` |
| 104 | Metadata aligned | Per-check boolean arrays positionally aligned with SMILES | `pytest tests/test_posebusters_wrapper.py::TestScoreMolecules::test_metadata_aligned_with_smiles -v` |
| 105 | Partial failure | First molecule succeeds, second fails → `[0.85, None]` | `pytest tests/test_posebusters_wrapper.py::TestScoreMolecules::test_partial_failure -v` |
| 106 | Reference paths forwarded | `reference_paths` passed through to PoseBusters for redock mode | `pytest tests/test_posebusters_wrapper.py::TestScoreMolecules::test_reference_paths_forwarded -v` |

#### Endpoint Validation

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 107 | Health endpoint | `GET /health` returns `{"status": "ok"}` | `pytest tests/test_posebusters_wrapper.py::TestHealthEndpoint -v` |
| 108 | Invalid mode → 400 | Unknown mode string returns HTTP 400 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointValidation::test_invalid_mode -v` |
| 109 | dock needs protein_pdb | `dock` mode without `protein_pdb_path` returns 400 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointValidation::test_dock_mode_requires_protein_pdb -v` |
| 110 | redock needs references | `redock` mode without `reference_paths` returns 400 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointValidation::test_redock_mode_requires_reference_paths -v` |
| 111 | SMILES/paths mismatch → 400 | Different-length `smiles` and `structure_paths` returns 400 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointValidation::test_smiles_structure_paths_length_mismatch -v` |
| 112 | Reference paths mismatch → 400 | Different-length `reference_paths` returns 400 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointValidation::test_reference_paths_length_mismatch -v` |
| 113 | mol mode no protein | `mol` mode does not require protein → 200 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointValidation::test_mol_mode_does_not_require_protein -v` |

#### Endpoint Success & Failure

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 114 | Successful scoring | Returns scores, metadata, and correct HTTP 200 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointSuccess::test_returns_scores_and_metadata -v` |
| 115 | Internal error → 500 | RuntimeError in scoring returns HTTP 500 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointSuccess::test_internal_error_returns_500 -v` |
| 116 | Null scores for failures | Failed molecules have `null` in scores array | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointSuccess::test_null_scores_for_failed_molecules -v` |

#### Mode Constants

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 117 | Protein modes valid | `MODES_REQUIRING_PROTEIN` is a subset of `VALID_MODES` | `pytest tests/test_posebusters_wrapper.py::TestModeConstants -v` |
| 118 | Reference modes valid | `MODES_REQUIRING_REFERENCE` subset of `VALID_MODES` and `MODES_REQUIRING_PROTEIN` | same class |

### External Scoring Integration (`tests/test_external_scoring.py`)

Tests for the ExternalRESTScorer REINVENT4 plugin, `list_external_services` MCP tool, pipeline artifact injection, and NaN propagation. All HTTP calls are mocked.

**Dependencies**: `numpy`, `requests`

#### ExternalRESTScorer

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 119 | Init extracts URL/service | `service_url` and `service` unpacked from list params | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_init_extracts_url_and_service -v` |
| 120 | Init builds request params | `protein_sequence`, `affinity_mode`, etc. extracted into `request_params` dict | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_init_builds_request_params -v` |
| 121 | Init skips None values | `None` params (e.g., `msa_path`) excluded from `request_params` | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_init_skips_none_values -v` |
| 122 | Artifacts initially None | `last_artifacts` is `None` before any call | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_last_artifacts_initially_none -v` |
| 123 | Successful scoring | HTTP 200 response → correct scores array and `last_artifacts` populated | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_successful_scoring -v` |
| 124 | Null → NaN conversion | JSON `null` in scores → `np.nan` in numpy array | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_null_to_nan_conversion -v` |
| 125 | All NaN on connection error | `ConnectionError` → all-NaN scores (RL loop continues) | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_all_nan_on_connection_error -v` |
| 126 | All NaN on timeout | `requests.Timeout` → all-NaN scores | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_all_nan_on_timeout -v` |
| 127 | All NaN on HTTP error | HTTP 500 → all-NaN scores | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_all_nan_on_http_error -v` |
| 128 | HTTP payload structure | POST body has `{smiles: [...], params: {...}}` with correct timeout | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_http_call_payload -v` |
| 129 | No artifacts in response | `last_artifacts` stays `None` when response omits `artifacts` | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_last_artifacts_none_when_response_has_no_artifacts -v` |
| 130 | No metadata in response | `metadata` is `None` when response omits it | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_metadata_none_when_absent -v` |

#### list_external_services

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 131 | Healthy service | `GET /health` returns `{"status": "ok"}` → status `"ok"` | `pytest tests/test_external_scoring.py::TestListExternalServices::test_healthy_service -v` |
| 132 | Unhealthy service | HTTP error on health check → status `"unhealthy"` | `pytest tests/test_external_scoring.py::TestListExternalServices::test_unhealthy_service -v` |
| 133 | Unreachable service | `ConnectionError` → status `"unreachable"` | `pytest tests/test_external_scoring.py::TestListExternalServices::test_unreachable_service -v` |
| 134 | Multiple services | Mixed healthy/unreachable correctly reported per service | `pytest tests/test_external_scoring.py::TestListExternalServices::test_multiple_services -v` |
| 135 | No services configured | Empty config → empty result, no HTTP calls | `pytest tests/test_external_scoring.py::TestListExternalServices::test_no_services_configured -v` |

#### Artifact Injection (`_ArtifactInjectingWrapper`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 136 | Injects structure paths | Upstream `last_artifacts.structures` injected into dependent's `request_params` | `pytest tests/test_external_scoring.py::TestArtifactInjectingWrapper::test_injects_structure_paths -v` |
| 137 | Injects protein PDB | `protein_pdb_path` from TTP target injected into dependent | `pytest tests/test_external_scoring.py::TestArtifactInjectingWrapper::test_injects_protein_pdb_from_ttp_target -v` |
| 138 | No artifacts → delegates | Upstream has no artifacts; dependent still called (may return NaN) | `pytest tests/test_external_scoring.py::TestArtifactInjectingWrapper::test_delegates_when_no_artifacts -v` |
| 139 | No structures key | Artifacts dict present but missing `structures` → no injection | `pytest tests/test_external_scoring.py::TestArtifactInjectingWrapper::test_delegates_when_artifacts_have_no_structures -v` |
| 140 | Attribute forwarding | `service`, `timeout` etc. forwarded to wrapped instance via `__getattr__` | `pytest tests/test_external_scoring.py::TestArtifactInjectingWrapper::test_getattr_forwards_to_wrapped -v` |

#### PipelineAwareScorer

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 141 | No dependencies skips wiring | Scorer created without replacing any component instances | `pytest tests/test_external_scoring.py::TestPipelineAwareScorer::test_no_dependencies_skips_wiring -v` |
| 142 | Wires dependent components | PoseBusters instance replaced with `_ArtifactInjectingWrapper` | `pytest tests/test_external_scoring.py::TestPipelineAwareScorer::test_wires_dependent_components -v` |
| 143 | compute_results delegates | `compute_results` calls through to underlying Scorer | `pytest tests/test_external_scoring.py::TestPipelineAwareScorer::test_compute_results_delegates -v` |
| 144 | Missing upstream warns | `depends_on` referencing nonexistent component logs warning | `pytest tests/test_external_scoring.py::TestPipelineAwareScorer::test_missing_upstream_logs_warning -v` |

#### NaN Propagation

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 145 | Service failure → all NaN | Connection error produces all-NaN array without crashing | `pytest tests/test_external_scoring.py::TestNaNPropagation::test_service_failure_produces_all_nan -v` |
| 146 | Partial null → partial NaN | Mix of scores and `null` → mix of floats and NaN | `pytest tests/test_external_scoring.py::TestNaNPropagation::test_partial_null_produces_partial_nan -v` |
| 147 | _format_results handles NaN | NaN total scores → `n_valid` excludes NaN molecules | `pytest tests/test_external_scoring.py::TestNaNPropagation::test_format_results_handles_nan_total_scores -v` |
| 148 | Empty artifacts → delegates | Wrapper still delegates to downstream even when upstream has no artifacts | `pytest tests/test_external_scoring.py::TestNaNPropagation::test_wrapper_delegates_even_with_empty_upstream_artifacts -v` |

### Optimization & Job Management (`tests/test_optimization.py`)

Tests for Phase 4: background job manager, TOML config generation, score trajectory parsing, convergence diagnostics, transfer learning, and MCP tool routing. All subprocess calls are mocked.

**Dependencies**: `pandas`, `numpy`

#### TOML Value Serialization

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 149 | Bool true/false | `_toml_bool` produces `"true"` / `"false"` | `pytest tests/test_optimization.py::TestTomlValueSerialization -v` |
| 150 | Int, float, string | Correct TOML literals for basic types | same class |
| 151 | String with quotes | Backslash-escaped inner quotes | same class |
| 152 | Lists | `[1, 2, 3]` and `["a", "b"]` serialization | same class |
| 153 | Nested lists | `[["CCO"]]` serialization | same class |

#### TOML Config Generation (Staged Learning)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 154 | Minimal config | `run_type`, `prior_file`, `agent_file`, `max_steps`, aggregation, components | `pytest tests/test_optimization.py::TestTOMLGeneration -v` |
| 155 | Agent file defaults | Agent file matches prior when not specified | same class |
| 156 | Checkpoint resume | `use_checkpoint = true` and agent_file points to checkpoint | same class |
| 157 | Full config with externals | `ExternalRESTScorer` entries with service params | same class |
| 158 | Transforms in TOML | `double_sigmoid` and `reverse_sigmoid` transforms | same class |
| 159 | Constraints in TOML | `CustomAlerts` with SMARTS params | same class |
| 160 | Diversity filter | Filter type, bucket size, minscore | same class |
| 161 | Learning strategy | DAP sigma and learning rate | same class |
| 162 | Device setting | Custom device string | same class |

#### TOML Config Generation (Transfer Learning)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 163 | Minimal TL config | `run_type = "transfer_learning"`, input/output model, SMILES file | `pytest tests/test_optimization.py::TestTLTOMLGeneration -v` |
| 164 | Custom TL parameters | num_epochs, batch_size, save_every_n_epochs, device | same class |
| 165 | With validation file | `validation_smiles_file` present when provided | same class |
| 166 | No validation file | `validation_smiles_file` absent when not provided | same class |
| 167 | No RL sections | TL config has no `[[stage]]`, `[learning_strategy]`, `[diversity_filter]` | same class |

#### Model Resolution

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 168 | File path unchanged | Absolute and relative paths returned as-is | `pytest tests/test_optimization.py::TestModelResolution -v` |
| 169 | Registry key fallback | `.reinvent` returned as-is when REINVENT4 unavailable | same class |

#### Safe Float Helper

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 170 | Normal float | `0.5` → `0.5` | `pytest tests/test_optimization.py::TestSafeFloat -v` |
| 171 | NaN → None | `float("nan")` → `None` | same class |
| 172 | None → None | `None` → `None` | same class |
| 173 | String number | `"0.75"` → `0.75` | same class |
| 174 | Bad string | `"not_a_number"` → `None` | same class |

#### Per-Component Stats

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 175 | Basic stats | Mean, max, min, mean_raw computed correctly | `pytest tests/test_optimization.py::TestPerComponentStats -v` |
| 176 | Skips non-float columns | Text columns excluded | same class |

#### Step Trajectory

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 177 | Basic trajectory | Per-step mean/max scores, molecule counts | `pytest tests/test_optimization.py::TestStepTrajectory -v` |
| 178 | None when no columns | Returns None for missing columns | same class |
| 179 | Sampling large trajectory | 200 steps sampled to 20 points | same class |

#### Per-Component Trajectory

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 180 | Basic per-component | Step-by-step mean per component (QED, MW) | `pytest tests/test_optimization.py::TestPerComponentTrajectory -v` |
| 181 | Skips non-score columns | SMILES, total_score excluded | same class |
| 182 | None when no step col | Returns None | same class |
| 183 | Sampling large trajectory | 100 steps sampled to 10 points | same class |

#### Component Health

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 184 | Healthy components | NaN rate 0%, status `"healthy"` | `pytest tests/test_optimization.py::TestComponentHealth -v` |
| 185 | Failing component (high NaN) | NaN rate 80% → status `"failing"` | same class |
| 186 | Degraded component | NaN rate 40% → status `"degraded"` | same class |
| 187 | Skips raw columns | `QED (raw)` excluded from health report | same class |
| 188 | High zero rate degraded | 80% zero scores → `"degraded"` | same class |

#### Convergence Diagnostics

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 189 | Early stage (few steps) | `convergence_status = "early"` with < 3 steps | `pytest tests/test_optimization.py::TestConvergenceDiagnostics -v` |
| 190 | Improving trajectory | Upward trend → `convergence_status = "improving"`, positive improvement_rate | same class |
| 191 | Plateau detection | Flat tail → `plateau_detected = true`, `convergence_status = "plateau"` | same class |
| 192 | Diverging trajectory | Declining scores → `diverging = true`, `convergence_status = "diverging"` | same class |
| 193 | Completed job | `convergence_status = "completed"` for COMPLETED/STOPPED jobs | same class |
| 194 | Trend computed | `early_mean < recent_mean`, positive `overall_improvement` | same class |
| 195 | Progress percentage | 50 of 100 steps → `progress_pct = 50.0` | same class |
| 196 | Component health included | Healthy QED + failing Boltz detected | same class |
| 197 | Recent validity rate | Validity rate computed over recent window | same class |
| 198 | None when no total col | Returns None | same class |
| 199 | None when empty df | Returns None | same class |

#### Warnings and Recommendations

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 200 | Plateau warning | `type = "score_plateau"` generated | `pytest tests/test_optimization.py::TestWarningsAndRecommendations -v` |
| 201 | Diverging warning | `type = "diverging"`, severity `"high"` | same class |
| 202 | Component failure warning | `type = "component_failure"` with component name | same class |
| 203 | Low validity warning | `type = "low_validity"`, severity based on rate | same class |
| 204 | Plateau recommendation | Suggests stopping and adjusting TTP | same class |
| 205 | Diverging recommendation | Suggests lowering learning rate / increasing sigma | same class |
| 206 | Failing component recommendation | Lists failing components, suggests checking services | same class |
| 207 | Improving near end | Suggests letting the run finish | same class |
| 208 | Early recommendation | Suggests continuing to monitor | same class |

#### Integration: get_status with Diagnostics

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 209 | Status includes diagnostics | Full `get_status` includes `diagnostics` with convergence, health, trend, progress | `pytest tests/test_optimization.py::TestGetStatusWithDiagnostics -v` |
| 210 | Per-component trajectory | `per_component_trajectory` with named components | same class |
| 211 | Early run no crash | Status with 2 steps returns diagnostics without error | same class |
| 212 | Failing component detection | Boltz with 87.5% NaN flagged as `"failing"` in diagnostics | same class |

#### JobInfo

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 213 | CSV path | `results_1.csv` in working dir | `pytest tests/test_optimization.py::TestJobInfo -v` |
| 214 | Checkpoint path | `stage.chkpt` in working dir | same class |
| 215 | to_dict | Correct serialization of all fields | same class |

#### JobManager Lifecycle

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 216 | Create and start job | TOML and metadata files written, correct TOML content | `pytest tests/test_optimization.py::TestJobManagerLifecycle -v` |
| 217 | Max concurrent enforced | Exceeding limit returns error with running job IDs | same class |
| 218 | Unknown job error | `get_status("nonexistent")` returns error | same class |
| 219 | Status with CSV | Trajectory parsed: step count, molecule count, validity rate | same class |
| 220 | List jobs | Returns all tracked jobs | same class |

#### JobManager Stop

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 221 | Stop non-running job | Returns error with current state | `pytest tests/test_optimization.py::TestJobManagerStop -v` |

#### JobManager Top Candidates

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 222 | Top candidates | Ranked by score, deduplicated, with component breakdowns | `pytest tests/test_optimization.py::TestJobManagerTopCandidates -v` |
| 223 | With min_score | Only candidates above threshold returned | same class |
| 224 | No CSV returns error | Error when results CSV doesn't exist | same class |

#### JobManager Checkpoint Resume

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 225 | Resume from checkpoint | `use_checkpoint = true` and agent_file set in TOML | `pytest tests/test_optimization.py::TestJobManagerResume -v` |
| 226 | Nonexistent checkpoint | Returns error | same class |

#### JobManager Metadata Persistence

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 227 | Metadata written on create | `job_metadata.json` exists with correct fields | `pytest tests/test_optimization.py::TestJobMetadataPersistence -v` |
| 228 | Restore jobs on init | Jobs restored from disk metadata on server restart | same class |
| 229 | Running → failed on restore | Jobs running at server crash time marked as failed | same class |

#### Tools: run_optimization

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 230 | Run with stored TTP | TTP resolved, scorer composed, job launched | `pytest tests/test_optimization.py::TestRunOptimizationTool -v` |
| 231 | Bad TTP returns error | Nonexistent TTP ID → error | same class |
| 232 | Empty scorer config | TTP with no components → error | same class |

#### Tools: stop_optimization

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 233 | Stop delegates to manager | Calls `manager.stop_job()` | `pytest tests/test_optimization.py::TestStopOptimizationTool -v` |

#### Tools: get_optimization_status

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 234 | Status delegates to manager | Calls `manager.get_status()` | `pytest tests/test_optimization.py::TestGetOptimizationStatusTool -v` |

#### Tools: get_top_candidates

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 235 | From job_id | Delegates to manager | `pytest tests/test_optimization.py::TestGetTopCandidatesTool -v` |
| 236 | From csv_path | Parses CSV directly with component breakdowns | same class |
| 237 | Both job_id and csv_path | Returns error (ambiguous) | same class |

#### Tools: read_results

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 238 | Read results | n_rows, n_unique_smiles, validity_rate, total_score_stats | `pytest tests/test_optimization.py::TestReadResultsTool -v` |
| 239 | Missing CSV | Returns error | same class |

#### Transfer Learning: JobManager

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 240 | Create TL job | SMILES file and TOML written, `run_type = "transfer_learning"` | `pytest tests/test_optimization.py::TestJobManagerTL -v` |
| 241 | TL from file | SMILES file path embedded in TOML | same class |
| 242 | TL requires SMILES | Error when neither `smiles` nor `smiles_file` provided | same class |
| 243 | TL rejects both inputs | Error when both `smiles` and `smiles_file` provided | same class |
| 244 | TL with validation SMILES | Validation file written and referenced in TOML | same class |
| 245 | TL max concurrent enforced | Respects `max_concurrent_jobs` limit. **Failed until 2026-10-08**: the cap was dead config and starting a job stopped every running one instead | same class |
| 246 | TL job to_dict | `job_type = "transfer_learning"`, `num_epochs`, `output_model_file` present | same class |
| 247 | TL metadata persistence | TL job restored correctly from disk on restart | same class |

#### Job Concurrency Slots

`max_concurrent_jobs` was defined in config and never read: `create_and_start`
stopped every running job first, so one RL run was the hard ceiling. An RL step
is mostly spent waiting on the scoring HTTP call, and placement/queueing is now
the boltz gateway's job, so several runs share a GPU.

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 314 | Jobs run concurrently | A second job starts while the first keeps running, untouched and un-checkpointed | `pytest tests/test_optimization.py::TestJobManagerLifecycle -v` |
| 315 | Cap enforced | The job past the limit is refused with "Maximum concurrent jobs", for RL and TL alike — they share the GPU so they share the slots | same class |
| 316 | Queued holds a slot | A job registered but not yet reporting `running` still counts, closing the window where two callers both pass the check | same class |
| 317 | Atomic claim | Four callers racing on two slots: exactly two start. The count and the registration share one lock acquisition | same class |
| 318 | Slots free up | A completed, failed, or stopped job releases its slot | same class |
| 319 | Refusal is actionable | The error lists the active job ids to stop, and names the setting to raise | same class |
| 320 | No orphan on refusal | A refused job leaves no working directory behind | same class |
| 321 | Restart frees slots | Both `queued` and `running` jobs are reset on restore, so an interrupted job cannot hold a slot forever | same class |

#### Tools: run_transfer_learning

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 248 | Run with SMILES list | Delegates to `manager.create_and_start_tl` | `pytest tests/test_optimization.py::TestRunTransferLearningTool -v` |
| 249 | Run with SMILES file | `smiles_file` forwarded, `smiles` is None | same class |
| 250 | No input returns error | Error when no SMILES provided | same class |

#### get_top_candidates: Component Breakdowns (CSV mode)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 251 | CSV with components | Per-component transformed + raw scores in output | `pytest tests/test_optimization.py::TestGetTopCandidatesComponentBreakdown -v` |
| 252 | CSV without raw columns | Transformed scores only, no raw key | same class |

### Scoring Latency Harvest (`tests/test_scoring_latency.py`)

Tests for `k8s/scripts/analyze-scoring-latency.py`, which reads
`external_scoring.jsonl` back to recover how long a Boltz batch takes and how
often molecules repeat. Records are synthesized in the shape
`ExternalRESTScorer._log_call` writes, so no cluster, log file, or GPU is
needed.

**Dependencies**: none beyond `pytest`

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 257 | Record parsing | Batch size recovered from `request.smiles`; a 2xx with no `error` is usable | `pytest tests/test_scoring_latency.py::TestRecordParsing -v` |
| 258 | Failure classification | `HTTP 500` → `http_500`; a capped `request_failed` → `timeout`; missing `elapsed_seconds` is unusable | same class |
| 259 | Cache-key digest | Predict params change the digest; per-call `structure_paths` do not | same class |
| 260 | Timeouts excluded | Nine 600s calls plus nine capped 1800s timeouts leave p95 at 600s | `pytest tests/test_scoring_latency.py::TestFailureExclusion -v` |
| 261 | Latency percentiles | Linear interpolation; per-molecule latency is `elapsed_seconds / batch size` | `pytest tests/test_scoring_latency.py::TestLatencyStats -v` |
| 262 | Per-batch-size split | Separate p50 per batch size; the modal size breaks ties towards the larger batch | same class |
| 263 | Cache hit rate | Repeated SMILES under one protein sequence are hits; the same SMILES under another target is a miss | `pytest tests/test_scoring_latency.py::TestCacheHitRate -v` |
| 264 | Full cache key | Differing `recycling_steps` turns a repeat into two misses | same class |
| 265 | Fleet sizing | A 10-minute batch against a 5-minute step gives `M = 2N`; a hit rate lowers `M`; 100 steps is 16.7 h on a dedicated replica | `pytest tests/test_scoring_latency.py::TestFleetSizing -v` |
| 266 | Assumption warnings | Too-short window → shrink the task unit; a slow CPU scorer → "not negligible"; mixed targets flagged | `pytest tests/test_scoring_latency.py::TestWarnings -v` |
| 267 | CLI | `--json`, `--service`, `--since`, `--batch-size`, directory input, and unparseable lines | `pytest tests/test_scoring_latency.py::TestCLI -v` |
| 268 | No usable records | An empty or all-failed log exits 2 and prints the single-call fallback | same class |

### Boltz Kernel A/B Harness (`tests/test_boltz_kernels_ab.py`)

Tests for `k8s/scripts/ab-boltz-kernels.py`, which times one `/score` call with
and without Boltz's optimized kernels. The molecule panel is checked against
the wrapper's own validator, so neither arm can silently score a smaller batch.
`kubectl` is stubbed and no arm is run, so no cluster, service, or GPU is
needed.

**Dependencies**: `rdkit`, `pydantic` (both already required to import the wrapper)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 269 | Panel validity | 16 unique molecules, each accepted by `_validate_smiles`, none dropped by `_partition_smiles` | `pytest tests/test_boltz_kernels_ab.py::TestPanel -v` |
| 270 | BRD4 BD1 sequence | The baked-in sequence is foldable and matches the query record of `brd4_bd1.a3m` (skipped when that gitignored file is absent) | same class |
| 271 | Request shape | Carries `msa_path` and no `use_msa_server`, keeps production affinity settings, and validates as a `ScoreRequest` | `pytest tests/test_boltz_kernels_ab.py::TestPayload -v` |
| 272 | Replica arithmetic | A 10-minute batch against a 5-minute step gives `M = 2N`; counts round up | `pytest tests/test_boltz_kernels_ab.py::TestUrlsAndMath -v` |
| 273 | Deployment env | `BOLTZ_NO_KERNELS` read from the right container; an originally unset value is restored by removing it | `pytest tests/test_boltz_kernels_ab.py::TestDeploymentEnv -v` |
| 274 | CLI guards | A two-arm run without a restart mechanism, conflicting arm flags, and a batch larger than the panel are all refused | `pytest tests/test_boltz_kernels_ab.py::TestCLIGuards -v` |

### Kubernetes Manifests (`tests/test_k8s_manifests.py`)

Tests for the split manifests under `k8s/manifests/`. The six Compose
containers were one pod with loopback scorer URLs; they are now one Deployment
plus one ClusterIP Service each. These parse the YAML directly, so no cluster,
`kubectl`, or GPU is needed.

**Dependencies**: `PyYAML`

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 275 | Pod split | Each of the five scorers has its own Deployment, every Deployment runs exactly one container, and `dd-agentic` carries only `dd-agent` | `pytest tests/test_k8s_manifests.py::TestPodSplit -v` |
| 276 | GPU accounting | Still two `amd.com/gpu` in total, and every GPU Deployment uses `Recreate` so a surge pod cannot ask for a third | same class |
| 277 | Boltz replica cap | `dd-agentic-boltz` stays at one replica while a plain Service would round-robin onto a busy replica (§5.2) | same class |
| 278 | Scorer Services | Each scorer has a ClusterIP Service whose selector matches its own pods and no one else's, on the container's named port | `pytest tests/test_k8s_manifests.py::TestScorerServices -v` |
| 279 | Cluster DNS URLs | No `127.0.0.1`/`localhost` anywhere in the manifests; every `service_url` resolves to a Service in the set, on the right port, unqualified so the namespace stays free | `pytest tests/test_k8s_manifests.py::TestConfigMapServiceUrls -v` |
| 280 | Node pinning | `apply.sh` patches `nodeSelector[kubernetes.io/hostname]` onto Deployments with no name or label restriction, so future workloads are pinned too | `pytest tests/test_k8s_manifests.py::TestSharedVolumePinning -v` |
| 281 | Shared volume | The `/shared` users are exactly the agent, Boltz, PoseBusters, and the interaction profiler, all mounting the one claim at the same path; ADMET and RetroDMTA mount nothing | same class |
| 282 | Never `ReadWriteOncePod` | The claim stays `ReadWriteOnce`, which is per node and admits the whole pinned fleet | same class |
| 283 | `/shared` is writable | No `/shared` mount is `readOnly`: PoseBusters extracts the ligand *beside* the PDB Boltz handed it, and a read-only mount fails that exactly like a missing file | same class |
| 284 | Kustomization | Every manifest file is a listed resource, images are remapped by tag and never pulled from a registry, and the `part-of` label `apply.sh` selects on is stamped | `pytest tests/test_k8s_manifests.py::TestKustomization -v` |
| 285 | Script targets | `ab-boltz-kernels.py` defaults to `dd-agentic-boltz`/`boltz`, and `uat-scoring.sh` and `switch-aim-model.py` still target `dd-agentic`/`dd-agent` | `pytest tests/test_k8s_manifests.py::TestScriptTargets -v` |
| 286 | Handoff harness targets | Every Deployment/container pair `verify-artifact-handoff.py` execs into exists and is one of the three that mount `/shared` | same class |
| 287 | Gateway placement | `balance leastconn` plus `maxconn 1` per replica, `http-reuse never` so an idle connection cannot hold the slot, and `BOLTZ_MAX_CONCURRENCY=1` on the replicas to match | `pytest tests/test_k8s_manifests.py::TestBoltzGateway -v` |
| 288 | Gateway timeouts | `client`, `server`, and `queue` all exceed `PREDICT_TIMEOUT_SECONDS`, and a connection failure redispatches instead of becoming a NaN batch | same class |
| 289 | Gateway discovery | Replicas come from the headless Service by DNS with `init-addr none`, enough slots for every replica, and no cluster DNS address or namespace baked in | same class |
| 290 | Gateway shape | Exactly one gateway (one view of who is busy), no GPU, no `/shared`, readiness that fails with no healthy replica but liveness that does not, and 10 min grace so a live batch is not killed | same class |
| 291 | Several replicas | More than one Boltz replica is only allowed when the ConfigMap dials the gateway; GPU demand is the agent plus the replicas; no GPU Deployment surges | `pytest tests/test_k8s_manifests.py::TestPodSplit -v` |

### Boltz Gateway (`tests/test_boltz_gateway.py`)

Tests for `services/boltz-gateway/app.py`: the per-molecule result cache, batch
coalescing, in-flight de-duplication, `/stats`, and the `202`+poll contract. The
upstream Boltz call is stubbed and the cache points at a `tmp_path`, so no GPU,
cluster, or service is needed.

**Dependencies**: `fastapi`, `pydantic`, `rdkit`

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 298 | Cache keying | The same molecule spelled differently shares a key (`randomize_smiles` makes that the common case); target and predict params do not; policy flags and per-call plumbing are excluded | `pytest tests/test_boltz_gateway.py::TestCacheKeying -v` |
| 299 | MSA keyed by content | Two paths holding one alignment share the whole key, editing an `.a3m` invalidates it, and the path is still forwarded so Boltz can find it | same class |
| 300 | Cache behaviour | A repeat batch predicts nothing, a randomised spelling still hits, a duplicate inside one batch is predicted once, and a partly cached batch only predicts the rest | `pytest tests/test_boltz_gateway.py::TestCacheBehaviour -v` |
| 301 | Tiers | `validation` bypasses the read but still populates; `cache_policy: none` does neither | same class |
| 302 | Structures survive a hit | The cached structure path comes back, and an entry naming a deleted file is dropped rather than served — downstream scorers re-open these by path | same class |
| 303 | What is cacheable | A deterministic skip (too many heavy atoms) is kept; a failed prediction never is, so one bad run cannot poison a molecule | same class |
| 304 | Coalescing | Concurrent batches share one upstream call; different targets or fidelities are never merged; a full batch dispatches without waiting and an oversized one splits | `pytest tests/test_boltz_gateway.py::TestCoalescing -v` |
| 305 | In-flight de-duplication | Three concurrent requests for one molecule cost one prediction — the cross-supervisor saving caching alone cannot give | same class |
| 306 | Response shape | Scores, per-molecule metadata and structures all come back in the caller's order and the right length; an upstream failure is a 502 and leaves nothing cached | `pytest tests/test_boltz_gateway.py::TestResponseShape -v` |
| 307 | Async contract | `202` + `task_id`, `202` while pending, `200` once, then `404`; a failure surfaces as 502 | `pytest tests/test_boltz_gateway.py::TestAsyncApi -v` |
| 308 | Stats | Hit rate, molecules per upstream call, latency percentiles, queue depth and duplicate/invalid counts | `pytest tests/test_boltz_gateway.py::TestStats -v` |
| 309 | Replica view | Healthy replicas and the proxy's **backend** queue row; an unreachable proxy still allows one call rather than wedging | `pytest tests/test_boltz_gateway.py::TestReplicaView -v` |
| 310 | Drop-in contract | The same request body validates against both the gateway and the Boltz wrapper, and the response has the three keys the scorer reads | `pytest tests/test_boltz_gateway.py::TestContract -v` |

### Scorer Timeout and Polling (`tests/test_external_scoring.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 311 | Configurable timeout | Default unchanged at 1800 s; component config beats `EXTERNAL_SCORER_TIMEOUT_SECONDS`; a nonsense env value falls back instead of disabling the timeout | `pytest tests/test_external_scoring.py::TestTimeoutConfiguration -v` |
| 312 | Submit then poll | `?mode=async`, the result URL derived from the score URL, backoff between polls, and the overall deadline still bounding it | `pytest tests/test_external_scoring.py::TestSubmitThenPoll -v` |
| 313 | Async misconfiguration is visible | A service that answers the submit normally instead of 202 is honoured as-is; a 202 with no `task_id` fails rather than hanging | same class |

### REINVENT Runner (`tests/test_reinvent_runner.py`)

Tests for `services/reinvent-runner/app.py` and the client that makes it a
drop-in for the local `JobManager`. REINVENT4 is not installed in the test
environment, so the manager and tool functions are patched out — which is also
the point: nothing above the seam should need them.

**Dependencies**: `fastapi`, `requests`, `PyYAML`

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 322 | Endpoints pass through | `/jobs`, `/tl-jobs`, status, stop, candidates, lists and the registries each delegate to the code that already exists, with `extra="forbid"` so a stale client cannot silently drop a parameter | `pytest tests/test_reinvent_runner.py::TestRunnerEndpoints -v` |
| 323 | Refusals stay refusals | "Maximum concurrent jobs" comes back as `200` with an error dict, because that is the manager's own contract and preserving it keeps the client a drop-in; only unexpected failures are 5xx | same class |
| 324 | Health names the real risk | Reports `reinvent_importable`, since a false there means the image is wrong rather than the service being down | same class |
| 325 | TTPs stay agent-side | `/score` takes the TTP inline and has no `ttp_id` parameter at all | same class |
| 326 | The client is a drop-in | Offers exactly the seven methods callers use, all of which exist on the local manager too | `pytest tests/test_reinvent_runner.py::TestRemoteJobManager -v` |
| 327 | Failures keep their shape | An unreachable runner, a 5xx, and an unparseable body all become `{"error": ...}` rather than raising — callers would crash on a raise | same class |
| 328 | List methods degrade safely | `list_jobs`/`get_alerts` return `[]` on failure, since an error dict would break every caller that iterates | same class |
| 329 | Job ids are URL-encoded | A path-traversal-shaped id cannot escape its endpoint | same class |
| 330 | The seam decides | `get_job_manager()` returns the local manager by default and the remote one when `runner.enabled`, with no restart needed; discovery and scoring follow the same switch | `pytest tests/test_reinvent_runner.py::TestSeam -v` |
| 331 | A ttp_id is resolved first | Sent inline to the runner; an unresolvable id never reaches it | same class |
| 332 | The runner never proxies to itself | `REINVENT_RUNNER_SELF` forces the local path, is set in both the image and the manifest, and the job endpoints hold their own manager so they cannot loop regardless | same class |

#### Image Split

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 333 | Agent image is slim | No `COPY REINVENT4/`, no Zenodo download, no `REINVENT4_ROOT`, and not built on `rocm/pytorch` — 164 MB against 40.8 GB unpacked | `pytest tests/test_reinvent_runner.py::TestImageSplit -v` |
| 334 | It stays slim | The build asserts `reinvent` and `torch` are **absent**, so a transitive dependency pulling them back fails the build instead of restoring a 40 GB image | same class |
| 335 | Runner owns the heavy half | REINVENT4, the priors and the ROCm base are there, `externalrestscorer` is still verified as registered, and `reinvent4_agent` is not installed | same class |
| 336 | No longer FROM dd-agent | The runner builds from its own base with the repo root as context, since it copies `REINVENT4/` and `reinvent4_mcp/` | same class |
| 337 | Agent needs no GPU | `amd.com/gpu` is absent from the agent Deployment's requests and limits; only Boltz and the runner ask for one | same class, plus `pytest tests/test_k8s_manifests.py::TestPodSplit -v` |
| 338 | The runner is mandatory | Both the ConfigMap and the baked `reinvent4_mcp_config.yaml` enable it, because a slim agent has no local REINVENT4 to fall back on | same class |
| 339 | Compose moved the GPU too | The agent service drops the ROCm devices and the 128 g shm and waits on the runner, which has them; the runner still has no `ttps` mount | same class |

### Boltz Placement Harness (`tests/test_boltz_placement.py`)

Tests for `k8s/scripts/verify-boltz-placement.py`, which proves concurrent
batches land on idle replicas instead of stacking on a busy one. The stats
endpoint and the HTTP calls are stubbed, so no cluster, service, or GPU is
needed.

**Dependencies**: `PyYAML`, `rdkit` (via the Boltz wrapper the payload is
validated against)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 292 | Stats parsing | Columns indexed by name, only this backend's rows, and unresolved slots excluded from "ready" | `pytest tests/test_boltz_placement.py::TestStatsParsing -v` |
| 293 | Backend queue row | An unplaceable request waits in the **backend** queue, so per-server `qcur` stays 0 — reading only server rows reports "nothing queued" while a batch is queued | same class |
| 294 | Verdict | Doubling up, an idle replica left unused, and a 503 queue timeout are each reported as their own failure; a missing replica header is inconclusive | `pytest tests/test_boltz_placement.py::TestVerdict -v` |
| 295 | Request shape | Boltz accepts the payload, the panel molecules all fold and are distinct so a cache cannot answer them, and the defaults are cheaper than production | `pytest tests/test_boltz_placement.py::TestRequest -v` |
| 296 | Harness vs manifest | Backend name, both ports, and the `X-Boltz-Replica` header match the deployed gateway, and the client timeout outlasts the gateway's queue | `pytest tests/test_boltz_placement.py::TestAgainstTheManifest -v` |
| 297 | CLI | Defaults to one more request than replicas so the queue is exercised; refuses zero requests, or more than the panel holds | `pytest tests/test_boltz_placement.py::TestCLI -v` |

### Cross-Pod Artifact Handoff (`tests/test_artifact_handoff.py`)

Tests for the handoff itself and for
`k8s/scripts/verify-artifact-handoff.py`, which proves it against a live
cluster. Boltz hands the consumers an absolute path on `/shared` and never the
structure, so the split only works if the consumer pods can open the
producer's files. kubectl and the HTTP calls are stubbed, so no cluster,
service, or GPU is needed.

**Dependencies**: `PyYAML`, `pydantic`, `rdkit`, `pandas` (the last two come
with the service modules under test)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 287 | Path portability | A relative path, or one outside `/shared`, is refused: neither means the same thing in two pods | `pytest tests/test_artifact_handoff.py::TestPathAudit -v` |
| 288 | Index alignment | A short list, a scored molecule with no structure, and a structure sitting where Boltz skipped one are all reported — the last is the off-by-one that pairs a pose with the wrong SMILES | same class |
| 289 | Payload parity | The payloads the harness relays are byte-identical to what `ExternalRESTScorer` injects, holes included rather than compacted | `pytest tests/test_artifact_handoff.py::TestRelayedPayloads -v` |
| 290 | Consumers accept them | PoseBusters' and the interaction profiler's own models validate the index-aligned list with `None` holes; Boltz is asked for `pdb`, since PoseBusters reads HETATM records | same class |
| 291 | Deliberate hole | The panel passes Boltz's validator, the 60-heavy-atom molecule is skipped for an explicit size reason, and it is still a parseable SMILES so the profiler fails it for the right reason | `pytest tests/test_artifact_handoff.py::TestBatch -v` |
| 292 | Extracted-ligand path | `derived_sdf_path` matches where `_boltz_pdb_to_sdf` really writes, which is inside the producer's own output directory | `pytest tests/test_artifact_handoff.py::TestDerivedSdfPath -v` |
| 293 | Failure classification | 200-with-nulls is a storage failure, a 4xx is a config one, a 5xx is neither, and a scored hole is misalignment | `pytest tests/test_artifact_handoff.py::TestConsumerClassification -v` |
| 294 | Verdict | A blind consumer or an unportable path disproves it; nothing produced or an unreachable service is inconclusive; an invisible `_lig.sdf` means the pods resolved one path on two filesystems | `pytest tests/test_artifact_handoff.py::TestVerdict -v` |
| 295 | Same-pod guard | A green run where producer and consumer share a pod exits 2, so a re-run of the in-pod proof cannot pass as the cross-pod one | same class |
| 296 | Findings | Each failure names its next step — the object-storage fallback, the config fix, or `--namespace` | `pytest tests/test_artifact_handoff.py::TestFindings -v` |
| 297 | kubectl probes | The pod is resolved from the `name` label and only when `Running`; a missing kubectl degrades to "unknown"; a zero-byte SDF counts as missing | `pytest tests/test_artifact_handoff.py::TestKubectlProbes -v` |
| 298 | Health gate | An unreachable service, or a profiler reporting `shared_dir_exists: false`, stops the run before the GPU call | `pytest tests/test_artifact_handoff.py::TestHealthGate -v` |
| 299 | Endpoints | The cluster URLs are the ConfigMap's `service_url`s and the documented port-forwards map the same ports; replay mode refuses a structure without its SMILES | `pytest tests/test_artifact_handoff.py::TestEndpointsAndCli -v` |
| 300 | End to end | `main` drives producer → consumers → verdict: three services dialled once each, the paths relayed verbatim, and exit 0/1/2 for proven/disproven/inconclusive | `pytest tests/test_artifact_handoff.py::TestEndToEnd -v` |

### Run All Unit Tests

```bash
python -m pytest tests/ -v
```

Expected: all 650+ tests pass in a few seconds.

---

## Tier 2: Integration Tests (require REINVENT4)

These tests use the real REINVENT4 scoring engine. Run them on a machine where REINVENT4 is installed (or `REINVENT4_ROOT` is set).

### Prerequisites

```bash
export REINVENT4_ROOT=/path/to/REINVENT4
# Verify:
python -c "import sys; sys.path.insert(0, '$REINVENT4_ROOT'); import reinvent; print('OK')"
```

### Config Loading

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 1 | Default config loads | `python -c "from reinvent4_mcp.config import load_config; c = load_config(); print(c.reinvent.device)"` | Prints `cuda:0` (or your device) |
| 2 | Env var config path | `REINVENT4_MCP_CONFIG=reinvent4_mcp_config.yaml python -c "from reinvent4_mcp.config import load_config; c = load_config(); print(c.scoring.aggregation)"` | Prints `geometric_mean` |
| 3 | Invalid config rejected | Create a YAML with `reinvent: {batch_size: -1}` and load it | `ValidationError` raised |
| 4 | Unknown key rejected | Create a YAML with `reinvent: {unknown_field: true}` and load it | `ValidationError` raised |

### Discovery Tools

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 5 | List scoring components | `python -c "from reinvent4_mcp.tools.discovery import list_scoring_components; cs = list_scoring_components(); print(f'{len(cs)} components'); print([c['name'] for c in cs[:5]])"` | Lists 40+ components including `QED`, `MolecularWeight`, `SAScore` |
| 6 | List available models | `python -c "from reinvent4_mcp.tools.discovery import list_available_models; ms = list_available_models(); print([(m['key'], m['exists']) for m in ms])"` | Lists model keys with `exists` status |
| 7 | List transforms | `python -c "from reinvent4_mcp.tools.discovery import list_transforms; ts = list_transforms(); print(f'{len(ts)} transforms'); print([t['name'] for t in ts])"` | Lists transform types (`sigmoid`, `reverse_sigmoid`, etc.) |

### Direct Scoring (Real REINVENT4)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 8 | Score with QED | `python -c "from reinvent4_mcp.tools.scoring import score_molecules; r = score_molecules(['CCO', 'c1ccccc1', 'CC(=O)OC1=CC=CC=C1C(=O)O'], components=[{'type': 'QED'}]); print([(m['smiles'], m['total_score']) for m in r['molecules']])"` | Three molecules with QED scores between 0 and 1 |
| 9 | Score with multiple components | `python -c "from reinvent4_mcp.tools.scoring import score_molecules; r = score_molecules(['CC(=O)OC1=CC=CC=C1C(=O)O'], components=[{'type': 'QED', 'weight': 0.5}, {'type': 'MolecularWeight', 'weight': 0.5, 'transform': {'type': 'double_sigmoid', 'high': 500, 'low': 200, 'coef_div': 500, 'coef_si': 20, 'coef_se': 20}}]); print(r['molecules'][0])"` | Aspirin scored with both QED and MW, total score is aggregated |
| 10 | Score with invalid SMILES | `python -c "from reinvent4_mcp.tools.scoring import score_molecules; r = score_molecules(['not_a_smiles'], components=[{'type': 'QED'}]); print(r)"` | Returns result (may show NaN for invalid SMILES) |
| 11 | Score with SAScore | `python -c "from reinvent4_mcp.tools.scoring import score_molecules; r = score_molecules(['c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34'], components=[{'type': 'SAScore'}]); print(r['molecules'][0])"` | Returns SA score for a complex molecule |
| 12 | Score with invalid component | `python -c "from reinvent4_mcp.tools.scoring import score_molecules; r = score_molecules(['CCO'], components=[{'type': 'NonExistentComponent'}]); print(r)"` | Returns `{"error": "Failed to create scorer: ..."}` |

### TTP-Based Scoring (Real REINVENT4)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 13 | Score with inline TTP (native only) | `python -c "from reinvent4_mcp.tools.scoring import score_molecules; ttp = {'id': 'test', 'name': 'Test', 'scoring': {'objectives': [{'name': 'QED', 'type': 'native', 'component': 'QED'}, {'name': 'SA', 'type': 'native', 'component': 'SAScore'}]}}; r = score_molecules(['CCO', 'c1ccccc1'], ttp=ttp); print(r['molecules'])"` | Two molecules with QED and SA scores |
| 14 | Score with TTP + pre-scoring | Same as above but add a `pre_scoring` section with `MolecularWeight` | Pre-scoring component included in results |
| 15 | TTP with transforms | Use transforms (e.g., `reverse_sigmoid` on SA score) | Transformed scores between 0-1, raw scores in original units |
| 16 | Mixed TTP notes externals | Use a TTP with both native and external objectives | `note` field lists skipped external objectives |

### TTP CRUD (Real File Persistence)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 17 | Create TTP | `python -c "from reinvent4_mcp.tools.ttp import create_ttp, reset_store; reset_store(); r = create_ttp({'id': 'test-manual', 'name': 'Manual Test', 'scoring': {'objectives': [{'name': 'SA', 'type': 'native', 'component': 'SAScore'}]}}); print(r)"` | `{"status": "created", "id": "test-manual", "version": 1, ...}` |
| 18 | Get TTP | After creating, run `get_ttp("test-manual")` | Returns full TTP JSON |
| 19 | Update TTP | `update_ttp("test-manual", {"name": "Updated"})` | Version incremented to 2 |
| 20 | List TTPs | `list_ttps()` | Shows all stored TTPs |
| 21 | Delete TTP | `delete_ttp("test-manual")` | `{"status": "deleted"}` |

### External Service Discovery (requires network)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 22 | List external services (none running) | `python -c "from reinvent4_mcp.tools.external import list_external_services; print(list_external_services())"` | All configured services report `"unreachable"` |
| 23 | List external services (Boltz running) | Start Boltz service first, then run the above | Boltz reports `"ok"`, PoseBusters `"unreachable"` |

### TTP-Based Scoring with External Objectives (requires REINVENT4 + services)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 24 | Score mixed TTP (externals down) | `python -c "from reinvent4_mcp.tools.scoring import score_molecules; ttp = {'id': 'test', 'name': 'Test', 'target': {'protein_sequence': 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH'}, 'scoring': {'objectives': [{'name': 'QED', 'type': 'native', 'component': 'QED'}, {'name': 'Binding', 'type': 'external', 'service': 'boltz', 'service_url': 'http://localhost:8091/score'}]}}; r = score_molecules(['CCO'], ttp=ttp); print(r.get('note', 'no note'))"` | Returns native scores + `note` listing skipped externals |
| 25 | Score mixed TTP (externals up) | Same TTP but with Boltz + PoseBusters running | All objectives scored, no `note` field, artifacts present |

### RL Optimization (requires REINVENT4 + GPU)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 26 | Run optimization with native TTP | Create a TTP with QED + SA, call `run_optimization(ttp_id=..., max_steps=10, batch_size=4)` | Returns `job_id`, TOML config written, REINVENT4 subprocess starts |
| 27 | Get status during run | Call `get_optimization_status(job_id)` while step < max_steps | `state: "running"`, `trajectory` with current_step, mean_total_score, per_component stats |
| 28 | Convergence diagnostics | Poll status after 10+ steps | `trajectory.diagnostics` present with `convergence_status`, `component_health`, `trend` |
| 29 | Stop and checkpoint | Call `stop_optimization(job_id)` | `state: "stopped"`, `checkpoint_path` present and file exists |
| 30 | Resume from checkpoint | Call `run_optimization(ttp_id=..., resume_from="<checkpoint_path>")` | New job starts with `use_checkpoint = true`, agent_file points to checkpoint |
| 31 | Get top candidates | After run completes: `get_top_candidates(job_id, top_n=5)` | Ranked list with SMILES, total_score, per-component breakdowns |
| 32 | Read results CSV | Call `read_results(csv_path)` on the output CSV | Summary stats: n_rows, n_unique_smiles, validity_rate, score distribution |
| 33 | List jobs | Call `list_jobs()` | All tracked jobs with states, TTP IDs, timestamps |

### Transfer Learning (requires REINVENT4 + GPU)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 34 | Run TL with inline SMILES | `run_transfer_learning(smiles=["CCO", "c1ccccc1", ...], num_epochs=5)` | Job starts, SMILES written to `train.smi`, TOML has `run_type = "transfer_learning"` |
| 35 | Run TL with file | Write SMILES to a file, call `run_transfer_learning(smiles_file="path.smi")` | Job starts, TOML references the provided file path |
| 36 | TL with validation data | Provide `validation_smiles=[...]` | Validation file written, TOML includes `validation_smiles_file` |
| 37 | Monitor TL progress | Poll `get_optimization_status(job_id)` | State transitions from `running` to `completed` |
| 38 | TL output model | After completion: check `output_model_file` in job dict | Model file exists in working directory, epoch checkpoints present |
| 39 | Use TL model in RL | Run `run_optimization(model="<tl_output_path>", ttp_id=...)` | RL starts with TL model as the agent |

---

## Tier 3: End-to-End Tests (full agent stack)

These tests verify the complete `dd-agent` -> `dd-mcp` -> REINVENT4 pipeline.

### Prerequisites

- REINVENT4 installed or `REINVENT4_ROOT` set
- `dd-mcp` and `dd-agent` installed (`pip install ./reinvent4_mcp ./reinvent4_agent`)
- LLM API access (API key or gateway key)

### MCP Server Smoke Test

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 1 | MCP Inspector | `npx @modelcontextprotocol/inspector dd-mcp` | Opens web UI at `localhost:6274`. Lists all 9 tools. Can call `list_scoring_components` and get results. |
| 2 | Server starts | `dd-mcp 2>server.log &` (sends JSON-RPC over stdin/stdout) | Server logs `Starting Drug Discovery MCP server v0.1.0` to stderr |

### Agent CLI Smoke Test

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 3 | Agent starts and connects | `dd-agent --backend anthropic --model claude-sonnet-4.6 --base-url https://llm-api.amd.com/Anthropic --gateway-key "$LLM_GATEWAY_KEY" --no-verify-ssl` | Prints welcome banner, lists 9 tools, shows `You>` prompt |
| 4 | Help command | Type `/help` | Shows available commands (`/help`, `/reset`, `/exit`) |
| 5 | List components | Ask: `What scoring components are available?` | Agent calls `list_scoring_components` tool, lists 40+ components |
| 6 | List models | Ask: `What generative models are available?` | Agent calls `list_available_models`, lists model priors with availability |
| 7 | Score molecules | Ask: `Score aspirin (CC(=O)OC1=CC=CC=C1C(=O)O) for QED and molecular weight` | Agent calls `score_molecules`, renders a score table |
| 8 | Multi-component scoring | Ask: `Score CCO and c1ccccc1 for QED, SA score, and LogP` | Agent composes a multi-component request, returns all scores |
| 9 | Create TTP | Ask: `Create a TTP called "test-ttp" for lead optimization with QED and SA score objectives` | Agent calls `create_ttp`, confirms creation with ID and version |
| 10 | Score with TTP | Ask: `Score aspirin using the test-ttp profile` | Agent calls `score_molecules` with `ttp_id`, returns TTP-based scores |
| 11 | Update TTP | Ask: `Update test-ttp to add molecular weight with weight 0.3` | Agent calls `update_ttp`, confirms version 2 |
| 12 | Score with updated TTP | Score again after update | Now includes MW in the scoring |
| 13 | List TTPs | Ask: `What TTPs do I have?` | Agent calls `list_ttps`, shows stored profiles |
| 14 | Delete TTP | Ask: `Delete the test-ttp profile` | Agent calls `delete_ttp`, confirms deletion |
| 15 | Conversation reset | Type `/reset`, then ask a follow-up referencing earlier context | Agent has no memory of previous conversation |
| 16 | Exit | Type `/exit` or `/quit` | Agent exits cleanly |

### Agent with Logging

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 17 | Session logging | `dd-agent --log-dir ./test-logs ...` | After a conversation: `test-logs/session_*.log` (human-readable) and `session_*.jsonl` (structured transcript) exist |
| 18 | Transcript structure | Inspect the `.jsonl` file | Each line is valid JSON with `ts`, `turn`, `event` fields |

### Docker Deployment Test (dd-agent)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 19 | Docker build | `docker build -f services/dd-agent/Dockerfile -t dd-agent .` | Image builds successfully |
| 20 | Docker smoke test | `docker run --rm --entrypoint bash dd-agent -c 'echo $REINVENT4_ROOT && python -c "import reinvent; print(reinvent.__file__)"'` | Prints `/app/REINVENT4` and the reinvent package path |
| 21 | Docker dd-mcp check | `docker run --rm --entrypoint bash dd-agent -c 'which dd-mcp && which dd-agent'` | Both commands found |
| 22 | Docker interactive | `docker run -it --rm -e LLM_GATEWAY_KEY="$LLM_GATEWAY_KEY" dd-agent` | Agent starts inside the container |

### Docker Deployment Test (Boltz2 + PoseBusters services)

Requires Docker (with GPU support for Boltz) and the `docker-compose.yml` in the project root.

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 23 | Boltz2 image builds | `docker compose build boltz` | Image builds from `services/boltz2/Dockerfile` |
| 24 | PoseBusters image builds | `docker compose build posebusters` | Image builds from `services/posebusters/Dockerfile` |
| 25 | Boltz2 health check | `docker compose up -d boltz && sleep 30 && curl http://localhost:8091/health` | `{"status": "ok", "no_kernels": true}` |
| 26 | PoseBusters health check | `docker compose up -d posebusters && sleep 10 && curl http://localhost:8082/health` | `{"status": "ok"}` |
| 27 | Full stack up | `docker compose up -d` | All three services (`reinvent4-mcp`, `boltz`, `posebusters`) healthy |
| 28 | Boltz2 score endpoint | `curl -X POST http://localhost:8091/score -H 'Content-Type: application/json' -d '{"smiles": ["CCO"], "params": {"protein_sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH", "use_msa_server": true}}'` | JSON with `scores`, `metadata`, `artifacts` |
| 29 | PoseBusters score endpoint | `curl -X POST http://localhost:8082/score -H 'Content-Type: application/json' -d '{"smiles": ["CCO"], "params": {"mode": "mol", "structure_paths": ["/shared/predictions/mol_0/mol_0_model_0.pdb"]}}'` | JSON with `scores` and per-check `metadata` |
| 30 | Shared volume | `docker compose exec boltz ls /shared && docker compose exec posebusters ls /shared` | Both containers can see the shared volume |
| 31 | Boltz2 MSA validation | `curl -X POST http://localhost:8091/score -H 'Content-Type: application/json' -d '{"smiles": ["CCO"], "params": {"protein_sequence": "MVLSPADKTN"}}'` | HTTP 400 with error about missing MSA config |
| 32 | Compose down | `docker compose down -v` | All services and volumes removed cleanly |

### Agent with External Scoring (Boltz + PoseBusters)

These tests verify the agent can interact with Boltz2 and PoseBusters via the MCP tools. Requires Docker services running (`docker compose up -d`).

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 33 | Check external services | Ask: `What external scoring services are available?` | Agent calls `list_external_services`, reports Boltz and PoseBusters status |
| 34 | Create TTP with Boltz | Ask: `Create a TTP for binding affinity optimization of tyrosine kinase. Use Boltz for binding prediction and PoseBusters for pose quality validation (depends on Boltz) with protein sequence MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH and QED as a native objective.` | Agent creates TTP with both native and external objectives |
| 35 | Score with Boltz TTP | Ask: `Score aspirin using the TTP` | If Boltz is running: returns affinity scores, confidence metadata, and structure artifacts. If Boltz is down: returns native scores + note about skipped externals |
| 36 | Create pipeline TTP | Ask: `Create a TTP that uses Boltz for binding affinity and PoseBusters for pose quality validation (depends on Boltz), plus QED as native.` | TTP created with `depends_on` linking PoseBusters to Boltz |
| 37 | Score pipeline TTP | Ask: `Score CCO using the pipeline TTP` | Boltz runs first, structures passed to PoseBusters, all scores returned |

### Agent RL Optimization

These tests verify the agent can run RL optimization campaigns end-to-end. Requires REINVENT4 + GPU.

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 38 | Launch RL campaign | Ask: `Create a TTP with QED and SA score, then run an optimization with max_steps=20` | Agent creates TTP, calls `run_optimization`, returns job_id |
| 39 | Monitor running job | Ask: `What's the status of job <id>?` | Agent calls `get_optimization_status`, reports step count, scores, convergence status |
| 40 | Interpret diagnostics | Wait for 10+ steps, then ask: `How is the optimization going?` | Agent interprets diagnostics, discusses convergence and component health |
| 41 | Stop and adjust | Ask: `Stop the optimization and reduce the SA weight to 0.05` | Agent calls `stop_optimization`, then `update_ttp`, reports checkpoint saved |
| 42 | Resume from checkpoint | Ask: `Resume the optimization with the updated TTP` | Agent calls `run_optimization` with `resume_from` pointing to checkpoint |
| 43 | Extract results | After completion, ask: `Show me the top 10 candidates` | Agent calls `get_top_candidates`, displays molecules with component scores |
| 44 | List all jobs | Ask: `What optimization jobs have I run?` | Agent calls `list_jobs`, lists all tracked jobs with states |

### Agent RL Optimization with External Scoring (Boltz2 + PoseBusters)

These tests verify the agent can run RL optimization campaigns that include Boltz2 and PoseBusters in the scoring loop. Requires REINVENT4 + GPU + Docker services running (`docker compose up -d`).

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 45 | Launch RL with full pipeline | Ask: `Create a TTP for lead optimization against tyrosine kinase with protein sequence MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH. Include QED as a native objective with weight 0.3, Boltz2 for binding affinity prediction as an external objective with weight 0.5 and use_msa_server enabled, and PoseBusters in dock mode for pose quality validation with weight 0.2 (depends on Boltz). Then run an RL optimization with max_steps=30 and batch_size=8.` | Agent creates TTP with native + external objectives and pipeline dependency, calls `run_optimization`, returns job_id. TOML config includes `ExternalRESTScorer` entries for both services. |
| 46 | Monitor external component health | After 5+ steps, ask: `How is the optimization going? Are the external scoring services working?` | Agent calls `get_optimization_status`, reports per-component health. Boltz and PoseBusters NaN rates visible in `component_health`. If services are healthy: NaN rate near 0%. If degraded: agent flags the specific failing component. |
| 47 | Interpret mixed diagnostics | After 15+ steps, ask: `Give me a detailed analysis of the optimization. How are the native vs external objectives performing?` | Agent interprets diagnostics showing convergence status, per-component trajectories for QED (native) vs Boltz (external) vs PoseBusters (external). Discusses whether molecules are improving on binding affinity alongside drug-likeness. |
| 48 | Stop, adjust weights, resume | Ask: `Stop the optimization. The Boltz binding scores look promising but PoseBusters scores are low. Increase the PoseBusters weight to 0.4 and reduce QED to 0.1, then resume.` | Agent calls `stop_optimization`, `update_ttp` with new weights, then `run_optimization` with `resume_from` pointing to checkpoint. New TOML reflects updated weights. |
| 49 | Top candidates with full breakdowns | After completion, ask: `Show me the top 10 candidates with their binding affinity, pose quality, and QED breakdowns.` | Agent calls `get_top_candidates`, displays ranked molecules with per-component scores for QED, Boltz (affinity), and PoseBusters (pose quality) including both transformed and raw values. |
| 50 | External service failure mid-run | While RL is running, stop Boltz (`docker compose stop boltz`). Then ask: `How is the optimization going?` | Agent reports Boltz component as `"failing"` with high NaN rate in diagnostics. PoseBusters also degraded (depends on Boltz for structure artifacts). Agent warns about external service issues and recommends checking service health or stopping the run. |
| 51 | Recover and resume after service restart | Restart Boltz (`docker compose start boltz`), stop the degraded run, then ask: `Resume the optimization now that Boltz is back up.` | Agent verifies services are healthy via `list_external_services`, then resumes from checkpoint. Subsequent steps show Boltz and PoseBusters NaN rates recovering to normal. |

### Agent Transfer Learning

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 52 | Run transfer learning | Ask: `Fine-tune a model on these SMILES: CCO, c1ccccc1, CC(=O)O, CCCCCC, c1ccc(O)cc1` | Agent calls `run_transfer_learning` with inline SMILES |
| 53 | Use TL model in RL | After TL completes, ask: `Now run an optimization using the fine-tuned model` | Agent uses the TL output model path in `run_optimization` |

### Slurm Deployment Test

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 54 | Setup script | `bash services/slurm/setup.sh` | Virtualenv created, all 4 packages installed, `dd-mcp` and `dd-agent` on PATH |
| 55 | Interactive session | `srun --partition=batch --cpus-per-task=4 --mem=16G --time=1:00:00 --pty bash services/slurm/run_interactive.sh` | Agent starts on compute node |
| 56 | Batch job | `echo "List all scoring components" \| sbatch services/slurm/agent.sbatch` | Job output file contains component list |

### Scorer Measurement (requires a campaign's worth of logs, or Docker + GPU)

These produce the two numbers Boltz replica counts are derived from. Neither
needs a dedicated benchmark run: the first reads history, the second is one
`/score` call per arm.

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 57 | Harvest latency and cache hit rate | `./k8s/scripts/analyze-scoring-latency.py data/logs/external_scoring.jsonl` | Per-service latency per batch and per molecule, the breakdown by batch size, the cache hit rate, and the replica counts that follow. Exit 2 when the log has no usable Boltz call |
| 58 | Measure `t_boltz_per_batch` | `docker compose up -d boltz && ./k8s/scripts/ab-boltz-kernels.py --baseline-only` | 16/16 scored against BRD4 BD1 using `brd4_bd1.a3m`, with the per-batch and per-molecule time and the replica count it implies. Measured 386 s on MI300 |
| 59 | A/B the optimized kernels | `./k8s/scripts/ab-boltz-kernels.py --restart-command 'docker compose up -d boltz'` | On ROCm the kernels arm fails with `ModuleNotFoundError: cuequivariance_torch` and the script reports keeping `BOLTZ_NO_KERNELS=1`. On a CUDA node both arms score 16/16 and a speedup is reported |

### Split Scorer Deployments (requires a cluster)

After `apply.sh`, the scorers are separate pods reached over cluster DNS rather
than loopback inside one pod.

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 60 | All six Deployments are up | `kubectl -n "${PROJECT}" get deployments,pods --selector app.kubernetes.io/part-of=dd-agentic --output wide` | Six Deployments `1/1`, every pod on the `--node` host, and two `amd.com/gpu` requested in total. Matching no pods means the `labels` transformer lost `includeTemplates` |
| 61 | Every Service has an endpoint | `kubectl -n "${PROJECT}" get endpointslices --selector app.kubernetes.io/component=scorer` | Five scorer Services, each with exactly one ready address, on ports 8091/8082/8083/8084/8085 |
| 62 | Scorers resolve from the agent pod | The one-liner under [Check scorer health](k8s/README.md#check-scorer-health) | `{"status": "ok"}` from all five cluster-DNS URLs. Boltz adds `no_kernels` only if `boltz2-wrapper:local` was rebuilt since that field landed |
| 63 | Scoring still works across pods | `kubectl -n "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- python -c 'import json; from reinvent4_mcp.tools.external import list_external_services; print(json.dumps(list_external_services(), indent=2))'` | All five report `status: ok` against the ConfigMap's `dd-agentic-*` URLs, which exercises the same config `ExternalRESTScorer` reads |

### Cross-Pod Artifact Handoff (requires a cluster + GPU)

The riskiest assumption in the split: Boltz returns absolute `/shared` paths
and PoseBusters and the interaction profiler re-open them, so the consumers
must see the producer's filesystem. Proven in-pod already; this is the same
handoff across pod boundaries. See
[Prove the structure handoff across pods](k8s/README.md#prove-the-structure-handoff-across-pods).

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 64 | Handoff across pods | `kubectl -n "${PROJECT}" exec -i deployment/dd-agentic -c dd-agent -- python - --in-cluster < k8s/scripts/verify-artifact-handoff.py` | `PROVEN`, exit 0: two structures written under `/shared`, both consumers scoring real values off them, the skipped molecule still `--` in every column. Without `--namespace` the verdict says pod identity was not checked |
| 65 | Same handoff, pods named | Three port-forwards (8091/8082/8083), then `./k8s/scripts/verify-artifact-handoff.py --namespace "${PROJECT}"` | The same, plus the pod and node that answered for each service, and PoseBusters' extracted `_lig.sdf` visible from the Boltz pod. Exit 2 if any two share a pod — that is the in-pod proof, not this one |
| 66 | Re-prove after a storage change | `./k8s/scripts/verify-artifact-handoff.py --namespace "${PROJECT}" --structure <path from run 64> --smiles <its SMILES>` | Same verdict with no GPU time: the consumers re-open a pose already on the volume |
| 67 | Same handoff on Compose | `docker compose up -d boltz posebusters interaction-profiler && ./k8s/scripts/verify-artifact-handoff.py` | `PROVEN` against the published 8091/8082/8083. Three separate containers over one host directory, so it rehearses the mechanism before a cluster is involved |

### Boltz Gateway Placement (requires a cluster + 2 GPUs)

Each replica serves one prediction at a time, so the gateway has to place work
on an idle one and queue the surplus. See
[Place batches on idle Boltz replicas](k8s/README.md#place-batches-on-idle-boltz-replicas).

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 68 | Gateway discovered the replicas | Port-forward 8404, then `curl -s "http://localhost:8404/stats;csv" \| cut -d, -f1,2,18` | One `UP` row per Boltz replica; the remaining slots sit in `MAINT (resolution)` until DNS fills them, which is how scaling needs no config change |
| 69 | Least-busy placement | `./k8s/scripts/verify-boltz-placement.py --namespace "${PROJECT}"` | `PROVEN`: each concurrent batch on its own replica, peak in-flight 1 per replica, the surplus queued. Measured at M=2: 2.11x speedup, third batch waited 197.6 s |
| 70 | Queueing is not failure | `kubectl -n "${PROJECT}" logs deployment/dd-agentic-boltz-gateway \| tail` | `%Tw` non-zero and `queue=0/1` on the surplus request, with status `200`. A `503` means `timeout queue` is below the prediction time |
| 71 | Scoring still works through the gateway | `./k8s/scripts/verify-artifact-handoff.py --namespace "${PROJECT}"` | `PROVEN` with Boltz reported as "2 replicas" and the extracted ligands visible from all of them |
| 72 | The RL path dials the gateway | `kubectl -n "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- python -c "from reinvent4_mcp.tools.external import list_external_services as l; [print('%-22s %-10s %s' % (n, i['status'], i['url'])) for n, i in l().items()]"` | `boltz ok http://dd-agentic-boltz-gateway:8091/score` and the four CPU scorers direct — the ConfigMap path `ExternalRESTScorer` actually takes |
| 73 | Scaling needs no proxy change | `kubectl -n "${PROJECT}" scale deployment/dd-agentic-boltz --replicas=3`, then watch the stats CSV | A third slot leaves `MAINT (resolution)` and turns `UP` **about 30 s after the pod is Ready** (10 s DNS hold + `rise 2 inter 10s`). Checking immediately shows the old count and looks like a failure |

### REINVENT Runner (requires a cluster + GPU)

The RL lifecycle can only really be exercised where REINVENT4 is installed. See
[Move RL off the agent](k8s/README.md#move-rl-off-the-agent-reinvent-runner).

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 78 | Runner is healthy | `kubectl -n "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- python -c "import urllib.request; print(urllib.request.urlopen('http://dd-agentic-reinvent-runner:8086/health').read().decode())"` | `reinvent_importable: true` and `max_concurrent_jobs` matching the config. A false there means the image lacks REINVENT4, which is a different problem from the service being down |
| 79 | Registries over HTTP | `GET /components`, `/models`, `/transforms` | Measured: 61 scoring components, 9 priors, 8 transforms — the real registries, not stubs |
| 80 | A real job over HTTP | `POST /jobs` with a QED-only scorer config and `max_steps: 2`, then poll `GET /jobs/{id}` | `running` then `completed`, and the job appears in `GET /jobs`. Proves REINVENT accepts the generated TOML and the subprocess lifecycle survives the move |
| 81 | Images are split | `docker images dd-agent` after `build-images.sh` | **164 MB** on `python:3.12-slim`, and `docker run dd-agent:local python -c "import importlib.util as u; print(u.find_spec('reinvent'), u.find_spec('torch'))"` prints `None None`. The build fails outright if either is present |
| 82 | The seam is invisible | Set `runner.enabled`, then drive the same job through `get_job_manager()` | Returns `RemoteJobManager`; same result as the direct call, and the third concurrent job is refused naming the two to stop — the slot limit holds across the HTTP boundary |

### Boltz Gateway: Cache and Coalescing (requires a cluster + GPU)

The gateway reduces how much prediction is asked for at all. See
[Cache results and coalesce batches](k8s/README.md#cache-results-and-coalesce-batches).

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 74 | The chain is wired | `kubectl -n "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- python -c "import json,urllib.request; print(urllib.request.urlopen('http://dd-agentic-boltz-gateway:8091/health').read().decode())"` | `cache_enabled: true`, `upstream` the proxy, `healthy_replicas` matching the Boltz replica count, `upstream_reachable: true` |
| 75 | Cold then warm | `./k8s/scripts/verify-artifact-handoff.py --namespace "${PROJECT}"` twice | Measured 3 m 35 s then **12 s**, with identical affinities and the *same* structure paths, and both consumers re-opening the cached poses. The molecule Boltz skipped hits too — a deterministic skip is cached |
| 76 | Hit rate is reported | `curl -s http://localhost:8091/stats` through a port-forward | `cache.hit_rate` 0.5 after one cold and one warm run, `coalescing.molecules_per_call`, `queue.*` and latency percentiles. This is the number §5.4's admission control has always had to assume |
| 77 | Submit and poll | `curl -s -X POST "http://localhost:8091/score?mode=async" -d @batch.json`, then `GET /result/{task_id}` | `202` with a `task_id`, `202` while queued or predicting, `200` with the scores once, then `404` — results are served once so the registry cannot grow without bound |

---

## Test Matrix Summary

| Tier | Tests | Needs REINVENT4 | Needs LLM API | Needs Docker | Needs GPU | Needs Slurm |
|------|-------|-----------------|---------------|--------------|-----------|-------------|
| 1 - Unit | 650+ | No | No | No | No | No |
| 2 - Integration | 39 | Yes | No | Partially | For RL/TL | No |
| 3 - E2E | 82 | Yes | Yes | Partially | For RL/TL | Partially |

### Recommended Test Sequence

1. **Local development**: Run Tier 1 (`python -m pytest tests/ -v`) after every change
2. **After setup on cluster**: Run Tier 2 integration tests to verify REINVENT4 wiring
3. **Docker services**: Build and start Boltz2 + PoseBusters services (`docker compose up -d`), then run Tier 2 external service tests
4. **RL / Transfer Learning**: Run Tier 2 optimization + TL tests on a GPU node with REINVENT4 installed
5. **Before demo/release**: Run Tier 3 E2E tests for the full agent experience including optimization campaigns and external scoring pipeline
