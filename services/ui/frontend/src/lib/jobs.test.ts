// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import {
  bestScore,
  canResume,
  componentSeriesNames,
  continueCost,
  continueSteps,
  errorSummary,
  isActive,
  isMol2Mol,
  isScoringComponent,
  linePath,
  linear,
  parseAlerts,
  parseCandidates,
  parseJobs,
  parseRunConfig,
  parseStatus,
  progress,
  resultsFile,
  resumeArguments,
  runningBest,
  scoreDomain,
  scoringComponents,
  startedJobId,
  stepTicks,
  stepsDone,
  ticks,
  trajectoryPoints,
  toolError,
  uniqueAlerts,
} from "./jobs";

const job = {
  job_id: "a1b2c3d4",
  ttp_id: "jak2-lead",
  ttp_version: 3,
  model: ".reinvent",
  job_type: "optimization",
  state: "stopped",
  working_dir: "/data/jobs/job_a1b2c3d4",
  pid: 123,
  created_at: "2026-10-26T10:00:00+00:00",
  started_at: "2026-10-26T10:00:05+00:00",
  finished_at: "2026-10-26T10:30:05+00:00",
  error: null,
  exit_code: -2,
  max_steps: 300,
  csv_path: "/data/jobs/job_a1b2c3d4/results_1.csv",
  checkpoint_path: "/data/jobs/job_a1b2c3d4/stage.chkpt",
  has_checkpoint: true,
};

describe("parseJobs", () => {
  it("normalises list_jobs, newest first, and drops entries without an id", () => {
    const jobs = parseJobs([
      job,
      { job_id: "ffff0000", job_type: "transfer_learning", state: "weird", created_at: "2026-10-26T11:00:00+00:00", num_epochs: 50 },
      { state: "running" },
    ]);
    expect(jobs.map((j) => j.id)).toEqual(["ffff0000", "a1b2c3d4"]);
    expect(jobs[0].state).toBe("unknown");
    expect(jobs[0].numEpochs).toBe(50);
    expect(jobs[1]).toMatchObject({ ttpId: "jak2-lead", ttpVersion: 3, maxSteps: 300, hasCheckpoint: true, state: "stopped" });
  });

  it("is empty for anything that is not a list", () => {
    expect(parseJobs({ error: "x" })).toEqual([]);
    expect(parseJobs(null)).toEqual([]);
  });
});

describe("resultsFile", () => {
  it("is the CSV's file name, or REINVENT's default", () => {
    const [parsed] = parseJobs([job]);
    expect(resultsFile(parsed)).toBe("results_1.csv");
    expect(resultsFile({ ...parsed, csvPath: "C:\\jobs\\job_1\\stage2.csv" })).toBe("stage2.csv");
    expect(resultsFile({ ...parsed, csvPath: "/x/../etc/passwd" })).toBe("results_1.csv");
    expect(resultsFile({ ...parsed, csvPath: null })).toBe("results_1.csv");
  });
});

describe("parseStatus", () => {
  const status = {
    ...job,
    state: "running",
    trajectory: {
      current_step: 120,
      max_steps: 300,
      n_molecules_total: 7680,
      n_valid: 7300,
      validity_rate: 0.95,
      mean_total_score: 0.41,
      best_total_score: 0.88,
      latest_mean_score: 0.52,
      per_component: { QED: { mean: 0.6, max: 0.9, min: 0.1 } },
      score_trajectory: [
        { step: 10, mean_score: 0.3, max_score: 0.6, n_molecules: 64 },
        { step: 0, mean_score: 0.2, max_score: 0.5, n_molecules: 64 },
        { step: "bad", mean_score: 0.1 },
      ],
      per_component_trajectory: { QED: [{ step: 0, mean_score: 0.5 }], empty: [] },
      diagnostics: {
        convergence_status: "plateau",
        plateau_detected: true,
        plateau_since_step: 90,
        diverging: false,
        improvement_rate: 0.0004,
        progress_pct: 40,
        trend: { early_mean: 0.2, mid_mean: 0.4, recent_mean: 0.5, overall_improvement: 0.3 },
        component_health: { QED: { status: "healthy", nan_rate: 0 }, Boltz: { status: "failing", nan_rate: 0.7 } },
        warnings: [{ type: "score_plateau", severity: "medium", message: "Plateaued since step 90" }, { type: "x" }],
        recommendations: ["Stop and adjust", 3],
      },
    },
  };

  it("charts the total score component when the run reports no score trajectory", () => {
    const raw = { ...status, trajectory: { ...status.trajectory, score_trajectory: [], per_component_trajectory: { Score: [{ step: 2, mean_score: 0.7 }], QED: [{ step: 2, mean_score: 0.5 }] } } };
    const fallback = trajectoryPoints(parseStatus(raw)!);
    expect(fallback.from).toBe("Score");
    expect(fallback.points).toEqual([{ step: 2, mean: 0.7, max: null, molecules: null }]);

    const direct = trajectoryPoints(parseStatus(status)!);
    expect(direct.from).toBeNull();
    expect(direct.points.map((p) => p.step)).toEqual([0, 10]);
    expect(trajectoryPoints(parseStatus({ ...status, trajectory: {} })!)).toEqual({ points: [], from: null });
  });

  it("reads the trajectory, sorted by step, skipping malformed points", () => {
    const s = parseStatus(status)!;
    expect(s.currentStep).toBe(120);
    expect(s.trajectory.map((p) => p.step)).toEqual([0, 10]);
    expect(s.components).toEqual([{ name: "QED", mean: 0.6, max: 0.9, min: 0.1 }]);
    expect(Object.keys(s.componentTrajectories)).toEqual(["QED"]);
    expect(progress(s)).toBeCloseTo(0.4);
    expect(isActive(s.state)).toBe(true);
  });

  it("reads the diagnostics", () => {
    const d = parseStatus(status)!.diagnostics!;
    expect(d.convergence).toBe("plateau");
    expect(d.plateauSince).toBe(90);
    expect(d.health.find((h) => h.name === "Boltz")?.status).toBe("failing");
    expect(d.warnings).toHaveLength(1);
    expect(d.recommendations).toEqual(["Stop and adjust"]);
    expect(d.trend?.overall).toBe(0.3);
  });

  it("copes with a job that has no CSV yet", () => {
    const s = parseStatus({ ...job, state: "queued" })!;
    expect(s.trajectory).toEqual([]);
    expect(s.diagnostics).toBeNull();
    expect(s.currentStep).toBeNull();
  });

  it("is null for an error", () => {
    expect(parseStatus({ error: "Job 'x' not found" })).toBeNull();
    expect(toolError({ error: "Job 'x' not found" })).toBe("Job 'x' not found");
    expect(toolError("Error executing tool")).toBe("Error executing tool");
    expect(toolError({ job_id: "a" })).toBeNull();
  });
});

describe("alerts and candidates", () => {
  it("parses get_job_alerts", () => {
    const alerts = parseAlerts([
      { job_id: "a1b2c3d4", severity: "high", type: "stalled_progress", message: "stalled" },
      { job_id: "a1b2c3d4" },
    ]);
    expect(alerts).toEqual([{ jobId: "a1b2c3d4", severity: "high", type: "stalled_progress", message: "stalled" }]);
  });

  it("parses get_top_candidates", () => {
    const c = parseCandidates({
      candidates: [{ smiles: "CCO", total_score: 0.8, components: { QED: { transformed: 0.7, raw: 0.42 } } }, { total_score: 1 }],
      n_total: 2,
    });
    expect(c).toEqual([{ smiles: "CCO", total: 0.8, components: [{ name: "QED", transformed: 0.7, raw: 0.42 }] }]);
    expect(parseCandidates({ error: "no CSV" })).toEqual([]);
  });

  it("takes REINVENT 4's Score as the total and drops likelihoods and component details", () => {
    const [c] = parseCandidates({
      candidates: [
        {
          smiles: "CCO",
          components: {
            Agent: { transformed: 4.16 },
            Prior: { transformed: 4.16 },
            Target: { transformed: 71.5 },
            Score: { transformed: 0.59 },
            "Aqueous solubility": { transformed: 0.59, raw: -4.4 },
            "std (Aqueous solubility)": { transformed: 0.38 },
          },
        },
      ],
    });
    expect(c).toEqual({ smiles: "CCO", total: 0.59, components: [{ name: "Aqueous solubility", transformed: 0.59, raw: -4.4 }] });
  });

  it("shows each alert message once, at its most severe", () => {
    const merged = uniqueAlerts([
      { severity: "medium", message: "Plateaued since step 90", type: "score_plateau" },
      { severity: "high", message: "Plateaued  since step 90 " },
      { severity: "low", message: "Validity is low" },
    ]);
    expect(merged).toEqual([
      { severity: "high", message: "Plateaued  since step 90", type: "score_plateau" },
      { severity: "low", message: "Validity is low", type: "alert" },
    ]);
  });

  it("summarises a traceback by its last line", () => {
    const error =
      'Process exited with code 1. Last stderr: Traceback (most recent call last):\n  File "x.py", line 62, in <module>\n    main()\nRuntimeError: No HIP GPUs are available\n';
    expect(errorSummary(error)).toBe("Process exited with code 1: RuntimeError: No HIP GPUs are available");
    expect(errorSummary("Traceback (most recent call last):\n  File x\nValueError: bad")).toBe("ValueError: bad");
    expect(errorSummary("Server restarted while job was running")).toBe("Server restarted while job was running");
  });
});

describe("resume", () => {
  it("reads the learning settings back from config.toml", () => {
    const toml = `[parameters]\nbatch_size = 64\n\n[learning_strategy]\ntype = "dap"\nsigma = 128\nrate = 0.0001\n`;
    expect(parseRunConfig(toml)).toEqual({ batchSize: 64, sigma: 128, learningRate: 0.0001, mol2mol: false });
    expect(parseRunConfig("")).toEqual({ batchSize: null, sigma: null, learningRate: null, mol2mol: false });
    const m2m = `[parameters]\nprior_file = "/app/priors/mol2mol_medium_similarity.prior"\nsmiles_file = "/data/seed.smi"\n`;
    expect(parseRunConfig(m2m).mol2mol).toBe(true);
  });

  it("continues for the remaining steps from the checkpoint", () => {
    // current_step is the next step to run: 120 means 119 are done.
    const s = parseStatus({ ...job, trajectory: { current_step: 120, max_steps: 300 } })!;
    expect(canResume(s)).toBe(true);
    expect(stepsDone(s)).toBe(119);
    expect(continueSteps(s)).toBe(181);
    expect(resumeArguments(s, { config: { batchSize: 64, sigma: null, learningRate: 0.0001 } })).toEqual({
      ttp_id: "jak2-lead",
      model: ".reinvent",
      max_steps: 181,
      resume_from: "/data/jobs/job_a1b2c3d4/stage.chkpt",
      batch_size: 64,
      learning_rate: 0.0001,
    });
    expect(startedJobId({ job_id: "deadbeef", working_dir: "/x" })).toBe("deadbeef");
  });

  it("extends a completed run with the same TTP and Mol2Mol seeds", () => {
    const [done] = parseJobs([{ ...job, state: "completed", model: ".m2m_medium", started_at: "2026-10-27T07:00:00Z", finished_at: "2026-10-27T07:30:00Z" }]);
    expect(canResume(done)).toBe(true);
    expect(continueSteps(done)).toBe(100);
    const ttp = { id: "jak2-lead", version: 2, scoring: { objectives: [] } };
    expect(resumeArguments(done, { steps: 50, ttp, seeds: ["CCO"], config: { batchSize: 64, sigma: 128, learningRate: null, mol2mol: true } })).toEqual({
      ttp,
      max_steps: 50,
      resume_from: "/data/jobs/job_a1b2c3d4/stage.chkpt",
      model: ".m2m_medium",
      batch_size: 64,
      sigma: 128,
      seed_smiles: ["CCO"],
    });
    expect(isMol2Mol(done)).toBe(true);
    // 300 steps took 30 min, so 100 more take ~10.
    expect(continueCost(done, 100, 64)).toBe(`${(6400).toLocaleString()} molecules · ~10 min at the last pace`);
    expect(continueCost(done, 100, null)).toBe("~10 min at the last pace");
  });

  it("is not offered without a checkpoint or while running", () => {
    const [noCheckpoint] = parseJobs([{ ...job, has_checkpoint: false }]);
    const [running] = parseJobs([{ ...job, state: "running" }]);
    expect(canResume(noCheckpoint)).toBe(false);
    expect(canResume(running)).toBe(false);
  });
});

describe("chart geometry", () => {
  it("keeps 0..1 for normal scores and widens for others", () => {
    expect(scoreDomain([0.2, 0.8])).toEqual([0, 1]);
    expect(scoreDomain([-0.5, 1.4])).toEqual([-0.5, 1.4]);
    expect(scoreDomain([])).toEqual([0, 1]);
  });

  it("draws a path and ticks", () => {
    const x = linear([0, 10], [0, 100]);
    const y = linear([0, 1], [100, 0]);
    expect(linePath([{ x: 0, y: 0 }, { x: 10, y: 1 }], x, y)).toBe("M0.0,100.0L100.0,0.0");
    expect(ticks(0, 1, 5)).toEqual([0, 0.25, 0.5, 0.75, 1]);
  });

  it("spaces the step axis on round steps inside the run", () => {
    // 51 steps used to tick 0, 10, 20, 31, 41, 51, and 3 steps repeated a tick.
    expect(stepTicks(50, 6)).toEqual([0, 10, 20, 30, 40, 50]);
    expect(stepTicks(51, 6)).toEqual([0, 20, 40]);
    expect(stepTicks(300, 6)).toEqual([0, 100, 200, 300]);
    expect(stepTicks(30, 6)).toEqual([0, 10, 20, 30]);
    expect(stepTicks(3, 6)).toEqual([0, 1, 2, 3]);
    expect(stepTicks(1, 6)).toEqual([0, 1]);
  });

  it("never ticks past the last step, nor twice on one", () => {
    for (const last of [1, 2, 3, 7, 12, 29, 50, 51, 99, 100, 137, 300, 1000, 2048]) {
      const t = stepTicks(last, 6);
      expect(t[0]).toBe(0);
      expect(Math.max(...t)).toBeLessThanOrEqual(last);
      expect(t.every(Number.isInteger)).toBe(true);
      expect(new Set(t).size).toBe(t.length);
      expect(t.length).toBeLessThanOrEqual(6);
    }
    expect(stepTicks(0, 6)).toEqual([0]);
    expect(stepTicks(-5, 6)).toEqual([0]);
    expect(stepTicks(NaN, 6)).toEqual([0]);
  });

  it("never lets the best score go down", () => {
    const best = runningBest([
      { step: 0, mean: 0.1, max: 0.5, molecules: 1 },
      { step: 1, mean: 0.2, max: 0.3, molecules: 1 },
      { step: 2, mean: 0.3, max: null, molecules: 1 },
      { step: 3, mean: 0.3, max: 0.9, molecules: 1 },
    ]);
    expect(best.map((p) => p.y)).toEqual([0.5, 0.5, 0.5, 0.9]);
  });
});

describe("scoring components", () => {
  // What the MCP server answers for a completed run: no score_trajectory, no best_total_score.
  const status = parseStatus({
    ...job,
    state: "completed",
    trajectory: {
      current_step: 301,
      max_steps: 300,
      per_component: {
        Agent: { mean: 13.5179, max: 40.0668, min: 2.2821 },
        Prior: { mean: 34.5732, max: 104.4353, min: 2.2821 },
        Target: { mean: 76.6245, max: 125.4626, min: -92.3977 },
        Score: { mean: 0.8687, max: 0.9487, min: 0 },
        QED: { mean: 0.74, max: 0.93, min: 0.2 },
        "Synthetic accessibility": { mean: 0.6, max: 0.8, min: 0.1 },
      },
      per_component_trajectory: {
        Agent: [{ step: 1, mean_score: 6.2256 }],
        Prior: [{ step: 1, mean_score: 6.2256 }],
        Target: [{ step: 1, mean_score: 71.5 }],
        Score: [{ step: 1, mean_score: 0.7065 }],
        QED: [{ step: 1, mean_score: 0.74 }],
        "Synthetic accessibility": [{ step: 1, mean_score: 0.6 }],
      },
    },
  })!;

  it("drops REINVENT's likelihoods and the total from the plottable series", () => {
    expect(isScoringComponent("Agent")).toBe(false);
    expect(isScoringComponent("prior")).toBe(false);
    expect(isScoringComponent(" Target ")).toBe(false);
    expect(isScoringComponent("Score")).toBe(false);
    expect(isScoringComponent("Boltz affinity")).toBe(true);
    expect(componentSeriesNames(status)).toEqual(["QED", "Synthetic accessibility"]);
    expect(scoringComponents(status).map((c) => c.name)).toEqual(["QED", "Synthetic accessibility"]);
    expect(scoringComponents(null)).toEqual([]);
  });

  it("keeps the Score series reachable as the chart's own line", () => {
    expect(trajectoryPoints(status).from).toBe("Score");
  });

  it("takes the best score from the total component, then the candidates, else nothing", () => {
    expect(bestScore(status)).toBe(0.9487);
    const withBest = parseStatus({ ...job, trajectory: { best_total_score: 0.99 } })!;
    expect(bestScore(withBest)).toBe(0.99);

    const bare = parseStatus({ ...job, trajectory: { per_component: { QED: { mean: 0.5 } } } })!;
    expect(bestScore(bare)).toBeNull();
    const candidates = parseCandidates({ candidates: [{ smiles: "CCO", components: { Score: { transformed: 0.59 } } }, { smiles: "CCN", components: { Score: { transformed: 0.81 } } }] });
    expect(bestScore(bare, candidates)).toBe(0.81);
    expect(bestScore(null)).toBeNull();
  });
});
