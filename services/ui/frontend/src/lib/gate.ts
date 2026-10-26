// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { StructureMetrics } from "../types/api";

/**
 * The confidence gate.
 *
 * Boltz always returns an affinity, including for poses it is not confident
 * in. A number shown next to a bad pose gets quoted anyway, so the UI only
 * shows an affinity when the pose passes all three checks.
 */
export const GATE = { confidence: 0.6, ligandIptm: 0.5 } as const;

export interface GateCheck {
  label: string;
  value: string;
  rule: string;
  /** null when the value is missing. */
  ok: boolean | null;
}

export interface Gate {
  /** "unknown" when a value needed for the verdict is missing. */
  status: "pass" | "fail" | "unknown";
  checks: GateCheck[];
  reasons: string[];
}

const num = (value: number | undefined, digits = 2) => (value === undefined ? "—" : value.toFixed(digits));

export function gate(metrics: StructureMetrics | undefined): Gate {
  const m = metrics ?? {};
  const pbKnown = m.posebustersTotal !== undefined && m.posebustersPassed !== undefined;
  const checks: GateCheck[] = [
    {
      label: "Confidence",
      value: num(m.confidence),
      rule: `> ${GATE.confidence}`,
      ok: m.confidence === undefined ? null : m.confidence > GATE.confidence,
    },
    {
      label: "Ligand ipTM",
      value: num(m.ligandIptm),
      rule: `> ${GATE.ligandIptm}`,
      ok: m.ligandIptm === undefined ? null : m.ligandIptm > GATE.ligandIptm,
    },
    {
      label: "PoseBusters",
      value: pbKnown ? `${m.posebustersPassed}/${m.posebustersTotal}` : "—",
      rule: "all checks pass",
      ok: pbKnown ? m.posebustersPassed === m.posebustersTotal : null,
    },
  ];
  const reasons = checks
    .filter((check) => check.ok === false)
    .map((check) => `${check.label} ${check.value} (needs ${check.rule})`);
  if (m.failedChecks?.length) reasons.push(`failed: ${m.failedChecks.join(", ")}`);

  // PoseBusters is often not part of the objective; confidence alone then
  // decides, but the verdict says what it did not check.
  const decisive = checks.filter((check) => check.ok !== null);
  const status =
    decisive.some((check) => check.ok === false) ? "fail" : checks[0].ok && checks[1].ok ? "pass" : "unknown";
  return { status, checks, reasons };
}

/** Affinity as text: a bracket when the two ensemble members disagree by more than a log unit. */
export function affinityText(m: StructureMetrics): string | null {
  if (m.affinity === undefined) return null;
  if (m.affinity1 !== undefined && m.affinity2 !== undefined && Math.abs(m.affinity1 - m.affinity2) > 1) {
    return `${Math.min(m.affinity1, m.affinity2).toFixed(2)} to ${Math.max(m.affinity1, m.affinity2).toFixed(2)}`;
  }
  return m.affinity.toFixed(2);
}

/** log10(IC50 / 1 µM) as a concentration a chemist reads: "≈ 63 nM". */
export function affinityConcentration(value: number): string {
  const micromolar = 10 ** value;
  if (micromolar >= 1000) return `≈ ${(micromolar / 1000).toPrecision(2)} mM`;
  if (micromolar >= 1) return `≈ ${micromolar.toPrecision(2)} µM`;
  return `≈ ${(micromolar * 1000).toPrecision(2)} nM`;
}

/** pIC50 = 6 − log10(IC50 / 1 µM). */
export const pIC50 = (value: number) => 6 - value;
