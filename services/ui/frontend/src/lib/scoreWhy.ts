// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Why a molecule scored what it did: each TTP objective's raw value, its
 * desirability after the transform, its weight, and how much it pulls the
 * total down.
 *
 * REINVENT aggregates desirabilities d with weights w as a weighted geometric
 * mean, prod(d_i ^ (w_i / W)), or an arithmetic one, sum(w_i d_i) / W. The
 * drag of an objective is what it costs the total: -(w/W) ln d for the
 * geometric mean, (w/W)(1 - d) for the arithmetic one.
 */

import { callTool } from "./api";
import { asTtp, evaluateTransform, objectivesOf, parseResult, type Transform, type Ttp } from "./ttp";

export interface WhyItem {
  name: string;
  weight: number;
  /** Weight as a share of all weights. */
  share: number;
  raw: number | null;
  /** 0..1 after the transform, null when the component reported nothing. */
  desirability: number | null;
  /** What the objective costs the total; Infinity when it zeroes it. */
  drag: number;
  transform: Transform | null;
}

export interface ScoreWhy {
  items: WhyItem[];
  /** The total the TTP gives for these desirabilities. */
  total: number | null;
  verdict: string;
}

/** Values of one molecule by column name: "<objective>" transformed, "<objective> (raw)" raw. */
export type ScoreValues = Record<string, number | null | undefined>;

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

function rawOf(values: ScoreValues, name: string): number | null {
  for (const key of [`${name} (raw)`, `${name}_raw`, `${name} raw`, `raw_${name}`]) {
    const v = values[key];
    if (finite(v)) return v;
  }
  return null;
}

/** The objectives that score, pre-scoring components first, as REINVENT combines them. */
function scored(ttp: Ttp) {
  return [...objectivesOf(ttp, "pre_scoring"), ...objectivesOf(ttp, "scoring")].filter((o) => o.weight > 0);
}

export function explainScore(ttp: Ttp, values: ScoreValues): ScoreWhy | null {
  const objectives = scored(ttp);
  if (!objectives.length) return null;
  const geometric = (ttp.scoring.aggregation ?? "geometric_mean") === "geometric_mean";
  const totalWeight = objectives.reduce((sum, o) => sum + o.weight, 0);
  let known = 0;
  const items = objectives.map((o): WhyItem => {
    const raw = rawOf(values, o.name);
    const reported = values[o.name];
    const transform = o.transform ?? null;
    let d: number | null = finite(reported) ? reported : null;
    if (d === null && raw !== null) d = transform ? evaluateTransform(transform, raw) : raw;
    if (d !== null) {
      known += 1;
      d = Math.min(1, Math.max(0, d));
    }
    const share = o.weight / totalWeight;
    const drag = d === null ? Infinity : geometric ? (d <= 0 ? Infinity : -share * Math.log(d)) : share * (1 - d);
    return { name: o.name, weight: o.weight, share, raw, desirability: d, drag, transform };
  });
  if (!known) return null;
  items.sort((a, b) => b.drag - a.drag || b.share - a.share);

  const ds = items.map((i) => i.desirability ?? 0);
  const total = geometric
    ? items.reduce((p, i, k) => p * ds[k] ** i.share, 1)
    : items.reduce((s, i, k) => s + ds[k] * i.share, 0);
  return { items, total, verdict: verdict(items, total) };
}

/** One line: what holds the score back, or that nothing does. */
export function verdict(items: WhyItem[], total: number): string {
  const [first, second] = items;
  if (!first) return "";
  if (first.desirability === null) return `${first.name} reported no value`;
  if (first.desirability < 0.01) return `Zeroed by ${first.name}`;
  if (total >= 0.9 || first.drag < 0.05) return items.length > 1 ? "Meets every objective" : "Close to ideal";
  const both = second && second.desirability !== null && Number.isFinite(second.drag) && second.drag >= first.drag * 0.6 && second.drag > 0.05;
  return both ? `Held back by ${first.name} and ${second.name}` : `Held back by ${first.name}`;
}

const cache = new Map<string, Promise<Ttp | null>>();

/** A TTP at a version, fetched once per session of the page. */
export function loadTtp(id: string, version?: number | null): Promise<Ttp | null> {
  const key = `${id}@${version ?? "latest"}`;
  let pending = cache.get(key);
  if (!pending) {
    pending = callTool("get_ttp", version != null ? { ttp_id: id, version } : { ttp_id: id })
      .then(({ result }) => asTtp(parseResult(result)))
      .catch(() => null);
    pending.then((ttp) => {
      if (!ttp) cache.delete(key);
    });
    cache.set(key, pending);
  }
  return pending;
}
