// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "./api";
import { loadModels, modelLabel, modelOf, pickEffort, switchNote, switchTo } from "./models";
import type { ModelsState } from "../types/api";

const READY: ModelsState = {
  current: "qwen",
  state: "ready",
  switchable: true,
  models: [
    { key: "qwen", id: "Qwen/Qwen3.8-27B", label: "Qwen3.8 27B", efforts: ["low", "medium", "high"], defaultEffort: "low" },
    { key: "gemma", id: "google/gemma-4-31B-it", label: "Gemma 4 31B", efforts: [], defaultEffort: null },
  ],
};

/** Answer every request with *status* and *body*, and report the calls. */
function respond(status: number, body: unknown = {}) {
  const fetcher = vi.fn<(path: string, init?: RequestInit) => Promise<Response>>(async () => new Response(JSON.stringify(body), { status, statusText: `status ${status}` }));
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}

afterEach(() => vi.unstubAllGlobals());

describe("loadModels", () => {
  it("reads the served models from the gateway", async () => {
    const fetcher = respond(200, READY);
    await expect(loadModels()).resolves.toEqual(READY);
    expect(fetcher.mock.calls[0][0]).toBe("/api/models");
  });

  it("rejects when the gateway has no endpoint, so the UI can fall back", async () => {
    respond(404, { detail: "Not Found" });
    await expect(loadModels()).rejects.toMatchObject({ status: 404 });
  });
});

describe("switchTo", () => {
  it("posts the key and reports the restart as loading", async () => {
    const fetcher = respond(202, { ...READY, current: "gemma", state: "loading" });
    await expect(switchTo("gemma")).resolves.toMatchObject({ state: "loading" });
    const [path, init] = fetcher.mock.calls[0];
    expect(path).toBe("/api/models");
    expect(init?.method).toBe("POST");
    expect(init?.body).toBe(JSON.stringify({ key: "gemma" }));
  });

  it("fails with the gateway's status when a turn is running", async () => {
    respond(409, { detail: "a turn is running" });
    await expect(switchTo("gemma")).rejects.toBeInstanceOf(ApiError);
  });
});

describe("switchNote", () => {
  it("explains a refused switch in one line", () => {
    expect(switchNote(new ApiError(409, "busy"))).toMatch(/turn is running/);
    expect(switchNote(new ApiError(503, "no cluster"))).toMatch(/not available here/);
    expect(switchNote(new ApiError(400, "nope"))).toMatch(/not served/);
    expect(switchNote(new Error("offline"))).toBe("offline");
  });
});

describe("pickEffort", () => {
  const efforts = READY.models[0].efforts;

  it("keeps the remembered level when the model offers it", () => {
    expect(pickEffort(efforts, "high", "low")).toBe("high");
  });

  it("falls back to the model's default when the remembered one is not offered", () => {
    expect(pickEffort(["low"], "high", "low")).toBe("low");
    expect(pickEffort(["medium", "high"], "low", null)).toBe("medium");
  });

  it("never applies a level to a model without any", () => {
    expect(pickEffort([], "high", "low")).toBeNull();
  });
});

describe("modelOf", () => {
  it("names a served model and leaves an unknown key as it is", () => {
    expect(modelOf(READY, "gemma")?.id).toBe("google/gemma-4-31B-it");
    expect(modelLabel(READY, "gemma")).toBe("Gemma 4 31B");
    expect(modelLabel(null, "qwen")).toBe("qwen");
    expect(modelLabel(READY, null)).toBe("");
  });
});
