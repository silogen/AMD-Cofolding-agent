// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { CHAT_MIN_WIDTH, DOCK_FLOOR_WIDTH, DOCK_TABS, clampWidth, closeTab, dockRoom, dockTabOf, loadTabs, loadWidth, openTab, saveTabs, saveWidth } from "./dock";

function memory() {
  const map = new Map<string, string>();
  return { getItem: (key: string) => map.get(key) ?? null, setItem: (key: string, value: string) => void map.set(key, value) };
}

describe("openTab", () => {
  it("appends new tabs once", () => {
    expect(openTab(["a"], "b")).toEqual(["a", "b"]);
    expect(openTab(["a", "b"], "a")).toEqual(["a", "b"]);
  });
});

describe("dockTabOf", () => {
  it("maps views to the tab that shows them", () => {
    expect(dockTabOf("candidates")).toBe("molecules");
    expect(dockTabOf("properties")).toBe("molecules");
    expect(dockTabOf("activity")).toBe("jobs");
    expect(dockTabOf("ttp")).toBe("ttp");
  });
});

describe("closeTab", () => {
  it("activates the right neighbour, then the left one", () => {
    expect(closeTab(["a", "b", "c"], "b", "b")).toEqual({ tabs: ["a", "c"], active: "c" });
    expect(closeTab(["a", "b", "c"], "c", "c")).toEqual({ tabs: ["a", "b"], active: "b" });
    expect(closeTab(["a"], "a", "a")).toEqual({ tabs: [], active: null });
  });
  it("keeps the active tab when closing another", () => {
    expect(closeTab(["a", "b"], "a", "b")).toEqual({ tabs: ["b"], active: "b" });
  });
});

describe("persistence", () => {
  it("round-trips tabs per session and drops unknown ids", () => {
    const store = memory();
    saveTabs("s1", ["jobs", "bogus", "jobs"], store);
    expect(loadTabs("s1", ["jobs", "ttp"], store)).toEqual(["jobs"]);
    expect(loadTabs("s2", ["jobs"], store)).toBeNull();
  });
  it("opens the merged tab for ids that used to be tabs", () => {
    const store = memory();
    saveTabs("s1", ["structure", "candidates", "properties", "activity", "jobs"], store);
    expect(loadTabs("s1", DOCK_TABS, store)).toEqual(["structure", "molecules", "jobs"]);
  });
  it("opens run results in the jobs tab", () => {
    const store = memory();
    saveTabs("s1", ["results", "jobs"], store);
    expect(loadTabs("s1", DOCK_TABS, store)).toEqual(["jobs"]);
    expect(dockTabOf("results")).toBe("jobs");
  });
  it("tolerates garbage", () => {
    const store = memory();
    store.setItem("cofold.dock.tabs:s1", "{nope");
    expect(loadTabs("s1", ["jobs"], store)).toBeNull();
  });
  it("round-trips the width", () => {
    const store = memory();
    expect(loadWidth(store)).toBeNull();
    saveWidth(612.4, store);
    expect(loadWidth(store)).toBe(612);
  });
});

describe("clampWidth", () => {
  it("bounds to 380px and to the room left beside the chat", () => {
    expect(clampWidth(100, 1000)).toBe(380);
    expect(clampWidth(900, 1000)).toBe(640);
    expect(clampWidth(500, 1000)).toBe(500);
    expect(clampWidth(1500, 2000)).toBe(1500);
    expect(clampWidth(1900, 2000)).toBe(1560);
  });
  it("keeps the chat at least 360px wide, after the session rail", () => {
    expect(dockRoom(1600, 260)).toBe(1600 - 260 - CHAT_MIN_WIDTH);
    expect(clampWidth(1400, 1600, dockRoom(1600, 260))).toBe(980);
    expect(dockRoom(2000)).toBe(1560);
  });
  it("shrinks the dock below its usual minimum on a small window", () => {
    expect(dockRoom(1100, 260)).toBe(480);
    expect(dockRoom(900, 260)).toBe(DOCK_FLOOR_WIDTH);
    expect(clampWidth(500, 1000, 340)).toBe(340);
    expect(clampWidth(100, 1000, 340)).toBe(340);
  });
});
