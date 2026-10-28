// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { DOCK_TABS, clampWidth, closeTab, dockTabOf, loadTabs, loadWidth, openTab, saveTabs, saveWidth } from "./dock";

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
  it("bounds to 380px and 78% of the viewport", () => {
    expect(clampWidth(100, 1000)).toBe(380);
    expect(clampWidth(900, 1000)).toBe(780);
    expect(clampWidth(500, 1000)).toBe(500);
  });
});
