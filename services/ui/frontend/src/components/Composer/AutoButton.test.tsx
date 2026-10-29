// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { AutoButton } from "./AutoButton";
import { BUDGETS, budgetLabel, countdown } from "./autoBudget";

const render = (until: string | null) => renderToStaticMarkup(<AutoButton until={until} onSet={() => undefined} />);
const inMinutes = (minutes: number) => new Date(Date.now() + minutes * 60_000).toISOString();

describe("countdown", () => {
  it("stays short at every scale", () => {
    expect(countdown(4 * 3600_000)).toBe("4h 00m");
    expect(countdown(95 * 60_000)).toBe("1h 35m");
    expect(countdown(9 * 60_000)).toBe("9m");
    expect(countdown(30_000)).toBe("30s");
    expect(countdown(-5_000)).toBe("0s");
  });

  it("names the offered budgets in hours once they are long", () => {
    expect(BUDGETS.map(budgetLabel)).toEqual(["30 min", "1 h", "2 h", "4 h"]);
  });
});

describe("AutoButton", () => {
  it("is one composer button until the agent is given a budget", () => {
    const html = render(null);
    expect(html).toContain('aria-label="Work on its own"');
    expect(html).toContain('aria-haspopup="menu"');
    expect(html).not.toContain("Auto ");
    expect(html).not.toContain('aria-label="Stop working on its own"');
  });

  it("counts the budget down, with a way to stop it", () => {
    const html = render(inMinutes(95));
    expect(html).toContain("Auto 1h 3");
    expect(html).toContain('aria-label="Stop working on its own"');
    // While it counts down the button that offers budgets is out of the way.
    expect(html).not.toContain('aria-label="Work on its own"');
  });

  it("goes back to the button once the budget has run out", () => {
    const html = render(inMinutes(-1));
    expect(html).toContain('aria-label="Work on its own"');
    expect(html).not.toContain('aria-label="Stop working on its own"');
  });
});
