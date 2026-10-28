// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { defaultObjectives, objectiveCatalogue, objectiveGroups } from "../../lib/ttp";
import { ObjectivePicker } from "./ObjectivePicker";

const catalogue = objectiveCatalogue({}, []);
const render = () => renderToStaticMarkup(<ObjectivePicker groups={objectiveGroups(catalogue)} objectives={defaultObjectives(catalogue)} onToggle={() => undefined} />);

describe("ObjectivePicker", () => {
  it("titles the groups without a group checkbox", () => {
    const html = render();
    expect(html).not.toContain('type="checkbox"');
    for (const name of ["Cofold", "Drug-likeness", "ADMET"]) expect(html).toContain(`>${name}</span>`);
    const checkboxes = html.match(/role="checkbox"/g) ?? [];
    expect(checkboxes).toHaveLength(catalogue.filter((e) => !e.extra).length);
  });

  it("ticks the default objectives, tinted by their group", () => {
    const html = render();
    expect(html.match(/aria-checked="true"/g)).toHaveLength(defaultObjectives(catalogue).length);
    expect(html).toContain("background:var(--color-structure)");
    expect(html).toContain("background:var(--color-ok)");
    expect(html).toContain("background:var(--color-table)");
  });
});
