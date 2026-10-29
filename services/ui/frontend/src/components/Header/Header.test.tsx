// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Header } from "./Header";

const noop = () => undefined;
const render = (busy: boolean) =>
  renderToStaticMarkup(
    <Header
      theme="light"
      onToggleTheme={noop}
      connection="open"
      agent={null}
      gatewayError={null}
      title="EGFR binders"
      busy={busy}
      onReconnect={noop}
      railOpen
      onToggleRail={noop}
      workbenchOpen
      onToggleWorkbench={noop}
      onOpenPalette={noop}
    />,
  );

describe("Header", () => {
  it("names the conversation without a target line", () => {
    const html = render(false);
    expect(html).toContain("EGFR binders");
    expect(html).not.toContain("No target yet");
    expect(html).not.toContain("working…");
  });

  it("says when the agent is working", () => {
    expect(render(true)).toContain("working…");
  });
});
