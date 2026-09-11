import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AgentLog, appendAgentLogEvent } from "./AgentLog";
import type { RunEvent } from "./types";

describe("AgentLog", () => {
  it("renders arriving worksheet events in Polish and retains only the last 200", () => {
    const view = render(<AgentLog events={[]} />);
    let events: RunEvent[] = [];

    for (let index = 0; index < 201; index += 1) {
      events = appendAgentLogEvent(events, {
        kind: "worksheet",
        role: index % 2 === 0 ? "analyst" : "verifier",
        entry_kind: index % 2 === 0 ? "candidate" : "verdict",
        unit_id: `unit-${index}`,
      });
    }
    view.rerender(<AgentLog events={events} />);

    expect(screen.getAllByTestId("agent-log-entry")).toHaveLength(200);
    expect(screen.queryByText("unit-0")).not.toBeInTheDocument();
    expect(screen.getByText("unit-200")).toBeInTheDocument();
    expect(screen.getAllByText("Analityk").length).toBeGreaterThan(0);
    expect(screen.getAllByText("ocena relacji").length).toBeGreaterThan(0);
  });

  it("ignores non-worksheet and incomplete worksheet events", () => {
    const events = [
      { kind: "status", status: "running" },
      { kind: "worksheet", role: "analyst", entry_kind: "search" },
    ] as RunEvent[];

    expect(events.reduce(appendAgentLogEvent, [] as RunEvent[])).toHaveLength(
      0,
    );
  });
});
