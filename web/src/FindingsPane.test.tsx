import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { FindingsPane, describeInterruption } from "./FindingsPane";
import type { Run } from "./types";
import {
  MOCK_RUN_STANDARD,
  MOCK_RUN_INTERRUPTED,
} from "./test-support/fixtures";
import { MockApiService } from "./test-support/mockApi";

describe("FindingsPane Component", () => {
  const mockApi = new MockApiService();

  it("displays expired session notice banner when session is expired (410)", () => {
    render(
      <FindingsPane
        run={MOCK_RUN_STANDARD}
        onSelectFinding={vi.fn()}
        api={mockApi}
        isExpired={true}
      />,
    );

    expect(screen.getByTestId("expired-session-notice")).toBeInTheDocument();
    expect(
      screen.getByText(/Sesja dokumentu wygasła \(410\)/i),
    ).toBeInTheDocument();
  });

  it("does not name a cause for an interruption that crossed no limit", () => {
    const onSelect = vi.fn();
    render(
      <FindingsPane
        run={{
          ...MOCK_RUN_INTERRUPTED,
          interruption_reason: undefined,
          metrics: { ...MOCK_RUN_INTERRUPTED.metrics!, elapsed_ms: 1_000 },
        }}
        onSelectFinding={onSelect}
        api={mockApi}
      />,
    );

    const banner = screen.getByTestId("interruption-banner");
    expect(banner).toBeInTheDocument();
    expect(banner.textContent).toContain(
      "Przerwanie analizy — wyczerpano budżet",
    );
    expect(banner).not.toHaveTextContent("Zużycie tokenów");
    expect(banner).not.toHaveTextContent("Czas analizy");
    expect(screen.getByTestId("units-not-processed-count").textContent).toBe(
      "7",
    );
  });

  it("does not report wall-time exhaustion without a configured wall budget", () => {
    render(
      <FindingsPane
        run={{
          ...MOCK_RUN_INTERRUPTED,
          interruption_reason: "wall_time" as const,
          wall_budget_seconds: undefined,
        }}
        onSelectFinding={vi.fn()}
        api={mockApi}
      />,
    );

    const banner = screen.getByTestId("interruption-banner");
    expect(banner).toHaveTextContent("Przerwanie analizy — wyczerpano budżet");
    expect(banner).not.toHaveTextContent("Czas analizy");
  });

  it("falls back to the general banner when interruption metrics are absent", () => {
    render(
      <FindingsPane
        run={{
          ...MOCK_RUN_INTERRUPTED,
          interruption_reason: "wall_time" as const,
          metrics: null,
        }}
        onSelectFinding={vi.fn()}
        api={mockApi}
      />,
    );

    const banner = screen.getByTestId("interruption-banner");
    expect(banner).toHaveTextContent("Przerwanie analizy — wyczerpano budżet");
    expect(banner).not.toHaveTextContent("Czas analizy");
  });

  it("identifies a wall-time interruption and shows used versus configured time", () => {
    render(
      <FindingsPane
        run={{
          ...MOCK_RUN_INTERRUPTED,
          interruption_reason: "wall_time" as const,
        }}
        onSelectFinding={vi.fn()}
        api={mockApi}
      />,
    );

    const banner = screen.getByTestId("interruption-banner");
    expect(banner).toHaveTextContent(
      "Przerwanie analizy — wyczerpano limit czasu",
    );
    expect(banner).toHaveTextContent("Czas analizy: 15.2 s / 15.0 s");
  });

  it("shows the server error for a failed run instead of prompting to start analysis", () => {
    const failedRun = {
      ...MOCK_RUN_STANDARD,
      status: "failed" as const,
      findings: [],
      error: {
        code: "model_transport_error" as const,
        message_pl: "Model nie odpowiedział w wyznaczonym limicie czasu.",
      },
    };

    render(
      <FindingsPane run={failedRun} onSelectFinding={vi.fn()} api={mockApi} />,
    );

    const paneText = screen.getByTestId("findings-pane").textContent;
    expect(paneText).toContain(
      "Model nie odpowiedział w wyznaczonym limicie czasu.",
    );
    expect(paneText).not.toContain("Uruchom analizę, aby uzyskać wyniki");
  });

  it("shows a cancelled state distinct from failure and completed without findings", () => {
    render(
      <FindingsPane
        run={{
          ...MOCK_RUN_STANDARD,
          status: "cancelled",
          findings: [],
          error: null,
        }}
        onSelectFinding={vi.fn()}
        api={mockApi}
      />,
    );

    const paneText = screen.getByTestId("findings-pane").textContent;
    expect(paneText).toContain("Analiza została anulowana.");
    expect(paneText).not.toContain("Analiza zakończyła się bez ustaleń.");
  });

  it("shows the completed without findings state", () => {
    render(
      <FindingsPane
        run={{ ...MOCK_RUN_STANDARD, findings: [] }}
        onSelectFinding={vi.fn()}
        api={mockApi}
      />,
    );

    expect(screen.getByTestId("findings-pane").textContent).toContain(
      "Analiza zakończyła się bez ustaleń.",
    );
  });

  it("shows that a run is still in progress", () => {
    render(
      <FindingsPane
        run={{
          ...MOCK_RUN_STANDARD,
          status: "running",
          findings: [],
          finished_at: null,
          error: null,
        }}
        onSelectFinding={vi.fn()}
        api={mockApi}
      />,
    );

    expect(screen.getByTestId("findings-pane").textContent).toContain(
      "Analiza jest w toku.",
    );
  });

  it("shows all new agent counters with the same metrics styling", () => {
    render(
      <FindingsPane
        run={MOCK_RUN_STANDARD}
        onSelectFinding={vi.fn()}
        api={mockApi}
      />,
    );

    const metrics = screen.getByLabelText("Metryki analizy");
    // The label says search turns, because that is what the server counts: the
    // analyst's characterise task never touches this counter.
    expect(metrics).toHaveTextContent("Tury wyszukiwania Wyszukującego: 12");
    expect(metrics).not.toHaveTextContent("Tury Analityka:");
    expect(metrics).toHaveTextContent("Wyszukiwania Wyszukującego: 7");
    expect(metrics).toHaveTextContent(
      "Jednostki z wyczerpanym limitem wyszukiwania: 0",
    );
    expect(metrics).toHaveTextContent("Tury Weryfikatora: 9");
    expect(metrics).toHaveTextContent("Odczyty przepisów: 4");
    // The counter 8/11 made durable, shown where the reader can see that a
    // characterisation was closed by the system rather than settled by a role.
    expect(metrics).toHaveTextContent("Automatycznie domknięte charaktery: 1");
  });

  it("renders a child run streamed under the parent when a message starts one", async () => {
    const childFinding = {
      ...MOCK_RUN_STANDARD.findings[3],
      id: "child-finding",
    };
    const childRun = {
      ...MOCK_RUN_STANDARD,
      id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
      status: "running" as const,
      measurement_valid: false,
      parent_run_id: MOCK_RUN_STANDARD.id,
      interaction: "contest" as const,
      findings: [],
      metrics: null,
    };
    const completedChild = {
      ...childRun,
      status: "completed" as const,
      findings: [childFinding],
      metrics: MOCK_RUN_STANDARD.metrics,
    };
    vi.spyOn(mockApi, "sendMessage").mockResolvedValueOnce({
      intent: "contest",
      run: childRun,
    });
    const streamRunEvents = vi
      .spyOn(mockApi, "streamRunEvents")
      .mockImplementation((_id, onEvent, onComplete) => {
        onEvent({
          kind: "worksheet",
          role: "verifier",
          entry_kind: "challenge",
          unit_id: "u-4",
        });
        onComplete(completedChild);
        return () => {};
      });
    render(
      <FindingsPane
        run={MOCK_RUN_STANDARD}
        onSelectFinding={vi.fn()}
        api={mockApi}
      />,
    );

    await userEvent.type(
      screen.getByLabelText("Wiadomość do czatu"),
      "Nie zgadzam się z tym ustaleniem.",
    );
    await userEvent.click(screen.getByLabelText("Wyślij"));

    expect(streamRunEvents).toHaveBeenCalledWith(
      childRun.id,
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
    );
    const child = await screen.findByTestId(`interaction-run-${childRun.id}`);
    expect(child).toHaveTextContent("Przebieg interaktywny (niemierzony)");
    expect(child).toHaveTextContent(
      "Ten przebieg nie zmienia zmierzonego wyniku analizy nadrzędnej.",
    );
    expect(child).toHaveTextContent("Weryfikator");
    const card = within(child).getByTestId("finding-card-child-finding");
    expect(card).toBeInTheDocument();
    expect(card).not.toHaveClass("selected");
  });
});

describe("describeInterruption", () => {
  it("returns null for non-interrupted runs or null/undefined", () => {
    expect(describeInterruption(null)).toBeNull();
    expect(describeInterruption(undefined)).toBeNull();
    expect(describeInterruption(MOCK_RUN_STANDARD)).toBeNull();
  });

  it("identifies wall time interruption", () => {
    const run: Run = {
      ...MOCK_RUN_STANDARD,
      interrupted: true,
      interruption_reason: "wall_time",
      wall_budget_seconds: 60,
      metrics: {
        ...MOCK_RUN_STANDARD.metrics!,
        elapsed_ms: 65_000,
      },
    };
    const desc = describeInterruption(run);
    expect(desc).toEqual({
      title: "Przerwanie analizy — wyczerpano limit czasu",
      kind: "wall_time",
      tokensUsed:
        MOCK_RUN_STANDARD.metrics!.input_tokens +
        MOCK_RUN_STANDARD.metrics!.output_tokens,
      elapsedSeconds: 65,
      wallBudgetSeconds: 60,
      unitsNotProcessedCount: 0,
    });
  });

  it("identifies general resource exhaustion", () => {
    const run: Run = {
      ...MOCK_RUN_STANDARD,
      interrupted: true,
      interruption_reason: null,
      metrics: {
        ...MOCK_RUN_STANDARD.metrics!,
        units_not_processed: 3,
        elapsed_ms: 10_000,
        input_tokens: 100,
        output_tokens: 100,
      },
    };
    const desc = describeInterruption(run);
    expect(desc?.kind).toBe("general");
    expect(desc?.title).toBe("Przerwanie analizy — wyczerpano budżet");
    expect(desc?.unitsNotProcessedCount).toBe(3);
  });
});
