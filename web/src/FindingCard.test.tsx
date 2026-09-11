import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { FindingCard } from "./FindingCard";
import {
  MOCK_RUN_STANDARD,
  MOCK_WORKSHEET_STANDARD,
} from "./test-support/fixtures";
import type { WorksheetState } from "./WorksheetView";

describe("FindingCard Component", () => {
  const defaultWorksheetState: WorksheetState = {};
  const noop = () => {};

  it("renders all prominence levels correctly with exact data-level attributes", () => {
    const criticalFinding = MOCK_RUN_STANDARD.findings.find(
      (f) => f.prominence === "critical",
    )!;
    const warningFinding = MOCK_RUN_STANDARD.findings.find(
      (f) => f.prominence === "warning",
    )!;
    const neutralFinding = MOCK_RUN_STANDARD.findings.find(
      (f) => f.prominence === "neutral",
    )!;

    const { rerender } = render(
      <FindingCard
        finding={criticalFinding}
        selected={false}
        onSelect={noop}
        worksheetState={defaultWorksheetState}
        onLoadWorksheet={noop}
      />,
    );

    const critCard = screen.getByTestId(`finding-card-${criticalFinding.id}`);
    expect(critCard).toHaveAttribute("data-level", "critical");
    expect(critCard.textContent).toContain("Sprzeczność z normą prawną");

    rerender(
      <FindingCard
        finding={warningFinding}
        selected={false}
        onSelect={noop}
        worksheetState={defaultWorksheetState}
        onLoadWorksheet={noop}
      />,
    );
    const warnCard = screen.getByTestId(`finding-card-${warningFinding.id}`);
    expect(warnCard).toHaveAttribute("data-level", "warning");
    expect(warnCard.textContent).toContain("Dopuszczalna modyfikacja normy");

    rerender(
      <FindingCard
        finding={neutralFinding}
        selected={false}
        onSelect={noop}
        worksheetState={defaultWorksheetState}
        onLoadWorksheet={noop}
      />,
    );
    const neutCard = screen.getByTestId(`finding-card-${neutralFinding.id}`);
    expect(neutCard).toHaveAttribute("data-level", "neutral");
    expect(neutCard.textContent).toContain("Zgodność z podstawą prawną");
  });

  it("displays plain message for unresolved anchor without approximate position", () => {
    // f-9 has anchor_resolved = false
    const f9 = MOCK_RUN_STANDARD.findings.find((f) => f.id === "f-9")!;
    render(
      <FindingCard
        finding={f9}
        selected={false}
        onSelect={noop}
        worksheetState={defaultWorksheetState}
        onLoadWorksheet={noop}
      />,
    );

    const f9Card = screen.getByTestId("finding-card-f-9");
    expect(f9Card).toBeInTheDocument();
    expect(f9Card.textContent).toContain("Zakotwiczenie nierozstrzygnięte");
    expect(f9Card.textContent).toContain("nie jest przybliżana");
  });

  it("renders provision character in Polish without the wire token", () => {
    const f1 = MOCK_RUN_STANDARD.findings.find((f) => f.id === "f-1")!;
    render(
      <FindingCard
        finding={f1}
        selected={false}
        onSelect={noop}
        worksheetState={defaultWorksheetState}
        onLoadWorksheet={noop}
      />,
    );

    const card = screen.getByTestId("finding-card-f-1");
    expect(card).not.toHaveTextContent("imperative");
    expect(card).toHaveTextContent("Charakter normy: imperatywny");
  });

  it("hides raw confidence while retaining the uncertainty cause", () => {
    const f5 = MOCK_RUN_STANDARD.findings.find((f) => f.id === "f-5")!;
    render(
      <FindingCard
        finding={f5}
        selected={false}
        onSelect={noop}
        worksheetState={defaultWorksheetState}
        onLoadWorksheet={noop}
      />,
    );

    const card = screen.getByTestId("finding-card-f-5");
    // The value as it would actually appear. Asserting "62%" guarded nothing:
    // rendering the field raw prints 0.62, which does not contain that string.
    expect(card).not.toHaveTextContent(String(f5.raw_confidence));
    expect(card).not.toHaveTextContent("62%");
    expect(card).toHaveTextContent(
      "System nie rozstrzygnął charakteru normy prawnej (imperatywny/dyspozytywny/semiimperatywny).",
    );
  });

  it("invokes onSelect callback when a finding card is clicked", async () => {
    const f4 = MOCK_RUN_STANDARD.findings.find((f) => f.id === "f-4")!;
    const onSelect = vi.fn();
    render(
      <FindingCard
        finding={f4}
        selected={false}
        onSelect={onSelect}
        worksheetState={defaultWorksheetState}
        onLoadWorksheet={noop}
      />,
    );

    const card = screen.getByTestId("finding-card-f-4");
    await userEvent.click(card);
    expect(onSelect).toHaveBeenCalledWith(f4);
  });

  it("invokes onSelect callback on Enter and Space key presses", async () => {
    const f4 = MOCK_RUN_STANDARD.findings.find((f) => f.id === "f-4")!;
    const onSelect = vi.fn();
    render(
      <FindingCard
        finding={f4}
        selected={false}
        onSelect={onSelect}
        worksheetState={defaultWorksheetState}
        onLoadWorksheet={noop}
      />,
    );

    const card = screen.getByTestId("finding-card-f-4");
    card.focus();

    await userEvent.keyboard("{Enter}");
    expect(onSelect).toHaveBeenCalledWith(f4);

    onSelect.mockClear();
    await userEvent.keyboard(" ");
    expect(onSelect).toHaveBeenCalledWith(f4);
  });

  it("expands and collapses worksheet when clicking toggle button", async () => {
    const f4 = MOCK_RUN_STANDARD.findings.find((f) => f.id === "f-4")!;
    const onLoadWorksheet = vi.fn();
    const loadedWorksheetState: WorksheetState = {
      data: {
        ...MOCK_WORKSHEET_STANDARD,
        units: [
          {
            unit_id: "u-4",
            entries: [...MOCK_WORKSHEET_STANDARD.units[0].entries].reverse(),
          },
        ],
      },
    };

    const { rerender } = render(
      <FindingCard
        finding={f4}
        selected={false}
        onSelect={noop}
        worksheetState={{}}
        onLoadWorksheet={onLoadWorksheet}
      />,
    );

    const card = screen.getByTestId("finding-card-f-4");
    const toggle = within(card).getByRole("button", {
      name: "Pokaż pracę agentów",
    });
    await userEvent.click(toggle);

    expect(onLoadWorksheet).toHaveBeenCalledOnce();

    // Rerender with data loaded
    rerender(
      <FindingCard
        finding={f4}
        selected={false}
        onSelect={noop}
        worksheetState={loadedWorksheetState}
        onLoadWorksheet={onLoadWorksheet}
      />,
    );

    const transcript = await screen.findByTestId("worksheet-entries-u-4");
    expect(transcript).toHaveTextContent("Analityk");
    expect(transcript).toHaveTextContent("Weryfikator");
    expect(transcript).toHaveTextContent("bez uprzedniego wezwania");

    // Close worksheet
    const hideToggle = within(card).getByRole("button", {
      name: "Ukryj pracę agentów",
    });
    await userEvent.click(hideToggle);
    expect(screen.queryByTestId("worksheet-entries-u-4")).toBeNull();
  });

  it("renders the content-expired message when the worksheet returns 410", async () => {
    const f4 = MOCK_RUN_STANDARD.findings.find((f) => f.id === "f-4")!;
    const worksheetState: WorksheetState = {
      error: {
        code: "content_expired",
        message_pl: "Sesja dokumentu wygasła.",
      },
    };

    render(
      <FindingCard
        finding={f4}
        selected={false}
        onSelect={noop}
        worksheetState={worksheetState}
        onLoadWorksheet={noop}
      />,
    );

    const card = screen.getByTestId("finding-card-f-4");
    await userEvent.click(
      within(card).getByRole("button", {
        name: "Pokaż pracę agentów",
      }),
    );

    expect(
      await screen.findByText(
        "Sesja dokumentu wygasła (410). Praca agentów jest niedostępna.",
      ),
    ).toBeInTheDocument();
  });
});
