import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeAll } from "vitest";
import { DocumentPane } from "./DocumentPane";
import {
  MOCK_DOC_CONTENT_INTERRUPTED,
  MOCK_DOC_CONTENT_STANDARD,
  MOCK_RUN_INTERRUPTED,
  MOCK_RUN_STANDARD,
} from "./test-support/fixtures";
import type { Finding } from "./types";

beforeAll(() => {
  if (typeof Element.prototype.scrollIntoView === "undefined") {
    Element.prototype.scrollIntoView = vi.fn();
  }
});

describe("DocumentPane Component", () => {
  it("renders canonical blocks for text/Word input", () => {
    render(
      <DocumentPane
        content={MOCK_DOC_CONTENT_STANDARD}
        selectedFinding={null}
      />,
    );

    expect(screen.getByTestId("canonical-blocks")).toBeInTheDocument();
    expect(
      screen.getByText(/UMOWA NAJMU LOKALU UŻYTKOWEGO/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/§ 1. Przedmiot umowy/i)).toBeInTheDocument();
  });

  it("marks corresponding exact span when finding with resolved anchor is selected", () => {
    const selectedFinding: Finding = {
      id: "f-1",
      unit_id: "u-1",
      code: "consistent",
      prominence: "neutral",
      anchor_resolved: true,
      anchor: {
        start_offset: 166,
        end_offset: 224,
        page: 1,
        bbox: null,
        line_start: 3,
        line_end: 5,
      },
    };

    render(
      <DocumentPane
        content={MOCK_DOC_CONTENT_STANDARD}
        selectedFinding={selectedFinding}
      />,
    );

    const mark = screen.getByTestId("anchor-highlight-span");
    expect(mark).toBeInTheDocument();
    expect(mark.tagName.toLowerCase()).toBe("mark");
    expect(mark.textContent).toBe(
      "Wynajmujący oddaje Najemcy do używania lokal użytkowy nr 4",
    );

    const block1 = screen.getByTestId("block-1");
    expect(block1).toContainElement(mark);
    expect(block1.textContent).toContain("§ 1. Przedmiot umowy. ");
  });

  it("marks exactly one span across the document when finding with resolved anchor is selected", () => {
    const selectedFinding: Finding = {
      id: "f-1",
      unit_id: "u-1",
      code: "consistent",
      prominence: "neutral",
      anchor_resolved: true,
      anchor: {
        start_offset: 166,
        end_offset: 224,
        page: 1,
        bbox: null,
        line_start: 3,
        line_end: 5,
      },
    };

    const { container } = render(
      <DocumentPane
        content={MOCK_DOC_CONTENT_STANDARD}
        selectedFinding={selectedFinding}
      />,
    );

    const marks = container.querySelectorAll(
      '[data-testid="anchor-highlight-span"]',
    );
    expect(marks).toHaveLength(1);
    expect(screen.getByTestId("block-1")).toContainElement(
      marks[0] as HTMLElement,
    );
    expect(
      screen
        .getByTestId("block-0")
        .querySelector('[data-testid="anchor-highlight-span"]'),
    ).toBeNull();
  });

  it("marks only strict sub-span of a block with text before and after remaining unmarked", () => {
    const selectedFinding: Finding = {
      id: "f-sub",
      unit_id: "u-1",
      code: "consistent",
      prominence: "neutral",
      anchor_resolved: true,
      anchor: {
        start_offset: 166,
        end_offset: 188,
        page: 1,
      },
    };

    render(
      <DocumentPane
        content={MOCK_DOC_CONTENT_STANDARD}
        selectedFinding={selectedFinding}
      />,
    );

    const block1 = screen.getByTestId("block-1");
    const mark = screen.getByTestId("anchor-highlight-span");
    expect(mark.textContent).toBe("Wynajmujący oddaje Naj");
    expect(block1.textContent).toContain("§ 1. Przedmiot umowy. ");
    expect(block1.textContent).toContain(
      "emcy do używania lokal użytkowy nr 4",
    );
  });

  it("renders explicit unresolved statement when selected finding has anchor_resolved: false", () => {
    const unresolvedFinding: Finding = {
      id: "f-unresolved",
      unit_id: "u-9",
      code: "uncertain",
      prominence: "warning",
      anchor_resolved: false,
    };

    render(
      <DocumentPane
        content={MOCK_DOC_CONTENT_STANDARD}
        selectedFinding={unresolvedFinding}
      />,
    );

    const note = screen.getByTestId("unresolved-anchor-note");
    expect(note).toBeInTheDocument();
    expect(note.textContent).toContain("Zakotwiczenie nierozstrzygnięte:");
    expect(note.textContent).toContain(
      "System nie powiązał ustalenia z konkretnym fragmentem tekstu. Pozycja w dokumencie nie jest przybliżana.",
    );
    expect(screen.queryByTestId("anchor-highlight-span")).toBeNull();
  });

  it("renders no mark when finding offsets do not lie inside any block", () => {
    const outOfBoundsFinding: Finding = {
      id: "f-oob",
      unit_id: "u-1",
      code: "consistent",
      prominence: "neutral",
      anchor_resolved: true,
      anchor: {
        start_offset: 9000,
        end_offset: 9050,
      },
    };

    const { container } = render(
      <DocumentPane
        content={MOCK_DOC_CONTENT_STANDARD}
        selectedFinding={outOfBoundsFinding}
      />,
    );

    expect(
      container.querySelector('[data-testid="anchor-highlight-span"]'),
    ).toBeNull();
  });

  it("renders no mark when finding span crosses block boundaries or is invalid", () => {
    const crossBlockFinding: Finding = {
      id: "f-cross",
      unit_id: "u-1",
      code: "consistent",
      prominence: "neutral",
      anchor_resolved: true,
      anchor: {
        start_offset: 100,
        end_offset: 200,
      },
    };

    const { container } = render(
      <DocumentPane
        content={MOCK_DOC_CONTENT_STANDARD}
        selectedFinding={crossBlockFinding}
      />,
    );

    expect(
      container.querySelector('[data-testid="anchor-highlight-span"]'),
    ).toBeNull();
  });

  it("displays expired notice when session is purged / expired (410)", () => {
    render(
      <DocumentPane content={null} selectedFinding={null} isExpired={true} />,
    );

    expect(screen.getByTestId("document-expired-pane")).toBeInTheDocument();
    expect(screen.getByText(/Treść dokumentu wygasła/i)).toBeInTheDocument();
  });

  describe("DocumentPane Expandable Finding Blocks", () => {
    it.each([
      [["critical", "warning", "neutral"], "status-critical", "Krytyczne"],
      [["warning", "neutral"], "status-warning", "Ostrzeżenie"],
      [["neutral"], "status-neutral", "Informacja"],
    ] as const)(
      "shows the most severe of %j as %s",
      (prominences, expectedClass, expectedLabel) => {
        const block = MOCK_DOC_CONTENT_STANDARD.blocks![0];
        const findings = prominences.map((prominence, index) => ({
          ...MOCK_RUN_STANDARD.findings[0],
          id: `p-${index}`,
          block_id: block.id,
          prominence,
        })) as Finding[];

        render(
          <DocumentPane
            content={MOCK_DOC_CONTENT_STANDARD}
            selectedFinding={null}
            run={{ ...MOCK_RUN_STANDARD, findings }}
          />,
        );

        const rendered = screen.getByTestId(`block-${block.id!.split("-")[1]}`);
        expect(rendered).toHaveClass(expectedClass);
        expect(rendered).toHaveTextContent(expectedLabel);
      },
    );

    // Two codes say the corpus was searched and answered nothing; every other
    // non-adjudicable code is a result about the clause and stays neutral.
    it.each([
      ["no_basis_found", "status-not_found"],
      ["no_relation", "status-not_found"],
      ["unit_not_adjudicable", "status-neutral"],
      ["basis_not_in_force", "status-neutral"],
    ])("badges a %s finding as %s", (code, expectedClass) => {
      const block = MOCK_DOC_CONTENT_STANDARD.blocks![0];
      render(
        <DocumentPane
          content={MOCK_DOC_CONTENT_STANDARD}
          selectedFinding={null}
          run={{
            ...MOCK_RUN_STANDARD,
            findings: [
              {
                ...MOCK_RUN_STANDARD.findings[0],
                id: "nb-1",
                block_id: block.id,
                code,
                prominence: "neutral",
              },
            ] as Finding[],
          }}
        />,
      );

      const rendered = screen.getByTestId(`block-${block.id!.split("-")[1]}`);
      expect(rendered).toHaveClass(expectedClass);
    });

    it("renders status badges and icons on blocks with findings", () => {
      render(
        <DocumentPane
          content={MOCK_DOC_CONTENT_STANDARD}
          selectedFinding={null}
          run={MOCK_RUN_STANDARD}
        />,
      );

      // Block 4 (f-4) is critical (contradictory)
      const block4 = screen.getByTestId("block-4");
      expect(block4).toHaveClass("has-findings");
      expect(block4).toHaveClass("status-critical");
      expect(block4).toHaveTextContent("Krytyczne");

      // Block 3 (f-3) is warning (permissible_departure)
      const block3 = screen.getByTestId("block-3");
      expect(block3).toHaveClass("has-findings");
      expect(block3).toHaveClass("status-warning");
      expect(block3).toHaveTextContent("Ostrzeżenie");

      // Block 1 (f-1) is neutral (consistent)
      const block1 = screen.getByTestId("block-1");
      expect(block1).toHaveClass("has-findings");
      expect(block1).toHaveClass("status-neutral");
      expect(block1).toHaveTextContent("Informacja");

      // Block 0 is the contract header/title and has no findings
      const block0 = screen.getByTestId("block-0");
      expect(block0).not.toHaveClass("has-findings");
    });

    it("expands and collapses block details when clicking the header or toggle", async () => {
      const onSelectFinding = vi.fn();
      render(
        <DocumentPane
          content={MOCK_DOC_CONTENT_STANDARD}
          selectedFinding={null}
          onSelectFinding={onSelectFinding}
          run={MOCK_RUN_STANDARD}
        />,
      );

      const block4Findings = screen.getByTestId("block-findings-4");
      expect(block4Findings).toHaveClass("collapsed");

      // Click toggle button on block 4
      const block4 = screen.getByTestId("block-4");
      const toggleBtn = block4.querySelector(".btn-toggle-expand")!;
      await userEvent.click(toggleBtn);

      expect(block4Findings).toHaveClass("open");
      expect(block4Findings).not.toHaveClass("collapsed");

      // Details are now visible inside the block
      const card4 = screen.getByTestId("finding-card-f-4");
      expect(card4).toBeInTheDocument();
      expect(card4).toHaveTextContent("Sprzeczność z normą prawną");

      // Click header again to collapse
      const header4 = block4.querySelector(".doc-block-header")!;
      await userEvent.click(header4);
      expect(block4Findings).toHaveClass("collapsed");
    });

    it("expands all blocks and collapses all blocks with toolbar actions", async () => {
      render(
        <DocumentPane
          content={MOCK_DOC_CONTENT_STANDARD}
          selectedFinding={null}
          run={MOCK_RUN_STANDARD}
        />,
      );

      const expandAllBtn = screen.getByRole("button", {
        name: /^Rozwiń wszystkie$/i,
      });
      await userEvent.click(expandAllBtn);

      expect(screen.getByTestId("block-findings-1")).toHaveClass("open");
      expect(screen.getByTestId("block-findings-3")).toHaveClass("open");
      expect(screen.getByTestId("block-findings-4")).toHaveClass("open");

      const collapseAllBtn = screen.getByRole("button", {
        name: /^Zwiń wszystkie$/i,
      });
      await userEvent.click(collapseAllBtn);

      expect(screen.getByTestId("block-findings-1")).toHaveClass("collapsed");
      expect(screen.getByTestId("block-findings-3")).toHaveClass("collapsed");
      expect(screen.getByTestId("block-findings-4")).toHaveClass("collapsed");
    });

    it("filters blocks by prominence when filter buttons are clicked", async () => {
      render(
        <DocumentPane
          content={MOCK_DOC_CONTENT_STANDARD}
          selectedFinding={null}
          run={MOCK_RUN_STANDARD}
        />,
      );

      // Filter by 'Krytyczne' (2 findings)
      const critBtn = screen.getByRole("button", { name: /Krytyczne \(2\)/i });
      await userEvent.click(critBtn);

      // Only blocks with critical findings should be shown
      expect(screen.getByTestId("block-4")).toBeInTheDocument();
      expect(screen.queryByTestId("block-1")).toBeNull();
      expect(screen.queryByTestId("block-2")).toBeNull();

      // Reset to all
      const allBtn = screen.getByRole("button", { name: /Wszystkie \(9\)/i });
      await userEvent.click(allBtn);
      expect(screen.getByTestId("block-1")).toBeInTheDocument();
      expect(screen.getByTestId("block-4")).toBeInTheDocument();
    });

    it("keeps a filtered block's other findings visible beside the match", async () => {
      // The filter selects blocks, not cards: a block survives because it holds
      // one finding of the chosen prominence, and is then shown whole. Reading a
      // clause with its critical finding but without the neutral one sitting on
      // the same clause would misrepresent what the system said about it. This is
      // now the only prominence filter in the interface, so the rule is pinned.
      const critical = MOCK_RUN_STANDARD.findings.find(
        (finding) => finding.prominence === "critical",
      )!;
      const neutral = MOCK_RUN_STANDARD.findings.find(
        (finding) => finding.prominence === "neutral",
      )!;
      const sharedBlock = {
        ...MOCK_RUN_STANDARD,
        findings: [critical, { ...neutral, block_id: critical.block_id }],
      };

      render(
        <DocumentPane
          content={MOCK_DOC_CONTENT_STANDARD}
          selectedFinding={null}
          run={sharedBlock}
        />,
      );

      await userEvent.click(
        screen.getByRole("button", { name: /Krytyczne \(1\)/i }),
      );

      const block = screen.getByTestId(`${critical.block_id}`);
      expect(
        block.querySelector(`[data-testid="finding-card-${critical.id}"]`),
      ).not.toBeNull();
      expect(
        block.querySelector(`[data-testid="finding-card-${neutral.id}"]`),
      ).not.toBeNull();
    });

    it("auto-expands block containing selectedFinding", () => {
      const finding4 = MOCK_RUN_STANDARD.findings.find((f) => f.id === "f-4")!;
      render(
        <DocumentPane
          content={MOCK_DOC_CONTENT_STANDARD}
          selectedFinding={finding4}
          run={MOCK_RUN_STANDARD}
        />,
      );

      const block4Findings = screen.getByTestId("block-findings-4");
      expect(block4Findings).toHaveClass("open");
    });
  });
});

describe("the fixtures a document pane is rendered from", () => {
  it("places every finding in a block its own document has", () => {
    // MOCK_RUN_INTERRUPTED carried block-0..2 while its document opens at
    // block-19, so every finding of the interrupted pair was unplaced and the
    // tests using it grouped nothing. Nothing failed, because those tests assert
    // on the banner; the grouping simply had no input.
    const pairs = [
      ["standard", MOCK_RUN_STANDARD, MOCK_DOC_CONTENT_STANDARD],
      ["interrupted", MOCK_RUN_INTERRUPTED, MOCK_DOC_CONTENT_INTERRUPTED],
    ] as const;

    for (const [name, run, content] of pairs) {
      const blockIds = new Set((content.blocks ?? []).map((b) => b.id));
      const placed = (run.findings ?? []).filter((f) => f.block_id);
      expect(
        placed.length,
        `${name}: no finding carries a block_id`,
      ).toBeGreaterThan(0);
      expect(
        placed.filter((f) => !blockIds.has(f.block_id!)).map((f) => f.id),
        `${name}: findings placed in blocks the document does not have`,
      ).toEqual([]);
    }
  });
});
