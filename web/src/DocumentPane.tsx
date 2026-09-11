import React, { useEffect, useMemo, useRef, useState } from "react";
import type { DocumentContent, Finding, Prominence, Run } from "./types";
import {
  FileText,
  AlertOctagon,
  AlertTriangle,
  SearchX,
  Info,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { FindingCard } from "./FindingCard";
import type { WorksheetState } from "./WorksheetView";
type BlockStatusType = "critical" | "warning" | "not_found" | "neutral";

const BLOCK_STATUS_META = {
  critical: {
    icon: AlertOctagon,
    color: "var(--critical-text)",
    label: "Krytyczne",
    title: "Krytyczne ustalenie",
  },
  warning: {
    icon: AlertTriangle,
    color: "var(--warning-text)",
    label: "Ostrzeżenie",
    title: "Ostrzeżenie",
  },
  not_found: {
    icon: SearchX,
    color: undefined,
    label: "Brak podstawy",
    title: "Brak podstawy w korpusie",
  },
  neutral: {
    icon: Info,
    color: "var(--neutral-text)",
    label: "Informacja",
    title: "Informacja",
  },
} as const satisfies Record<BlockStatusType, unknown>;

function getBlockStatus(findings: Finding[]): {
  status: BlockStatusType | null;
  primaryFinding: Finding | null;
  count: number;
} {
  if (!findings || findings.length === 0) {
    return { status: null, primaryFinding: null, count: 0 };
  }

  const count = findings.length;
  const critical = findings.find((f) => f.prominence === "critical");
  if (critical) return { status: "critical", primaryFinding: critical, count };

  const warning = findings.find((f) => f.prominence === "warning");
  if (warning) return { status: "warning", primaryFinding: warning, count };

  // Only these two say the corpus was searched and answered nothing. Other
  // non-adjudicable codes are results about the clause, so they stay neutral.
  const notFound = findings.find(
    (f) => f.code === "no_basis_found" || f.code === "no_relation",
  );
  if (notFound) return { status: "not_found", primaryFinding: notFound, count };

  const neutral = findings.find((f) => f.prominence === "neutral");
  return { status: "neutral", primaryFinding: neutral ?? findings[0], count };
}

interface DocumentPaneProps {
  content: DocumentContent | null;
  selectedFinding: Finding | null;
  onSelectFinding?: (finding: Finding | null) => void;
  isExpired?: boolean;
  run?: Run | null;
  worksheets?: Record<string, WorksheetState>;
  onLoadWorksheet?: (runId: string) => void;
}

const noop = () => {};

/** What a worksheet is once the retention window that held it has closed. */
const EXPIRED_WORKSHEET: WorksheetState = {
  error: {
    code: "content_expired",
    message_pl: "Sesja dokumentu wygasła. Praca agentów jest niedostępna.",
  },
};

export const DocumentPane: React.FC<DocumentPaneProps> = ({
  content,
  selectedFinding,
  onSelectFinding,
  isExpired,
  run,
  worksheets,
  onLoadWorksheet,
}) => {
  const blockRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  const [filter, setFilter] = useState<Prominence | "all">("all");
  const [expandedBlockKeys, setExpandedBlockKeys] = useState<Set<string>>(
    new Set(),
  );

  const findingsCount = run?.findings?.length ?? 0;

  const { findingsByBlock, prominenceCounts } = useMemo(() => {
    const map = new Map<string, Finding[]>();
    const counts: Record<Prominence, number> = {
      critical: 0,
      warning: 0,
      neutral: 0,
    };
    if (run?.findings) {
      for (const f of run.findings) {
        counts[f.prominence] = (counts[f.prominence] || 0) + 1;
        if (f.block_id) {
          const list = map.get(f.block_id);
          if (list) {
            list.push(f);
          } else {
            map.set(f.block_id, [f]);
          }
        }
      }
    }
    return { findingsByBlock: map, prominenceCounts: counts };
  }, [run?.findings]);

  const count = (p: Prominence) => prominenceCounts[p] ?? 0;

  const renderFindingCard = (finding: Finding) => (
    <FindingCard
      key={finding.id}
      finding={finding}
      selected={selectedFinding?.id === finding.id}
      onSelect={onSelectFinding || noop}
      worksheetState={worksheets?.[run?.id ?? ""] || {}}
      onLoadWorksheet={() => run?.id && onLoadWorksheet?.(run.id)}
    />
  );

  const toggleBlockExpanded = (
    key: string,
    primaryFinding?: Finding | null,
  ) => {
    const willExpand = !expandedBlockKeys.has(key);
    setExpandedBlockKeys((prev) => {
      const next = new Set(prev);
      if (willExpand) {
        next.add(key);
      } else {
        next.delete(key);
      }
      return next;
    });
    // Selecting on collapse would trigger auto-expansion and reopen the block.
    if (
      willExpand &&
      primaryFinding &&
      onSelectFinding &&
      (!selectedFinding || selectedFinding.id !== primaryFinding.id)
    ) {
      onSelectFinding(primaryFinding);
    }
  };

  const expandAllBlocks = () => {
    if (!content?.blocks) return;
    const allKeys = new Set<string>();
    content.blocks.forEach((b, i) => {
      const key = b.id || `block-${i}`;
      allKeys.add(key);
    });
    setExpandedBlockKeys(allKeys);
  };

  const collapseAllBlocks = () => {
    setExpandedBlockKeys(new Set());
  };

  useEffect(() => {
    if (!selectedFinding?.block_id || !content?.blocks) return;
    const blocks = content.blocks;
    setExpandedBlockKeys((prev) => {
      const index = blocks.findIndex(
        (block, idx) =>
          (block.id || `block-${idx}`) === selectedFinding.block_id,
      );
      if (index === -1) return prev;
      const key = blocks[index].id || `block-${index}`;
      if (prev.has(key)) return prev;
      return new Set(prev).add(key);
    });
  }, [selectedFinding, content?.blocks]);

  // RF-13 permits highlighting only server-resolved anchors.
  const resolvedAnchor = selectedFinding?.anchor_resolved
    ? selectedFinding.anchor
    : null;

  useEffect(() => {
    if (selectedFinding?.block_id) {
      const el = blockRefs.current.get(selectedFinding.block_id);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  }, [selectedFinding, content]);

  // Do not guess a block for findings the server could not place.
  const unplacedFindings = useMemo(() => {
    const unplaced = (run?.findings ?? []).filter((f) => !f.block_id);
    // Keep a selected finding visible even when the run list is unavailable.
    if (
      selectedFinding &&
      !selectedFinding.block_id &&
      !unplaced.some((f) => f.id === selectedFinding.id)
    ) {
      return [...unplaced, selectedFinding];
    }
    return unplaced;
  }, [run?.findings, selectedFinding]);

  const blocksToRender = useMemo(() => {
    if (!content?.blocks) return [];
    if (filter === "all" || findingsCount === 0) {
      return content.blocks.map((block, idx) => ({ block, idx }));
    }
    return content.blocks
      .map((block, idx) => ({ block, idx }))
      .filter(({ block }) =>
        (findingsByBlock.get(block.id) ?? []).some(
          (f) => f.prominence === filter,
        ),
      );
  }, [content?.blocks, findingsCount, filter, findingsByBlock]);

  if (isExpired) {
    return (
      <div className="pane-content" data-testid="document-expired-pane">
        <div className="expired-banner" role="alert">
          <strong>Treść dokumentu wygasła</strong>
          <p>
            Treść dokumentu została usunięta z pamięci serwera zgodnie z
            polityką prywatności sesji (410 ContentExpired).
          </p>
        </div>
        {run && findingsCount > 0 && (
          <div className="doc-blocks-container" data-testid="canonical-blocks">
            <div data-testid="unplaced-findings">
              <div className="doc-block-header">
                <strong>Ustalenia z wygasłej sesji ({findingsCount})</strong>
              </div>
              <div className="doc-blocks-list" data-testid="findings-list">
                {run?.findings?.map((finding) => (
                  <FindingCard
                    key={finding.id}
                    finding={finding}
                    selected={selectedFinding?.id === finding.id}
                    onSelect={(f) => onSelectFinding?.(f)}
                    // Expired worksheets return 410; show that state without refetching.
                    worksheetState={EXPIRED_WORKSHEET}
                    onLoadWorksheet={noop}
                  />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  if (!content) {
    return (
      <div
        className="pane-content"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
        }}
      >
        <div style={{ textAlign: "center", color: "var(--text-muted)" }}>
          <FileText size={48} style={{ opacity: 0.3, marginBottom: "1rem" }} />
          <p>
            Prześlij dokument (.txt, .docx, .doc, .pdf), aby rozpocząć analizę.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="pane-content" tabIndex={0} aria-label="Podgląd dokumentu">
      {content?.read_mode && (
        <div className="disclosure-banner" style={{ marginBottom: "1rem" }}>
          <strong>Widok kanoniczny:</strong> System pokazuje bloki tekstu
          odczytane z dokumentu, a nie jego oryginalny układ graficzny.
          Ustalenia są zakotwiczone w tym tekście.
        </div>
      )}

      {
        <div className="doc-blocks-container" data-testid="canonical-blocks">
          {unplacedFindings.length > 0 && (
            <div data-testid="unplaced-findings">
              <div className="doc-block-header">
                <strong>Ustalenia bez rozwiązanej kotwicy</strong>
              </div>
              {unplacedFindings.map(renderFindingCard)}
            </div>
          )}

          {run && run.findings && run.findings.length > 0 && (
            <div
              className="doc-blocks-toolbar"
              role="toolbar"
              aria-label="Filtrowanie bloków i ustaleń"
            >
              <div
                className="filter-bar"
                role="toolbar"
                aria-label="Filtrowanie ustaleń"
              >
                {(
                  [
                    ["all", "Wszystkie", findingsCount],
                    ["critical", "Krytyczne", count("critical")],
                    ["warning", "Ostrzeżenia", count("warning")],
                    ["neutral", "Neutralne", count("neutral")],
                  ] as const
                ).map(([value, label, amount]) => (
                  <button
                    key={value}
                    type="button"
                    className={`filter-btn ${filter === value ? "active" : ""}`}
                    onClick={() => setFilter(value)}
                  >
                    {label} ({amount})
                  </button>
                ))}
              </div>
              <div className="doc-expand-actions">
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={expandAllBlocks}
                  title="Rozwiń wszystkie bloki ze szczegółami ustaleń"
                >
                  Rozwiń wszystkie
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={collapseAllBlocks}
                  title="Zwiń wszystkie bloki"
                >
                  Zwiń wszystkie
                </button>
              </div>
            </div>
          )}

          <div
            className="doc-blocks-list"
            data-testid={run && findingsCount > 0 ? "findings-list" : undefined}
          >
            {blocksToRender.length > 0 ? (
              blocksToRender.map(({ block, idx }) => {
                const blockKey = block.id || `block-${idx}`;

                const blockFindings = findingsByBlock.get(block.id) ?? [];
                const hasFindings = blockFindings.length > 0;
                const blockStatus = getBlockStatus(blockFindings);
                const isExpanded = expandedBlockKeys.has(blockKey);

                const hasOffsets =
                  resolvedAnchor &&
                  typeof resolvedAnchor.start_offset === "number" &&
                  typeof resolvedAnchor.end_offset === "number" &&
                  typeof block.anchor.start_offset === "number" &&
                  typeof block.anchor.end_offset === "number";

                const spanStart = hasOffsets
                  ? resolvedAnchor.start_offset - block.anchor.start_offset
                  : -1;
                const spanEnd = hasOffsets
                  ? resolvedAnchor.end_offset - block.anchor.start_offset
                  : -1;

                // Highlight only when the full quote span fits entirely within the block.
                const isValidSpan = Boolean(
                  hasOffsets &&
                  spanStart >= 0 &&
                  spanEnd <= block.text.length &&
                  spanEnd > spanStart,
                );

                const isHighlighted = isValidSpan;

                return (
                  <div
                    key={blockKey || idx}
                    ref={(el) => {
                      if (el && blockKey) blockRefs.current.set(blockKey, el);
                    }}
                    className={`doc-block-item ${isHighlighted ? "highlighted" : ""} ${hasFindings ? `has-findings status-${blockStatus.status}` : ""}`}
                    data-testid={`block-${idx}`}
                  >
                    <div
                      className={`doc-block-header ${hasFindings ? "interactive" : ""}`}
                      onClick={
                        hasFindings
                          ? () =>
                              toggleBlockExpanded(
                                blockKey,
                                blockStatus.primaryFinding,
                              )
                          : undefined
                      }
                      onKeyDown={
                        hasFindings
                          ? (e) => {
                              if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                toggleBlockExpanded(
                                  blockKey,
                                  blockStatus.primaryFinding,
                                );
                              }
                            }
                          : undefined
                      }
                      role={hasFindings ? "button" : undefined}
                      tabIndex={hasFindings ? 0 : undefined}
                      aria-expanded={
                        hasFindings ? Boolean(isExpanded) : undefined
                      }
                      aria-controls={
                        hasFindings ? `block-details-${blockKey}` : undefined
                      }
                    >
                      <div className="doc-block-title">
                        <span className="doc-block-id">Blok #{idx + 1}</span>
                        {block.anchor.page ? (
                          <span className="doc-block-meta">
                            · Strona {block.anchor.page}
                          </span>
                        ) : null}
                        {block.anchor.line_start ? (
                          <span className="doc-block-meta">
                            · Linie {block.anchor.line_start}–
                            {block.anchor.line_end}
                          </span>
                        ) : null}
                      </div>

                      {blockStatus.status && (
                        <div
                          className="doc-block-status-indicator"
                          data-testid={`block-status-${idx}`}
                        >
                          {(() => {
                            const meta = BLOCK_STATUS_META[blockStatus.status];
                            const StatusIcon = meta.icon;
                            return (
                              <span
                                className={`block-status-badge status-${blockStatus.status}`}
                                title={meta.title}
                              >
                                <StatusIcon
                                  size={16}
                                  className="status-icon"
                                  color={meta.color}
                                  aria-hidden="true"
                                />
                                <span className="status-label">
                                  {meta.label}
                                </span>
                                {blockFindings.length > 1 && (
                                  <span className="status-count">
                                    ({blockFindings.length})
                                  </span>
                                )}
                              </span>
                            );
                          })()}
                          <button
                            type="button"
                            className="btn-toggle-expand"
                            aria-label={
                              isExpanded ? "Zwiń szczegóły" : "Rozwiń szczegóły"
                            }
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleBlockExpanded(
                                blockKey,
                                blockStatus.primaryFinding,
                              );
                            }}
                          >
                            {isExpanded ? (
                              <ChevronDown size={18} />
                            ) : (
                              <ChevronRight size={18} />
                            )}
                          </button>
                        </div>
                      )}
                    </div>

                    <div className="doc-block-text">
                      {isValidSpan ? (
                        <>
                          {block.text.slice(0, spanStart)}
                          <mark
                            className="anchor-highlight"
                            data-testid="anchor-highlight-span"
                          >
                            {block.text.slice(spanStart, spanEnd)}
                          </mark>
                          {block.text.slice(spanEnd)}
                        </>
                      ) : (
                        block.text
                      )}
                    </div>

                    {hasFindings && (
                      <div
                        id={`block-details-${blockKey}`}
                        className={`doc-block-findings ${isExpanded ? "open" : "collapsed"}`}
                        data-testid={`block-findings-${idx}`}
                      >
                        {blockFindings.map(renderFindingCard)}
                      </div>
                    )}
                  </div>
                );
              })
            ) : (
              <p style={{ color: "var(--text-muted)", padding: "1rem" }}>
                {filter !== "all"
                  ? "Brak bloków spełniających wybrane kryterium filtra."
                  : "Brak treści kanonicznej do wyświetlenia."}
              </p>
            )}
          </div>
        </div>
      }
    </div>
  );
};
