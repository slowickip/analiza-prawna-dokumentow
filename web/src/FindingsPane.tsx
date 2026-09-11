import React, { useEffect, useRef, useState } from "react";
import type { Finding, Run, RunEvent } from "./types";
import { INTERACTION_ERROR_MESSAGES_PL, RUN_STATUS_LABELS_PL } from "./types";
import { AlertOctagon, Info } from "lucide-react";
import { AgentLog, appendAgentLogEvent } from "./AgentLog";
import { ChatSection, type ChatMessage } from "./ChatSection";
import { FindingCard } from "./FindingCard";
import type { IApiClient } from "./api";
import { useWorksheetCache } from "./useWorksheetCache";

interface FindingsPaneProps {
  run: Run | null;
  onSelectFinding: (finding: Finding | null) => void;
  api: IApiClient;
  isExpired?: boolean;
  onContentExpired?: () => void;
  onApiError?: (error: unknown, fallbackMessage: string) => void;
  chatMessages?: Record<string, ChatMessage[]>;
  onChatMessagesChange?: (runId: string, messages: ChatMessage[]) => void;
}

interface ChildRunState {
  run: Run;
  events: RunEvent[];
  selectedFinding: Finding | null;
}

interface InterruptionDescription {
  title: string;
  kind: "wall_time" | "general";
  tokensUsed: number | null;
  elapsedSeconds: number | null;
  wallBudgetSeconds: number | null;
  unitsNotProcessedCount: number;
}

export function describeInterruption(
  run: Run | null | undefined,
): InterruptionDescription | null {
  if (!run) return null;
  const metrics = run.metrics;
  const unitsNotProcessedCount = metrics?.units_not_processed ?? 0;
  const isInterrupted =
    Boolean(run.interrupted) ||
    run.interruption_reason != null ||
    unitsNotProcessedCount > 0;

  if (!isInterrupted) return null;

  const tokensUsed = metrics
    ? metrics.input_tokens + metrics.output_tokens
    : null;
  const wallBudget = run.wall_budget_seconds ?? null;
  const wallLimitReached =
    metrics != null &&
    wallBudget !== null &&
    metrics.elapsed_ms >= wallBudget * 1000;

  const wallBudgetInterruption =
    wallBudget !== null &&
    metrics != null &&
    (run.interruption_reason === "wall_time" ||
      (run.interruption_reason == null && wallLimitReached));

  const title = wallBudgetInterruption
    ? "Przerwanie analizy — wyczerpano limit czasu"
    : "Przerwanie analizy — wyczerpano budżet";

  const kind = wallBudgetInterruption ? "wall_time" : "general";

  return {
    title,
    kind,
    tokensUsed,
    elapsedSeconds: metrics ? metrics.elapsed_ms / 1000 : null,
    wallBudgetSeconds: wallBudget,
    unitsNotProcessedCount,
  };
}

const MetricsSummary: React.FC<{ run: Run }> = ({ run }) => {
  const metrics = run.metrics;
  if (!metrics) return null;

  const totalTokens = metrics.input_tokens + metrics.output_tokens;
  // Counter names map directly to server-side telemetry metrics.
  const items: [string, string | number | null | undefined][] = [
    ["Czas", `${(metrics.elapsed_ms / 1000).toFixed(1)}s`],
    ["Tokeny", totalTokens],
    ["Tury wyszukiwania Wyszukującego", metrics.finder_tool_turns],
    ["Wyszukiwania Wyszukującego", metrics.finder_search_calls],
    [
      "Jednostki z wyczerpanym limitem wyszukiwania",
      metrics.finder_budget_exhausted_units,
    ],
    ["Tury Weryfikatora", metrics.verifier_tool_turns],
    ["Odczyty przepisów", metrics.provision_reads],
    ["Automatycznie domknięte charaktery", metrics.defaulted_characterisations],
  ];

  return (
    <div className="metrics-summary" aria-label="Metryki analizy">
      {items
        .filter(([, value]) => value !== null && value !== undefined)
        .map(([label, value]) => (
          <span key={label}>
            {label}: {value}
          </span>
        ))}
    </div>
  );
};

export const FindingsPane: React.FC<FindingsPaneProps> = ({
  run,
  onSelectFinding,
  api,
  isExpired,
  onContentExpired,
  onApiError,
  chatMessages,
  onChatMessagesChange,
}) => {
  const { worksheets, loadWorksheet, clearWorksheets } = useWorksheetCache(
    api,
    onContentExpired,
  );
  const [childRuns, setChildRuns] = useState<ChildRunState[]>([]);
  const childStreamCancels = useRef(new Map<string, () => void>());

  useEffect(() => {
    clearWorksheets();
    setChildRuns([]);
    childStreamCancels.current.forEach((cancel) => cancel());
    childStreamCancels.current.clear();
  }, [run?.id, clearWorksheets]);

  useEffect(
    () => () => {
      childStreamCancels.current.forEach((cancel) => cancel());
    },
    [],
  );

  const reportInteractionError = (error: unknown) => {
    const code = (error as { code?: string })?.code;
    const fallback =
      INTERACTION_ERROR_MESSAGES_PL[
        code as keyof typeof INTERACTION_ERROR_MESSAGES_PL
      ] || "Nie udało się uruchomić interakcji z agentami.";
    onApiError?.(error, fallback);
  };

  /**
   * Follow a child run the composer started. The run already exists by the time
   * it reaches here, so this only attaches the event stream that reports it.
   */
  const trackChildRun = (child: Run) => {
    setChildRuns((current) => [
      ...current.filter((item) => item.run.id !== child.id),
      { run: child, events: [], selectedFinding: null },
    ]);
    const cancel = api.streamRunEvents(
      child.id,
      (event) => {
        setChildRuns((current) =>
          current.map((item) => {
            if (item.run.id !== child.id) return item;
            const nextEvents = appendAgentLogEvent(item.events, event);
            return {
              ...item,
              events: nextEvents,
            };
          }),
        );
      },
      (completedChild) => {
        setChildRuns((current) =>
          current.map((item) =>
            item.run.id === completedChild.id
              ? { ...item, run: completedChild }
              : item,
          ),
        );
      },
      (err) => {
        reportInteractionError(err);
      },
    );
    childStreamCancels.current.set(child.id, cancel);
  };

  const findings = run?.findings || [];
  const interruption = describeInterruption(run);

  const statusMessage = !run
    ? isExpired
      ? null
      : "Brak ustaleń do wyświetlenia. Uruchom analizę, aby uzyskać wyniki."
    : run.status === "running"
      ? "Analiza jest w toku."
      : run.status === "failed"
        ? run.error?.message_pl || "Analiza zakończyła się niepowodzeniem."
        : run.status === "cancelled"
          ? "Analiza została anulowana."
          : findings.length === 0
            ? "Analiza zakończyła się bez ustaleń."
            : null;

  return (
    <div className="pane-content" data-testid="findings-pane">
      {interruption && (
        <div
          className="interruption-banner"
          data-testid="interruption-banner"
          role="alert"
        >
          <AlertOctagon size={24} className="no-shrink" />
          <div>
            <strong>{interruption.title}</strong>
            <p>
              {interruption.kind === "wall_time" && (
                <>
                  Czas analizy: {interruption.elapsedSeconds?.toFixed(1)} s /{" "}
                  {interruption.wallBudgetSeconds?.toFixed(1)} s.{" "}
                </>
              )}
              {interruption.kind === "general" && (
                <>Przekroczono skonfigurowany limit zasobów. </>
              )}
              Liczba nieprzetworzonych jednostek:{" "}
              <strong data-testid="units-not-processed-count">
                {interruption.unitsNotProcessedCount}
              </strong>
              .
            </p>
          </div>
        </div>
      )}

      {isExpired && (
        <div
          className="expired-banner"
          role="status"
          data-testid="expired-session-notice"
        >
          <strong>Sesja dokumentu wygasła (410)</strong>: Metryki pozostają
          tutaj, a ustalenia w panelu dokumentu; treść umowy i czat zostały
          usunięte.
        </div>
      )}

      {run && (
        <div className="run-telemetry-summary" style={{ marginBottom: "1rem" }}>
          <MetricsSummary run={run} />
        </div>
      )}

      {statusMessage && (
        <div
          className="findings-status"
          role={run?.status === "failed" ? "alert" : "status"}
          style={{ marginBottom: "1rem" }}
        >
          {run?.status === "failed" ? (
            <AlertOctagon size={36} />
          ) : (
            <Info size={36} />
          )}
          <p>{statusMessage}</p>
        </div>
      )}

      <ChatSection
        runId={run?.id ?? null}
        allFindings={findings}
        api={api}
        onSelectFinding={onSelectFinding}
        isExpired={isExpired}
        onContentExpired={onContentExpired}
        onChildRun={trackChildRun}
        messages={run?.id && chatMessages ? chatMessages[run.id] : undefined}
        onMessagesChange={
          run?.id && onChatMessagesChange
            ? (msgs) => onChatMessagesChange(run.id, msgs)
            : undefined
        }
      />

      {childRuns.map((child) => (
        <section
          key={child.run.id}
          className="interaction-run"
          data-testid={`interaction-run-${child.run.id}`}
        >
          <h3>Przebieg interaktywny (niemierzony)</h3>
          <p className="interaction-run-note">
            Ten przebieg nie zmienia zmierzonego wyniku analizy nadrzędnej.
          </p>
          <div className="interaction-run-status">
            Status: {RUN_STATUS_LABELS_PL[child.run.status]}
          </div>
          <AgentLog events={child.events} />
          <MetricsSummary run={child.run} />
          {child.run.findings.length > 0 ? (
            <div className="findings-container">
              {child.run.findings.map((finding) => (
                <FindingCard
                  key={finding.id}
                  finding={finding}
                  selected={child.selectedFinding?.id === finding.id}
                  onSelect={(selected) =>
                    setChildRuns((current) =>
                      current.map((item) =>
                        item.run.id === child.run.id
                          ? { ...item, selectedFinding: selected }
                          : item,
                      ),
                    )
                  }
                  worksheetState={worksheets[child.run.id] || {}}
                  onLoadWorksheet={() => void loadWorksheet(child.run.id)}
                />
              ))}
            </div>
          ) : (
            <p className="interaction-run-empty">
              Brak ustaleń do wyświetlenia.
            </p>
          )}
        </section>
      ))}
    </div>
  );
};
