import React from "react";
import type { RunEvent } from "./types";
import { AGENT_ROLE_LABELS_PL, WORKSHEET_KIND_LABELS_PL } from "./types";

const AGENT_LOG_LIMIT = 200;

export function appendAgentLogEvent(
  events: RunEvent[],
  event: RunEvent,
): RunEvent[] {
  if (
    event.kind !== "worksheet" ||
    !event.role ||
    !event.entry_kind ||
    !event.unit_id
  ) {
    return events;
  }
  return [...events, event].slice(-AGENT_LOG_LIMIT);
}

export const AgentLog: React.FC<{ events: RunEvent[] }> = ({ events }) => (
  <section className="agent-log" aria-label="Dziennik agentów">
    <h3>Dziennik agentów</h3>
    {events.length === 0 ? (
      <p className="agent-log-empty">Brak zarejestrowanych wpisów agentów.</p>
    ) : (
      <ol
        data-testid="agent-log-entries"
        aria-live="polite"
        aria-relevant="additions"
      >
        {events.slice(-AGENT_LOG_LIMIT).map((event, index) => (
          <li key={`${event.unit_id}-${index}`} data-testid="agent-log-entry">
            <strong>
              {AGENT_ROLE_LABELS_PL[event.role || ""] || event.role}
            </strong>
            <span>
              {WORKSHEET_KIND_LABELS_PL[event.entry_kind || ""] ||
                event.entry_kind}
            </span>
            <code>{event.unit_id}</code>
          </li>
        ))}
      </ol>
    )}
  </section>
);
