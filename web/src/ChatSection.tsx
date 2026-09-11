import React, { useRef, useState } from "react";
import type { Finding, ApiError, Run } from "./types";
import { FINDING_CODE_LABELS_PL, INTERACTION_ERROR_MESSAGES_PL } from "./types";
import type { IApiClient } from "./api";
import { MessageSquare, Send, AlertTriangle } from "lucide-react";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  isRefusal?: boolean;
  citedFindingIds?: string[];
  corpusConsulted?: boolean;
}

interface ChatSectionProps {
  runId: string | null;
  allFindings: Finding[];
  api: IApiClient;
  onSelectFinding: (finding: Finding | null) => void;
  isExpired?: boolean;
  onContentExpired?: () => void;
  /** A message the system read as an objection or a re-analysis started this run. */
  onChildRun?: (run: Run) => void;
  messages?: ChatMessage[];
  onMessagesChange?: (messages: ChatMessage[]) => void;
}

const FindingChips: React.FC<{
  findingIds: string[];
  allFindings: Finding[];
  onSelectFinding: (finding: Finding | null) => void;
}> = ({ findingIds, allFindings, onSelectFinding }) => (
  <div className="chat-finding-chips">
    {findingIds.map((findingId) => {
      const finding = allFindings.find(
        (candidate) => candidate.id === findingId,
      );
      return finding ? (
        <button
          key={findingId}
          type="button"
          className="btn btn-sm"
          onClick={() => onSelectFinding(finding)}
        >
          {FINDING_CODE_LABELS_PL[finding.code]}
        </button>
      ) : (
        <div key={findingId} role="alert">
          Odpowiedź odwołuje się do ustalenia niedostępnego w bieżącej analizie.
        </div>
      );
    })}
  </div>
);

export const ChatSection: React.FC<ChatSectionProps> = ({
  runId,
  allFindings,
  api,
  onSelectFinding,
  isExpired,
  onContentExpired,
  onChildRun,
  messages: externalMessages,
  onMessagesChange,
}) => {
  const [internalMessages, setInternalMessages] = useState<ChatMessage[]>([]);
  const messages =
    externalMessages !== undefined ? externalMessages : internalMessages;
  // Ref mirrors messages to prevent stale-closure loss during asynchronous operations.
  const latestMessages = useRef(messages);
  latestMessages.current = messages;
  const setMessages = (updater: (prev: ChatMessage[]) => ChatMessage[]) => {
    const next = updater(latestMessages.current);
    latestMessages.current = next;
    if (onMessagesChange) {
      onMessagesChange(next);
    } else {
      setInternalMessages(next);
    }
  };
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const sendInFlightRef = useRef(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  if (!runId) {
    return null;
  }

  if (isExpired) {
    return (
      <div className="chat-container">
        <div className="chat-disclaimer" role="alert">
          Czat wyjaśniający jest niedostępny — sesja dokumentu wygasła (410
          ContentExpired).
        </div>
      </div>
    );
  }

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    // Ref guard prevents duplicate in-flight sends before re-render.
    if (!input.trim() || sendInFlightRef.current) return;
    sendInFlightRef.current = true;

    const written = input.trim();
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: written,
    };

    const historyPayload = latestMessages.current.slice(-20).map((m) => ({
      role: m.role,
      content: m.content,
    }));
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const response = await api.sendMessage(runId, {
        message: written,
        history: historyPayload,
      });

      if (response.intent === "ask" && response.answer) {
        const { answer } = response;
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: answer.answer,
            citedFindingIds: answer.cited_finding_ids,
            corpusConsulted: answer.corpus_consulted,
          },
        ]);
      } else if (response.run) {
        // The child run reports the result; chat only acknowledges its start.
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content:
              response.intent === "contest"
                ? "System przyjął zastrzeżenie i ponownie sprawdza to ustalenie."
                : "System ponownie analizuje wskazaną jednostkę.",
          },
        ]);
        onChildRun?.(response.run);
      }
    } catch (err) {
      const apiErr = err as ApiError;
      if (apiErr?.code === "content_expired") {
        onContentExpired?.();
      } else if (apiErr?.code === "out_of_scope") {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content:
              apiErr.message_pl ||
              "System odmawia odpowiedzi: zapytanie wykracza poza zakres wyjaśnień.",
            isRefusal: true,
          },
        ]);
      } else {
        setErrorMessage(
          INTERACTION_ERROR_MESSAGES_PL[apiErr?.code] ||
            apiErr?.message_pl ||
            "Wystąpił błąd podczas pobierania odpowiedzi.",
        );
      }
    } finally {
      sendInFlightRef.current = false;
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container" data-testid="chat-section">
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          marginBottom: "0.5rem",
        }}
      >
        <MessageSquare size={18} />
        <h3 style={{ fontSize: "0.95rem", fontWeight: 600 }}>
          Czat wyjaśniający ustalenia
        </h3>
      </div>

      <div className="chat-disclaimer">
        <strong>Informacja:</strong> Czat służy wyłącznie do wyjaśniania
        zidentyfikowanych ustaleń i cytowanych przepisów prawa. System nie
        świadczy porad prawnych, nie ocenia szans procesowych ani nie
        rekomenduje podpisania umowy.
      </div>

      {messages.length > 0 && (
        <div className="chat-history" data-testid="chat-history">
          {messages.map((m) => (
            <div
              key={m.id}
              className={`chat-message ${m.role} ${m.isRefusal ? "refusal" : ""}`}
              data-testid={`chat-message-${m.role}`}
            >
              {m.isRefusal && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.35rem",
                    fontWeight: 600,
                    marginBottom: "0.25rem",
                  }}
                >
                  <AlertTriangle size={14} />
                  Odmowa odpowiedzi (zakres niedozwolony)
                </div>
              )}
              <div>{m.content}</div>
              {m.corpusConsulted && (
                // Includes both direct corpus reads and delegated lookups.
                <div className="chat-corpus-note">
                  Odpowiedź korzystała z zamrożonego korpusu
                </div>
              )}
              {m.citedFindingIds && m.citedFindingIds.length > 0 && (
                <FindingChips
                  findingIds={m.citedFindingIds}
                  allFindings={allFindings}
                  onSelectFinding={onSelectFinding}
                />
              )}
            </div>
          ))}
        </div>
      )}

      {errorMessage && (
        <div
          role="alert"
          style={{
            color: "#dc2626",
            fontSize: "0.8rem",
            marginBottom: "0.5rem",
          }}
        >
          {errorMessage}
        </div>
      )}

      <form onSubmit={handleSend} className="chat-input-row">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Napisz wiadomość: pytanie, zastrzeżenie albo prośbę o ponowną analizę..."
          maxLength={2000}
          disabled={isLoading}
          aria-label="Wiadomość do czatu"
        />
        <button
          type="submit"
          className="btn btn-primary"
          disabled={isLoading || !input.trim()}
          aria-label="Wyślij"
        >
          <Send size={16} />
        </button>
      </form>
    </div>
  );
};
