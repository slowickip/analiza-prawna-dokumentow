import type { DocumentDescriptor, Run, ArmCode } from "./types";
import type { WorksheetState } from "./WorksheetView";
import type { ChatMessage } from "./ChatSection";

export const SESSION_STORAGE_KEY = "contract_analyzer_session";

export interface CachedSession {
  version: number;
  documentDesc: DocumentDescriptor | null;
  documentFileName: string | null;
  currentRun: Run | null;
  selectedArm: ArmCode;
  disclosureAccepted: boolean;
  selectedFindingId: string | null;
  worksheets: Record<string, WorksheetState>;
  chatMessages: Record<string, ChatMessage[]>;
  isExpired?: boolean;
}

/** Save the tab's analysis and chat state for reloads. */
export function saveSession(session: CachedSession): void {
  if (typeof window === "undefined" || !window.sessionStorage) {
    return;
  }

  try {
    const serialized = JSON.stringify(session);
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, serialized);
  } catch {
    // Quota or privacy restrictions must not interrupt the current session.
  }
}

/** Load a snapshot only if its schema version matches. */
export function loadSession(): CachedSession | null {
  if (typeof window === "undefined" || !window.sessionStorage) {
    return null;
  }

  try {
    const raw = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (!raw) return null;

    const parsed = JSON.parse(raw) as Partial<CachedSession>;
    if (!parsed || typeof parsed !== "object" || parsed.version !== 1) {
      return null;
    }

    return {
      version: 1,
      documentDesc: parsed.documentDesc ?? null,
      documentFileName: parsed.documentFileName ?? null,
      currentRun: parsed.currentRun ?? null,
      selectedArm: parsed.selectedArm ?? "mid",
      disclosureAccepted: Boolean(parsed.disclosureAccepted),
      selectedFindingId: parsed.selectedFindingId ?? null,
      worksheets: parsed.worksheets ?? {},
      chatMessages: parsed.chatMessages ?? {},
      isExpired: Boolean(parsed.isExpired),
    };
  } catch {
    return null;
  }
}

export function clearSession(): void {
  if (typeof window === "undefined" || !window.sessionStorage) {
    return;
  }

  try {
    window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
  } catch {
    // Storage restrictions must not prevent clearing the current UI session.
  }
}
