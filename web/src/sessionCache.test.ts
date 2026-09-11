import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  saveSession,
  loadSession,
  clearSession,
  SESSION_STORAGE_KEY,
  type CachedSession,
} from "./sessionCache";
import {
  MOCK_DOC_DESCRIPTOR_STANDARD,
  MOCK_RUN_STANDARD,
} from "./test-support/fixtures";

describe("sessionCache module", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    vi.restoreAllMocks();
  });

  it("returns null when no session is stored", () => {
    expect(loadSession()).toBeNull();
  });

  it("saves and loads a valid session snapshot", () => {
    const session: CachedSession = {
      version: 1,
      documentFileName: "umowa.txt",
      documentDesc: MOCK_DOC_DESCRIPTOR_STANDARD,
      currentRun: MOCK_RUN_STANDARD,
      selectedArm: "mid",
      disclosureAccepted: true,
      selectedFindingId: MOCK_RUN_STANDARD.findings[0]?.id ?? null,
      worksheets: {},
      chatMessages: {
        [MOCK_RUN_STANDARD.id]: [
          { id: "msg-1", role: "user", content: "Pytanie o karę umowną" },
          { id: "msg-2", role: "assistant", content: "Odpowiedź na pytanie" },
        ],
      },
      isExpired: false,
    };

    saveSession(session);
    const loaded = loadSession();

    expect(loaded).not.toBeNull();
    expect(loaded?.documentDesc?.id).toBe(MOCK_DOC_DESCRIPTOR_STANDARD.id);
    expect(loaded?.currentRun?.id).toBe(MOCK_RUN_STANDARD.id);
    expect(loaded?.selectedArm).toBe("mid");
    expect(loaded?.disclosureAccepted).toBe(true);
    expect(loaded?.chatMessages[MOCK_RUN_STANDARD.id]).toHaveLength(2);
    expect(loaded?.selectedFindingId).toBe(MOCK_RUN_STANDARD.findings[0]?.id);
  });

  it("clears session data from sessionStorage", () => {
    const session: CachedSession = {
      version: 1,
      documentFileName: "umowa.txt",
      documentDesc: MOCK_DOC_DESCRIPTOR_STANDARD,
      currentRun: null,
      selectedArm: "off",
      disclosureAccepted: false,
      selectedFindingId: null,
      worksheets: {},
      chatMessages: {},
    };

    saveSession(session);
    expect(window.sessionStorage.getItem(SESSION_STORAGE_KEY)).not.toBeNull();

    clearSession();
    expect(window.sessionStorage.getItem(SESSION_STORAGE_KEY)).toBeNull();
    expect(loadSession()).toBeNull();
  });

  it("returns null when sessionStorage contains invalid JSON or wrong version", () => {
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, "invalid-json{{");
    expect(loadSession()).toBeNull();

    window.sessionStorage.setItem(
      SESSION_STORAGE_KEY,
      JSON.stringify({ version: 99, documentDesc: {} }),
    );
    expect(loadSession()).toBeNull();
  });
});
