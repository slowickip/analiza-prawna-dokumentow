import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { App } from "./App";
import { MockApiService } from "./test-support/mockApi";
import { HttpApiClient, type IApiClient } from "./api";
import { ControllableEventSource } from "./test-support/controllableEventSource";
import {
  MOCK_DOC_DESCRIPTOR_STANDARD,
  MOCK_DOC_CONTENT_STANDARD,
  MOCK_RUN_STANDARD,
  MOCK_DOC_DESCRIPTOR_INTERRUPTED,
  MOCK_DOC_CONTENT_INTERRUPTED,
  MOCK_RUN_INTERRUPTED,
  MOCK_WORKSHEET_STANDARD,
} from "./test-support/fixtures";
import type {
  ArmCode,
  ApiError,
  Finding,
  Run,
  RunEvent,
  DocumentDescriptor,
  DocumentContent,
} from "./types";
import { ARM_LABELS } from "./types";
import { saveSession, SESSION_STORAGE_KEY } from "./sessionCache";

function createSyncApi(
  descriptor: DocumentDescriptor = MOCK_DOC_DESCRIPTOR_STANDARD,
  content: DocumentContent = MOCK_DOC_CONTENT_STANDARD,
  run: Run = MOCK_RUN_STANDARD,
  extras: Partial<MockApiService> = {},
): MockApiService {
  const api = new MockApiService();
  api.uploadDocument = async () => descriptor;
  api.getDocumentContent = async () => content;
  api.createRun = async () => run;
  api.getRun = async () => run;
  api.streamRunEvents = (_id, _onEvent, onComplete) => {
    onComplete(run);
    return () => {};
  };
  Object.assign(api, extras);
  return api;
}

async function renderApp(api?: IApiClient) {
  const view = render(<App api={api} />);
  await act(async () => {});
  return view;
}

async function uploadAndRun(
  api: IApiClient,
  fileName = "umowa.txt",
  fileContent = "dummy contract content",
) {
  await renderApp(api);

  const file = new File([fileContent], fileName, { type: "text/plain" });
  const input = screen.getByTestId("file-upload-input");
  await userEvent.upload(input, file);

  const checkbox = screen.getByLabelText(
    /Przyjmuję do wiadomości informacje o przetwarzaniu zewnętrznym/i,
  );
  await userEvent.click(checkbox);

  const runBtn = screen.getByRole("button", { name: /Uruchom analizę/i });
  await userEvent.click(runBtn);
}

describe("Web Interface Contract Tests", () => {
  let mockApi: MockApiService;

  beforeEach(() => {
    mockApi = new MockApiService();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    ControllableEventSource.reset();
  });

  it("keeps the upload input focusable", async () => {
    await renderApp(mockApi);

    const input = screen.getByTestId("file-upload-input");
    expect(input).toHaveClass("visually-hidden");
    input.focus();

    expect(document.activeElement).toBe(input);
  });

  it("opens the file picker with Enter", async () => {
    const user = userEvent.setup();
    await renderApp(mockApi);

    const input = screen.getByTestId("file-upload-input");
    const clickSpy = vi.spyOn(input, "click");

    await user.tab();
    await user.keyboard("{Enter}");

    expect(clickSpy).toHaveBeenCalledOnce();
  });

  it("opens the file picker with Space", async () => {
    const user = userEvent.setup();
    await renderApp(mockApi);

    const input = screen.getByTestId("file-upload-input");
    const clickSpy = vi.spyOn(input, "click");

    await user.tab();
    await user.keyboard(" ");

    expect(clickSpy).toHaveBeenCalledOnce();
  });

  it("arms the dropzone while a file is dragged over it", async () => {
    await renderApp(mockApi);

    const dropzone = screen
      .getByText("Wybierz lub upuść plik umowy")
      .closest(".upload-dropzone");
    const dragAllowed = fireEvent.dragOver(dropzone!);

    expect(dragAllowed).toBe(false);
    expect(dropzone).toHaveTextContent("Upuść plik tutaj");
  });

  it("uploads a dropped accepted file through the document upload path", async () => {
    const uploadDocument = vi.fn(async () => MOCK_DOC_DESCRIPTOR_STANDARD);
    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      MOCK_RUN_STANDARD,
      { uploadDocument },
    );
    await renderApp(testApi);

    const file = new File(["treść umowy"], "umowa.txt", {
      type: "text/plain",
    });
    const dropzone = screen
      .getByText("Wybierz lub upuść plik umowy")
      .closest(".upload-dropzone");
    fireEvent.drop(dropzone!, { dataTransfer: { files: [file] } });

    expect(uploadDocument).toHaveBeenCalledWith(file);
    expect(await screen.findByText(/umowa\.txt/)).toBeInTheDocument();
  });

  it("rejects a dropped file outside the accepted formats", async () => {
    const uploadDocument = vi.fn(async () => MOCK_DOC_DESCRIPTOR_STANDARD);
    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      MOCK_RUN_STANDARD,
      { uploadDocument },
    );
    await renderApp(testApi);

    const file = new File(["binary"], "program.exe", {
      type: "application/octet-stream",
    });
    const dropzone = screen
      .getByText("Wybierz lub upuść plik umowy")
      .closest(".upload-dropzone");
    fireEvent.drop(dropzone!, { dataTransfer: { files: [file] } });

    const alert = screen.queryByRole("alert");
    expect(alert).not.toBeNull();
    expect(alert).toHaveTextContent("Nieobsługiwany format pliku.");
    expect(uploadDocument).not.toHaveBeenCalled();
  });

  it("uses the HTTP API when rendered without an api prop", async () => {
    const fetchStub = vi.fn(() => new Promise<Response>(() => undefined));
    vi.stubGlobal("fetch", fetchStub);
    await renderApp();

    const file = new File(["treść umowy"], "umowa.txt", {
      type: "text/plain",
    });
    await userEvent.upload(screen.getByTestId("file-upload-input"), file);

    expect(fetchStub).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/documents"),
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("shows a Polish error without findings when the HTTP transport fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
    );
    await renderApp();

    const file = new File(["treść umowy"], "umowa.txt", {
      type: "text/plain",
    });
    await userEvent.upload(screen.getByTestId("file-upload-input"), file);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Nie udało się połączyć z serwerem.",
    );
    expect(screen.queryByTestId("findings-list")).not.toBeInTheDocument();
    expect(screen.queryByTestId(/finding-card-/)).not.toBeInTheDocument();
  });

  it("shows a recoverable Polish error and unlocks controls when stream recovery is exhausted", async () => {
    const runningRun: Run = {
      ...MOCK_RUN_STANDARD,
      status: "running",
      finished_at: null,
    };
    let runStateRequests = 0;
    vi.stubGlobal("EventSource", ControllableEventSource);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/documents") && init?.method === "POST") {
          return Response.json(MOCK_DOC_DESCRIPTOR_STANDARD);
        }
        if (url.endsWith("/content")) {
          return Response.json(MOCK_DOC_CONTENT_STANDARD);
        }
        if (url.endsWith("/runs") && init?.method === "POST") {
          return Response.json(runningRun);
        }
        if (url.endsWith(`/runs/${runningRun.id}`)) {
          runStateRequests += 1;
          throw new TypeError("Failed to fetch");
        }
        throw new Error(`Unexpected request: ${url}`);
      }),
    );
    await uploadAndRun(new HttpApiClient());
    expect(ControllableEventSource.instances).toHaveLength(1);
    vi.useFakeTimers();

    await act(async () => {
      ControllableEventSource.instances[0].emitError();
      await vi.runAllTimersAsync();
    });

    expect.soft(runStateRequests).toBe(5);
    expect
      .soft(screen.queryByRole("alert"))
      .toHaveTextContent(
        "Utracono połączenie z analizą. Spróbuj uruchomić ją ponownie.",
      );
    expect
      .soft(screen.queryByRole("button", { name: /Uruchom analizę/i }))
      .toBeEnabled();
  });

  it("keeps the text of a malformed stream message out of the console", async () => {
    const runningRun: Run = {
      ...MOCK_RUN_STANDARD,
      status: "running",
      finished_at: null,
    };
    vi.stubGlobal("EventSource", ControllableEventSource);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/documents") && init?.method === "POST") {
          return Response.json(MOCK_DOC_DESCRIPTOR_STANDARD);
        }
        if (url.endsWith("/content")) {
          return Response.json(MOCK_DOC_CONTENT_STANDARD);
        }
        if (url.endsWith("/runs") && init?.method === "POST") {
          return Response.json(runningRun);
        }
        if (url.endsWith(`/runs/${runningRun.id}`)) {
          throw new TypeError("Failed to fetch");
        }
        throw new Error(`Unexpected request: ${url}`);
      }),
    );
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});
    await uploadAndRun(new HttpApiClient());
    vi.useFakeTimers();

    await act(async () => {
      ControllableEventSource.instances[0].emitMessage("tajna tresc dokumentu");
      await vi.runAllTimersAsync();
    });

    const printed = consoleError.mock.calls.flat().map(String).join(" ");
    expect(printed).not.toContain("tajna");
    consoleError.mockRestore();
  });

  it("shows mechanism labels without a quality ranking or grading", async () => {
    await renderApp(mockApi);

    // Verify exact labels with correct Polish diacritics
    expect(
      screen.getByRole("radio", { name: "OFF — cały dokument" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("radio", { name: "MID — jednostki" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("radio", { name: "ON — jednostki + odwołania" }),
    ).toBeInTheDocument();

    // Verify no quality grading, tiers or "better/worse" adjectives exist
    expect(
      screen.queryByText(
        /najlepsz|dokładniejsz|polecan|rekomendowan|tier|gwiazd/i,
      ),
    ).toBeNull();
  });

  it("uses exactly critical and warning alert levels, and neutral detail without green/compliance checkmarks", async () => {
    const testApi = createSyncApi();
    await uploadAndRun(testApi);

    await waitFor(() => {
      expect(screen.getByTestId("findings-list")).toBeInTheDocument();
    });

    // Check prominence levels present on finding cards
    const findingCards = screen.getAllByRole("article");
    const levels = new Set(
      findingCards.map((card) => card.getAttribute("data-level")),
    );

    expect(levels).toEqual(new Set(["critical", "warning", "neutral"]));

    // Verify critical findings
    const criticalCards = screen
      .getAllByTestId(/finding-card-/)
      .filter((el) => el.getAttribute("data-level") === "critical");
    expect(criticalCards.length).toBeGreaterThan(0);

    // Verify warning findings
    const warningCards = screen
      .getAllByTestId(/finding-card-/)
      .filter((el) => el.getAttribute("data-level") === "warning");
    expect(warningCards.length).toBeGreaterThan(0);

    // Verify neutral findings: MUST NOT have green ticks, checkmarks or "compliant" wording
    const neutralCards = screen
      .getAllByTestId(/finding-card-/)
      .filter((el) => el.getAttribute("data-level") === "neutral");
    expect(neutralCards.length).toBeGreaterThan(0);

    neutralCards.forEach((card) => {
      // Must not contain positive certification badges
      expect(card.textContent).not.toMatch(
        /zgodny z prawem|certyfikat|zweryfikowano pozytywnie|brak naruszeń|pełny audyt/i,
      );
    });
  });

  it("renders run status events in Polish without the wire token", async () => {
    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      MOCK_RUN_STANDARD,
      {
        streamRunEvents: (_id, onEvent) => {
          onEvent({ kind: "status", status: "completed" });
          return () => {};
        },
      },
    );

    await uploadAndRun(testApi);

    const progress = await screen.findByText(/^Status:/);
    expect(progress).not.toHaveTextContent("completed");
    expect(progress).toHaveTextContent("Status: Zakończono");
  });

  async function startStreamingRun(arm: ArmCode) {
    let emitCounter: ((event: RunEvent) => void) | null = null;
    const runningRun: Run = {
      ...MOCK_RUN_STANDARD,
      arm,
      status: "running",
      finished_at: null,
    };
    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      runningRun,
      {
        createRun: async () => runningRun,
        streamRunEvents: (_id, onEvent) => {
          emitCounter = onEvent;
          return () => {};
        },
      },
    );

    await renderApp(testApi);
    const file = new File(["treść umowy"], "umowa.txt", { type: "text/plain" });
    await userEvent.upload(screen.getByTestId("file-upload-input"), file);
    await userEvent.click(
      screen.getByLabelText(
        /Przyjmuję do wiadomości informacje o przetwarzaniu zewnętrznym/i,
      ),
    );
    if (arm !== "mid") {
      await userEvent.click(
        screen.getByRole("radio", { name: ARM_LABELS[arm] }),
      );
    }
    await userEvent.click(
      screen.getByRole("button", { name: /Uruchom analizę/i }),
    );

    return {
      cancelButton: screen.getByRole("button", { name: /Przerwij analizę/i }),
      emitCount: (count: number) =>
        act(() => emitCounter?.({ kind: "counter", count })),
    };
  }

  it("renders a unit total of 1 for OFF running progress and cancellation counter", async () => {
    const { cancelButton, emitCount } = await startStreamingRun("off");
    expect(cancelButton).toHaveTextContent("Przerwij analizę (0 / 1)");

    emitCount(1);

    expect(
      screen.getByText("Przetwarzanie jednostek: 1 / 1"),
    ).toBeInTheDocument();
    expect(cancelButton).toHaveTextContent("Przerwij analizę (1 / 1)");
  });

  it("retains the document unit count denominator for MID running progress and cancellation counter", async () => {
    const { cancelButton, emitCount } = await startStreamingRun("mid");
    expect(cancelButton).toHaveTextContent("Przerwij analizę (0 / 9)");

    emitCount(1);

    expect(
      screen.getByText("Przetwarzanie jednostek: 1 / 9"),
    ).toBeInTheDocument();
    expect(cancelButton).toHaveTextContent("Przerwij analizę (1 / 9)");
  });

  it("plainly displays an unresolved anchor message when anchor_resolved is false and does not approximate", async () => {
    const findingWithUnresolvedAnchor: Finding = {
      id: "f-unresolved",
      unit_id: "u-9",
      code: "uncertain",
      prominence: "warning",
      uncertain_cause: "relation_below_threshold",
      raw_confidence: 0.35,
      anchor_resolved: false,
    };

    const runWithUnresolved: Run = {
      ...MOCK_RUN_STANDARD,
      findings: [findingWithUnresolvedAnchor],
    };

    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      runWithUnresolved,
    );

    await uploadAndRun(testApi);

    await waitFor(() => {
      expect(screen.getByTestId("unresolved-anchor-note")).toBeInTheDocument();
    });

    const note = screen.getByTestId("unresolved-anchor-note");
    expect(note.textContent).toContain("Zakotwiczenie nierozstrzygnięte");
    expect(note.textContent).toContain("nie jest przybliżana");
  });

  it("labels each force record with its own scope and never states provision-level force", async () => {
    const findingWithBasis: Finding = {
      id: "f-basis",
      unit_id: "u-1",
      code: "consistent",
      prominence: "neutral",
      anchor_resolved: false,
      basis: {
        provision_locator:
          "https://api.sejm.gov.pl/eli/acts/DU/2026/795/art/659",
        act_identifier: "DU/2026/795",
        act_force: {
          value: "in_force",
          scope: "act",
          snapshot_date: "2026-05-19",
          source_locator: "https://api.sejm.gov.pl/eli/acts/DU/2026/795",
        },
        provision_force: {
          value: "undetermined",
          scope: "provision",
          snapshot_date: "2026-05-19",
          source_locator: "https://api.sejm.gov.pl/eli/acts/DU/2026/795",
        },
        character_kind: "imperative",
      },
    };

    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      { ...MOCK_RUN_STANDARD, findings: [findingWithBasis] },
    );

    await uploadAndRun(testApi);

    await waitFor(() => {
      expect(screen.getByTestId("legal-basis-force")).toBeInTheDocument();
    });

    const force = screen.getByTestId("legal-basis-force");
    expect(force.textContent).toContain("Stan ustawy (zakres: akt)");
    expect(force.textContent).toContain("obowiązuje w dacie obserwacji");
    expect(force.textContent).toContain("Stan przepisu (zakres: przepis)");
    expect(force.textContent).toContain("tekst jednolity tego nie rozstrzyga");
  });

  it("displays a critical document-level banner with exact units_not_processed count when interrupted", async () => {
    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_INTERRUPTED,
      MOCK_DOC_CONTENT_INTERRUPTED,
      MOCK_RUN_INTERRUPTED,
    );

    await uploadAndRun(testApi, "interrupted.txt", "serwis");

    await waitFor(() => {
      expect(screen.getByTestId("interruption-banner")).toBeInTheDocument();
    });

    const countElem = screen.getByTestId("units-not-processed-count");
    expect(countElem.textContent).toBe("7");
    expect(screen.getByText(/Wyczerpano limit czasu/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Czas analizy: 15.2 s \/ 15.0 s/i),
    ).toBeInTheDocument();
  });

  it("displays the refusal verbatim when a chat question is out of scope", async () => {
    const testApi = createSyncApi();
    await uploadAndRun(testApi);

    await waitFor(() => {
      expect(screen.getByTestId("chat-section")).toBeInTheDocument();
    });

    await userEvent.click(screen.getByTestId("finding-card-f-4"));
    const chatInput = screen.getByLabelText("Wiadomość do czatu");
    await userEvent.type(chatInput, "Czy warto podpisać tę umowę?");

    const sendBtn = screen.getByLabelText("Wyślij");
    await userEvent.click(sendBtn);

    await waitFor(() => {
      expect(screen.getByText(/Odmowa odpowiedzi/i)).toBeInTheDocument();
    });

    expect(
      screen.getByText(
        /System odmawia odpowiedzi: zapytanie dotyczy porady prawnej lub rekomendacji/i,
      ),
    ).toBeInTheDocument();
  });

  it("returns to an empty workspace when the session is purged", async () => {
    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      MOCK_RUN_STANDARD,
      { deleteDocument: async () => {} },
    );

    await renderApp(testApi);

    const file = new File(["test"], "umowa.txt", { type: "text/plain" });
    const input = screen.getByTestId("file-upload-input");
    await userEvent.upload(input, file);

    const purgeBtn = screen.getByTitle(
      /Usuń treść dokumentu z pamięci serwera/i,
    );
    await userEvent.click(purgeBtn);

    // A purge erases the analysis as well as the text, so what is left is not an
    // expired session but no session: showing the expired pane afterwards would
    // claim findings the server no longer holds.
    await waitFor(() => {
      expect(screen.getByTestId("file-upload-input")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("document-expired-pane")).toBeNull();
    expect(screen.queryByTestId("findings-list")).toBeNull();
  });

  it("derives expiry from a run whose content is unavailable", async () => {
    const expiredRun: Run = {
      ...MOCK_RUN_STANDARD,
      content_available: false,
    };
    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      expiredRun,
    );

    await uploadAndRun(testApi);

    await waitFor(() => {
      expect(screen.queryByTestId("expired-session-notice")).not.toBeNull();
    });
    expect(screen.getByTestId("findings-list")).toBeInTheDocument();
  });

  it("allows a second upload after purging the first document", async () => {
    const uploadDocument = vi.fn(async (file: File) => ({
      ...MOCK_DOC_DESCRIPTOR_STANDARD,
      id: `document-${file.name}`,
    }));
    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      MOCK_RUN_STANDARD,
      {
        uploadDocument,
        deleteDocument: async () => {},
      },
    );
    await renderApp(testApi);

    await userEvent.upload(
      screen.getByTestId("file-upload-input"),
      new File(["pierwsza"], "pierwsza.txt", { type: "text/plain" }),
    );
    await userEvent.click(
      screen.getByTitle(/Usuń treść dokumentu z pamięci serwera/i),
    );

    await waitFor(() => {
      expect(screen.queryByTestId("file-upload-input")).not.toBeNull();
    });
    const secondInput = screen.getByTestId("file-upload-input");
    await userEvent.upload(
      secondInput,
      new File(["druga"], "druga.txt", { type: "text/plain" }),
    );

    expect(uploadDocument).toHaveBeenCalledTimes(2);
    expect(await screen.findByText(/druga\.txt/)).toBeInTheDocument();
  });

  it("derives expiry from a content_expired response", async () => {
    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      MOCK_RUN_STANDARD,
      {
        getDocumentContent: async () => {
          throw {
            code: "content_expired",
            message_pl: "Sesja wygasła podczas pobierania treści.",
          };
        },
      },
    );
    await renderApp(testApi);

    await userEvent.upload(
      screen.getByTestId("file-upload-input"),
      new File(["treść"], "umowa.txt", { type: "text/plain" }),
    );

    await waitFor(() => {
      expect(screen.queryByTestId("expired-session-notice")).not.toBeNull();
    });
  });

  it("fetches the worksheet once per run, however many cards ask for it", async () => {
    // Every finding card carries its own worksheet toggle, but the transcript is
    // per run, not per finding: without the guard in loadWorksheet, opening a
    // second card refetches the same worksheet the first one already has.
    const getWorksheet = vi.fn(async () => MOCK_WORKSHEET_STANDARD);
    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      MOCK_RUN_STANDARD,
      { getWorksheet },
    );
    await uploadAndRun(testApi);

    const toggles = await screen.findAllByRole("button", {
      name: /Pokaż pracę agentów/i,
    });
    expect(toggles.length).toBeGreaterThan(1);

    // Both toggles fire before React re-renders, so neither card can see that
    // the other already started the fetch: only loadWorksheet's own guard can
    // keep this to one request.
    await act(async () => {
      toggles[0].click();
      toggles[1].click();
    });

    await waitFor(() => expect(getWorksheet).toHaveBeenCalled());
    expect(getWorksheet).toHaveBeenCalledTimes(1);
    expect(getWorksheet).toHaveBeenCalledWith(MOCK_RUN_STANDARD.id);
  });

  it("still reports the closed session when a worksheet is expanded again", async () => {
    // Expiry wipes worksheet state, so a run left marked as already-requested
    // would answer the next expand with neither data nor error -- an empty
    // worksheet where the truth is a closed session.
    const getWorksheet = vi.fn(async () => {
      throw { code: "content_expired", message_pl: "Sesja dokumentu wygasła." };
    });
    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      MOCK_RUN_STANDARD,
      { getWorksheet },
    );
    await uploadAndRun(testApi);

    const open = async () => {
      const toggles = await screen.findAllByRole("button", {
        name: /Pokaż pracę agentów/i,
      });
      await userEvent.click(toggles[0]);
    };

    await open();
    await waitFor(() =>
      expect(screen.queryByTestId("expired-session-notice")).not.toBeNull(),
    );
    await waitFor(() => expect(getWorksheet).toHaveBeenCalledTimes(1));

    // Expiry re-renders the pane, so the card comes back collapsed; opening it
    // again is what the reader does next.
    await open();

    // No second request: the worksheet is stated as gone, not fetched again.
    expect(getWorksheet).toHaveBeenCalledTimes(1);
    expect(
      screen.queryAllByText(/Praca agentów jest niedostępna/i).length,
    ).toBeGreaterThan(0);
  });

  it("derives expiry when the worksheet fetch returns content_expired", async () => {
    // A 410 on the worksheet is the retention window closing under the reader,
    // so it has to reach the whole view, not just the card that asked.
    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      MOCK_RUN_STANDARD,
      {
        getWorksheet: async () => {
          throw {
            code: "content_expired",
            message_pl: "Sesja dokumentu wygasła.",
          };
        },
      },
    );
    await uploadAndRun(testApi);

    const toggles = await screen.findAllByRole("button", {
      name: /Pokaż pracę agentów/i,
    });
    await userEvent.click(toggles[0]);

    await waitFor(() => {
      expect(screen.queryByTestId("expired-session-notice")).not.toBeNull();
    });
  });

  it("derives expiry when a reader message returns content_expired", async () => {
    // The client reaches the server through sendMessage, so the expiry has to be
    // raised there: overriding the mock's internal ask helper would exercise a
    // path the browser no longer takes.
    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      MOCK_RUN_STANDARD,
      {
        sendMessage: async () => {
          throw {
            code: "content_expired",
            message_pl: "Sesja wygasła podczas rozmowy.",
          };
        },
      },
    );
    await uploadAndRun(testApi);

    await userEvent.click(screen.getByTestId("finding-card-f-4"));
    await userEvent.type(
      screen.getByLabelText("Wiadomość do czatu"),
      "Wyjaśnij ustalenie.",
    );
    await userEvent.click(screen.getByLabelText("Wyślij"));

    await waitFor(() => {
      expect(screen.queryByTestId("expired-session-notice")).not.toBeNull();
    });

    // Expiry ends the retention window the interface tells the reader about, so
    // nothing it covers may outlive it in the snapshot: the canonical text, the
    // worksheet prose, the chat this pane says was deleted, and the synthesis,
    // which carries the document's retention.
    await waitFor(() => {
      const raw = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
      expect(raw).not.toBeNull();
      const snapshot = JSON.parse(raw as string);
      // The key is absent, not null: the snapshot never carries canonical text,
      // so asserting "?? null" would pass even if it started carrying it again.
      expect("documentContent" in snapshot).toBe(false);
      expect("agentEvents" in snapshot).toBe(false);
      expect(snapshot.worksheets).toEqual({});
      expect(snapshot.chatMessages).toEqual({});
      expect(snapshot.currentRun?.synthesis ?? null).toBeNull();
    });
  });

  it("disables the run button until external processing consent is accepted", async () => {
    const testApi = createSyncApi();
    await renderApp(testApi);

    await userEvent.upload(
      screen.getByTestId("file-upload-input"),
      new File(["treść"], "umowa.txt", { type: "text/plain" }),
    );

    const runTrigger = screen.getByRole("button", {
      name: /Uruchom analizę/i,
    });
    expect(runTrigger).toBeDisabled();
    expect(runTrigger).toHaveAttribute(
      "title",
      expect.stringMatching(/zaznacz zgodę/i),
    );

    await userEvent.click(
      screen.getByLabelText(
        /Przyjmuję do wiadomości informacje o przetwarzaniu zewnętrznym/i,
      ),
    );

    expect(runTrigger).toBeEnabled();
  });

  it("derives expiry from a descriptor whose expiry is in the past", async () => {
    const expiredDescriptor: DocumentDescriptor = {
      ...MOCK_DOC_DESCRIPTOR_STANDARD,
      expires_at: new Date(Date.now() - 1000).toISOString(),
    };
    const testApi = createSyncApi(expiredDescriptor, MOCK_DOC_CONTENT_STANDARD);
    await renderApp(testApi);

    await userEvent.upload(
      screen.getByTestId("file-upload-input"),
      new File(["treść"], "umowa.txt", { type: "text/plain" }),
    );

    await waitFor(() => {
      expect(screen.queryByTestId("expired-session-notice")).not.toBeNull();
    });
  });

  it("displays the honest upload size limit read from public config in the dropzone", async () => {
    await renderApp(mockApi);

    expect(
      await screen.findByText(
        /Obsługiwane formaty: \.txt, \.docx, \.doc, \.pdf \(maks\. 25 MB\)/i,
      ),
    ).toBeInTheDocument();
  });

  it("surfaces the server 413 error message in the alert banner when an oversized file upload is rejected", async () => {
    const uploadDocument = vi.fn().mockRejectedValue({
      code: "limit_breach",
      message_pl: "Plik przekracza dopuszczalny limit rozmiaru.",
    });
    const testApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      MOCK_RUN_STANDARD,
      { uploadDocument },
    );

    await renderApp(testApi);

    const oversizedFile = new File(["dummy content"], "large_contract.pdf", {
      type: "application/pdf",
    });
    Object.defineProperty(oversizedFile, "size", { value: 26214401 });

    const input = screen.getByTestId("file-upload-input");
    await userEvent.upload(input, oversizedFile);

    expect(uploadDocument).toHaveBeenCalledWith(oversizedFile);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      "Plik przekracza dopuszczalny limit rozmiaru.",
    );
  });

  it.each([
    [
      "run_not_finished",
      "Interakcję można uruchomić dopiero po zakończeniu analizy.",
    ],
    [
      "run_already_active",
      "Inna analiza jest już aktywna. Poczekaj na jej zakończenie.",
    ],
  ] as const)(
    "shows the Polish interaction message for %s through the global error path",
    async (code, expectedMessage) => {
      const testApi = createSyncApi(
        MOCK_DOC_DESCRIPTOR_STANDARD,
        MOCK_DOC_CONTENT_STANDARD,
        MOCK_RUN_STANDARD,
        {
          sendMessage: async () => {
            throw {
              code,
              message_pl: "Backend interaction error.",
            } as ApiError;
          },
        },
      );
      await uploadAndRun(testApi);

      await userEvent.type(
        screen.getByLabelText("Wiadomość do czatu"),
        "Przeanalizuj tę jednostkę ponownie.",
      );
      await userEvent.click(screen.getByLabelText("Wyślij"));

      expect(await screen.findByRole("alert")).toHaveTextContent(
        expectedMessage,
      );
    },
  );

  it("rehydrates document, run, and chat from sessionStorage on mount (simulating page refresh)", async () => {
    saveSession({
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
          { id: "msg-1", role: "user", content: "Poprzednie pytanie z czatu" },
          {
            id: "msg-2",
            role: "assistant",
            content: "Poprzednia odpowiedź asystenta",
          },
        ],
      },
      isExpired: false,
    });

    const mockApi = createSyncApi();
    await renderApp(mockApi);

    // Document and findings are immediately rendered without uploading
    expect(screen.getByTestId("canonical-blocks")).toBeInTheDocument();
    expect(
      screen.getByText(/UMOWA NAJMU LOKALU UŻYTKOWEGO/i),
    ).toBeInTheDocument();
    expect(screen.getByText("Poprzednie pytanie z czatu")).toBeInTheDocument();
    expect(
      screen.getByText("Poprzednia odpowiedź asystenta"),
    ).toBeInTheDocument();
  });

  it("resumes monitoring via streamRunEvents if rehydrated run status is running", async () => {
    const runningRun: Run = {
      ...MOCK_RUN_STANDARD,
      status: "running",
      findings: [],
    };

    saveSession({
      version: 1,
      documentFileName: "umowa.txt",
      documentDesc: MOCK_DOC_DESCRIPTOR_STANDARD,
      currentRun: runningRun,
      selectedArm: "mid",
      disclosureAccepted: true,
      selectedFindingId: null,
      worksheets: {},
      chatMessages: {},
      isExpired: false,
    });

    let sseConnected = false;
    let completeHandler: ((run: Run) => void) | null = null;

    const mockApi = createSyncApi(
      MOCK_DOC_DESCRIPTOR_STANDARD,
      MOCK_DOC_CONTENT_STANDARD,
      runningRun,
      {
        getRun: async () => runningRun,
        streamRunEvents: (_id, _onEvent, onComplete) => {
          sseConnected = true;
          completeHandler = onComplete;
          return () => {};
        },
      },
    );

    await renderApp(mockApi);

    await waitFor(() => {
      expect(sseConnected).toBe(true);
      expect(
        screen.getByText(/Wznawianie monitorowania analizy/i),
      ).toBeInTheDocument();
    });

    // Complete the run
    act(() => {
      completeHandler?.(MOCK_RUN_STANDARD);
    });

    await waitFor(() => {
      expect(
        screen.getByText(/UMOWA NAJMU LOKALU UŻYTKOWEGO/i),
      ).toBeInTheDocument();
    });
  });

  it("leaves no session snapshot behind when the session is purged", async () => {
    const testApi = createSyncApi();
    await uploadAndRun(testApi);

    const deleteBtn = screen.getByRole("button", {
      name: /Usuń sesję/i,
    });
    await userEvent.click(deleteBtn);

    // Not a snapshot recording an expired session -- no snapshot. The reader
    // asked for it to stop existing, and the tab keeps its own copy.
    await waitFor(() =>
      expect(window.sessionStorage.getItem(SESSION_STORAGE_KEY)).toBeNull(),
    );
  });
});
