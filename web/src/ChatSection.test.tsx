import React, { useState } from "react";
import { render, screen, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { ChatSection, type ChatMessage } from "./ChatSection";
import { MockApiService } from "./test-support/mockApi";
import { MOCK_RUN_STANDARD } from "./test-support/fixtures";
import type { ApiError } from "./types";

const COMPOSER = "Wiadomość do czatu";
const SEND = "Wyślij";

function renderChat(
  mockApi: MockApiService,
  overrides: Partial<React.ComponentProps<typeof ChatSection>> = {},
) {
  return render(
    <ChatSection
      runId={MOCK_RUN_STANDARD.id}
      allFindings={MOCK_RUN_STANDARD.findings}
      api={mockApi}
      onSelectFinding={vi.fn()}
      {...overrides}
    />,
  );
}

async function send(text: string) {
  await userEvent.type(screen.getByLabelText(COMPOSER), text);
  await userEvent.click(screen.getByLabelText(SEND));
}

describe("ChatSection Component", () => {
  it("sends without the reader first choosing a finding", async () => {
    const mockApi = new MockApiService();
    const sendMessage = vi.spyOn(mockApi, "sendMessage");
    renderChat(mockApi);

    // No selection was ever made: typing alone is enough to enable sending.
    await userEvent.type(
      screen.getByLabelText(COMPOSER),
      "Dlaczego system tak ocenił tę umowę?",
    );
    expect(screen.getByLabelText(SEND)).not.toBeDisabled();
    await userEvent.click(screen.getByLabelText(SEND));

    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1));
    expect(sendMessage.mock.calls[0][1].message).toBe(
      "Dlaczego system tak ocenił tę umowę?",
    );
  });

  it("spends one run when the reader submits twice before the first lands", async () => {
    // isLoading is state, so it is not set until React re-renders: two submits in
    // the same tick both read the old value. On this endpoint a second submit is
    // not a repeated message but a second contest or re-analysis, each a full
    // model run that is billed and recorded against the parent.
    const mockApi = new MockApiService();
    const sendMessage = vi.spyOn(mockApi, "sendMessage");
    renderChat(mockApi);

    await userEvent.type(
      screen.getByLabelText(COMPOSER),
      "Nie zgadzam sie z tym ustaleniem.",
    );
    const form = screen.getByLabelText(SEND).closest("form");
    expect(form).not.toBeNull();

    // Both events are dispatched inside one batch, so React has not re-rendered
    // between them: this is the double-click the reader can actually perform.
    await act(async () => {
      const submit = () =>
        form!.dispatchEvent(
          new Event("submit", { bubbles: true, cancelable: true }),
        );
      submit();
      submit();
    });

    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1));
    expect(sendMessage).toHaveBeenCalledTimes(1);
  });

  it("sends the turns before the message as history, never the message itself", async () => {
    const mockApi = new MockApiService();
    const sendMessage = vi.spyOn(mockApi, "sendMessage");
    const Host: React.FC = () => {
      const [messages, setMessages] = useState<ChatMessage[]>([]);
      return (
        <ChatSection
          runId={MOCK_RUN_STANDARD.id}
          allFindings={MOCK_RUN_STANDARD.findings}
          api={mockApi}
          onSelectFinding={vi.fn()}
          messages={messages}
          onMessagesChange={setMessages}
        />
      );
    };
    render(<Host />);

    await send("Pierwsze pytanie.");
    await screen.findByTestId("chat-message-assistant");
    expect(sendMessage.mock.calls[0][1].history).toEqual([]);

    await send("Drugie pytanie.");
    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(2));
    const history = sendMessage.mock.calls[1][1].history ?? [];
    expect(history.map((turn) => turn.role)).toEqual(["user", "assistant"]);
    expect(history[0].content).toBe("Pierwsze pytanie.");
    expect(history.some((turn) => turn.content === "Drugie pytanie.")).toBe(
      false,
    );
  });

  it("keeps the reader's own message in the transcript after the answer", async () => {
    // The composer appends twice around one await. A controlled parent must not
    // lose the first append when the second lands.
    const mockApi = new MockApiService();
    const Host: React.FC = () => {
      const [messages, setMessages] = useState<ChatMessage[]>([]);
      return (
        <ChatSection
          runId={MOCK_RUN_STANDARD.id}
          allFindings={MOCK_RUN_STANDARD.findings}
          api={mockApi}
          onSelectFinding={vi.fn()}
          messages={messages}
          onMessagesChange={setMessages}
        />
      );
    };
    render(<Host />);

    await send("Moje pytanie o czynsz.");

    await screen.findByTestId("chat-message-assistant");
    expect(screen.getByText("Moje pytanie o czynsz.")).toBeInTheDocument();
  });

  it("no longer opens with a grouping the reader has to pick from", () => {
    const mockApi = new MockApiService();
    renderChat(mockApi);

    expect(screen.queryByTestId("chat-synthesis")).not.toBeInTheDocument();
    expect(
      screen.queryByText(/Najpierw wybierz ustalenie/i),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/Kontekst zapytania/i)).not.toBeInTheDocument();
  });

  it("renders the scoped explanation disclaimer", () => {
    const mockApi = new MockApiService();
    renderChat(mockApi);

    expect(
      screen.getByText(/Czat służy wyłącznie do wyjaśniania/i),
    ).toBeInTheDocument();
  });

  it("renders a cited finding as a control with its Polish code label", async () => {
    const mockApi = new MockApiService();
    vi.spyOn(mockApi, "sendMessage").mockResolvedValueOnce({
      intent: "ask",
      answer: {
        answer: "Wyjaśnienie odpowiedzi.",
        cited_finding_ids: ["f-4"],
        interaction_run_id: "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb",
        corpus_consulted: false,
      },
    });
    renderChat(mockApi);

    await send("Wyjaśnij ustalenie.");

    await screen.findByText("Wyjaśnienie odpowiedzi.");
    const cited = screen.queryByRole("button", {
      name: "Sprzeczność z normą prawną",
    });
    expect(cited).not.toBeNull();
    expect(cited!).not.toHaveTextContent("f-4");
    expect(
      screen.queryByText("Odpowiedź korzystała z zamrożonego korpusu"),
    ).not.toBeInTheDocument();
  });

  it("selects a cited finding through the findings selection callback", async () => {
    const mockApi = new MockApiService();
    vi.spyOn(mockApi, "sendMessage").mockResolvedValueOnce({
      intent: "ask",
      answer: {
        answer: "Wyjaśnienie odpowiedzi.",
        cited_finding_ids: ["f-4"],
        interaction_run_id: "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb",
        corpus_consulted: false,
      },
    });
    const onSelectFinding = vi.fn();
    renderChat(mockApi, { onSelectFinding });

    await send("Wyjaśnij ustalenie.");
    await screen.findByText("Wyjaśnienie odpowiedzi.");
    await userEvent.click(
      screen.getByRole("button", { name: "Sprzeczność z normą prawną" }),
    );

    expect(onSelectFinding).toHaveBeenCalledWith(MOCK_RUN_STANDARD.findings[3]);
  });

  it("reports an unknown cited finding without rendering a broken control", async () => {
    const mockApi = new MockApiService();
    vi.spyOn(mockApi, "sendMessage").mockResolvedValueOnce({
      intent: "ask",
      answer: {
        answer: "Wyjaśnienie odpowiedzi.",
        cited_finding_ids: ["unknown-finding"],
        interaction_run_id: "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb",
        corpus_consulted: false,
      },
    });
    renderChat(mockApi);

    await send("Wyjaśnij ustalenie.");
    await screen.findByText("Wyjaśnienie odpowiedzi.");

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /ustalenia niedostępnego w bieżącej analizie/i,
    );
  });

  it("renders refusal verbatim without answering when the server refuses scope", async () => {
    const mockApi = new MockApiService();
    const refusal: ApiError = {
      code: "out_of_scope",
      message_pl:
        "System odmawia odpowiedzi: zapytanie dotyczy porady prawnej.",
    };
    vi.spyOn(mockApi, "sendMessage").mockRejectedValueOnce(refusal);
    renderChat(mockApi);

    await send("Czy warto to podpisać?");

    expect(
      await screen.findByText(
        "System odmawia odpowiedzi: zapytanie dotyczy porady prawnej.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Odmowa odpowiedzi \(zakres niedozwolony\)/i),
    ).toBeInTheDocument();
  });

  it("labels an answer only when the corpus was consulted", async () => {
    const mockApi = new MockApiService();
    vi.spyOn(mockApi, "sendMessage").mockResolvedValueOnce({
      intent: "ask",
      answer: {
        answer: "Wyjaśnienie z korpusu.",
        cited_finding_ids: [],
        interaction_run_id: "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb",
        corpus_consulted: true,
      },
    });
    renderChat(mockApi);

    await send("Wyjaśnij podstawę prawną.");

    await screen.findByText("Wyjaśnienie z korpusu.");
    expect(
      screen.getByText("Odpowiedź korzystała z zamrożonego korpusu"),
    ).toBeInTheDocument();
  });

  it("hands up the child run when a message is read as an objection", async () => {
    const mockApi = new MockApiService();
    const onChildRun = vi.fn();
    renderChat(mockApi, { onChildRun });

    await send("Nie zgadzam się z tym ustaleniem.");

    await waitFor(() => expect(onChildRun).toHaveBeenCalledTimes(1));
    expect(
      screen.getByText(/System przyjął zastrzeżenie/i),
    ).toBeInTheDocument();
  });

  it("hands up the child run when a message asks for a re-analysis", async () => {
    const mockApi = new MockApiService();
    const onChildRun = vi.fn();
    renderChat(mockApi, { onChildRun });

    await send("Przeanalizuj tę klauzulę jeszcze raz.");

    await waitFor(() => expect(onChildRun).toHaveBeenCalledTimes(1));
    expect(screen.getByText(/System ponownie analizuje/i)).toBeInTheDocument();
  });
});
