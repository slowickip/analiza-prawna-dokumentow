import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  MOCK_RUN_ID_STANDARD_MID,
  MOCK_RUN_STANDARD,
  MOCK_CONFIG,
  MOCK_WORKSHEET_STANDARD,
} from "./test-support/fixtures";
import { ControllableEventSource } from "./test-support/controllableEventSource";
import type { PublicConfig, Run } from "./types";
import { ApiTransportError, HttpApiClient } from "./api";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  ControllableEventSource.reset();
});

const RUNNING_RUN: Run = {
  ...MOCK_RUN_STANDARD,
  status: "running",
  finished_at: null,
};

describe("HttpApiClient run stream recovery", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("EventSource", ControllableEventSource);
  });

  it("keeps tracking after a non-terminal stream disconnect", async () => {
    const client = new HttpApiClient();
    const getRun = vi.spyOn(client, "getRun").mockResolvedValue(RUNNING_RUN);
    const cancel = client.streamRunEvents(
      MOCK_RUN_ID_STANDARD_MID,
      vi.fn(),
      vi.fn(),
    );

    ControllableEventSource.instances[0].emitError();
    await Promise.resolve();
    expect(getRun).toHaveBeenCalledOnce();

    await vi.advanceTimersToNextTimerAsync();
    expect(getRun).toHaveBeenCalledTimes(2);
    cancel();
  });

  it("completes recovery when polling observes a completed run", async () => {
    const completedRun: Run = { ...MOCK_RUN_STANDARD, status: "completed" };
    const client = new HttpApiClient();
    const getRun = vi
      .spyOn(client, "getRun")
      .mockResolvedValueOnce(RUNNING_RUN)
      .mockResolvedValueOnce(completedRun);
    const onComplete = vi.fn();
    client.streamRunEvents(MOCK_RUN_ID_STANDARD_MID, vi.fn(), onComplete);

    ControllableEventSource.instances[0].emitError();
    await Promise.resolve();
    await vi.advanceTimersToNextTimerAsync();

    expect(onComplete).toHaveBeenCalledOnce();
    expect(onComplete).toHaveBeenCalledWith(completedRun);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(getRun).toHaveBeenCalledTimes(2);
  });

  it("completes immediately when the disconnect check observes a terminal run", async () => {
    const terminalRun: Run = { ...MOCK_RUN_STANDARD, status: "failed" };
    const client = new HttpApiClient();
    const getRun = vi.spyOn(client, "getRun").mockResolvedValue(terminalRun);
    const onComplete = vi.fn();
    client.streamRunEvents(MOCK_RUN_ID_STANDARD_MID, vi.fn(), onComplete);

    ControllableEventSource.instances[0].emitError();
    await Promise.resolve();

    expect(onComplete).toHaveBeenCalledOnce();
    expect(onComplete).toHaveBeenCalledWith(terminalRun);
    expect(getRun).toHaveBeenCalledOnce();
  });

  it("calls onError when a terminal status event arrives and getRun rejects", async () => {
    const client = new HttpApiClient();
    const getRunError = new ApiTransportError("Serwer niedostępny.");
    const getRun = vi.spyOn(client, "getRun").mockRejectedValue(getRunError);
    const onComplete = vi.fn();
    const onError = vi.fn();
    client.streamRunEvents(
      MOCK_RUN_ID_STANDARD_MID,
      vi.fn(),
      onComplete,
      onError,
    );

    ControllableEventSource.instances[0].emitMessage(
      JSON.stringify({ kind: "status", status: "completed" }),
    );
    await Promise.resolve();

    expect(onError).toHaveBeenCalledOnce();
    expect(onError).toHaveBeenCalledWith(getRunError);
    expect(onComplete).not.toHaveBeenCalled();
    expect(getRun).toHaveBeenCalledOnce();
  });

  it("calls onError when an SSE payload is malformed JSON", async () => {
    const client = new HttpApiClient();
    const getRun = vi.spyOn(client, "getRun");
    const onComplete = vi.fn();
    const onError = vi.fn();
    client.streamRunEvents(
      MOCK_RUN_ID_STANDARD_MID,
      vi.fn(),
      onComplete,
      onError,
    );

    ControllableEventSource.instances[0].emitMessage("not-json");
    await Promise.resolve();

    expect(onError).toHaveBeenCalledOnce();
    expect(onError.mock.calls[0][0]).toBeInstanceOf(SyntaxError);
    expect(onComplete).not.toHaveBeenCalled();
    expect(getRun).not.toHaveBeenCalled();
  });

  it("propagates an exception thrown by onComplete without routing to onError", async () => {
    const completedRun: Run = { ...MOCK_RUN_STANDARD, status: "completed" };
    const client = new HttpApiClient();
    vi.spyOn(client, "getRun").mockResolvedValue(completedRun);
    const onCompleteError = new Error("onComplete blew up");
    const onComplete = vi.fn(() => {
      throw onCompleteError;
    });
    const onError = vi.fn();
    client.streamRunEvents(
      MOCK_RUN_ID_STANDARD_MID,
      vi.fn(),
      onComplete,
      onError,
    );

    await expect(
      ControllableEventSource.instances[0].emitMessage(
        JSON.stringify({ kind: "status", status: "completed" }),
      ),
    ).rejects.toThrow(onCompleteError);

    expect(onComplete).toHaveBeenCalledOnce();
    expect(onError).not.toHaveBeenCalled();
  });

  it("cancels pending recovery polling during teardown", async () => {
    const client = new HttpApiClient();
    const getRun = vi.spyOn(client, "getRun").mockResolvedValue(RUNNING_RUN);
    const onComplete = vi.fn();
    const onError = vi.fn();
    const cancel = client.streamRunEvents(
      MOCK_RUN_ID_STANDARD_MID,
      vi.fn(),
      onComplete,
      onError,
    );

    ControllableEventSource.instances[0].emitError();
    await Promise.resolve();
    await vi.advanceTimersToNextTimerAsync();
    expect(getRun).toHaveBeenCalledTimes(2);

    cancel();
    await vi.advanceTimersByTimeAsync(60_000);

    expect(getRun).toHaveBeenCalledTimes(2);
    expect(onComplete).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });
});

describe("HttpApiClient transport errors", () => {
  it("turns a rejected fetch into a Polish client-side error without a wire code", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
    );

    const error = await new HttpApiClient().getRun("run-1").then(
      () => null,
      (reason: unknown) => reason,
    );

    expect(error).toMatchObject({
      name: "ApiTransportError",
      message_pl: "Nie udało się połączyć z serwerem.",
    });
    expect(error).toBeInstanceOf(ApiTransportError);
    expect(error).not.toHaveProperty("code");
  });

  it("turns an invalid wire error into a Polish client-side error without a wire code", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ message: "Bad gateway" }), {
          status: 502,
          statusText: "Bad Gateway",
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    const error = await new HttpApiClient().getRun("run-1").then(
      () => null,
      (reason: unknown) => reason,
    );

    expect(error).toMatchObject({
      name: "ApiTransportError",
      message_pl: "Serwer zwrócił nieprawidłową odpowiedź (HTTP 502).",
    });
    expect(error).toBeInstanceOf(ApiTransportError);
    expect(error).not.toHaveProperty("code");
  });
});

describe("HttpApiClient operations", () => {
  it("sendMessage posts message request and returns message response", async () => {
    const mockResponse = {
      intent: "ask" as const,
      answer: {
        answer: "Odpowiedź na pytanie",
        cited_finding_ids: ["f-4"],
        interaction_run_id: "int-1",
        corpus_consulted: true,
      },
    };
    const fetchStub = vi.fn().mockResolvedValue(Response.json(mockResponse));
    vi.stubGlobal("fetch", fetchStub);

    const result = await new HttpApiClient().sendMessage(
      MOCK_RUN_ID_STANDARD_MID,
      {
        message: "Dlaczego ta jednostka jest sprzeczna?",
      },
    );

    expect(result).toEqual(mockResponse);
    expect(fetchStub).toHaveBeenCalledWith(
      `/api/v1/runs/${MOCK_RUN_ID_STANDARD_MID}/message`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: "Dlaczego ta jednostka jest sprzeczna?",
        }),
      },
    );
  });

  it("fetches public configuration via HttpApiClient.getConfig", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(MOCK_CONFIG), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    const client = new HttpApiClient();
    const config: PublicConfig = await client.getConfig();
    expect(config).toEqual(MOCK_CONFIG);
    expect(config.limits.max_input_bytes).toBe(26214400);
  });

  it("fetches the retained worksheet from the run endpoint", async () => {
    const fetchStub = vi
      .fn()
      .mockResolvedValue(Response.json(MOCK_WORKSHEET_STANDARD));
    vi.stubGlobal("fetch", fetchStub);

    const worksheet = await new HttpApiClient().getWorksheet(
      MOCK_RUN_ID_STANDARD_MID,
    );

    expect(worksheet).toEqual(MOCK_WORKSHEET_STANDARD);
    expect(fetchStub).toHaveBeenCalledWith(
      `/api/v1/runs/${MOCK_RUN_ID_STANDARD_MID}/worksheet`,
      undefined,
    );
  });
});
