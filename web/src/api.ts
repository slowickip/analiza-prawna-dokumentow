import type {
  DocumentDescriptor,
  DocumentContent,
  Run,
  RunEvent,
  CreateRunRequest,
  ApiError,
  PublicConfig,
  WorksheetResponse,
  MessageRequest,
  MessageResponse,
} from "./types";

export interface IApiClient {
  getConfig(): Promise<PublicConfig>;
  uploadDocument(file: File): Promise<DocumentDescriptor>;
  getDocumentContent(documentId: string): Promise<DocumentContent>;
  deleteDocument(documentId: string): Promise<void>;
  createRun(request: CreateRunRequest): Promise<Run>;
  getRun(runId: string): Promise<Run>;
  cancelRun(runId: string): Promise<Run>;
  streamRunEvents(
    runId: string,
    onEvent: (event: RunEvent) => void,
    onComplete: (run: Run) => void,
    onError?: (err: unknown) => void,
  ): () => void;
  sendMessage(runId: string, request: MessageRequest): Promise<MessageResponse>;
  getWorksheet(runId: string): Promise<WorksheetResponse>;
}

export class ApiTransportError extends Error {
  constructor(public readonly message_pl: string) {
    super(message_pl);
    this.name = "ApiTransportError";
  }
}

function isApiError(value: unknown): value is ApiError {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.code === "string" &&
    typeof candidate.message_pl === "string"
  );
}

function isTerminalRun(run: Run): boolean {
  return (
    run.status === "completed" ||
    run.status === "failed" ||
    run.status === "cancelled"
  );
}

// Chosen by argument, not measurement: 2s polling with bounded exponential backoff.
const RUN_RECOVERY_POLL_INTERVAL_MS = 2_000;
const RUN_RECOVERY_RETRY_DELAYS_MS = [1_000, 2_000, 4_000, 8_000] as const;

export class HttpApiClient implements IApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = "/api/v1") {
    this.baseUrl = baseUrl;
  }

  private async request<T>(url: string, init?: RequestInit): Promise<T> {
    let response: Response;
    try {
      response = await fetch(url, init);
    } catch {
      throw new ApiTransportError("Nie udało się połączyć z serwerem.");
    }
    return this.handleResponse<T>(response);
  }

  private async handleResponse<T>(res: Response): Promise<T> {
    if (!res.ok) {
      let errorData: unknown;
      try {
        errorData = await res.json();
      } catch {
        throw new ApiTransportError(
          `Serwer zwrócił nieprawidłową odpowiedź (HTTP ${res.status}).`,
        );
      }
      if (!isApiError(errorData)) {
        throw new ApiTransportError(
          `Serwer zwrócił nieprawidłową odpowiedź (HTTP ${res.status}).`,
        );
      }
      throw errorData;
    }
    if (res.status === 204) {
      return undefined as unknown as T;
    }
    try {
      return await res.json();
    } catch {
      throw new ApiTransportError(
        `Serwer zwrócił nieprawidłową odpowiedź (HTTP ${res.status}).`,
      );
    }
  }

  async uploadDocument(file: File): Promise<DocumentDescriptor> {
    const formData = new FormData();
    formData.append("file", file);
    return this.request<DocumentDescriptor>(`${this.baseUrl}/documents`, {
      method: "POST",
      body: formData,
    });
  }

  async getDocumentContent(documentId: string): Promise<DocumentContent> {
    return this.request<DocumentContent>(
      `${this.baseUrl}/documents/${documentId}/content`,
    );
  }

  async deleteDocument(documentId: string): Promise<void> {
    return this.request<void>(`${this.baseUrl}/documents/${documentId}`, {
      method: "DELETE",
    });
  }

  async createRun(request: CreateRunRequest): Promise<Run> {
    return this.request<Run>(`${this.baseUrl}/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  }

  async getRun(runId: string): Promise<Run> {
    return this.request<Run>(`${this.baseUrl}/runs/${runId}`);
  }

  async cancelRun(runId: string): Promise<Run> {
    return this.request<Run>(`${this.baseUrl}/runs/${runId}/cancel`, {
      method: "POST",
    });
  }

  streamRunEvents(
    runId: string,
    onEvent: (event: RunEvent) => void,
    onComplete: (run: Run) => void,
    onError?: (err: unknown) => void,
  ): () => void {
    const sseUrl = `${this.baseUrl}/runs/${runId}/events`;
    let isStopped = false;
    let eventSource: EventSource | null = null;
    let recoveryTimer: ReturnType<typeof setTimeout> | null = null;
    let consecutiveFailures = 0;

    const stopTracking = () => {
      isStopped = true;
      eventSource?.close();
      eventSource = null;
      if (recoveryTimer !== null) {
        clearTimeout(recoveryTimer);
        recoveryTimer = null;
      }
    };

    const complete = (run: Run) => {
      if (isStopped) return;
      stopTracking();
      onComplete(run);
    };

    const fail = (err: unknown) => {
      if (isStopped) return;
      stopTracking();
      if (onError) onError(err);
    };

    const schedulePoll = (delayMs: number) => {
      if (isStopped) return;
      recoveryTimer = setTimeout(() => {
        recoveryTimer = null;
        void pollRun();
      }, delayMs);
    };

    const pollRun = async () => {
      if (isStopped) return;
      try {
        const currentRun = await this.getRun(runId);
        if (isStopped) return;
        consecutiveFailures = 0;
        if (isTerminalRun(currentRun)) {
          complete(currentRun);
        } else {
          schedulePoll(RUN_RECOVERY_POLL_INTERVAL_MS);
        }
      } catch (err) {
        if (isStopped) return;
        if (isApiError(err) && err.code === "content_expired") {
          fail(err);
          return;
        }

        const retryDelay = RUN_RECOVERY_RETRY_DELAYS_MS[consecutiveFailures];
        if (retryDelay === undefined) {
          fail(
            new ApiTransportError(
              "Utracono połączenie z analizą. Spróbuj uruchomić ją ponownie.",
            ),
          );
          return;
        }
        consecutiveFailures += 1;
        schedulePoll(retryDelay);
      }
    };

    try {
      eventSource = new EventSource(sseUrl);

      eventSource.onmessage = async (e) => {
        if (isStopped) return;
        let event: RunEvent;
        try {
          event = JSON.parse(e.data);
        } catch (err) {
          fail(err);
          return;
        }

        onEvent(event);

        if (
          event.kind === "status" &&
          (event.status === "completed" ||
            event.status === "failed" ||
            event.status === "cancelled")
        ) {
          eventSource?.close();
          let fullRun: Run;
          try {
            fullRun = await this.getRun(runId);
          } catch (err) {
            fail(err);
            return;
          }
          complete(fullRun);
        }
      };

      eventSource.onerror = async () => {
        if (isStopped) return;
        // On reconnect / SSE disconnect, fall back to GET /runs/{runId} for authoritative state
        eventSource?.close();
        eventSource = null;
        await pollRun();
      };
    } catch {
      fail(
        new ApiTransportError(
          "Utracono połączenie z analizą. Spróbuj uruchomić ją ponownie.",
        ),
      );
    }

    return stopTracking;
  }

  async sendMessage(
    runId: string,
    request: MessageRequest,
  ): Promise<MessageResponse> {
    return this.request<MessageResponse>(
      `${this.baseUrl}/runs/${runId}/message`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      },
    );
  }

  async getWorksheet(runId: string): Promise<WorksheetResponse> {
    return this.request<WorksheetResponse>(
      `${this.baseUrl}/runs/${runId}/worksheet`,
    );
  }

  async getConfig(): Promise<PublicConfig> {
    return this.request<PublicConfig>(`${this.baseUrl}/config`);
  }
}
