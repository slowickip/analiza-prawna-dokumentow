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
} from "../types";
import type { IApiClient } from "../api";
import { ACCEPTED_FILE_EXTENSIONS, getFileExtension } from "../UploadDropzone";

type MockInteractionRequest =
  | { kind: "contest"; note: string; finding_id: string }
  | { kind: "analyse"; note: string; unit_id: string };
import {
  MOCK_DOC_ID_STANDARD,
  MOCK_DOC_ID_INTERRUPTED,
  MOCK_DOC_ID_EXPIRED,
  MOCK_DOC_DESCRIPTOR_STANDARD,
  MOCK_DOC_CONTENT_STANDARD,
  MOCK_RUN_STANDARD,
  MOCK_DOC_DESCRIPTOR_INTERRUPTED,
  MOCK_DOC_CONTENT_INTERRUPTED,
  MOCK_RUN_INTERRUPTED,
  MOCK_CHAT_RESPONSE,
  MOCK_RUN_ID_STANDARD_MID,
  MOCK_CONFIG,
  MOCK_WORKSHEET_STANDARD,
} from "./fixtures";

export class MockApiService implements IApiClient {
  private documents = new Map<
    string,
    { descriptor: DocumentDescriptor; content: DocumentContent | null }
  >();
  private runs = new Map<string, Run>();

  constructor() {
    this.reset();
  }

  public reset() {
    this.documents.clear();
    this.runs.clear();

    // Standard document
    this.documents.set(MOCK_DOC_ID_STANDARD, {
      descriptor: { ...MOCK_DOC_DESCRIPTOR_STANDARD },
      content: { ...MOCK_DOC_CONTENT_STANDARD },
    });
    this.runs.set(MOCK_RUN_ID_STANDARD_MID, { ...MOCK_RUN_STANDARD });

    // Interrupted document
    this.documents.set(MOCK_DOC_ID_INTERRUPTED, {
      descriptor: { ...MOCK_DOC_DESCRIPTOR_INTERRUPTED },
      content: { ...MOCK_DOC_CONTENT_INTERRUPTED },
    });
    this.runs.set(MOCK_RUN_INTERRUPTED.id, { ...MOCK_RUN_INTERRUPTED });

    // Expired document
    this.documents.set(MOCK_DOC_ID_EXPIRED, {
      descriptor: {
        id: MOCK_DOC_ID_EXPIRED,
        content_hash:
          "sha256:expired000000000000000000000000000000000000000000000000000000000",
        read_mode: "native_pdf",
        conversion: null,
        unit_count: 1,
        expires_at: new Date(Date.now() - 3600000).toISOString(),
      },
      content: null, // purged
    });
  }

  public async uploadDocument(file: File): Promise<DocumentDescriptor> {
    const ext = getFileExtension(file.name);
    if (!ACCEPTED_FILE_EXTENSIONS.includes(ext)) {
      const error: ApiError = {
        code: "unsupported_input",
        message_pl: `Format pliku ${ext || "unknown"} nie jest obsługiwany. Dopuszczalne formaty: ${ACCEPTED_FILE_EXTENSIONS.join(", ")}.`,
      };
      throw error;
    }

    if (file.size === 0) {
      const error: ApiError = {
        code: "empty_input",
        message_pl: "Przesłany plik jest pusty.",
      };
      throw error;
    }

    const id = crypto.randomUUID();
    const isPdf = ext === ".pdf";

    // If file name contains "interrupted" simulate interrupted
    const isInterruptedFixture =
      file.name.toLowerCase().includes("interrupted") ||
      file.name.toLowerCase().includes("przerw");

    const descriptor: DocumentDescriptor = {
      id,
      content_hash: `sha256:${crypto.randomUUID().replaceAll("-", "").repeat(2)}`,
      read_mode: isPdf ? "native_pdf" : null,
      conversion:
        ext === ".doc"
          ? {
              converter: "LibreOffice",
              converter_version: "24.2.5",
              output_hash: "sha256:doc123",
            }
          : null,
      unit_count: isInterruptedFixture ? 10 : 9,
      expires_at: new Date(Date.now() + 3600000).toISOString(),
    };

    const baseContent = isInterruptedFixture
      ? MOCK_DOC_CONTENT_INTERRUPTED
      : MOCK_DOC_CONTENT_STANDARD;
    const content: DocumentContent = {
      ...baseContent,
      document_id: id,
      read_mode: descriptor.read_mode,
    };

    this.documents.set(id, { descriptor, content });
    return descriptor;
  }

  public async getDocumentContent(
    documentId: string,
  ): Promise<DocumentContent> {
    const doc = this.documents.get(documentId);
    if (!doc) {
      const error: ApiError = {
        code: "not_found",
        message_pl:
          "Dokument o podanym identyfikatorze nie został odnaleziony.",
      };
      throw error;
    }

    if (!doc.content) {
      const error: ApiError = {
        code: "content_expired",
        message_pl:
          "Sesja dokumentu wygasła lub treść została usunięta z pamięci operacyjnej serwera.",
      };
      throw error;
    }

    return doc.content;
  }

  public async deleteDocument(documentId: string): Promise<void> {
    const doc = this.documents.get(documentId);
    if (doc) {
      doc.content = null; // Purge content
      for (const run of this.runs.values()) {
        if (run.document_id === documentId) {
          run.content_available = false;
        }
      }
    }
  }

  public async createRun(request: CreateRunRequest): Promise<Run> {
    const doc = this.documents.get(request.document_id);
    if (!doc) {
      const error: ApiError = {
        code: "not_found",
        message_pl: "Nie znaleziono wskazanego dokumentu.",
      };
      throw error;
    }

    if (!doc.content) {
      const error: ApiError = {
        code: "content_expired",
        message_pl:
          "Sesja dokumentu wygasła. Uruchomienie analizy wymaga ponownego przesłania pliku.",
      };
      throw error;
    }

    const runId = crypto.randomUUID();
    const isInterrupted = doc.descriptor.unit_count === 10;
    const templateRun = isInterrupted
      ? MOCK_RUN_INTERRUPTED
      : MOCK_RUN_STANDARD;

    const run: Run = {
      ...templateRun,
      id: runId,
      document_id: request.document_id,
      arm: request.arm,
      status: "running",
      created_at: new Date().toISOString(),
      finished_at: null,
      findings: [],
    };

    this.runs.set(runId, run);
    return run;
  }

  public async getRun(runId: string): Promise<Run> {
    const run = this.runs.get(runId);
    if (!run) {
      const error: ApiError = {
        code: "not_found",
        message_pl:
          "Uruchomienie (run) o podanym identyfikatorze nie istnieje.",
      };
      throw error;
    }
    return run;
  }

  public async cancelRun(runId: string): Promise<Run> {
    const run = this.runs.get(runId);
    if (!run) {
      const error: ApiError = {
        code: "not_found",
        message_pl: "Uruchomienie nie zostało znalezione.",
      };
      throw error;
    }

    if (run.status !== "running") {
      const error: ApiError = {
        code: "run_already_terminal",
        message_pl: "Uruchomienie osiągnęło już stan końcowy.",
      };
      throw error;
    }

    run.status = "cancelled";
    run.finished_at = new Date().toISOString();
    return run;
  }

  public streamRunEvents(
    runId: string,
    onEvent: (event: RunEvent) => void,
    onComplete: (run: Run) => void,
    _onError?: (err: unknown) => void,
  ): () => void {
    const run = this.runs.get(runId);
    let cancelled = false;

    if (!run) {
      return () => {};
    }

    const templateRun =
      run.document_id === MOCK_DOC_ID_INTERRUPTED
        ? MOCK_RUN_INTERRUPTED
        : MOCK_RUN_STANDARD;
    const targetFindings = templateRun.findings;

    let step = 0;
    const totalSteps = targetFindings.length;

    const emit = (event: RunEvent) => onEvent(event);

    emit({ kind: "status", status: "running" });

    const interval = setInterval(() => {
      if (cancelled) {
        clearInterval(interval);
        return;
      }

      if (step < totalSteps) {
        const finding = targetFindings[step];
        run.findings.push(finding);
        emit({
          kind: "worksheet",
          role: step % 2 === 0 ? "analyst" : "verifier",
          entry_kind: step % 2 === 0 ? "candidate" : "verdict",
          unit_id: finding.unit_id,
        });
        emit({ kind: "finding", finding_id: finding.id });
        emit({
          kind: "counter",
          counter_name: "processed_units",
          count: step + 1,
        });
        step++;
      } else {
        clearInterval(interval);
        run.status = "completed";
        run.finished_at = new Date().toISOString();
        run.metrics = templateRun.metrics;
        run.interrupted = templateRun.interrupted;
        emit({ kind: "status", status: "completed" });
        onComplete(run);
      }
    }, 150);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }

  /**
   * The single composer: the mock reads the message the way the backend router
   * does, so a test can drive an objection or a re-analysis by wording alone.
   */
  public async sendMessage(
    runId: string,
    request: MessageRequest,
  ): Promise<MessageResponse> {
    const run = this.runs.get(runId);
    if (!run) {
      const error: ApiError = {
        code: "not_found",
        message_pl: "Nie znaleziono wskazanego uruchomienia analizy.",
      };
      throw error;
    }
    const lower = request.message.toLowerCase();
    if (lower.includes("nie zgadzam") || lower.includes("kwestionuj")) {
      const finding = run.findings[0];
      return {
        intent: "contest",
        run: await this.createInteraction(runId, {
          kind: "contest",
          note: request.message,
          finding_id: finding.id,
        }),
      };
    }
    if (lower.includes("przeanalizuj") || lower.includes("jeszcze raz")) {
      const finding = run.findings[0];
      return {
        intent: "analyse",
        run: await this.createInteraction(runId, {
          kind: "analyse",
          note: request.message,
          unit_id: finding.unit_id,
        }),
      };
    }
    if (!run.content_available) {
      const error: ApiError = {
        code: "content_expired",
        message_pl:
          "Sesja dokumentu wygasła. Wyjaśnienia czatu są niedostępne.",
      };
      throw error;
    }
    if (
      lower.includes("podpisa") ||
      lower.includes("podpisze") ||
      lower.includes("rekomendac") ||
      lower.includes("dorad") ||
      lower.includes("porad") ||
      lower.includes("opłaca") ||
      lower.includes("warto")
    ) {
      const error: ApiError = {
        code: "out_of_scope",
        message_pl:
          "System odmawia odpowiedzi: zapytanie dotyczy porady prawnej lub rekomendacji biznesowej. Narzędzie badawcze służy wyłącznie do technicznego wyjaśniania zidentyfikowanych ustaleń i cytowanych podstaw prawnych.",
      };
      throw error;
    }
    return {
      intent: "ask",
      answer: {
        ...MOCK_CHAT_RESPONSE,
        cited_finding_ids:
          run.findings.length > 0 ? run.findings.map((f) => f.id) : ["f-4"],
        interaction_run_id: crypto.randomUUID(),
        corpus_consulted: true,
      },
    };
  }

  public async getWorksheet(runId: string): Promise<WorksheetResponse> {
    const run = this.runs.get(runId);
    if (!run || !run.content_available) {
      const error: ApiError = {
        code: "content_expired",
        message_pl: "Sesja dokumentu wygasła. Praca agentów jest niedostępna.",
      };
      throw error;
    }
    return { ...MOCK_WORKSHEET_STANDARD, run_id: runId };
  }

  private async createInteraction(
    runId: string,
    request: MockInteractionRequest,
  ): Promise<Run> {
    const parent = this.runs.get(runId);
    if (!parent) {
      throw {
        code: "not_found",
        message_pl: "Nie znaleziono wskazanej analizy.",
      } as ApiError;
    }
    if (parent.status !== "completed") {
      throw {
        code: "run_not_finished",
        message_pl: "Analiza nadrzędna nie została zakończona.",
      } as ApiError;
    }
    if (!parent.content_available) {
      throw {
        code: "content_expired",
        message_pl: "Sesja dokumentu wygasła.",
      } as ApiError;
    }

    const child: Run = {
      ...parent,
      id: crypto.randomUUID(),
      status: "running",
      measurement_valid: false,
      parent_run_id: parent.id,
      interaction: request.kind,
      findings: [],
      metrics: null,
      created_at: new Date().toISOString(),
      finished_at: null,
    };
    this.runs.set(child.id, child);
    return child;
  }

  public async getConfig(): Promise<PublicConfig> {
    return { ...MOCK_CONFIG };
  }
}
