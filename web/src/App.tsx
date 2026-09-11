import React, { useState, useEffect, useRef, useCallback } from "react";
import type {
  DocumentDescriptor,
  DocumentContent,
  Run,
  Finding,
  ArmCode,
  ApiError,
  PublicConfig,
} from "./types";
import {
  ARM_LABELS,
  INTERACTION_ERROR_MESSAGES_PL,
  RUN_STATUS_LABELS_PL,
  isRunStatus,
} from "./types";
import { HttpApiClient, type IApiClient } from "./api";
import { DocumentPane } from "./DocumentPane";
import { FindingsPane } from "./FindingsPane";
import { saveSession, loadSession, clearSession } from "./sessionCache";
import type { ChatMessage } from "./ChatSection";
import {
  UploadDropzone,
  ACCEPTED_FILE_EXTENSIONS,
  getFileExtension,
} from "./UploadDropzone";
import { useWorksheetCache } from "./useWorksheetCache";
import {
  Play,
  Square,
  Trash2,
  Scale,
  Loader2,
  FileCheck,
  AlertOctagon,
} from "lucide-react";

interface AppProps {
  api?: IApiClient;
}

const CONSENT_REQUIRED_MESSAGE =
  "Aby uruchomić analizę, zaznacz zgodę na przetwarzanie zewnętrzne.";

const httpApiClient = new HttpApiClient();

export const App: React.FC<AppProps> = ({ api = httpApiClient }) => {
  const initialSessionRef = useRef(loadSession());
  const initialSession = initialSessionRef.current;

  const [documentFileName, setDocumentFileName] = useState<string | null>(
    () => initialSession?.documentFileName ?? null,
  );
  const [documentDesc, setDocumentDesc] = useState<DocumentDescriptor | null>(
    () => initialSession?.documentDesc ?? null,
  );
  const [documentContent, setDocumentContent] =
    useState<DocumentContent | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isExpired, setIsExpired] = useState<boolean>(
    () => initialSession?.isExpired ?? false,
  );

  const [selectedArm, setSelectedArm] = useState<ArmCode>(
    () => initialSession?.selectedArm ?? "mid",
  );
  const [disclosureAccepted, setDisclosureAccepted] = useState<boolean>(
    () => initialSession?.disclosureAccepted ?? false,
  );
  const [currentRun, setCurrentRun] = useState<Run | null>(
    () => initialSession?.currentRun ?? null,
  );
  const [isRunning, setIsRunning] = useState(false);
  const [progressStatus, setProgressStatus] = useState<string | null>(null);
  const [processedCount, setProcessedCount] = useState<number>(0);

  const initialFinding =
    initialSession?.selectedFindingId && initialSession?.currentRun?.findings
      ? (initialSession.currentRun.findings.find(
          (f) => f.id === initialSession.selectedFindingId,
        ) ?? null)
      : null;
  const [selectedFinding, setSelectedFinding] = useState<Finding | null>(
    () => initialFinding,
  );
  const handleContentExpiredRef = useRef<() => void>(() => {});
  const { worksheets, loadWorksheet, clearWorksheets } = useWorksheetCache(
    api,
    () => handleContentExpiredRef.current(),
    initialSession?.worksheets,
  );
  const [chatMessages, setChatMessages] = useState<
    Record<string, ChatMessage[]>
  >(() => initialSession?.chatMessages ?? {});

  const prevRunIdRef = useRef<string | undefined>(
    initialSession?.currentRun?.id,
  );

  useEffect(() => {
    if (prevRunIdRef.current !== currentRun?.id) {
      prevRunIdRef.current = currentRun?.id;
      clearWorksheets();
    }
  }, [currentRun?.id, clearWorksheets]);

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [config, setConfig] = useState<PublicConfig | null>(null);

  useEffect(() => {
    let isCancelled = false;
    api
      .getConfig()
      .then((cfg) => {
        if (!isCancelled) {
          setConfig(cfg);
        }
      })
      .catch(() => {
        console.error("[config] configuration fetch failed");
      });
    return () => {
      isCancelled = true;
    };
  }, [api]);

  const sseCancelRef = useRef<(() => void) | null>(null);
  const descriptorExpired = Boolean(
    documentDesc?.expires_at &&
    Date.parse(documentDesc.expires_at) <= Date.now(),
  );
  const isDocumentExpired =
    isExpired || currentRun?.content_available === false || descriptorExpired;
  const canInvokeModel = Boolean(
    documentDesc && disclosureAccepted && !isDocumentExpired,
  );
  const modelActionDisabledReason = !disclosureAccepted
    ? CONSENT_REQUIRED_MESSAGE
    : undefined;

  const handleContentExpired = useCallback(() => {
    setIsExpired(true);
    setDocumentContent(null);
    clearWorksheets();
    setChatMessages({});
    setCurrentRun((run) => (run ? { ...run, synthesis: null } : run));
  }, [clearWorksheets]);
  handleContentExpiredRef.current = handleContentExpired;

  const handleApiError = useCallback(
    (err: unknown, fallbackMessage: string) => {
      const apiErr = err as ApiError;
      if (apiErr?.code === "content_expired") {
        handleContentExpired();
      }
      setErrorMessage(
        INTERACTION_ERROR_MESSAGES_PL[apiErr?.code] ||
          apiErr?.message_pl ||
          fallbackMessage,
      );
    },
    [handleContentExpired],
  );

  const subscribeToRun = useCallback(
    (runId: string, totalUnits: number) => {
      return api.streamRunEvents(
        runId,
        (event) => {
          if (event.kind === "status" && event.status) {
            const statusLabel = isRunStatus(event.status)
              ? RUN_STATUS_LABELS_PL[event.status]
              : event.status;
            setProgressStatus(`Status: ${statusLabel}`);
          } else if (
            event.kind === "counter" &&
            event.count !== undefined &&
            event.count !== null
          ) {
            setProcessedCount(event.count);
            setProgressStatus(
              `Przetwarzanie jednostek: ${event.count} / ${totalUnits}`,
            );
          }
        },
        (completedRun: Run) => {
          setCurrentRun(completedRun);
          setIsRunning(false);
          setProgressStatus(null);
        },
        (err: unknown) => {
          console.error(`[run-stream:${runId}] stream failed`);
          handleApiError(
            err,
            "Utracono połączenie z analizą. Spróbuj uruchomić ją ponownie.",
          );
          setIsRunning(false);
          setProgressStatus(null);
          sseCancelRef.current = null;
        },
      );
    },
    [api, handleApiError],
  );

  useEffect(() => {
    const session = initialSessionRef.current;
    if (!session?.documentDesc) return;

    let isCancelled = false;

    // Full document content is fetched on reload rather than cached in the tab.
    if (!session.isExpired && session.documentDesc) {
      const docId = session.documentDesc.id;
      api
        .getDocumentContent(docId)
        .then((content) => {
          if (!isCancelled) {
            setDocumentContent(content);
          }
        })
        .catch((err) => {
          if (!isCancelled) {
            if ((err as ApiError)?.code === "content_expired") {
              handleContentExpired();
            } else {
              console.error("[session-restore] document restore failed", docId);
            }
          }
        });
    }

    // A run that was still going when the tab reloaded is reconnected, not restarted.
    if (session.currentRun?.id && session.currentRun.status === "running") {
      const runId = session.currentRun.id;
      api
        .getRun(runId)
        .then((backendRun) => {
          if (isCancelled) return;
          setCurrentRun(backendRun);

          if (backendRun.status === "running") {
            setIsRunning(true);
            const totalUnits =
              backendRun.arm === "off"
                ? 1
                : (session.documentDesc?.unit_count ?? 1);
            setProgressStatus("Wznawianie monitorowania analizy...");

            sseCancelRef.current = subscribeToRun(backendRun.id, totalUnits);
          }
        })
        .catch((err) => {
          if (!isCancelled) {
            if ((err as ApiError)?.code === "content_expired") {
              handleContentExpired();
            } else {
              console.error("[session-restore] run restore failed", runId);
            }
          }
        });
    }

    return () => {
      isCancelled = true;
      if (sseCancelRef.current) {
        sseCancelRef.current();
        sseCancelRef.current = null;
      }
    };
  }, [api, handleContentExpired, subscribeToRun]);

  useEffect(() => {
    if (!documentDesc) {
      clearSession();
      return;
    }
    saveSession({
      version: 1,
      documentDesc,
      documentFileName,
      currentRun,
      selectedArm,
      disclosureAccepted,
      selectedFindingId: selectedFinding?.id ?? null,
      worksheets,
      chatMessages,
      isExpired,
    });
  }, [
    documentDesc,
    documentFileName,
    currentRun,
    selectedArm,
    disclosureAccepted,
    selectedFinding?.id,
    worksheets,
    chatMessages,
    isExpired,
  ]);

  const handleChatMessagesChange = (runId: string, messages: ChatMessage[]) => {
    setChatMessages((prev) => ({
      ...prev,
      [runId]: messages,
    }));
  };

  const handleFileUpload = async (file: File) => {
    const extension = getFileExtension(file.name);
    if (!ACCEPTED_FILE_EXTENSIONS.includes(extension)) {
      setErrorMessage("Nieobsługiwany format pliku.");
      return;
    }

    clearSession();

    setErrorMessage(null);
    setIsUploading(true);
    setIsExpired(false);
    setSelectedFinding(null);
    setCurrentRun(null);

    try {
      setDocumentFileName(file.name);
      const descriptor = await api.uploadDocument(file);
      setDocumentDesc(descriptor);

      if (
        descriptor.expires_at &&
        Date.parse(descriptor.expires_at) <= Date.now()
      ) {
        handleContentExpired();
        return;
      }

      const content = await api.getDocumentContent(descriptor.id);
      setDocumentContent(content);
    } catch (err: unknown) {
      handleApiError(err, "Wystąpił błąd podczas przesyłania dokumentu.");
      setDocumentFileName(null);
      setDocumentDesc(null);
      setDocumentContent(null);
    } finally {
      setIsUploading(false);
    }
  };

  const handleStartRun = async () => {
    const document = documentDesc;
    if (!canInvokeModel || !document || isRunning) return;

    setErrorMessage(null);
    setIsRunning(true);
    setProgressStatus("Inicjalizacja analizy...");
    setProcessedCount(0);
    setSelectedFinding(null);

    try {
      const run = await api.createRun({
        document_id: document.id,
        arm: selectedArm,
      });

      setCurrentRun(run);
      const totalUnits = selectedArm === "off" ? 1 : document.unit_count;

      sseCancelRef.current = subscribeToRun(run.id, totalUnits);
    } catch (err: unknown) {
      handleApiError(err, "Nie udało się uruchomić analizy.");
      setIsRunning(false);
      setProgressStatus(null);
    }
  };

  const handleCancelRun = async () => {
    if (!currentRun || !isRunning) return;

    if (sseCancelRef.current) {
      sseCancelRef.current();
      sseCancelRef.current = null;
    }

    try {
      const cancelledRun = await api.cancelRun(currentRun.id);
      setCurrentRun(cancelledRun);
    } catch (err: unknown) {
      handleApiError(err, "Nie udało się przerwać analizy.");
    } finally {
      setIsRunning(false);
      setProgressStatus(null);
    }
  };

  const handleDeleteSession = async () => {
    if (!documentDesc) return;
    try {
      await api.deleteDocument(documentDesc.id);
      if (sseCancelRef.current) {
        sseCancelRef.current();
        sseCancelRef.current = null;
      }
      setCurrentRun(null);
      setDocumentContent(null);
      setDocumentDesc(null);
      setDocumentFileName(null);
      setSelectedFinding(null);
      clearWorksheets();
      setChatMessages({});
      setIsExpired(false);
      setIsRunning(false);
      setProgressStatus(null);
      setProcessedCount(0);
      clearSession();
    } catch (err: unknown) {
      handleApiError(err, "Nie udało się usunąć sesji.");
    }
  };

  useEffect(() => {
    return () => {
      if (sseCancelRef.current) {
        sseCancelRef.current();
      }
    };
  }, []);

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="app-title-group">
          <h1>
            <Scale size={20} />
            Analizator Umów
          </h1>
        </div>

        <div className="header-actions">
          {documentDesc && !isDocumentExpired && (
            <>
              <button
                type="button"
                className="btn btn-sm btn-danger"
                onClick={handleDeleteSession}
                disabled={isRunning}
                title="Usuń treść dokumentu z pamięci serwera"
              >
                <Trash2 size={14} />
                Usuń sesję (Purge)
              </button>
            </>
          )}
        </div>
      </header>

      {errorMessage && (
        <div
          className="interruption-banner"
          style={{ margin: "0.5rem 1rem", borderRadius: "4px" }}
          role="alert"
        >
          <AlertOctagon size={20} />
          <div>{errorMessage}</div>
        </div>
      )}

      <main className="workspace-main">
        <section
          className="pane-document"
          aria-label="Panel dokumentu źródłowego"
        >
          <div className="pane-toolbar">
            <h2>Dokument źródłowy</h2>
            {documentDesc && !isDocumentExpired && (
              <span
                style={{
                  fontSize: "0.75rem",
                  color: "var(--text-muted)",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.25rem",
                }}
              >
                <FileCheck size={14} />
                {documentFileName} ({documentDesc.unit_count} jednostek)
                {documentDesc.read_mode &&
                  ` · ${documentDesc.read_mode === "native_pdf" ? "PDF Text" : "PDF OCR"}`}
              </span>
            )}
          </div>

          {!documentDesc && !isUploading ? (
            <div className="pane-content">
              <UploadDropzone
                onFileSelected={handleFileUpload}
                maxInputBytes={config?.limits?.max_input_bytes}
                label="Wybierz lub upuść plik umowy"
              />
            </div>
          ) : isUploading ? (
            <div
              className="pane-content"
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
              }}
            >
              <div style={{ textAlign: "center" }}>
                <Loader2
                  className="animate-spin"
                  size={36}
                  style={{ margin: "0 auto 1rem auto" }}
                />
                <p>Przetwarzanie i kanonizacja dokumentu...</p>
              </div>
            </div>
          ) : (
            <>
              {isDocumentExpired && (
                <div className="expired-upload">
                  <UploadDropzone
                    onFileSelected={handleFileUpload}
                    maxInputBytes={config?.limits?.max_input_bytes}
                    label="Wybierz lub upuść nowy plik umowy"
                  />
                </div>
              )}
              <DocumentPane
                content={documentContent}
                selectedFinding={selectedFinding}
                onSelectFinding={setSelectedFinding}
                isExpired={isDocumentExpired}
                run={currentRun}
                worksheets={worksheets}
                onLoadWorksheet={loadWorksheet}
              />
            </>
          )}
        </section>

        <section
          className="pane-findings"
          aria-label="Panel sterowania i czatu"
        >
          <div className="pane-toolbar">
            <h2>Konfiguracja i czat</h2>
            {isRunning && (
              <span
                style={{
                  fontSize: "0.8rem",
                  color: "#2563eb",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.35rem",
                }}
              >
                <Loader2 className="animate-spin" size={14} />
                {progressStatus || "Analiza w toku..."}
              </span>
            )}
          </div>

          <div style={{ padding: "1rem 1.25rem 0 1.25rem" }}>
            <fieldset className="arm-selector-group">
              <legend className="arm-selector-title">
                Wariant mechanizmu analizy
              </legend>
              <div className="arm-options" role="radiogroup">
                {(["off", "mid", "on"] as ArmCode[]).map((arm) => (
                  <label
                    key={arm}
                    className={`arm-option-label ${selectedArm === arm ? "active" : ""}`}
                  >
                    <input
                      type="radio"
                      name="analysis-arm"
                      value={arm}
                      checked={selectedArm === arm}
                      onChange={() => setSelectedArm(arm)}
                      disabled={isRunning}
                    />
                    <span>{ARM_LABELS[arm]}</span>
                  </label>
                ))}
              </div>
            </fieldset>

            <div className="disclosure-banner">
              <strong>Poufność i przetwarzanie danych:</strong> OCR działa
              lokalnie na serwerze. Wyszukiwanie przepisów korzysta z
              zewnętrznej usługi osadzeń (OpenRouter): trafiają do niej frazy
              wyszukiwawcze układane przez model, które mogą zawierać fragmenty
              dokumentu. Treść dokumentu — w całości albo w podziale na
              jednostki, zależnie od wybranego wariantu — jest przesyłana do
              skonfigurowanego dostawcy modelu analizującego. Dostawca jest
              zewnętrzny wobec instalacji, a warunki przetwarzania po jego
              stronie zależą od wybranego modelu i od ustawień konta. Narzędzie
              nie służy do przetwarzania danych osobowych.
              <label className="disclosure-checkbox">
                <input
                  type="checkbox"
                  checked={disclosureAccepted}
                  onChange={(e) => setDisclosureAccepted(e.target.checked)}
                  disabled={isRunning}
                />
                <span>
                  Przyjmuję do wiadomości informacje o przetwarzaniu zewnętrznym
                </span>
              </label>
            </div>

            <div
              style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}
            >
              {!isRunning ? (
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={handleStartRun}
                  disabled={!canInvokeModel}
                  title={modelActionDisabledReason}
                  aria-description={modelActionDisabledReason}
                  style={{ flex: 1 }}
                >
                  <Play size={16} />
                  Uruchom analizę ({selectedArm.toUpperCase()})
                </button>
              ) : (
                <button
                  type="button"
                  className="btn btn-danger"
                  onClick={handleCancelRun}
                  style={{ flex: 1 }}
                >
                  <Square size={16} />
                  Przerwij analizę ({processedCount} /{" "}
                  {selectedArm === "off" ? 1 : documentDesc?.unit_count})
                </button>
              )}
            </div>
          </div>

          <FindingsPane
            run={currentRun}
            onSelectFinding={setSelectedFinding}
            api={api}
            isExpired={isDocumentExpired}
            onContentExpired={handleContentExpired}
            onApiError={handleApiError}
            chatMessages={chatMessages}
            onChatMessagesChange={handleChatMessagesChange}
          />
        </section>
      </main>
    </div>
  );
};
