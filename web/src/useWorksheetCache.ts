import { useState, useRef, useCallback } from "react";
import type { IApiClient } from "./api";
import type { WorksheetState } from "./WorksheetView";

export function useWorksheetCache(
  api: IApiClient,
  onExpired?: () => void,
  initialWorksheets?: Record<string, WorksheetState>,
) {
  const [worksheets, setWorksheets] = useState<Record<string, WorksheetState>>(
    () => initialWorksheets ?? {},
  );
  const requestedWorksheets = useRef(new Set<string>());
  const onExpiredRef = useRef(onExpired);
  onExpiredRef.current = onExpired;

  const clearWorksheets = useCallback(() => {
    requestedWorksheets.current.clear();
    setWorksheets({});
  }, []);

  const loadWorksheet = useCallback(
    async (runId: string) => {
      if (requestedWorksheets.current.has(runId)) return;
      requestedWorksheets.current.add(runId);
      setWorksheets((current) => ({
        ...current,
        [runId]: { ...current[runId], loading: true },
      }));
      try {
        const data = await api.getWorksheet(runId);
        setWorksheets((current) => ({ ...current, [runId]: { data } }));
      } catch (error) {
        const apiError = error as WorksheetState["error"];
        setWorksheets((current) => ({
          ...current,
          [runId]: { error: apiError },
        }));
        // Failed requests may be retried.
        requestedWorksheets.current.delete(runId);
        if (apiError?.code === "content_expired") {
          onExpiredRef.current?.();
        }
      }
    },
    [api],
  );

  return { worksheets, loadWorksheet, clearWorksheets };
}
