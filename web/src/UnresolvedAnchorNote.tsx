import React from "react";
import { MapPinOff } from "lucide-react";
import type { QuoteResolution } from "./types";
import { QUOTE_RESOLUTION_LABELS_PL } from "./types";

interface UnresolvedAnchorNoteProps {
  quoteResolution?: QuoteResolution | null;
}

export const UnresolvedAnchorNote: React.FC<UnresolvedAnchorNoteProps> = ({
  quoteResolution,
}) => {
  const detail =
    quoteResolution && quoteResolution in QUOTE_RESOLUTION_LABELS_PL
      ? ` (${QUOTE_RESOLUTION_LABELS_PL[quoteResolution]})`
      : "";

  return (
    <div
      className="unresolved-anchor-note"
      data-testid="unresolved-anchor-note"
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.35rem",
        }}
      >
        <MapPinOff size={14} />
        <span>
          <strong>Zakotwiczenie nierozstrzygnięte:</strong> System nie powiązał
          ustalenia z konkretnym fragmentem tekstu{detail}. Pozycja w dokumencie
          nie jest przybliżana.
        </span>
      </div>
    </div>
  );
};
