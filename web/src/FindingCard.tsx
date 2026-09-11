import React from "react";
import type { Finding } from "./types";
import {
  ACT_FORCE_LABELS_PL,
  CHARACTER_KIND_LABELS_PL,
  FINDING_CODE_LABELS_PL,
  PROVISION_FORCE_LABELS_PL,
  UNCERTAIN_CAUSE_LABELS_PL,
} from "./types";
import { AlertOctagon, AlertTriangle, HelpCircle, Info } from "lucide-react";
import { LegalLocatorLink } from "./LegalLocatorLink";
import { UnresolvedAnchorNote } from "./UnresolvedAnchorNote";
import { WorksheetView, type WorksheetState } from "./WorksheetView";

const PROMINENCE_META = {
  critical: {
    icon: AlertOctagon,
    color: "var(--critical-text)",
    label: "Krytyczne",
  },
  warning: {
    icon: AlertTriangle,
    color: "var(--warning-text)",
    label: "Ostrzeżenie",
  },
  neutral: {
    icon: Info,
    color: "var(--neutral-text)",
    label: "Informacja",
  },
} as const;

export const FindingCard: React.FC<{
  finding: Finding;
  selected: boolean;
  onSelect: (finding: Finding) => void;
  worksheetState: WorksheetState;
  onLoadWorksheet: () => void;
}> = ({ finding, selected, onSelect, worksheetState, onLoadWorksheet }) => {
  const codeLabel = FINDING_CODE_LABELS_PL[finding.code] || finding.code;
  const meta = PROMINENCE_META[finding.prominence];
  const Icon = meta.icon;

  return (
    <div
      className={`finding-card ${selected ? "selected" : ""}`}
      data-level={finding.prominence}
      data-testid={`finding-card-${finding.id}`}
      tabIndex={0}
      role="article"
      aria-label={`Ustalenie: ${codeLabel}`}
      onClick={() => onSelect(finding)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect(finding);
        }
      }}
    >
      <div className="finding-header">
        <div className="finding-title">
          <Icon size={16} color={meta.color} aria-hidden="true" />
          <span className="finding-code-title">{codeLabel}</span>
        </div>

        <span className="finding-badge">{meta.label}</span>
      </div>

      {finding.code === "uncertain" && (
        <div className="finding-details" data-testid="uncertain-cause-details">
          <div className="finding-title">
            <HelpCircle size={14} />
            <span>
              <strong>Przyczyna niepewności:</strong>{" "}
              {finding.uncertain_cause
                ? UNCERTAIN_CAUSE_LABELS_PL[finding.uncertain_cause]
                : "Brak zarejestrowanej przyczyny"}
            </span>
          </div>
        </div>
      )}

      {(finding.basis ||
        (finding.legal_locators && finding.legal_locators.length > 0)) && (
        <div className="finding-details" data-testid="legal-basis-details">
          <div>
            <strong>Powołana podstawa prawna:</strong>
            {finding.basis ? (
              <div className="legal-basis">
                <LegalLocatorLink
                  locator={finding.basis.provision_locator}
                  label={
                    <>
                      {finding.basis.act_identifier} (
                      {finding.basis.provision_locator})
                    </>
                  }
                />
                <div className="legal-basis-meta">
                  Charakter normy:{" "}
                  <em>
                    {CHARACTER_KIND_LABELS_PL[finding.basis.character_kind]}
                  </em>
                  <div data-testid="legal-basis-force">
                    Stan ustawy (zakres: akt):{" "}
                    <em>
                      {ACT_FORCE_LABELS_PL[finding.basis.act_force.value]}
                    </em>{" "}
                    ({finding.basis.act_force.snapshot_date})
                    <br />
                    Stan przepisu (zakres: przepis):{" "}
                    <em>
                      {
                        PROVISION_FORCE_LABELS_PL[
                          finding.basis.provision_force.value
                        ]
                      }
                    </em>
                  </div>
                </div>
              </div>
            ) : (
              finding.legal_locators?.map((locator) => (
                <div key={locator} className="legal-basis">
                  <LegalLocatorLink locator={locator} />
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {finding.anchor_resolved === false && (
        <UnresolvedAnchorNote quoteResolution={finding.quote_resolution} />
      )}

      <WorksheetView
        unitId={finding.unit_id}
        scopeId={finding.id}
        state={worksheetState}
        onLoad={onLoadWorksheet}
      />
    </div>
  );
};
