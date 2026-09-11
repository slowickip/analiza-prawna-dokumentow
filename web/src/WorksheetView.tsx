import React, { useState } from "react";
import type { ApiError, WorksheetEntry, WorksheetResponse } from "./types";
import {
  AGENT_ROLE_LABELS_PL,
  CHARACTER_KIND_LABELS_PL,
  WORKSHEET_KIND_LABELS_PL,
  WORKSHEET_VALUE_LABELS_PL,
} from "./types";
import { LegalLocatorLink } from "./LegalLocatorLink";

export interface WorksheetState {
  data?: WorksheetResponse;
  error?: ApiError;
  loading?: boolean;
}

const FIELD_LABELS_PL: Record<string, string> = {
  locator: "Podstawa prawna",
  act_identifier: "Identyfikator aktu",
  snapshot_id: "Migawka korpusu",
  act_force: "Stan aktu",
  provision_force: "Stan przepisu",
  why: "Uzasadnienie",
  supporting_locators: "Dodatkowe podstawy",
  candidate_id: "Kandydat",
  note: "Notatka",
  character: "Charakter normy",
  kind: "Rodzaj",
  evidence: "Dowody",
  source_kind: "Rodzaj źródła",
  pinpoint: "Miejsce w przepisie",
  interpretive_methods: "Metody wykładni",
  rationale: "Uzasadnienie dowodu",
  semi_imperative_direction: "Kierunek semiimperatywny",
  protected_party_role: "Strona chroniona",
  relation: "Relacja",
  undetermined_reason: "Powód nierozstrzygnięcia",
  stage: "Etap",
  relevant: "Związek z jednostką",
  quote: "Cytat",
  departure: "Odchylenie",
  direction: "Kierunek",
  raw_confidence: "Pewność surowa",
  evidence_locators: "Dowody charakteru",
  based_on: "Na podstawie wpisów",
  phrase: "Wyszukiwana fraza",
  result_locators: "Wyniki wyszukiwania",
  purpose: "Cel",
  finding_id: "Ustalenie",
  value: "Wartość",
  scope: "Zakres",
  snapshot_date: "Data migawki",
  source_locator: "Źródło",
};

const LOCATOR_FIELDS = new Set([
  "locator",
  "source_locator",
  "locators",
  "supporting_locators",
  "evidence_locators",
  "result_locators",
]);

const CLOSED_VALUE_LABELS_PL: Record<string, Record<string, string>> = {
  act_force: WORKSHEET_VALUE_LABELS_PL["act_force.value"],
  provision_force: WORKSHEET_VALUE_LABELS_PL["provision_force.value"],
  kind: CHARACTER_KIND_LABELS_PL,
  ...WORKSHEET_VALUE_LABELS_PL,
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function fieldLabel(key: string): string {
  return FIELD_LABELS_PL[key] || key.replaceAll("_", " ");
}

/** The Polish reading of a wire value, or the value itself when there is none. */
function valueLabel(path: readonly string[], value: string): string {
  const full = path.join(".");
  const leaf = path[path.length - 1] ?? "";
  return (
    CLOSED_VALUE_LABELS_PL[full]?.[value] ??
    CLOSED_VALUE_LABELS_PL[leaf]?.[value] ??
    value
  );
}

function PlainValue({
  path = [],
  value,
}: {
  path?: readonly string[];
  value: unknown;
}) {
  if (value === null || value === undefined) return <>brak</>;
  if (typeof value === "boolean") return <>{value ? "tak" : "nie"}</>;
  if (typeof value === "string") return <>{valueLabel(path, value)}</>;
  if (typeof value === "number") return <>{String(value)}</>;
  return <>{JSON.stringify(value)}</>;
}

/** The fields whose value is the identifier of another entry on this worksheet. */
const ENTRY_REFERENCE_FIELDS = new Set(["based_on", "candidate_id"]);

/** Scoped DOM id ensuring unique anchors across multiple rendered cards. */
const anchorId = (scopeId: string, entryId: string): string =>
  `worksheet-entry-${scopeId}-${entryId}`;

function EntryReference({
  id,
  known,
  anchor,
}: {
  id: string;
  known: Set<string>;
  anchor: (entryId: string) => string;
}) {
  if (!known.has(id)) return <>{id}</>;
  return (
    <a className="worksheet-entry-ref" href={`#${anchor(id)}`}>
      {id}
    </a>
  );
}

function FieldValue({
  path,
  value,
  knownIds,
  anchor,
}: {
  path: readonly string[];
  value: unknown;
  knownIds: Set<string>;
  anchor: (entryId: string) => string;
}) {
  const field = path[path.length - 1] ?? "";

  if (ENTRY_REFERENCE_FIELDS.has(field)) {
    const references = Array.isArray(value) ? value : [value];
    return (
      <span className="worksheet-values">
        {references.map((reference, index) =>
          typeof reference === "string" ? (
            <EntryReference
              key={`${reference}-${index}`}
              id={reference}
              known={knownIds}
              anchor={anchor}
            />
          ) : (
            <PlainValue key={index} path={path} value={reference} />
          ),
        )}
      </span>
    );
  }

  if (LOCATOR_FIELDS.has(field)) {
    const locators = Array.isArray(value) ? value : [value];
    return (
      <span className="worksheet-values">
        {locators.map((locator, index) =>
          typeof locator === "string" ? (
            <LegalLocatorLink key={`${locator}-${index}`} locator={locator} />
          ) : (
            <PlainValue key={index} path={path} value={locator} />
          ),
        )}
      </span>
    );
  }

  if (Array.isArray(value)) {
    return (
      <span className="worksheet-values">
        {value.map((item, index) => (
          <span key={index}>
            {/* An item may itself be a record -- evidence is a list of them --
                so this recurses rather than stringifying the object. */}
            <FieldValue
              path={path}
              value={item}
              knownIds={knownIds}
              anchor={anchor}
            />
          </span>
        ))}
      </span>
    );
  }

  if (isRecord(value)) {
    return (
      <dl className="worksheet-fields worksheet-fields-nested">
        {Object.entries(value).map(([nestedField, nestedValue]) => (
          <React.Fragment key={nestedField}>
            <dt>{fieldLabel(nestedField)}</dt>
            <dd>
              <FieldValue
                path={[...path, nestedField]}
                value={nestedValue}
                knownIds={knownIds}
                anchor={anchor}
              />
            </dd>
          </React.Fragment>
        ))}
      </dl>
    );
  }

  return <PlainValue path={path} value={value} />;
}

const WorksheetEntryView: React.FC<{
  entry: WorksheetEntry;
  knownIds: Set<string>;
  anchor: (entryId: string) => string;
}> = ({ entry, knownIds, anchor }) => {
  const author = typeof entry.author === "string" ? entry.author : "system";
  const kind = typeof entry.kind === "string" ? entry.kind : "";
  const id = typeof entry.id === "string" ? entry.id : null;

  return (
    <li className="worksheet-entry" id={id ? anchor(id) : undefined}>
      <div className="worksheet-entry-header">
        {/* The entries reference each other by this identifier: a verdict says
            which entries it rests on, and a characterisation which candidate it
            describes. Hiding it left those references unresolvable. */}
        {id && <code className="worksheet-entry-id">{id}</code>}
        <strong>{AGENT_ROLE_LABELS_PL[author] || author}</strong>
        <span>{WORKSHEET_KIND_LABELS_PL[kind] || kind}</span>
      </div>
      <dl className="worksheet-fields">
        {Object.entries(entry)
          .filter(([field]) => !["id", "seq", "author", "kind"].includes(field))
          .map(([field, value]) => (
            <React.Fragment key={field}>
              <dt>{fieldLabel(field)}</dt>
              <dd>
                <FieldValue
                  path={[field]}
                  value={value}
                  knownIds={knownIds}
                  anchor={anchor}
                />
              </dd>
            </React.Fragment>
          ))}
      </dl>
    </li>
  );
};

export const WorksheetView: React.FC<{
  unitId: string;
  /** What this transcript belongs to, unique on the page: the finding's id. */
  scopeId: string;
  state: WorksheetState;
  onLoad: () => void;
}> = ({ unitId, scopeId, state, onLoad }) => {
  const [expanded, setExpanded] = useState(false);
  const entries = state.data?.units.find(
    (unit) => unit.unit_id === unitId,
  )?.entries;
  const anchor = (entryId: string) => anchorId(scopeId, entryId);
  const knownIds = new Set(
    (entries ?? [])
      .map((entry) => entry.id)
      .filter((id): id is string => typeof id === "string"),
  );

  const toggle = () => {
    const nextExpanded = !expanded;
    setExpanded(nextExpanded);
    // An error is a state worth retrying from: re-expanding asks again.
    if (nextExpanded && !state.data && !state.loading) {
      onLoad();
    }
  };

  return (
    <div
      className="worksheet-view"
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => event.stopPropagation()}
    >
      <button
        type="button"
        className="btn btn-sm"
        aria-expanded={expanded}
        onClick={toggle}
      >
        {expanded ? "Ukryj pracę agentów" : "Pokaż pracę agentów"}
      </button>
      {expanded && (
        <div className="worksheet-transcript">
          {state.loading ? (
            <p role="status">Pobieranie pracy agentów…</p>
          ) : state.error?.code === "content_expired" ? (
            <div className="expired-banner" role="alert">
              Sesja dokumentu wygasła (410). Praca agentów jest niedostępna.
            </div>
          ) : state.error ? (
            <div role="alert">
              {state.error.message_pl || "Nie udało się pobrać pracy agentów."}
            </div>
          ) : entries && entries.length > 0 ? (
            <ol data-testid={`worksheet-entries-${unitId}`}>
              {[...entries]
                .sort(
                  (left, right) =>
                    Number(left.seq ?? 0) - Number(right.seq ?? 0),
                )
                .map((entry, index) => (
                  <WorksheetEntryView
                    key={String(entry.id ?? `${unitId}-${index}`)}
                    entry={entry}
                    knownIds={knownIds}
                    anchor={anchor}
                  />
                ))}
            </ol>
          ) : (
            <p>Brak wpisów agentów dla tej jednostki.</p>
          )}
        </div>
      )}
    </div>
  );
};
