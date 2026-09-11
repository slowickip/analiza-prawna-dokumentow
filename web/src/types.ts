import type { components, paths } from "./schema";

export type ApiError = components["schemas"]["Error"];
export type ArmCode = components["schemas"]["ArmCode"];
export type FindingCode = components["schemas"]["FindingCode"];
export type UncertainCause = components["schemas"]["UncertainCause"];
export type Prominence = components["schemas"]["Prominence"];
export type RunStatus = components["schemas"]["RunStatus"];
export type CharacterKind =
  components["schemas"]["LegalBasisReference"]["character_kind"];
export type ForceState = components["schemas"]["ForceState"];
export type ForceValue = ForceState["value"];
export type DocumentDescriptor = components["schemas"]["DocumentDescriptor"];
export type DocumentContent = components["schemas"]["DocumentContent"];
export type Finding = components["schemas"]["Finding"];
export type QuoteResolution = components["schemas"]["QuoteResolution"];
export type Run = components["schemas"]["Run"];
export type RunEvent = components["schemas"]["RunEvent"];
export type ChatResponse = components["schemas"]["ChatResponse"];
export type WorksheetResponse = components["schemas"]["WorksheetResponse"];
export type WorksheetEntry =
  WorksheetResponse["units"][number]["entries"][number];
export type PublicConfig = components["schemas"]["PublicConfig"];

export type CreateRunRequest =
  paths["/runs"]["post"]["requestBody"]["content"]["application/json"];
export type MessageRequest =
  paths["/runs/{runId}/message"]["post"]["requestBody"]["content"]["application/json"];
export type MessageResponse =
  paths["/runs/{runId}/message"]["post"]["responses"]["200"]["content"]["application/json"];

export const INTERACTION_ERROR_MESSAGES_PL: Partial<
  Record<ApiError["code"], string>
> = {
  run_not_finished:
    "Interakcję można uruchomić dopiero po zakończeniu analizy.",
  run_already_active:
    "Inna analiza jest już aktywna. Poczekaj na jej zakończenie.",
  content_expired:
    "Sesja dokumentu wygasła. Praca agentów i interakcje są niedostępne.",
  message_not_routable:
    "Nie udało się odczytać, czego dotyczy wiadomość. Spróbuj napisać ją inaczej.",
  out_of_scope:
    "System odmawia odpowiedzi: zapytanie wykracza poza analizowany dokument.",
};

export const AGENT_ROLE_LABELS_PL: Record<string, string> = {
  researcher: "Wyszukujący",
  analyst: "Analityk",
  verifier: "Weryfikator",
  synthesizer: "Syntetyzator",
  explainer: "Objaśniający",
  user: "Użytkownik",
  system: "System",
};

export const WORKSHEET_KIND_LABELS_PL: Record<string, string> = {
  search: "wyszukiwanie",
  candidate: "kandydat na przepis",
  read: "odczyt przepisu",
  character: "charakter normy",
  verdict: "ocena relacji",
  user_note: "notatka użytkownika",
};

// Worksheet entries carry the wire values of the domain enums, so a reader
// otherwise sees `semi_imperative` or `against_permitted_direction` in the
// middle of Polish prose. Anything absent here is shown as it arrived, because
// inventing a translation for an unknown token would be worse than the token.
export const WORKSHEET_VALUE_LABELS_PL: Record<
  string,
  Record<string, string>
> = {
  // Keys are matched as a full field path first and then as the leaf field, so
  // `value` inside an act record reads differently from `value` inside a
  // provision record, which RF-05 requires them to.
  "act_force.value": {
    in_force: "obowiązuje w dacie obserwacji",
    not_in_force: "nie obowiązuje w dacie obserwacji",
    undetermined: "nieustalony",
  },
  "provision_force.value": {
    in_force: "stan niedopuszczalny dla rekordu przepisu",
    not_in_force: "oznaczony w tekście jako uchylony",
    undetermined: "nieustalony — tekst jednolity tego nie rozstrzyga",
  },
  scope: { act: "akt", provision: "przepis" },
  relation: { more_favourable_to: "korzystniejszy dla" },
  stage: { relevance: "trafność", relation: "relacja i kierunek" },
  departure: {
    none: "brak odstępstwa",
    present: "odstępstwo występuje",
    undetermined: "nierozstrzygnięte",
  },
  direction: {
    with_permitted_direction: "w dozwolonym kierunku",
    against_permitted_direction: "przeciw dozwolonemu kierunkowi",
    undetermined: "nierozstrzygnięty",
  },
  semi_imperative_direction: {
    with_permitted_direction: "w dozwolonym kierunku",
    against_permitted_direction: "przeciw dozwolonemu kierunkowi",
    undetermined: "nierozstrzygnięty",
  },
  undetermined_reason: {
    unusable_model_answer: "odpowiedź roli nie nadawała się do użycia",
  },
  purpose: {
    ask: "pytanie",
    contest: "sprzeciw",
    analyse: "ponowna analiza",
  },
  source_kind: {
    official_normative_text: "urzędowy tekst normatywny",
    judicial_decision: "orzeczenie",
    doctrinal_publication: "publikacja doktrynalna",
    legislative_material: "materiał legislacyjny",
  },
  interpretive_methods: {
    linguistic: "wykładnia językowa",
    systemic: "wykładnia systemowa",
    purposive: "wykładnia celowościowa",
  },
};

export const ARM_LABELS: Record<ArmCode, string> = {
  off: "OFF — cały dokument",
  mid: "MID — jednostki",
  on: "ON — jednostki + odwołania",
};

export const RUN_STATUS_LABELS_PL: Record<RunStatus, string> = {
  running: "W toku",
  completed: "Zakończono",
  failed: "Niepowodzenie",
  cancelled: "Anulowano",
};

export function isRunStatus(status: string): status is RunStatus {
  return Object.hasOwn(RUN_STATUS_LABELS_PL, status);
}

export const CHARACTER_KIND_LABELS_PL: Record<CharacterKind, string> = {
  imperative: "imperatywny",
  dispositive: "dyspozytywny",
  semi_imperative: "semiimperatywny",
  undetermined: "nierozstrzygnięty",
};

export const ACT_FORCE_LABELS_PL: Record<ForceValue, string> =
  WORKSHEET_VALUE_LABELS_PL["act_force.value"] as Record<ForceValue, string>;

export const PROVISION_FORCE_LABELS_PL: Record<ForceValue, string> =
  WORKSHEET_VALUE_LABELS_PL["provision_force.value"] as Record<
    ForceValue,
    string
  >;

export const FINDING_CODE_LABELS_PL: Record<FindingCode, string> = {
  consistent: "Zgodność z podstawą prawną",
  contradictory: "Sprzeczność z normą prawną",
  permissible_departure: "Dopuszczalna modyfikacja normy",
  no_relation: "Brak związku z odnalezionymi przepisami",
  no_basis_found: "Nie znaleziono podstawy prawnej w korpusie",
  basis_not_in_force: "Podstawa prawna nie obowiązuje",
  unit_not_adjudicable: "Jednostka nierozstrzygalna",
  not_processed: "Nie przetworzono (wyczerpany budżet)",
  uncertain: "Ocena niejednoznaczna (niepewność)",
};

export const UNCERTAIN_CAUSE_LABELS_PL: Record<UncertainCause, string> = {
  relation_below_threshold:
    "System ocenił pewność związku lub odchylenia poniżej progu rozstrzygalności.",
  force_state_undetermined:
    "System nie rozstrzygnął obowiązywania normy prawnej w korpusie.",
  provision_character_undetermined:
    "System nie rozstrzygnął charakteru normy prawnej (imperatywny/dyspozytywny/semiimperatywny).",
  permitted_direction_undetermined:
    "System nie rozstrzygnął dopuszczalnego kierunku modyfikacji normy semiimperatywnej.",
  departure_direction_undetermined:
    "System nie rozstrzygnął kierunku modyfikacji jednostki względem normy semiimperatywnej.",
};

export const QUOTE_RESOLUTION_LABELS_PL: Record<QuoteResolution, string> = {
  resolved: "Cytat powiązany z tekstem",
  quote_empty: "Cytat jest pusty",
  quote_fabricated: "Cytatu nie odnaleziono w tekście",
  quote_ambiguous: "Cytat występuje wielokrotnie w tekście",
};
