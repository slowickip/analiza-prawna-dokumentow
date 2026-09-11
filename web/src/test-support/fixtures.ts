import type {
  DocumentDescriptor,
  DocumentContent,
  Run,
  ChatResponse,
  PublicConfig,
  WorksheetResponse,
} from "../types";

export const MOCK_DOC_ID_STANDARD = "11111111-1111-4111-8111-111111111111";
export const MOCK_DOC_ID_INTERRUPTED = "22222222-2222-4222-8222-222222222222";
export const MOCK_DOC_ID_EXPIRED = "33333333-3333-4333-8333-333333333333";

export const MOCK_RUN_ID_STANDARD_MID = "44444444-4444-4444-8444-444444444444";
const MOCK_RUN_ID_INTERRUPTED = "77777777-7777-4777-8777-777777777777";

export const MOCK_DOC_DESCRIPTOR_STANDARD: DocumentDescriptor = {
  id: MOCK_DOC_ID_STANDARD,
  content_hash:
    "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  read_mode: "native_pdf",
  conversion: null,
  unit_count: 9,
  expires_at: new Date(Date.now() + 3600 * 1000).toISOString(),
};

function unitsFromBlocks(
  blocks: DocumentContent["blocks"],
  idPrefix: string,
): DocumentContent["units"] {
  return blocks.map((block, index) => ({
    id: `${idPrefix}${index + 1}`,
    text: block.text,
    anchor: block.anchor,
  }));
}

const STANDARD_BLOCKS: DocumentContent["blocks"] = [
  {
    id: "block-0",
    text: "UMOWA NAJMU LOKALU UŻYTKOWEGO\nzawarta w Warszawie w dniu 15 stycznia 2026 r. pomiędzy Fikcyjną Spółką Alfa Logistyka Sp. z o.o. a Beta Nieruchomości S.A.",
    anchor: {
      start_offset: 0,
      end_offset: 142,
      page: 1,
      bbox: [50, 40, 545, 90],
      line_start: 1,
      line_end: 2,
    },
  },
  {
    id: "block-1",
    text: "§ 1. Przedmiot umowy. Wynajmujący oddaje Najemcy do używania lokal użytkowy nr 4 o powierzchni 120 m2 położony w Warszawie przy ul. Przykładowej 10, z przeznaczeniem na prowadzenie działalności biurowej.",
    anchor: {
      start_offset: 144,
      end_offset: 350,
      page: 1,
      bbox: [50, 100, 545, 160],
      line_start: 3,
      line_end: 5,
    },
  },
  {
    id: "block-2",
    text: "§ 2. Czynsz i opłaty eksploatacyjne. Najemca zobowiązuje się płacić miesięczny czynsz najmu w wysokości 5000 PLN netto, płatny z góry do 10. dnia każdego miesiąca kalendarzowego na rachunek bankowy Wynajmującego.",
    anchor: {
      start_offset: 352,
      end_offset: 569,
      page: 1,
      bbox: [50, 170, 545, 230],
      line_start: 6,
      line_end: 8,
    },
  },
  {
    id: "block-3",
    text: "§ 3. Kaucja gwarancyjna. Najemca wpłaci kaucję w wysokości trzykrotności miesięcznego czynszu. Strony zgodnie ustalają, że kaucja podlega zwrotowi w terminie 14 dni od dnia protokolarnego zwrotu lokalu po potrąceniu bezspornych roszczeń.",
    anchor: {
      start_offset: 571,
      end_offset: 809,
      page: 1,
      bbox: [50, 240, 545, 305],
      line_start: 9,
      line_end: 12,
    },
  },
  {
    id: "block-4",
    text: "§ 4. Wypowiedzenie i natychmiastowe opróżnienie. Wynajmujący może wypowiedzieć umowę ze skutkiem natychmiastowym bez uprzedniego wezwania i bez wyznaczenia terminu dodatkowego w przypadku 3-dniowego opóźnienia w zapłacie jakiejkolwiek opłaty, z prawem do natychmiastowej wymiany zamków i zatrzymania rzeczy Najemcy.",
    anchor: {
      start_offset: 811,
      end_offset: 1133,
      page: 1,
      bbox: [50, 315, 545, 395],
      line_start: 13,
      line_end: 17,
    },
  },
  {
    id: "block-5",
    text: "§ 5. Odpowiedzialność za wady rzeczy najętej. Wynajmujący wyłącza całkowicie swoją odpowiedzialność za wady fizyczne i prawne lokalu, w tym wady uniemożliwiające umówiony użytek lokalu, z zastrzeżeniem § 3 ust. 2.",
    anchor: {
      start_offset: 1135,
      end_offset: 1353,
      page: 2,
      bbox: [50, 50, 545, 120],
      line_start: 18,
      line_end: 21,
    },
  },
  {
    id: "block-6",
    text: "§ 6. Przepisy uchylone i dawne regulacje. Do rozliczeń nakładów zastosowanie mają zasady określone w nieobowiązującym rozporządzeniu Ministra Gospodarki Przestrzennej z 1990 r. w sprawie lokali użytkowych.",
    anchor: {
      start_offset: 1355,
      end_offset: 1563,
      page: 2,
      bbox: [50, 130, 545, 195],
      line_start: 22,
      line_end: 25,
    },
  },
  {
    id: "block-7",
    text: "§ 7. Postanowienia końcowe. W sprawach nieuregulowanych niniejszą umową zastosowanie mają ogólne zasady prawa polskiego. Wszelkie spory będą rozstrzygane polubownie w terminie 30 dni.",
    anchor: {
      start_offset: 1565,
      end_offset: 1751,
      page: 2,
      bbox: [50, 205, 545, 270],
      line_start: 26,
      line_end: 29,
    },
  },
  {
    id: "block-8",
    text: "§ 8. Notatka kancelaryjna i dane rejestrowe. Sąd Rejonowy dla m.st. Warszawy, XII Wydział Gospodarczy KRS 0000000000, NIP 000-00-00-000, REGON 000000000.",
    anchor: {
      start_offset: 1753,
      end_offset: 1910,
      page: 2,
      bbox: [50, 280, 545, 330],
      line_start: 30,
      line_end: 32,
    },
  },
  {
    id: "block-9",
    text: "§ 9. Klauzula załącznika nr 3 (odesłanie zewnętrzne bez lokalizacji). Strony powołują się na regulamin obiektu handlowego określony w odrębnym dokumencie.",
    anchor: {
      start_offset: 1912,
      end_offset: 2068,
      page: 2,
      bbox: [50, 340, 545, 400],
      line_start: 33,
      line_end: 35,
    },
  },
];

export const MOCK_DOC_CONTENT_STANDARD: DocumentContent = {
  document_id: MOCK_DOC_ID_STANDARD,
  read_mode: "native_pdf",
  blocks: STANDARD_BLOCKS,
  units: unitsFromBlocks(STANDARD_BLOCKS.slice(1), "u-"),
  references: [
    {
      citing_unit_id: "u-5",
      reference_type: "full_internal",
      status: "resolved",
      raw_text: "§ 3 ust. 2",
      target_unit_id: "u-3",
    },
    {
      citing_unit_id: "u-9",
      reference_type: "annex",
      status: "outside_input",
      raw_text: "załącznik nr 3",
      target_unit_id: null,
    },
  ],
};

export const MOCK_RUN_STANDARD: Run = {
  id: MOCK_RUN_ID_STANDARD_MID,
  document_id: MOCK_DOC_ID_STANDARD,
  arm: "mid",
  status: "completed",
  measurement_valid: true,
  interrupted: false,
  content_available: true,
  requested_model: "deepseek-v4-flash",
  returned_model: "deepseek-v4-flash-2026-08",
  input_hash:
    "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  prompt_bundle_version: "v1.3.0-20260828",
  corpus_snapshot_id: "snapshot-2026-08-28-eli-pl",
  config_version: "default-1.0",
  tool_bundle_version: "tools-1.0",
  retry_policy: "bounded_exponential_max3",
  concurrency: 4,
  wall_budget_seconds: 120,
  created_at: new Date(Date.now() - 60000).toISOString(),
  finished_at: new Date(Date.now() - 45000).toISOString(),
  error: null,
  findings: [
    {
      id: "f-1",
      unit_id: "u-1",
      code: "consistent",
      prominence: "neutral",
      anchor_resolved: true,
      block_id: "block-1",
      anchor: {
        start_offset: 144,
        end_offset: 350,
        page: 1,
        bbox: [50, 100, 545, 160],
        line_start: 3,
        line_end: 5,
      },
      basis: {
        provision_locator:
          "https://api.sejm.gov.pl/eli/acts/DU/2023/725/art/659/ust/1",
        act_identifier: "DU/2023/725",
        act_force: {
          value: "in_force",
          scope: "act",
          snapshot_date: "2026-08-28",
          source_locator: "https://api.sejm.gov.pl/eli/acts/DU/2001/733",
        },
        provision_force: {
          value: "undetermined",
          scope: "provision",
          snapshot_date: "2026-08-28",
          source_locator: "https://api.sejm.gov.pl/eli/acts/DU/2001/733",
        },
        character_kind: "imperative",
      },
      legal_locators: [
        "https://api.sejm.gov.pl/eli/acts/DU/2023/725/art/659/ust/1",
      ],
    },
    {
      id: "f-2",
      unit_id: "u-2",
      code: "no_relation",
      prominence: "neutral",
      anchor_resolved: true,
      block_id: "block-2",
      anchor: {
        start_offset: 352,
        end_offset: 569,
        page: 1,
        bbox: [50, 170, 545, 230],
        line_start: 6,
        line_end: 8,
      },
    },
    {
      id: "f-3",
      unit_id: "u-3",
      code: "permissible_departure",
      prominence: "warning",
      anchor_resolved: true,
      block_id: "block-3",
      anchor: {
        start_offset: 571,
        end_offset: 809,
        page: 1,
        bbox: [50, 240, 545, 305],
        line_start: 9,
        line_end: 12,
      },
      basis: {
        provision_locator:
          "https://api.sejm.gov.pl/eli/acts/DU/2023/725/art/677",
        act_identifier: "DU/2023/725",
        act_force: {
          value: "in_force",
          scope: "act",
          snapshot_date: "2026-08-28",
          source_locator: "https://api.sejm.gov.pl/eli/acts/DU/2001/733",
        },
        provision_force: {
          value: "undetermined",
          scope: "provision",
          snapshot_date: "2026-08-28",
          source_locator: "https://api.sejm.gov.pl/eli/acts/DU/2001/733",
        },
        character_kind: "dispositive",
      },
      legal_locators: ["https://api.sejm.gov.pl/eli/acts/DU/2023/725/art/677"],
    },
    {
      id: "f-4",
      unit_id: "u-4",
      code: "contradictory",
      prominence: "critical",
      anchor_resolved: true,
      block_id: "block-4",
      anchor: {
        start_offset: 811,
        end_offset: 1133,
        page: 1,
        bbox: [50, 315, 545, 395],
        line_start: 13,
        line_end: 17,
      },
      basis: {
        provision_locator:
          "https://api.sejm.gov.pl/eli/acts/DU/2023/725/art/687",
        act_identifier: "DU/2023/725",
        act_force: {
          value: "in_force",
          scope: "act",
          snapshot_date: "2026-08-28",
          source_locator: "https://api.sejm.gov.pl/eli/acts/DU/2001/733",
        },
        provision_force: {
          value: "undetermined",
          scope: "provision",
          snapshot_date: "2026-08-28",
          source_locator: "https://api.sejm.gov.pl/eli/acts/DU/2001/733",
        },
        character_kind: "imperative",
      },
      legal_locators: ["https://api.sejm.gov.pl/eli/acts/DU/2023/725/art/687"],
    },
    {
      id: "f-5",
      unit_id: "u-5",
      code: "uncertain",
      prominence: "warning",
      uncertain_cause: "provision_character_undetermined",
      raw_confidence: 0.62,
      anchor_resolved: true,
      block_id: "block-5",
      anchor: {
        start_offset: 1135,
        end_offset: 1353,
        page: 2,
        bbox: [50, 50, 545, 120],
        line_start: 18,
        line_end: 21,
      },
      legal_locators: [
        "https://api.sejm.gov.pl/eli/acts/DU/2023/725/art/664/ust/3",
      ],
    },
    {
      id: "f-6",
      unit_id: "u-6",
      code: "basis_not_in_force",
      prominence: "critical",
      anchor_resolved: true,
      block_id: "block-6",
      anchor: {
        start_offset: 1355,
        end_offset: 1563,
        page: 2,
        bbox: [50, 130, 545, 195],
        line_start: 22,
        line_end: 25,
      },
      legal_locators: ["https://api.sejm.gov.pl/eli/acts/DU/1990/45/art/1"],
    },
    {
      id: "f-7",
      unit_id: "u-7",
      code: "no_basis_found",
      prominence: "neutral",
      anchor_resolved: true,
      block_id: "block-7",
      anchor: {
        start_offset: 1565,
        end_offset: 1751,
        page: 2,
        bbox: [50, 205, 545, 270],
        line_start: 26,
        line_end: 29,
      },
    },
    {
      id: "f-8",
      unit_id: "u-8",
      code: "unit_not_adjudicable",
      prominence: "warning",
      anchor_resolved: true,
      block_id: "block-8",
      anchor: {
        start_offset: 1753,
        end_offset: 1910,
        page: 2,
        bbox: [50, 280, 545, 330],
        line_start: 30,
        line_end: 32,
      },
    },
    {
      id: "f-9",
      unit_id: "u-9",
      code: "uncertain",
      prominence: "warning",
      uncertain_cause: "relation_below_threshold",
      raw_confidence: 0.41,
      anchor_resolved: false,
    },
  ],
  metrics: {
    input_tokens: 14250,
    output_tokens: 2840,
    elapsed_ms: 14820,
    units_total: 9,
    units_with_finding: 9,
    units_not_processed: 0,
    context_edge_count: 2,
    finder_tool_turns: 12,
    finder_search_calls: 7,
    finder_budget_exhausted_units: 0,
    verifier_tool_turns: 9,
    provision_reads: 4,
    defaulted_characterisations: 1,
    cost: {
      monetary_cost_microunits: 8500,
      price_table_date: "2026-08-28",
      price_table_hash: "sha256:abcd1234efgh5678",
      unknown_reason: null,
    },
    attempts: [
      {
        id: "att-1",
        requested_model: "deepseek-v4-flash",
        returned_model: "deepseek-v4-flash-2026-08",
        prompt_version: "basis-finder-v1.3",
        temperature: 0.0,
        parameters: {},
        input_tokens: 7200,
        output_tokens: 1400,
        latency_ms: 6200,
        status: "success",
        retry_number: 0,
        prompt_hash: "sha256:p1",
        response_hash: "sha256:r1",
      },
    ],
  },
};

export const MOCK_DOC_DESCRIPTOR_INTERRUPTED: DocumentDescriptor = {
  id: MOCK_DOC_ID_INTERRUPTED,
  content_hash:
    "sha256:7777777777777777777777777777777777777777777777777777777777777777",
  read_mode: "ocr_pdf",
  conversion: null,
  unit_count: 10,
  expires_at: new Date(Date.now() + 3600 * 1000).toISOString(),
};

const INTERRUPTED_BLOCKS: DocumentContent["blocks"] = [
  {
    id: "block-19",
    text: "§ 1. Zakres usług serwisowych.",
    anchor: {
      start_offset: 0,
      end_offset: 40,
      page: 1,
      bbox: [50, 50, 500, 90],
    },
  },
  {
    id: "block-20",
    text: "§ 2. Czas reakcji na zgłoszenie.",
    anchor: {
      start_offset: 42,
      end_offset: 90,
      page: 1,
      bbox: [50, 100, 500, 140],
    },
  },
  {
    id: "block-21",
    text: "§ 3. Kary umowne za opóźnienie.",
    anchor: {
      start_offset: 92,
      end_offset: 145,
      page: 1,
      bbox: [50, 150, 500, 190],
    },
  },
  {
    id: "block-22",
    text: "§ 4. Odpowiedzialność odszkodowawcza.",
    anchor: {
      start_offset: 147,
      end_offset: 200,
      page: 1,
      bbox: [50, 200, 500, 240],
    },
  },
  {
    id: "block-23",
    text: "§ 5. Poufność informacji.",
    anchor: {
      start_offset: 202,
      end_offset: 245,
      page: 1,
      bbox: [50, 250, 500, 290],
    },
  },
  {
    id: "block-24",
    text: "§ 6. Prawa autorskie.",
    anchor: {
      start_offset: 247,
      end_offset: 285,
      page: 1,
      bbox: [50, 300, 500, 340],
    },
  },
  {
    id: "block-25",
    text: "§ 7. Podwykonawcy.",
    anchor: {
      start_offset: 287,
      end_offset: 320,
      page: 1,
      bbox: [50, 350, 500, 390],
    },
  },
  {
    id: "block-26",
    text: "§ 8. Okres obowiązywania.",
    anchor: {
      start_offset: 322,
      end_offset: 360,
      page: 1,
      bbox: [50, 400, 500, 440],
    },
  },
  {
    id: "block-27",
    text: "§ 9. Rozwiązanie umowy.",
    anchor: {
      start_offset: 362,
      end_offset: 400,
      page: 1,
      bbox: [50, 450, 500, 490],
    },
  },
  {
    id: "block-28",
    text: "§ 10. Właściwość sądu.",
    anchor: {
      start_offset: 402,
      end_offset: 440,
      page: 1,
      bbox: [50, 500, 500, 540],
    },
  },
];

export const MOCK_DOC_CONTENT_INTERRUPTED: DocumentContent = {
  document_id: MOCK_DOC_ID_INTERRUPTED,
  read_mode: "ocr_pdf",
  blocks: INTERRUPTED_BLOCKS,
  units: unitsFromBlocks(INTERRUPTED_BLOCKS, "ui-"),
  references: [],
};

export const MOCK_RUN_INTERRUPTED: Run = {
  id: MOCK_RUN_ID_INTERRUPTED,
  document_id: MOCK_DOC_ID_INTERRUPTED,
  arm: "mid",
  status: "completed",
  measurement_valid: true,
  interrupted: true,
  interruption_reason: "wall_time",
  content_available: true,
  wall_budget_seconds: 15,
  created_at: new Date(Date.now() - 30000).toISOString(),
  finished_at: new Date(Date.now() - 15000).toISOString(),
  error: null,
  findings: [
    {
      id: "fi-1",
      unit_id: "ui-1",
      code: "consistent",
      prominence: "neutral",
      anchor_resolved: true,
      block_id: "block-19",
      anchor: {
        start_offset: 0,
        end_offset: 40,
        page: 1,
        bbox: [50, 50, 500, 90],
      },
    },
    {
      id: "fi-2",
      unit_id: "ui-2",
      code: "contradictory",
      prominence: "critical",
      anchor_resolved: true,
      block_id: "block-19",
      anchor: {
        start_offset: 42,
        end_offset: 90,
        page: 1,
        bbox: [50, 100, 500, 140],
      },
    },
    {
      id: "fi-3",
      unit_id: "ui-3",
      code: "permissible_departure",
      prominence: "warning",
      anchor_resolved: true,
      block_id: "block-19",
      anchor: {
        start_offset: 92,
        end_offset: 145,
        page: 1,
        bbox: [50, 150, 500, 190],
      },
    },
    {
      id: "fi-4",
      unit_id: "ui-4",
      code: "not_processed",
      prominence: "warning",
      anchor_resolved: true,
      block_id: "block-20",
      anchor: {
        start_offset: 147,
        end_offset: 200,
        page: 1,
        bbox: [50, 200, 500, 240],
      },
    },
    {
      id: "fi-5",
      unit_id: "ui-5",
      code: "not_processed",
      prominence: "warning",
      anchor_resolved: true,
      block_id: "block-20",
      anchor: {
        start_offset: 202,
        end_offset: 245,
        page: 1,
        bbox: [50, 250, 500, 290],
      },
    },
    {
      id: "fi-6",
      unit_id: "ui-6",
      code: "not_processed",
      prominence: "warning",
      anchor_resolved: true,
      block_id: "block-20",
      anchor: {
        start_offset: 247,
        end_offset: 285,
        page: 1,
        bbox: [50, 300, 500, 340],
      },
    },
    {
      id: "fi-7",
      unit_id: "ui-7",
      code: "not_processed",
      prominence: "warning",
      anchor_resolved: true,
      block_id: "block-20",
      anchor: {
        start_offset: 287,
        end_offset: 320,
        page: 1,
        bbox: [50, 350, 500, 390],
      },
    },
    {
      id: "fi-8",
      unit_id: "ui-8",
      code: "not_processed",
      prominence: "warning",
      anchor_resolved: true,
      block_id: "block-20",
      anchor: {
        start_offset: 322,
        end_offset: 360,
        page: 1,
        bbox: [50, 400, 500, 440],
      },
    },
    {
      id: "fi-9",
      unit_id: "ui-9",
      code: "not_processed",
      prominence: "warning",
      anchor_resolved: true,
      block_id: "block-21",
      anchor: {
        start_offset: 362,
        end_offset: 400,
        page: 1,
        bbox: [50, 450, 500, 490],
      },
    },
    {
      id: "fi-10",
      unit_id: "ui-10",
      code: "not_processed",
      prominence: "warning",
      anchor_resolved: true,
      block_id: "block-21",
      anchor: {
        start_offset: 402,
        end_offset: 440,
        page: 1,
        bbox: [50, 500, 500, 540],
      },
    },
  ],
  metrics: {
    input_tokens: 5120,
    output_tokens: 920,
    elapsed_ms: 15200,
    units_total: 10,
    units_with_finding: 3,
    units_not_processed: 7,
    context_edge_count: 0,
    finder_tool_turns: 5,
    finder_search_calls: 3,
    finder_budget_exhausted_units: 7,
    verifier_tool_turns: 2,
    provision_reads: 1,
    cost: {
      monetary_cost_microunits: 3100,
      price_table_date: "2026-08-28",
      price_table_hash: "sha256:abcd1234efgh5678",
      unknown_reason: null,
    },
    attempts: [],
  },
};

export const MOCK_CHAT_RESPONSE: ChatResponse = {
  answer:
    "System zakwalifikował jednostkę § 4 jako sprzeczną z art. 687 Kodeksu cywilnego, ponieważ zgodnie z tym przepisem wynajmujący może wypowiedzieć najem bez zachowania terminów wypowiedzenia dopiero po uprzedzeniu najemcy na piśmie i udzieleniu mu dodatkowego terminu miesięcznego do zapłaty zaległego czynszu. Postanowienie umowne wyłączające ten wymóg jest sprzeczne z bezwzględnie obowiązującą normą ustawową.",
  cited_finding_ids: ["f-4"],
  interaction_run_id: "99999999-9999-4999-8999-999999999999",
  corpus_consulted: true,
};

export const MOCK_WORKSHEET_STANDARD: WorksheetResponse = {
  run_id: MOCK_RUN_ID_STANDARD_MID,
  units: [
    {
      unit_id: "u-4",
      entries: [
        {
          id: "entry-1",
          seq: 1,
          author: "analyst",
          kind: "candidate",
          locator: "https://api.sejm.gov.pl/eli/acts/DU/2023/725/art/687",
          why: "Przepis dotyczy wypowiedzenia najmu po zaległości.",
        },
        {
          id: "entry-2",
          seq: 2,
          author: "verifier",
          kind: "verdict",
          quote: "bez uprzedniego wezwania",
          relevant: true,
        },
      ],
    },
  ],
};

export const MOCK_CONFIG: PublicConfig = {
  model_request_id: "deepseek-v4-flash",
  model_endpoint: "openrouter.ai",
  prompt_bundle_version: "v1.3.0-20260828",
  corpus_snapshot: "snapshot-2026-08-28-eli-pl",
  embedding_model: "qwen/qwen3-embedding-8b",
  embedding_space_fingerprint: "9f2c41ab77e05d13",
  corpus_source_format: "as_declared",
  tool_bundle_version: "worksheet-roles-8c6fa642eefb",
  arms: ["off", "mid", "on"],
  measured_mode: true,
  limits: {
    max_input_bytes: 26214400,
    max_pdf_pages: 200,
    content_ttl_seconds: 3600,
  },
};
