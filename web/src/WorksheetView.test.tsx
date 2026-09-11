import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { WorksheetView } from "./WorksheetView";
import type { WorksheetResponse } from "./types";

const UNIT_ID = "u-4";
const LOCATOR =
  "https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=11";

/** The transcript exactly as `GET /runs/{id}/worksheet` serialises it.
 *
 * Taken from `entry.model_dump(mode="json")` over a real worksheet, so the force
 * records are nested objects and the evidence is a list of them. A flattened
 * stand-in would let the renderer pass here and still print wire tokens in the
 * application.
 */
const AS_SERVED: WorksheetResponse = {
  run_id: "11111111-1111-4111-8111-111111111111",
  units: [
    {
      unit_id: UNIT_ID,
      entries: [
        {
          id: "e1",
          seq: 1,
          author: "analyst",
          kind: "candidate",
          locator: LOCATOR,
          act_identifier: "DU/2023/725",
          snapshot_id: "s1",
          act_force: {
            value: "in_force",
            scope: "act",
            snapshot_date: "2026-09-01",
            source_locator: LOCATOR,
          },
          provision_force: {
            value: "undetermined",
            scope: "provision",
            snapshot_date: "2026-09-01",
            source_locator: LOCATOR,
          },
          why: "Przepis reguluje wypowiedzenie najmu.",
          supporting_locators: [],
        },
        {
          id: "e2",
          seq: 2,
          author: "analyst",
          kind: "character",
          candidate_id: "e1",
          character: {
            kind: "semi_imperative",
            evidence: [
              {
                source_kind: "official_normative_text",
                locator: LOCATOR,
                pinpoint: "ust. 1",
                interpretive_methods: ["linguistic"],
                rationale: "Zamknięty katalog przyczyn wypowiedzenia.",
              },
            ],
            semi_imperative_direction: {
              relation: "more_favourable_to",
              protected_party_role: "najemca",
            },
            undetermined_reason: null,
          },
        },
        {
          id: "e3",
          seq: 3,
          author: "verifier",
          kind: "verdict",
          candidate_id: "e1",
          stage: "relation",
          departure: "present",
          direction: "against_permitted_direction",
          based_on: ["e1", "e2"],
        },
      ],
    },
  ],
};

const expand = async () => {
  await userEvent.click(
    screen.getByRole("button", { name: "Pokaż pracę agentów" }),
  );
};

const FINDING_ID = "f-4";

const renderTranscript = async (data: WorksheetResponse = AS_SERVED) => {
  render(
    <WorksheetView
      unitId={UNIT_ID}
      scopeId={FINDING_ID}
      state={{ data }}
      onLoad={vi.fn()}
    />,
  );
  await expand();
  return screen.getByTestId(`worksheet-entries-${UNIT_ID}`);
};

describe("WorksheetView", () => {
  const anchorFor = (entryId: string, scopeId = FINDING_ID) =>
    `worksheet-entry-${scopeId}-${entryId}`;

  it("shows the identifier each entry is referenced by", async () => {
    const transcript = await renderTranscript();

    expect(transcript).toHaveTextContent("Na podstawie wpisów");
    for (const id of ["e1", "e2", "e3"]) {
      expect(document.getElementById(anchorFor(id))).not.toBeNull();
    }
  });

  it("keeps anchors distinct when one unit is shown more than once", async () => {
    // A unit that produced several findings renders a card per finding, and a
    // re-analysis child sits beside its parent; every transcript restarts its
    // identifiers at e1. Two nodes under one DOM id would send every reference
    // to the first of them, in whichever card that happened to be.
    const secondFinding = "f-7";
    render(
      <>
        <WorksheetView
          unitId={UNIT_ID}
          scopeId={FINDING_ID}
          state={{ data: AS_SERVED }}
          onLoad={vi.fn()}
        />
        <WorksheetView
          unitId={UNIT_ID}
          scopeId={secondFinding}
          state={{ data: AS_SERVED }}
          onLoad={vi.fn()}
        />
      </>,
    );
    for (const button of screen.getAllByRole("button", {
      name: "Pokaż pracę agentów",
    })) {
      await userEvent.click(button);
    }

    // Same run, same unit, same entry identifiers: only the view differs.
    const anchors = [...document.querySelectorAll('[id^="worksheet-entry-"]')];
    expect(anchors).toHaveLength(6);
    expect(new Set(anchors.map((node) => node.id)).size).toBe(6);
    const references = screen.getAllByRole("link", { name: "e2" });
    expect(references.map((link) => link.getAttribute("href"))).toEqual([
      `#${anchorFor("e2")}`,
      `#${anchorFor("e2", secondFinding)}`,
    ]);
  });

  it("links a reference to the entry it names, and only when it exists", async () => {
    const withDanglingReference: WorksheetResponse = {
      ...AS_SERVED,
      units: [
        {
          unit_id: UNIT_ID,
          entries: [
            ...AS_SERVED.units[0].entries,
            {
              id: "e4",
              seq: 4,
              author: "verifier",
              kind: "verdict",
              candidate_id: "e9",
              stage: "relevance",
              relevant: true,
            },
          ],
        },
      ],
    };
    await renderTranscript(withDanglingReference);

    const reference = screen.getByRole("link", { name: "e2" });
    expect(reference).toHaveAttribute("href", `#${anchorFor("e2")}`);
    // e9 is on no entry here, so it stays text rather than offering a link
    // that scrolls nowhere.
    expect(screen.queryByRole("link", { name: "e9" })).not.toBeInTheDocument();
    expect(screen.getByText("e9")).toBeInTheDocument();
  });

  it("reads the nested force records in Polish, each in its own scope", async () => {
    const transcript = await renderTranscript();

    // RF-05: an act record admits and a provision record can only veto, so the
    // same wire value has to read differently under each.
    expect(transcript).toHaveTextContent("obowiązuje w dacie obserwacji");
    expect(transcript).toHaveTextContent(
      "nieustalony — tekst jednolity tego nie rozstrzyga",
    );
    // Read the scope rows themselves: "akt" is a substring of the act
    // identifier label and "przepis" of a kind label, so a transcript-wide
    // match would pass with no scope map at all.
    const scopes = screen
      .getAllByText("Zakres")
      .map((term) => term.nextElementSibling?.textContent);
    expect(scopes).toEqual(["akt", "przepis"]);
    expect(transcript).not.toHaveTextContent("in_force");
  });

  it("reads a character and its evidence in Polish rather than as JSON", async () => {
    const transcript = await renderTranscript();

    expect(transcript).toHaveTextContent("semiimperatywny");
    expect(transcript).toHaveTextContent("urzędowy tekst normatywny");
    expect(transcript).toHaveTextContent("wykładnia językowa");
    expect(transcript).toHaveTextContent("korzystniejszy dla");
    expect(transcript).toHaveTextContent("najemca");
    // Evidence is a list of records: stringifying it would put raw JSON in
    // front of the reader.
    expect(transcript).not.toHaveTextContent("source_kind");
    expect(transcript).not.toHaveTextContent('{"');
    for (const token of [
      "semi_imperative",
      "official_normative_text",
      "linguistic",
      "more_favourable_to",
    ]) {
      expect(transcript).not.toHaveTextContent(token);
    }
  });

  it("reads a verdict in Polish rather than as wire tokens", async () => {
    const transcript = await renderTranscript();

    expect(transcript).toHaveTextContent("relacja i kierunek");
    expect(transcript).toHaveTextContent("odstępstwo występuje");
    expect(transcript).toHaveTextContent("przeciw dozwolonemu kierunkowi");
    for (const token of ["against_permitted_direction", "present"]) {
      expect(transcript).not.toHaveTextContent(token);
    }
  });

  it("asks again after a failed fetch when the reader reopens it", async () => {
    const onLoad = vi.fn();
    render(
      <WorksheetView
        unitId={UNIT_ID}
        scopeId={FINDING_ID}
        state={{ error: { code: "internal_error", message_pl: "Błąd." } }}
        onLoad={onLoad}
      />,
    );

    await expand();

    expect(screen.getByRole("alert")).toHaveTextContent("Błąd.");
    expect(onLoad).toHaveBeenCalledTimes(1);
  });
});
