import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { LegalLocatorLink } from "./LegalLocatorLink";

const CORPUS_LOCATOR =
  "https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=11";

describe("LegalLocatorLink", () => {
  it("links a corpus locator and opens it without handing over the opener", () => {
    render(<LegalLocatorLink locator={CORPUS_LOCATOR} />);

    const link = screen.getByRole("link", { name: /arti=11/ });
    expect(link).toHaveAttribute("href", CORPUS_LOCATOR);
    expect(link).toHaveAttribute("rel", expect.stringContaining("noopener"));
  });

  it.each([
    ["javascript:alert(1)"],
    ["data:text/html,<script>alert(1)</script>"],
    ["art. 11 ustawy o ochronie praw lokatorów"],
  ])("renders %s as text rather than a link", (locator) => {
    // Worksheet entries are agent-authored strings, so a locator is data until
    // something says otherwise. Only https earns an href.
    render(<LegalLocatorLink locator={locator} />);

    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByText(locator)).toBeInTheDocument();
  });
});
