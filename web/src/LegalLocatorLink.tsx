import React from "react";
import { ExternalLink } from "lucide-react";

/** Whether this string is safe to put in an href.
 *
 * Locators reach the interface as agent-authored strings on worksheet entries,
 * so the value is data, not a URL the interface chose. Anything but https is
 * rendered as text: a `javascript:` or `data:` locator is then something the
 * reader can see rather than something one click runs.
 */
const isFollowable = (locator: string): boolean => {
  try {
    return new URL(locator).protocol === "https:";
  } catch {
    return false;
  }
};

export const LegalLocatorLink: React.FC<{
  locator: string;
  label?: React.ReactNode;
}> = ({ locator, label = locator }) =>
  isFollowable(locator) ? (
    <a
      href={locator}
      target="_blank"
      rel="noopener noreferrer"
      className="legal-locator-link"
      onClick={(event) => event.stopPropagation()}
    >
      {label}
      <ExternalLink size={12} />
    </a>
  ) : (
    <span className="legal-locator-link legal-locator-link--plain">
      {label}
    </span>
  );
