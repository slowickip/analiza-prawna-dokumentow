import "@testing-library/jest-dom";
import { beforeEach } from "vitest";

beforeEach(() => {
  if (typeof window !== "undefined" && window.sessionStorage) {
    window.sessionStorage.clear();
  }
});

Element.prototype.scrollIntoView = () => {};
