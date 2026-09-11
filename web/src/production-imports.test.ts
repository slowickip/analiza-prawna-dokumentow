import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const sourceRoot = path.dirname(fileURLToPath(import.meta.url));

function walk(directory: string): string[] {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(entryPath) : [entryPath];
  });
}

function isProductionFile(filePath: string): boolean {
  const relativePath = path.relative(sourceRoot, filePath);
  return (
    !relativePath.startsWith(`test-support${path.sep}`) &&
    !/\.test\.tsx?$/.test(relativePath) &&
    relativePath !== "test-setup.ts"
  );
}

function importedTestSupportModules(source: string): string[] {
  const moduleSpecifier =
    /(?:import|export)\s+(?:[\s\S]*?\s+from\s+)?["']([^"']+)["']|import\s*\(\s*["']([^"']+)["']/g;

  return Array.from(source.matchAll(moduleSpecifier), (match) =>
    String(match[1] ?? match[2]),
  ).filter((specifier) => specifier.split("/").includes("test-support"));
}

describe("production import boundary", () => {
  it("keeps test-support unreachable from production source files", () => {
    const violations = walk(sourceRoot)
      .filter(isProductionFile)
      .flatMap((filePath) =>
        importedTestSupportModules(fs.readFileSync(filePath, "utf8")).map(
          (specifier) =>
            `${path.relative(sourceRoot, filePath)} -> ${specifier}`,
        ),
      );

    expect(violations, violations.join("\n")).toEqual([]);
  });

  it("keeps public CDN hosts out of frontend source", () => {
    const forbiddenHosts = [
      ["cdnjs", "cloudflare", "com"].join("."),
      ["unpkg", "com"].join("."),
    ];
    const violations = walk(sourceRoot).flatMap((filePath) => {
      const source = fs.readFileSync(filePath, "utf8");
      return forbiddenHosts
        .filter((host) => source.includes(host))
        .map((host) => `${path.relative(sourceRoot, filePath)} -> ${host}`);
    });

    expect(violations, violations.join("\n")).toEqual([]);
  });
});
