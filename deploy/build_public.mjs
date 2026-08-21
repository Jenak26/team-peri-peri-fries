// Assemble the static bundle Vercel serves.
//
// The project page and the console are one file each; the examination engine runs
// elsewhere, because a serverless bundle cannot hold PyTorch. PERI_API is baked into
// the console at build time so the deployed page knows where its engine is.
//
// This is Node rather than Python because it runs inside Vercel's build image, where
// Node is the one runtime guaranteed to be present.

import { cp, mkdir, readFile, rm, writeFile, readdir, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PUBLIC = path.join(ROOT, "public");
const ENGINE = (process.env.PERI_API || "").replace(/\/+$/, "");

if (existsSync(PUBLIC)) await rm(PUBLIC, { recursive: true, force: true });
await mkdir(PUBLIC, { recursive: true });

// The project page and the real examination record it renders.
await cp(path.join(ROOT, "site", "index.html"), path.join(PUBLIC, "index.html"));
await cp(path.join(ROOT, "site", "demo"), path.join(PUBLIC, "demo"), { recursive: true });

// The live console, pointed at the engine.
let console_html = await readFile(path.join(ROOT, "web", "index.html"), "utf8");
if (ENGINE) {
  console_html = console_html.replace(
    "<script>",
    `<script>window.PERI_API = ${JSON.stringify(ENGINE)};</script>\n  <script>`,
  );
}
await writeFile(path.join(PUBLIC, "console.html"), console_html, "utf8");

console.log(`public/ built; engine = ${ENGINE || "not set (console calls same origin)"}`);
for (const entry of await readdir(PUBLIC, { recursive: true })) {
  const full = path.join(PUBLIC, entry);
  const info = await stat(full);
  if (info.isFile()) console.log(`  ${entry}  ${info.size} bytes`);
}
