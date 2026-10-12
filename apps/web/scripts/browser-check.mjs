/**
 * Browser check — the one thing curl cannot tell you.
 *
 * The failure this catches: a page that renders perfectly and does nothing. Next refuses to serve
 * its dev resources to an unrecognised origin (`allowedDevOrigins`), and a client that never
 * hydrates leaves every button, chip and file picker inert with no error anywhere. `curl` sees a
 * complete 200 response; a browser sees a screenshot.
 *
 * Usage (from apps/web):
 *
 *   npm install --no-save puppeteer-core @sparticuz/chromium
 *   node scripts/browser-check.mjs                     # checks http://127.0.0.1:3000
 *   BASE=https://your-preview.example node scripts/browser-check.mjs
 *
 * Exit code is non-zero if any check fails, so it can gate a deploy.
 */

import chromium from "@sparticuz/chromium";
import puppeteer from "puppeteer-core";

const BASE = process.env.BASE ?? "http://127.0.0.1:3000";
const ROUTES = ["/", "/churn", "/fraud", "/maintenance", "/pricing", "/rag", "/scraper", "/diagnostics"];
const BODIES = ["churn", "fraud", "predictive", "dynamic", "knowledge", "web", "explainable"];
const FILE = process.env.FILE;

const failures = [];
const ok = (label, condition, detail = "") => {
  console.log(`${condition ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${detail}` : ""}`);
  if (!condition) failures.push(label);
};

const browser = await puppeteer.launch({
  args: ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
  executablePath: await chromium.executablePath(),
  headless: true,
  env: { ...process.env, LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH ?? "" },
});

for (const route of ROUTES) {
  const page = await browser.newPage();
  const problems = [];
  page.on("pageerror", (e) => problems.push(String(e).slice(0, 120)));
  page.on("response", (r) => {
    if (r.url().includes("/api/") && r.status() >= 400) {
      problems.push(`${r.status()} ${new URL(r.url()).pathname}`);
    }
  });
  await page.goto(BASE + route, { waitUntil: "networkidle2", timeout: 90_000 });
  await new Promise((r) => setTimeout(r, 2500));
  const hydrated = await page.evaluate(() =>
    [...document.querySelectorAll("*")].some((el) =>
      Object.keys(el).some((k) => k.startsWith("__reactFiber")),
    ),
  );
  // A hydrated page with N controls is interactive; a static one is the bug this script exists for.
  const controls = await page.$$eval("button, select, input", (els) => els.length);
  ok(`${route} hydrates`, hydrated, hydrated ? "" : "page is static HTML — controls will do nothing");
  ok(`${route} has no failing API call`, problems.length === 0, problems.slice(0, 3).join(" | "));
  await page.close();
}

// The upload studio: every body must be selectable, and a file must actually be analysed.
const page = await browser.newPage();
await page.goto(BASE + "/", { waitUntil: "networkidle2", timeout: 90_000 });
await new Promise((r) => setTimeout(r, 2500));
const accepts = () =>
  page.evaluate(() => {
    const p = [...document.querySelectorAll("#upload p")].find((e) => /accepts ·/i.test(e.textContent ?? ""));
    return p?.textContent?.replace(/\s+/g, " ") ?? "";
  });
const seen = new Set();
for (const prefix of BODIES) {
  await page.evaluate((p) => {
    const b = [...document.querySelectorAll("#upload button")].find((x) =>
      x.textContent.trim().toLowerCase().startsWith(p),
    );
    b?.click();
  }, prefix);
  await new Promise((r) => setTimeout(r, 250));
  seen.add((await accepts()).slice(0, 40));
}
ok("every body is selectable in the upload studio", seen.size >= BODIES.length, `${seen.size} distinct responses`);

if (FILE) {
  const input = await page.$("#upload input[type=file]");
  await input.uploadFile(FILE);
  const analysed = await page
    .waitForFunction(
      () => {
        const t = document.querySelector("#upload")?.innerText ?? "";
        return /USED BY THE MODEL|WHAT THIS BODY NEEDS/i.test(t);
      },
      { timeout: 60_000 },
    )
    .then(() => true)
    .catch(() => false);
  const verdict = await page.evaluate(() => {
    const t = document.querySelector("#upload")?.innerText.replace(/\s+/g, " ") ?? "";
    return /USED BY THE MODEL/i.test(t) ? "mapped" : /WHAT THIS BODY NEEDS/i.test(t) ? "refused with columns" : "nothing";
  });
  ok("a dropped file is analysed", analysed, verdict);
}

await browser.close();
console.log(failures.length ? `\n${failures.length} check(s) failed` : "\nall browser checks passed");
process.exit(failures.length ? 1 : 0);
