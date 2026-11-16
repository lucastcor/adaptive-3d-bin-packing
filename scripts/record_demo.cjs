#!/usr/bin/env node
// Optional authoring tool: install Playwright separately; see docs/media.md.
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

async function main() {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, '.private/media-tools');
  const media = path.join(root, 'docs/assets/demo');
  fs.mkdirSync(output, { recursive: true });
  fs.mkdirSync(media, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
    args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'],
  });
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1,
    recordVideo: { dir: output, size: { width: 1920, height: 1080 } },
  });
  const page = await context.newPage().catch(async error => {
    await browser.close();
    throw error;
  });
  const errors = [];
  const recordingStarted = Date.now();
  page.on('pageerror', error => errors.push(error.message));
  let payload, solveRequest;
  page.on('request', request => {
    if (request.url().endsWith('/api/solve-jobs') && request.method() === 'POST') {
      solveRequest = request.postDataJSON();
    }
  });
  page.on('response', async response => {
    if (/\/api\/solve-jobs\/[^/]+$/.test(response.url()) && response.request().method() === 'GET') {
      const body = await response.json().catch(() => null);
      if (body?.status === 'completed' && body.result) payload = body;
    }
  });
  try {
    await page.goto(process.env.DEMO_URL || 'http://127.0.0.1:5173', { waitUntil: 'networkidle' });
    await page.getByRole('button', { name: 'Executar', exact: true }).waitFor();
    // Use the real editor, controls, API and solver. No synthetic responses.
    await page.locator('textarea').fill(fs.readFileSync(path.join(root, 'data/raw/demo_60.json'), 'utf8'));
    await page.getByRole('spinbutton', { name: /^Avaliações / }).fill('96000');
    await page.getByRole('spinbutton', { name: /^Tempo / }).fill('30');
    await page.getByRole('spinbutton', { name: /^Multi-start / }).fill('8');
    await page.getByRole('checkbox').uncheck();
    await page.getByRole('button', { name: 'C++', exact: true }).click();
    await page.getByRole('button', { name: 'Pool', exact: true }).click();
    await page.getByRole('spinbutton', { name: /^Workers / }).fill('4');
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(1600);
    const started = Date.now();
    const videoInSeconds = Math.max(0, (started - recordingStarted) / 1000 - 0.8);
    await page.getByRole('button', { name: 'Executar', exact: true }).click();
    await page.getByText('Execucao concluida; analise e solucao atualizadas.', { exact: true }).waitFor({ timeout: 90000 });
    await page.waitForTimeout(1200);
    if (!payload) throw new Error('Completed solver response was not captured');
    if (payload.result.status !== 'feasible') throw new Error('Demo must have a feasible solution');
    if (payload.result.solver !== 'simulated_annealing_cpp_core_parallel') {
      throw new Error('Expected the actual parallel native solver, without fallback');
    }
    if (solveRequest.config.parallel_workers !== 4 || solveRequest.config.multi_start_count !== 8) {
      throw new Error('Expected 4 workers and 8 starts');
    }
    const solvedVideoSeconds = (Date.now() - recordingStarted) / 1000;
    fs.writeFileSync(path.join(media, 'demo-run.json'), JSON.stringify({
      captured_on: new Date().toISOString().slice(0, 10),
      instance: 'data/raw/demo_60.json',
      config: solveRequest.config,
      seed: solveRequest.seed, persist: solveRequest.persist, result: payload.result,
      note: 'Functional demonstration, not a comparative benchmark. Video includes deliberate viewing pauses.',
    }, null, 2) + '\n');
    await page.screenshot({ path: path.join(media, 'app-desktop.png'), fullPage: true });
    const canvas = page.locator('.viewer-canvas canvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('3D canvas is missing');
    async function orbit(dx, dy) {
      await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width * 0.5 + dx, box.y + box.height * 0.5 + dy, { steps: 80 });
      await page.mouse.up();
      await page.waitForTimeout(2000);
    }
    await orbit(160, 35);
    const bins = page.getByRole('tablist', { name: 'Caixas', exact: true }).getByRole('tab');
    if (await bins.count() > 1) await bins.nth(1).click();
    await page.waitForTimeout(1200);
    await orbit(-130, -20);
    await page.getByRole('tab', { name: 'SA', exact: true }).click();
    await page.waitForTimeout(2500);
    await page.getByRole('tab', { name: 'Curva', exact: true }).click();
    await bins.first().click();
    await page.waitForTimeout(1800);
    await orbit(-80, 5);
    await page.evaluate(() => window.scrollTo({ top: 240, behavior: 'smooth' }));
    await page.waitForTimeout(2200);
    fs.writeFileSync(path.join(output, 'capture-info.json'), JSON.stringify({
      elapsed_ms: Date.now() - started, page_errors: errors,
      video_in_seconds: videoInSeconds, solved_video_seconds: solvedVideoSeconds,
      result: { bins: payload.result.best_found_bins, lower_bound: payload.result.lower_bound,
        evaluations: payload.result.evaluations, runtime_seconds: payload.result.runtime_seconds,
        solver: payload.result.solver, version: payload.result.solver_version },
    }, null, 2));
    if (errors.length) throw new Error(`Browser errors: ${errors.join('; ')}`);
    await context.close();
    const raw = await page.video().path();
    fs.copyFileSync(raw, path.join(output, 'capture.webm'));
    console.log(JSON.stringify({ raw: path.join(output, 'capture.webm'), errors, bins: payload.result.best_found_bins }));
  } finally {
    await context.close().catch(() => {});
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
