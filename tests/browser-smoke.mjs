/**
 * Serves docs and checks training, reset/stop races, probe controls,
 * CSV contents and small-screen layout in headless Chrome via CDP.
 * SMOKE_SHOTS=1 saves screenshots for separate visual inspection.
 */
import { spawn } from "node:child_process";
import {
  writeFileSync,
  readFileSync,
  mkdirSync,
  existsSync,
  rmSync,
  mkdtempSync,
} from "node:fs";

const HTTP_PORT = 8953;
const CDP_PORT = 9353;
const SHOTS = process.env.SMOKE_SHOTS === "1";
const root = new URL("..", import.meta.url).pathname;

const profile = mkdtempSync("/tmp/contour-chrome-");
const server = spawn(
  "python3",
  [
    "-m",
    "http.server",
    String(HTTP_PORT),
    "--bind",
    "127.0.0.1",
    "--directory",
    `${root}docs`,
  ],
  { stdio: "ignore", detached: true },
);
const chrome = spawn(
  "google-chrome",
  [
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    `--remote-debugging-port=${CDP_PORT}`,
    `--user-data-dir=${profile}`,
    "about:blank",
  ],
  { stdio: "ignore", detached: true },
);

const cleanup = async (code) => {
  for (const child of [chrome, server]) {
    try {
      process.kill(-child.pid, "SIGTERM");
    } catch {}
  }
  await Promise.all(
    [chrome, server].map((child) =>
      child.exitCode !== null
        ? Promise.resolve()
        : new Promise((r) => child.once("exit", r)),
    ),
  );
  rmSync(profile, { recursive: true, force: true });
  process.exit(code);
};
const timeout = setTimeout(() => {
  console.error("smoke: timeout");
  cleanup(1);
}, 90000);
process.on("SIGINT", () => cleanup(2));
process.on("uncaughtException", (e) => {
  console.error(e);
  cleanup(1);
});
process.on("unhandledRejection", (e) => {
  console.error(e);
  cleanup(1);
});

async function waitFor(fn, label, timeoutMs = 60000) {
  const start = Date.now();
  for (;;) {
    try {
      const v = await fn();
      if (v) return v;
    } catch {}
    if (Date.now() - start > timeoutMs)
      throw new Error(`timed out waiting for ${label}`);
    await new Promise((r) => setTimeout(r, 250));
  }
}

const checks = [];
const check = (name, ok, detail = "") => {
  checks.push({ name, ok, detail });
  console.log(
    `${ok ? "ok" : "NOT OK"} - ${name}${detail ? " - " + detail : ""}`,
  );
};

await waitFor(
  async () => (await fetch(`http://127.0.0.1:${CDP_PORT}/json/version`)).ok,
  "chrome devtools",
);
const target = await (
  await fetch(`http://127.0.0.1:${CDP_PORT}/json/new`, { method: "PUT" })
).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  ws.onopen = resolve;
  ws.onerror = reject;
});

let mid = 0;
const rpcPending = new Map();
const consoleErrors = [];
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.id && rpcPending.has(msg.id)) {
    rpcPending.get(msg.id)(msg);
    rpcPending.delete(msg.id);
  } else if (msg.method === "Runtime.exceptionThrown")
    consoleErrors.push(
      msg.params.exceptionDetails?.exception?.description ?? "page exception",
    );
};
const cdp = (method, params = {}) =>
  new Promise((resolve) => {
    const id = ++mid;
    rpcPending.set(id, resolve);
    ws.send(JSON.stringify({ id, method, params }));
  });
const evaluate = async (expression) => {
  const res = await cdp("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (res.result?.exceptionDetails)
    throw new Error(JSON.stringify(res.result.exceptionDetails));
  return res.result?.result?.value;
};
const shot = async (name) => {
  if (!SHOTS) return;
  await new Promise((r) => setTimeout(r, 900));
  const metrics = await cdp("Page.getLayoutMetrics");
  const size = metrics.result.cssContentSize;
  const r = await cdp("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: true,
    clip: { x: 0, y: 0, width: size.width, height: size.height, scale: 1 },
  });
  writeFileSync(
    `${root}smoke-${name}.png`,
    Buffer.from(r.result.data, "base64"),
  );
};

await cdp("Runtime.enable");
await cdp("Page.enable");
await cdp("Emulation.setDeviceMetricsOverride", {
  width: 1440,
  height: 960,
  deviceScaleFactor: 1,
  mobile: false,
});
await cdp("Page.navigate", { url: `http://127.0.0.1:${HTTP_PORT}/` });

await waitFor(
  () =>
    evaluate(
      `document.getElementById('slice-range').textContent.includes('BCE')`,
    ),
  "initial slice",
);
check(
  "boot exact initial loss",
  await evaluate(
    `document.getElementById('train-loss').textContent==='0.761712'`,
  ),
);
check(
  "no autoplay",
  await evaluate(`document.getElementById('epoch').textContent==='0'`),
);
await shot("initial");
await evaluate(`document.getElementById('run').click()`);
await waitFor(
  () =>
    evaluate(
      `document.getElementById('epoch').textContent==='600' && document.getElementById('slice-range').textContent.includes('BCE')`,
    ),
  "600 training steps",
);
check(
  "training expected loss",
  await evaluate(
    `document.getElementById('train-loss').textContent==='0.000269'`,
  ),
);
check(
  "heldout expected accuracy",
  await evaluate(`document.getElementById('accuracy').textContent==='100.0%'`),
);
check(
  "slice origin equals train loss",
  await evaluate(
    `document.getElementById('slice-probe').textContent.endsWith('0.000269')`,
  ),
);
await shot("trained");
await evaluate(
  `document.getElementById('alpha').value='1.5';document.getElementById('alpha').dispatchEvent(new Event('input'))`,
);
check(
  "probe updates",
  await evaluate(
    `document.getElementById('slice-probe').textContent.includes('α 1.5')`,
  ),
);
await evaluate(
  `document.getElementById('radius').value='6';document.getElementById('radius').dispatchEvent(new Event('change'))`,
);
await waitFor(
  () => evaluate(`!document.getElementById('alpha').disabled`),
  "radius slice",
);
check(
  "radius updates control",
  await evaluate(`document.getElementById('alpha').max==='6'`),
);
check(
  "radius leaves network unchanged",
  await evaluate(
    `document.getElementById('epoch').textContent==='600' && document.getElementById('train-loss').textContent==='0.000269'`,
  ),
);
await evaluate(`document.getElementById('alpha').focus()`);
await cdp("Input.dispatchKeyEvent", {
  type: "keyDown",
  key: "ArrowRight",
  code: "ArrowRight",
  windowsVirtualKeyCode: 39,
});
await cdp("Input.dispatchKeyEvent", {
  type: "keyUp",
  key: "ArrowRight",
  code: "ArrowRight",
  windowsVirtualKeyCode: 39,
});
check(
  "keyboard probe works",
  await evaluate(
    `document.getElementById('slice-probe').textContent.includes('α 0.1')`,
  ),
);
await evaluate(
  `(()=>{const c=document.getElementById('landscape'),r=c.getBoundingClientRect();c.dispatchEvent(new MouseEvent('click',{clientX:r.left+44+(r.width-58)*.75,clientY:r.top+16+(r.height-52)*.25}));})()`,
);
check(
  "click maps to correct coordinates",
  await evaluate(
    `document.getElementById('slice-probe').textContent.startsWith('α 3.0 / β 3.0')`,
  ),
);
mkdirSync(`${root}build/downloads`, { recursive: true });
rmSync(`${root}build/downloads/contour-loss-history.csv`, { force: true });
await cdp("Browser.setDownloadBehavior", {
  behavior: "allow",
  downloadPath: `${root}build/downloads`,
});
await evaluate(`document.getElementById('download').click()`);
await waitFor(
  () => existsSync(`${root}build/downloads/contour-loss-history.csv`),
  "CSV",
);
const rows = readFileSync(
  `${root}build/downloads/contour-loss-history.csv`,
  "utf8",
)
  .trim()
  .split(/\r?\n/);
check(
  "export has every 20-step checkpoint",
  rows.length === 32 && rows[31].split(",")[2] === "600",
);
check(
  "export exact rate and final loss",
  rows[31].split(",")[1] === "0.02" &&
    Math.abs(+rows[31].split(",")[3] - 0.00026857) < 1e-8,
);
await evaluate(
  `document.getElementById('run').click();document.getElementById('run').click()`,
);
await waitFor(
  () =>
    evaluate(
      `document.getElementById('status').textContent.startsWith('Stopped')`,
    ),
  "stop",
);
check(
  "stop interrupts chunked training",
  await evaluate(`+document.getElementById('epoch').textContent<1200`),
);
await evaluate(`document.getElementById('reset').click()`);
check(
  "reset restores exact state",
  await evaluate(
    `document.getElementById('epoch').textContent==='0' && document.getElementById('train-loss').textContent==='0.761712'`,
  ),
);
await evaluate(
  `document.getElementById('seed').value='7';document.getElementById('seed').dispatchEvent(new Event('change'))`,
);
check(
  "seed changes state",
  await evaluate(
    `document.getElementById('train-loss').textContent!=='0.761712'`,
  ),
);
await evaluate(`document.getElementById('step').click()`);
await waitFor(
  () =>
    evaluate(
      `document.getElementById('epoch').textContent==='20' && document.getElementById('slice-range').textContent.includes('BCE')`,
    ),
  "step20",
);
check(
  "single step chunk",
  await evaluate(`document.getElementById('epoch').textContent==='20'`),
);
for (const width of [390, 320]) {
  await cdp("Emulation.setDeviceMetricsOverride", {
    width,
    height: 844,
    deviceScaleFactor: 1,
    mobile: false,
  });
  check(
    `mobile ${width} no overflow`,
    await evaluate(`document.documentElement.scrollWidth<=innerWidth`),
  );
  await evaluate(`scrollTo(0,0)`);
  await shot(`mobile-${width}`);
  await evaluate(`document.getElementById('landscape').scrollIntoView()`);
  await shot(`slice-${width}`);
}
check("no page exceptions", consoleErrors.length === 0, consoleErrors[0] ?? "");
check(
  "no UI error",
  await evaluate(`document.getElementById('error').textContent===''`),
);
const failed = checks.filter((c) => !c.ok);
console.log(`${checks.length - failed.length} passed, ${failed.length} failed`);
clearTimeout(timeout);
await cleanup(failed.length ? 1 : 0);
