import {
  network,
  moons,
  forward,
  sigmoid,
  loss,
  accuracy,
  adam,
  directions,
  atSlice,
} from "./engine.js";
const $ = (id) => document.getElementById(id),
  train = moons(128, 17),
  test = moons(256, 91),
  dirs = directions(101);
let weights,
  optimizer,
  epoch,
  history,
  running = false,
  runToken = 0,
  sliceToken = 0,
  slice = null;
const next = () => new Promise((r) => setTimeout(r, 0));
const margin = { left: 44, right: 14, top: 16, bottom: 36 };
function context(id) {
  const c = $(id),
    d = Math.min(devicePixelRatio || 1, 2),
    w = c.clientWidth,
    h = c.clientHeight;
  c.width = Math.round(w * d);
  c.height = Math.round(h * d);
  const ctx = c.getContext("2d");
  ctx.scale(d, d);
  ctx.fillStyle = "#141719";
  ctx.fillRect(0, 0, w, h);
  return {
    c,
    ctx,
    w,
    h,
    pw: w - margin.left - margin.right,
    ph: h - margin.top - margin.bottom,
  };
}
function axes(p, xmin, xmax, ymin, ymax, xlabel, ylabel) {
  const { ctx, w, h, pw, ph } = p;
  ctx.strokeStyle = "#536158";
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, pw, ph);
  ctx.fillStyle = "#a5ada8";
  ctx.font = "12px ui-monospace,monospace";
  ctx.textAlign = "center";
  for (let i = 0; i <= 4; i++) {
    let x = margin.left + (pw * i) / 4;
    ctx.textAlign = i === 0 ? "left" : i === 4 ? "right" : "center";
    const tick = xmin + ((xmax - xmin) * i) / 4;
    ctx.fillText(
      xlabel === "step" ? String(Math.round(tick)) : tick.toFixed(1),
      x,
      h - 19,
    );
  }
  ctx.textAlign = "center";
  ctx.fillText(xlabel, margin.left + pw / 2, h - 2);
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    ctx.fillText(
      (ymax - ((ymax - ymin) * i) / 4).toFixed(1),
      margin.left - 7,
      margin.top + (ph * i) / 4 + 4,
    );
  }
  ctx.textAlign = "left";
  ctx.fillText(ylabel, 4, 11);
}
// Linear interpolation on each cell edge. Four-edge saddle cases use a fixed diagonal pairing.
function contours(ctx, grid, n, levels, p, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  for (const level of levels) {
    ctx.beginPath();
    for (let j = 0; j < n - 1; j++)
      for (let i = 0; i < n - 1; i++) {
        const v = [
            grid[j * n + i],
            grid[j * n + i + 1],
            grid[(j + 1) * n + i + 1],
            grid[(j + 1) * n + i],
          ],
          xy = [
            [i, j],
            [i + 1, j],
            [i + 1, j + 1],
            [i, j + 1],
          ],
          hits = [];
        for (let e = 0; e < 4; e++) {
          const f = (e + 1) % 4;
          if (v[e] < level === v[f] < level) continue;
          const t = (level - v[e]) / (v[f] - v[e]);
          hits.push([
            xy[e][0] + t * (xy[f][0] - xy[e][0]),
            xy[e][1] + t * (xy[f][1] - xy[e][1]),
          ]);
        }
        for (let k = 0; k + 1 < hits.length; k += 2) {
          const a = hits[k],
            b = hits[k + 1];
          ctx.moveTo(
            margin.left + (a[0] * p.pw) / (n - 1),
            margin.top + (a[1] * p.ph) / (n - 1),
          );
          ctx.lineTo(
            margin.left + (b[0] * p.pw) / (n - 1),
            margin.top + (b[1] * p.ph) / (n - 1),
          );
        }
      }
    ctx.stroke();
  }
}
function boundary() {
  const p = context("boundary"),
    { ctx, pw, ph } = p,
    n = 65,
    g = new Float64Array(n * n),
    colors = ["#64463e", "#513f39", "#303d36", "#3e5a4c", "#547762"];
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++)
      g[j * n + i] = sigmoid(
        forward(weights, -1.8 + (3.6 * i) / (n - 1), 1.4 - (2.8 * j) / (n - 1)),
      );
  for (let j = 0; j < n - 1; j++)
    for (let i = 0; i < n - 1; i++) {
      ctx.fillStyle = colors[Math.min(4, Math.floor(g[j * n + i] * 5))];
      ctx.fillRect(
        margin.left + (i * pw) / (n - 1),
        margin.top + (j * ph) / (n - 1),
        pw / (n - 1) + 0.5,
        ph / (n - 1) + 0.5,
      );
    }
  contours(ctx, g, n, [0.5], p, "#e7e9e6");
  for (const [x, y, label] of train) {
    const px = margin.left + ((x + 1.8) * pw) / 3.6,
      py = margin.top + ((1.4 - y) * ph) / 2.8;
    ctx.fillStyle = label ? "#93c9ad" : "#ef967f";
    ctx.strokeStyle = "#141719";
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (label) ctx.rect(px - 3, py - 3, 6, 6);
    else ctx.arc(px, py, 3.3, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }
  axes(p, -1.8, 1.8, -1.4, 1.4, "x₁", "x₂");
}
function historyPlot() {
  const p = context("history"),
    { ctx, pw, ph } = p,
    max = Math.max(...history.map((r) => r.loss), 0.01),
    end = Math.max(epoch, 20);
  ctx.strokeStyle = "#343b37";
  for (let i = 0; i <= 4; i++) {
    const y = margin.top + (ph * i) / 4;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + pw, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "#93c9ad";
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((r, i) => {
    const x = margin.left + (r.epoch / end) * pw,
      y = margin.top + (1 - r.loss / max) * ph;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  for (const r of history) {
    ctx.beginPath();
    ctx.arc(
      margin.left + (r.epoch / end) * pw,
      margin.top + (1 - r.loss / max) * ph,
      2,
      0,
      Math.PI * 2,
    );
    ctx.fillStyle = "#93c9ad";
    ctx.fill();
  }
  axes(p, 0, end, 0, max, "step", "BCE");
}
function drawSlice() {
  const p = context("landscape");
  if (!slice) {
    p.ctx.fillStyle = "#a5ada8";
    p.ctx.font = "12px system-ui";
    p.ctx.fillText("Calculating training-loss slice…", margin.left, 50);
    return;
  }
  const { ctx, pw, ph } = p,
    { grid, n, min, max, radius } = slice;
  for (let j = 0; j < n - 1; j++)
    for (let i = 0; i < n - 1; i++) {
      const t = (grid[j * n + i] - min) / (max - min || 1),
        c = [37, 46, 44].map((v, k) =>
          Math.round(v + t * ([218, 231, 223][k] - v)),
        );
      ctx.fillStyle = `rgb(${c.join(",")})`;
      ctx.fillRect(
        margin.left + (i * pw) / (n - 1),
        margin.top + (j * ph) / (n - 1),
        pw / (n - 1) + 0.5,
        ph / (n - 1) + 0.5,
      );
    }
  contours(
    ctx,
    grid,
    n,
    [0.2, 0.4, 0.6, 0.8].map((t) => min + t * (max - min)),
    p,
    "#71887b",
  );
  ctx.strokeStyle = "#ef967f";
  ctx.lineWidth = 2;
  const cx = margin.left + pw / 2,
    cy = margin.top + ph / 2;
  ctx.beginPath();
  ctx.moveTo(cx - 6, cy);
  ctx.lineTo(cx + 6, cy);
  ctx.moveTo(cx, cy - 6);
  ctx.lineTo(cx, cy + 6);
  ctx.stroke();
  const a = +$("alpha").value,
    b = +$("beta").value;
  ctx.strokeStyle = "#ef967f";
  ctx.beginPath();
  ctx.arc(
    margin.left + ((a + radius) / (2 * radius)) * pw,
    margin.top + ((radius - b) / (2 * radius)) * ph,
    5,
    0,
    2 * Math.PI,
  );
  ctx.stroke();
  axes(p, -radius, radius, -radius, radius, "α", "β");
}
function probe() {
  if (!slice) return;
  const a = +$("alpha").value,
    b = +$("beta").value,
    v = loss(atSlice(slice.center, dirs, a, b), train);
  $("slice-probe").textContent =
    `α ${a.toFixed(1)} / β ${b.toFixed(1)} / training BCE ${v.toFixed(6)}`;
  drawSlice();
}
async function calculateSlice() {
  const token = ++sliceToken,
    center = weights.slice(),
    radius = +$("radius").value,
    n = 41,
    grid = new Float64Array(n * n);
  slice = null;
  drawSlice();
  $("slice-range").textContent = "calculating…";
  $("slice-probe").textContent = "Calculating slice for the displayed network…";
  for (const id of ["alpha", "beta"]) {
    $(id).disabled = true;
    $(id).min = -radius;
    $(id).max = radius;
    $(id).value = 0;
  }
  let min = Infinity,
    max = -Infinity;
  for (let j = 0; j < n; j++) {
    if (token !== sliceToken) return;
    for (let i = 0; i < n; i++) {
      const v = loss(
        atSlice(
          center,
          dirs,
          -radius + (2 * radius * i) / (n - 1),
          radius - (2 * radius * j) / (n - 1),
        ),
        train,
      );
      grid[j * n + i] = v;
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
    await next();
  }
  if (token !== sliceToken) return;
  slice = { center, radius, n, grid, min, max };
  $("slice-range").textContent = `${min.toFixed(3)} to ${max.toFixed(3)} BCE`;
  $("alpha").disabled = false;
  $("beta").disabled = false;
  probe();
}
function render() {
  const tr = loss(weights, train),
    te = loss(weights, test),
    acc = accuracy(weights, test);
  $("epoch").textContent = epoch;
  $("train-loss").textContent = tr.toFixed(6);
  $("test-loss").textContent = te.toFixed(6);
  $("accuracy").textContent = (acc * 100).toFixed(1) + "%";
  $("boundary").setAttribute(
    "aria-label",
    `Prediction field at step ${epoch}. Test accuracy ${(acc * 100).toFixed(1)} percent. Points are training examples.`,
  );
  boundary();
  historyPlot();
}
function controls() {
  $("run").textContent = running ? "Stop training" : "Train 600 steps";
  $("step").disabled = running;
  $("seed").disabled = running;
  $("rate").disabled = running;
}
function reset() {
  $("error").textContent = "";
  runToken++;
  sliceToken++;
  running = false;
  epoch = 0;
  weights = network(+$("seed").value);
  optimizer = adam();
  history = [{ epoch: 0, loss: loss(weights, train), rate: 0 }];
  controls();
  $("status").textContent = "Ready";
  render();
  calculateSlice().catch(fail);
}
function fail(e) {
  running = false;
  controls();
  $("error").textContent =
    "This calculation failed. Reset the experiment to try again.";
  console.error(e);
}
async function trainSteps(count) {
  if (running) {
    runToken++;
    running = false;
    controls();
    $("status").textContent = `Stopped at step ${epoch}`;
    calculateSlice().catch(fail);
    return;
  }
  running = true;
  controls();
  sliceToken++;
  slice = null;
  drawSlice();
  $("slice-range").textContent = "waiting for training";
  $("slice-probe").textContent = "Slice recomputes when training stops.";
  $("alpha").disabled = true;
  $("beta").disabled = true;
  const token = ++runToken,
    rate = +$("rate").value;
  try {
    for (let i = 0; i < count; i += 20) {
      if (token !== runToken) return;
      for (let j = 0; j < Math.min(20, count - i); j++) {
        optimizer.step(weights, train, rate);
        epoch++;
      }
      history.push({ epoch, loss: loss(weights, train), rate });
      render();
      $("status").textContent = `Training: step ${epoch}`;
      await next();
    }
    if (token !== runToken) return;
    running = false;
    controls();
    $("status").textContent = `Finished at step ${epoch}`;
    calculateSlice().catch(fail);
  } catch (e) {
    fail(e);
  }
}
$("run").addEventListener("click", () => trainSteps(600));
$("step").addEventListener("click", () => trainSteps(20));
$("reset").addEventListener("click", reset);
$("seed").addEventListener("change", reset);
$("radius").addEventListener("change", () => {
  if (!running) calculateSlice().catch(fail);
});
for (const id of ["alpha", "beta"]) $(id).addEventListener("input", probe);
$("landscape").addEventListener("click", (e) => {
  if (!slice) return;
  const rect = e.currentTarget.getBoundingClientRect(),
    x =
      (e.clientX - rect.left - margin.left) /
      (rect.width - margin.left - margin.right),
    y =
      (e.clientY - rect.top - margin.top) /
      (rect.height - margin.top - margin.bottom);
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  $("alpha").value = (2 * x - 1) * slice.radius;
  $("beta").value = (1 - 2 * y) * slice.radius;
  probe();
});
$("download").addEventListener("click", () => {
  const text =
    "seed,learning_rate,step,training_bce\r\n" +
    history
      .map((r) => `${$("seed").value},${r.rate},${r.epoch},${r.loss}`)
      .join("\r\n") +
    "\r\n";
  const blob = new Blob([text], { type: "text/csv;charset=utf-8" }),
    url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = "contour-loss-history.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
let timer;
window.addEventListener("resize", () => {
  clearTimeout(timer);
  timer = setTimeout(() => {
    render();
    drawSlice();
  }, 100);
});
reset();
