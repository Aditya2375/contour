// Two-input, two hidden layer MLP. Every derivative and optimizer update is explicit.
export const WIDTH = 8,
  COUNT = 105;
export function rng(seed) {
  let s = seed | 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) | 0;
    return (s >>> 0) / 4294967296;
  };
}
export function moons(count, seed) {
  if (!Number.isInteger(count) || count < 2 || count % 2)
    throw new Error("Even balanced sample count required");
  const random = rng(seed),
    rows = [];
  for (let i = 0; i < count; i++) {
    const label = i % 2,
      a = random() * Math.PI,
      x = label === 0 ? Math.cos(a) : 1 - Math.cos(a),
      y = label === 0 ? Math.sin(a) : 0.5 - Math.sin(a);
    let nx = 0,
      ny = 0;
    for (let j = 0; j < 6; j++) {
      nx += random() - 0.5;
      ny += random() - 0.5;
    }
    rows.push([x + nx * 0.14 - 0.5, y + ny * 0.14 - 0.25, label]);
  }
  return rows;
}
export function network(seed) {
  const random = rng(seed),
    w = new Float64Array(COUNT);
  for (let j = 0; j < 8; j++)
    for (let i = 0; i < 2; i++)
      w[2 * j + i] = (random() * 2 - 1) * Math.sqrt(6 / 10);
  for (let j = 0; j < 8; j++)
    for (let i = 0; i < 8; i++)
      w[24 + j * 8 + i] = (random() * 2 - 1) * Math.sqrt(6 / 16);
  for (let i = 0; i < 8; i++) w[96 + i] = (random() * 2 - 1) * Math.sqrt(6 / 9);
  return w;
}
export function forward(
  w,
  x,
  y,
  a = new Float64Array(8),
  b = new Float64Array(8),
) {
  for (let j = 0; j < 8; j++)
    a[j] = Math.tanh(w[j * 2] * x + w[j * 2 + 1] * y + w[16 + j]);
  for (let j = 0; j < 8; j++) {
    let z = w[88 + j];
    for (let i = 0; i < 8; i++) z += w[24 + j * 8 + i] * a[i];
    b[j] = Math.tanh(z);
  }
  let z = w[104];
  for (let i = 0; i < 8; i++) z += w[96 + i] * b[i];
  return z;
}
export const sigmoid = (z) =>
  z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
export function loss(w, data, g = null) {
  if (g) g.fill(0);
  const a = new Float64Array(8),
    b = new Float64Array(8),
    da = new Float64Array(8);
  let total = 0;
  for (const row of data) {
    const z = forward(w, row[0], row[1], a, b),
      label = row[2];
    total += Math.max(z, 0) - z * label + Math.log1p(Math.exp(-Math.abs(z)));
    if (!g) continue;
    const dz = (sigmoid(z) - label) / data.length;
    g[104] += dz;
    da.fill(0);
    for (let j = 0; j < 8; j++) {
      g[96 + j] += dz * b[j];
      const db = dz * w[96 + j] * (1 - b[j] * b[j]);
      g[88 + j] += db;
      for (let i = 0; i < 8; i++) {
        g[24 + j * 8 + i] += db * a[i];
        da[i] += db * w[24 + j * 8 + i];
      }
    }
    for (let j = 0; j < 8; j++) {
      const u = da[j] * (1 - a[j] * a[j]);
      g[j * 2] += u * row[0];
      g[j * 2 + 1] += u * row[1];
      g[16 + j] += u;
    }
  }
  return total / data.length;
}
export function accuracy(w, data) {
  return (
    data.reduce(
      (n, r) => n + ((forward(w, r[0], r[1]) >= 0 ? 1 : 0) === r[2] ? 1 : 0),
      0,
    ) / data.length
  );
}
export function adam() {
  const m = new Float64Array(COUNT),
    v = new Float64Array(COUNT),
    g = new Float64Array(COUNT);
  let steps = 0;
  return {
    step(w, data, rate) {
      loss(w, data, g);
      steps++;
      for (let i = 0; i < COUNT; i++) {
        m[i] = 0.9 * m[i] + 0.1 * g[i];
        v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
        w[i] -=
          (rate * (m[i] / (1 - Math.pow(0.9, steps)))) /
          (Math.sqrt(v[i] / (1 - Math.pow(0.999, steps))) + 1e-8);
      }
    },
  };
}
// Orthonormal random directions with unit Euclidean norm in the complete parameter space.
// A slice is w + alpha*d1 + beta*d2. Not filter-normalized or the full loss surface.
export function directions(seed) {
  const random = rng(seed),
    a = Float64Array.from({ length: COUNT }, () => random() * 2 - 1),
    b = Float64Array.from({ length: COUNT }, () => random() * 2 - 1);
  const norm = (x) => Math.sqrt(x.reduce((s, v) => s + v * v, 0));
  const na = norm(a);
  for (let i = 0; i < COUNT; i++) a[i] /= na;
  const projection = b.reduce((s, v, i) => s + v * a[i], 0);
  for (let i = 0; i < COUNT; i++) b[i] -= projection * a[i];
  const nb = norm(b);
  for (let i = 0; i < COUNT; i++) b[i] /= nb;
  return [a, b];
}
export function atSlice(center, dirs, alpha, beta) {
  return center.map((v, i) => v + alpha * dirs[0][i] + beta * dirs[1][i]);
}
