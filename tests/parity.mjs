import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import {
  moons,
  network,
  loss,
  accuracy,
  adam,
  directions,
  atSlice,
  COUNT,
} from "../web/src/engine.js";
const refs = JSON.parse(
  execFileSync("java", ["-cp", "build/classes", "contour.RefDump"], {
    encoding: "utf8",
  }),
);
const train = moons(128, 17),
  test = moons(256, 91),
  model = network(42),
  optimizer = adam();
let epoch = 0,
  compared = 0,
  worst = 0;
const near = (a, b, path) => {
  const d = Math.abs(a - b);
  assert.ok(d < 1e-8, `${path}: ${d}`);
  worst = Math.max(worst, d);
  compared++;
};
for (const ref of refs) {
  while (epoch < ref.step) {
    optimizer.step(model, train, 0.02);
    epoch++;
  }
  const g = new Float64Array(COUNT);
  near(loss(model, train, g), ref.loss, "loss");
  near(loss(model, test), ref.testLoss, "testLoss");
  near(accuracy(model, test), ref.accuracy, "accuracy");
  model.forEach((v, i) => near(v, ref.weights[i], "weight " + i));
  g.forEach((v, i) => near(v, ref.gradient[i], "gradient " + i));
}
const dirs = directions(101);
near(
  dirs[0].reduce((s, v) => s + v * v, 0),
  1,
  "direction norm",
);
near(
  dirs[1].reduce((s, v) => s + v * v, 0),
  1,
  "direction norm",
);
near(
  dirs[0].reduce((s, v, i) => s + v * dirs[1][i], 0),
  0,
  "orthogonal directions",
);
assert.deepEqual(atSlice(model, dirs, 0, 0), model);
const g = new Float64Array(COUNT);
loss(model, train, g);
let maxGradientError = 0;
for (let i = 0; i < COUNT; i++) {
  const old = model[i],
    eps = 1e-5;
  model[i] = old + eps;
  const hi = loss(model, train);
  model[i] = old - eps;
  const lo = loss(model, train);
  model[i] = old;
  maxGradientError = Math.max(
    maxGradientError,
    Math.abs((hi - lo) / (2 * eps) - g[i]),
  );
}
assert.ok(maxGradientError < 1e-8);
console.log(
  `${compared} Java/JS comparisons; max difference ${worst}; gradient max error ${maxGradientError}`,
);
