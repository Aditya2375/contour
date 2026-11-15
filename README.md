# contour

[Open the live experiment](https://aditya2375.github.io/contour/)

A neural network from scratch, with a decision-boundary plot and an interactive two-dimensional slice of training loss. Java is the numerical reference. A dependency-free JavaScript port runs the browser experiment.

## The experiment

- A 2-8-8-1 tanh network with 105 trainable parameters.
- Stable binary cross-entropy from logits, explicit backpropagation and full-batch Adam.
- Seeded, balanced synthetic moons: 128 training examples and 256 independent test examples.
- Train 600 more steps, advance 20 steps, stop or reset. Choose initialization and learning rate.
- Probe a 41 by 41 training-loss slice through the current weights, with mouse or keyboard-accessible sliders. Changing its radius never changes the network.
- Export unsmoothed training-loss checkpoints to CSV. Each row records the rate used for that checkpoint's preceding 20 updates; step zero has rate zero.

Nothing is uploaded or saved. No machine-learning libraries, accounts, API keys, third-party runtime assets or build dependencies. All data is generated locally. JavaScript is necessary for the interactive demo; Java can run the numerical tests without a browser.

## Run and test

Requires JDK 21+ and Node 22+. Browser checks additionally need Python 3 and `google-chrome` on PATH.

```sh
javac -d build/classes src/contour/*.java tools/*.java tests/contour/*.java
java -cp build/classes contour.TestRunner
node tests/parity.mjs
node tools/build-web.mjs
node tests/browser-smoke.mjs
python3 -m http.server 8080 --directory docs
```

Open localhost:8080. `SMOKE_SHOTS=1 node tests/browser-smoke.mjs` also saves screenshots. Build output in `docs/` is reproducible and suited to a static host. CI checks Java tests, Java/JS parity, browser behavior, build drift and narrow static security rules.

The fixed seed-42, rate-0.02 example at step 600 has training BCE about 0.000269 and 100% accuracy on this one synthetic test draw. This is a reproducibility fixture, not a broad performance claim.

## What the loss map means

A position is `w + alpha*d1 + beta*d2`. The two seeded random directions are orthonormal under the Euclidean norm over all 105 parameters. They are **not** filter-normalized. The center is the currently displayed network, not a separately optimized minimum. Every update recenters the slice; no optimizer trajectory is projected onto it. Colors use a linear scale between this slice's minimum and maximum, so shades are not comparable between recentered slices or radius settings. Thin contours are interpolated from the sampled grid. The probe evaluates loss directly at its displayed coordinates.

A slice hides 103 dimensions and depends on parameterization and direction. It cannot prove global flatness, optimization quality or generalization. Test examples never enter training or the slice. Looking at test scores while changing settings makes this educational, not an unbiased model-selection benchmark. Data and seeds stay fixed across resets.

## Layout

- `src/contour/`: Java model, optimizer and synthetic data.
- `web/src/`: explicit JavaScript numerical port and Canvas 2D interface.
- `tests/`: analytical, finite-difference, full-state parity and browser checks.
- `tools/RefDump.java`: fresh Java snapshots consumed by parity tests.
- `NUMERICS.md`: equations, parameter layout and numerical conventions.

System fonts only. No external datasets or copied third-party assets.

## License

MIT OR Apache-2.0, at your option. See `LICENSE-MIT` and `LICENSE-APACHE`.
