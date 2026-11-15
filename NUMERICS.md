# Numerical conventions

## Parameter layout

The 105 doubles are flattened row-major by output neuron:

| Indices | Meaning |
| --- | --- |
| 0-15 | W1: 8 by 2 |
| 16-23 | b1: 8 |
| 24-87 | W2: 8 by 8 |
| 88-95 | b2: 8 |
| 96-103 | W3: 1 by 8 |
| 104 | b3: 1 |

Forward pass: a = tanh(W1*x+b1), b = tanh(W2*a+b2), z = W3*b+b3. The output probability is sigmoid(z). The predicted class is 1 when z >= 0, else 0.

For target y in {0,1}, loss is max(z,0)-z*y+log1p(exp(-abs(z))). This avoids exp(z) overflow. Loss is the unregularized arithmetic mean over examples. Its logit derivative is (sigmoid(z)-y)/N. Each dense-layer weight derivative is the downstream derivative times the input activation. Each hidden activation contributes a factor 1-tanh(u)^2 in the reverse pass.

All gradients are computed against the unchanged weight vector before any optimizer update. Adam uses beta1=0.9, beta2=0.999, epsilon=1e-8. Moments start at zero; step t starts at 1. Update: w -= learning_rate * (m/(1-beta1^t)) / (sqrt(v/(1-beta2^t)) + epsilon). Biases are updated like weights. There is no dropout, regularization, batch shuffling or learning-rate schedule. Changing the UI learning rate between runs preserves Adam moments; reset clears them.

Each dense weight matrix uses Xavier uniform bounds ±sqrt(6/(fan_in+fan_out)). Biases start at zero. The PRNG is a 32-bit LCG with state = 1664525*state + 1013904223 modulo 2^32; next = unsigned(state)/2^32. Model initialization seed is selectable (42, 7, 101).

## Synthetic data

Even indices have label zero and (cos(a),sin(a)); odd indices have label one and (1-cos(a),0.5-sin(a)), where a is uniform on [0,pi). Each coordinate gets noise 0.14 times a sum of six independent centered uniform draws, standard deviation approximately 0.099. Then x is shifted by -0.5 and y by -0.25. This is bounded Irwin-Hall noise, not Gaussian noise. The 128 train and 256 test points are distinct draws from seeds 17 and 91, not a shuffled split of one sample. Labels alternate but full-batch averaging does not use order.

## Slice and plots

LCG seed 101 supplies two vectors of 105 independent uniform values on [-1,1). Gram-Schmidt makes their dot product zero and Euclidean norms one. They stay fixed as training and initialization change. A 41 by 41 grid evaluates training BCE at w + alpha*d1 + beta*d2 with alpha,beta in [-radius,radius]. Rows run from positive to negative beta. Grid computation yields after every row and invalidates stale work on reset, training or radius changes. Values are never computed on test examples. The scale is linear per slice. Marching-cell edges are linearly interpolated; ambiguous four-edge saddle cells use a fixed pairing. Contours are approximate, not additional exact evaluations. The probe computes exact model loss at slider coordinates rounded to 0.1. The center is always an exact grid point.

The boundary uses a 65 by 65 probability grid on x1 in [-1.8,1.8], x2 in [-1.4,1.4], with bins of width 0.2 and interpolated P=0.5 contour. The graph is a finite view, not all possible inputs. Circle/square shapes distinguish classes without relying only on color. Training loss history is sampled every 20 updates, joined by straight lines, without smoothing. Accuracy is unweighted fraction correct on a balanced test set.

## Validation

Java tests compare all 105 analytic derivatives with central differences at epsilon=1e-5, test an all-zero network against ln(2), check an exact first Adam update, and verify extreme-logit stability. They also check decreasing finite loss for every selectable seed/rate combination. Fresh Java snapshots at steps 0,1,10,100,600 are compared against the JavaScript model: all weights, gradients, train/test loss and accuracy. JS derivatives and slice direction norms/orthogonality are checked separately. Parity alone is not evidence against a shared mathematical bug, which is why the analytical fixtures and finite differences are separate.

No claim of cryptographic randomness, convergence for arbitrary settings, or applicability to real-world classification is made.
