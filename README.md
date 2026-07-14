# hmmlearn-rs

A native Rust port of the Python library [`hmmlearn`](https://github.com/hmmlearn/hmmlearn):
Hidden Markov Models with a range of emission distributions, trained by
Expectation–Maximization or mean-field variational inference. Pure Rust — no
LAPACK/BLAS, no Python, no C.

## Features

- **Seven models**, each with a fluent builder:
  - `GaussianHmm` — Gaussian emissions (spherical / diagonal / full / tied covariance)
  - `GmmHmm` — Gaussian-mixture emissions
  - `CategoricalHmm` — discrete symbol emissions
  - `MultinomialHmm` — multinomial counts
  - `PoissonHmm` — Poisson counts
  - `VariationalGaussianHmm`, `VariationalCategoricalHmm` — Bayesian variants fit by variational inference
- **Inference**: forward–backward in both `log` and `scaling` implementations,
  Viterbi and MAP decoding, posterior probabilities, and sampling.
- **Training**: Expectation–Maximization (maximum likelihood) and mean-field
  variational inference, with a convergence monitor and configurable priors.
- **Model selection**: `score`, `aic`, and `bic` on any fitted model.
- Multi-sequence fitting via a `lengths` slice, matching `hmmlearn`'s API.

## Installation

```toml
[dependencies]
hmmlearn-rs = "1.0"
```

The public API is built on [`ndarray`](https://docs.rs/ndarray). It is re-exported
as `hmmlearn::ndarray` so you can construct inputs without adding (or version-matching)
`ndarray` yourself.

## Quick start

```rust
use hmmlearn::models::GaussianHmm;
use hmmlearn::ndarray::array;

// One feature per column; each row is an observation.
let x = array![[0.0], [0.2], [-0.1], [5.0], [5.2], [4.8]];

let model = GaussianHmm::new(2)
    .n_iter(20)
    .random_state(42)
    .fit(x.view(), None)?;

let states = model.predict(x.view(), None)?;   // most likely hidden state per row
let score = model.score(x.view(), None)?;      // log-likelihood of the data
# Ok::<(), hmmlearn::HmmError>(())
```

Independent sequences are concatenated along the rows and delimited by a
`lengths` slice that sums to the number of rows:

```rust
# use hmmlearn::models::GaussianHmm;
# use hmmlearn::ndarray::Array2;
# let x = Array2::<f64>::zeros((450, 1));
# let model = GaussianHmm::new(2).n_iter(1).fit(x.view(), Some(&[450]))?;
// three sequences of length 100, 200, 150
let states = model.predict(x.view(), Some(&[100, 200, 150]))?;
# Ok::<(), hmmlearn::HmmError>(())
```

## Examples

Runnable programs live in [`examples/`](examples/):

```console
cargo run --example casino         # discrete "dishonest casino" — decode loaded vs fair dice
cargo run --example gaussian_hmm   # sample from a known 4-state model and decode it back
```