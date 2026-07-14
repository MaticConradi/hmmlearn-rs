# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0]

Initial release: a native, pure-Rust port of Python `hmmlearn` 0.3.3.

### Added

- Seven models with fluent builders: `GaussianHmm`, `GmmHmm`, `CategoricalHmm`,
  `MultinomialHmm`, `PoissonHmm`, `VariationalGaussianHmm`, and
  `VariationalCategoricalHmm`.
- Forward–backward inference in `log` and `scaling` implementations; Viterbi and
  MAP decoding; posterior probabilities; sampling.
- Training by Expectation–Maximization and mean-field variational inference, with
  a convergence monitor and configurable priors.
- Model-selection helpers: `score`, `aic`, `bic`.
- Multi-sequence fitting via a `lengths` slice.
- Bit-exact port of NumPy's legacy `RandomState` for reproducible seeded runs,
  validated against golden files.
- `casino` and `gaussian_hmm` examples.

### Notes

Three documented, test-validated deviations from upstream, all confined to
initialization and sampling: k-means++ mean initialization, Cholesky-based
multivariate-normal sampling, and an inversion-only binomial sampler. Fitted
parameters and inference results match `hmmlearn`.