# Development Checklist

## Deferred Until After the MVP

### linear_solver.py
- [ ] Add package-level validation and a clear error message when a callable
      preconditioner returns a vector with the wrong shape. The MVP currently
      relies on SciPy to reject the incompatible output.

### MLMC runner reproducibility and parallelization

- [ ] Give every correction sample a deterministic RNG stream identified by:
  - base seed
  - fine level
  - sample index

- [ ] Construct each stream using:

      np.random.SeedSequence(
          [base_seed, fine_level, sample_index]
      )

- [ ] Ensure the runner owns RNG construction. Models must only use the
      generator passed to `sample_randomness()`.

- [ ] Draw randomness once per correction. Fine and coarse inputs must be
      constructed from the same random realization.

- [ ] Do not share one mutable `Generator` between parallel workers.

- [ ] Ensure correction samples are reproducible regardless of parallel task
      completion order.

- [ ] Accumulate parallel results in deterministic `(level, sample_index)`
      order to minimize floating-point ordering differences.

- [ ] Test that changing one level's sample count does not alter the random
      samples assigned to another level.