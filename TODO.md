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

### MLMC timing and cost semantics

- [ ] Preserve the existing per-correction cost measurement used by
      `LevelStatistics`. `SampleCorrection.elapsed_time` currently measures
      the model work from `sample_randomness()` through the final fine or
      coarse quantity-of-interest evaluation. This is the appropriate level
      cost for variance-and-cost-based MLMC sample allocation.

- [ ] Clarify or rename `MLMCResult.total_cost` to
      `total_correction_cost`. It is the sum of successful correction
      evaluation times, not the complete wall-clock duration experienced by
      the user.

- [ ] Add a separate runner-level cumulative wall-time measurement covering
      sample RNG construction, task scheduling or serial loop overhead,
      correction evaluation, and statistics accumulation for `run_fixed()`
      and every later `add_samples()` batch.

- [ ] Decide whether `MLMCResult` should expose only cumulative runner wall
      time or both `total_wall_time` and `last_batch_wall_time`. Returned
      snapshots must remain immutable after additional batches are run.

- [ ] Define and document whether validation, level-statistics
      initialization, result-snapshot construction, and time spent in failed
      batches belong in runner wall time. Use `try/finally` if failed work
      must contribute to the operational timing record.

- [ ] Do not add `LinearSolveResult.solve_time` to correction cost. Solver
      calls are already inside `SampleCorrection.elapsed_time`, so doing so
      would double-count solve work.

- [ ] Preserve the distinction under parallel execution:
  - total correction cost is the sum of individual successful task times;
  - runner wall time is the actual elapsed execution time;
  - total correction cost may be larger than wall time when tasks overlap.

- [ ] Add timing tests using a controlled or mocked clock. Verify initial and
      additional-batch accumulation, immutable snapshots, failure behavior,
      and the distinction between summed correction work and runner wall
      time without relying on real sleeps.

#### Parallel worker and model boundary

- [ ] Keep fine and coarse evaluations for one correction in the same worker.
      The worker must call `compute_sample_correction()` once so both
      evaluations use the same coupled random realization.

- [ ] Do not naively send the complete model with every process-pool task.
      Passing `model` to `compute_sample_correction()` is only an ordinary
      reference in serial or threaded execution, but process-based execution
      may serialize it for every submitted task.

- [ ] Initialize or construct the model once per worker process and reuse that
      worker-local model for multiple correction samples.

- [ ] Keep individual process task payloads lightweight. A task should be
      identified primarily by `(fine_level, sample_index, base_seed)` rather
      than carrying meshes, matrices, solver state, or solution arrays.

- [ ] Document and test the parallel model contract. User models and solver
      backends may contain non-picklable objects or mutable state and may not
      be thread-safe. Process workers may therefore need a model factory or
      worker initializer instead of a serialized model instance.

- [ ] Avoid returning complete `SampleCorrection` objects from process
      workers by default. They contain fine and coarse `LinearSolveResult`
      objects and potentially large solution arrays that would be serialized
      and copied back to the main process.

- [ ] Introduce a lightweight parallel result summary containing only the
      correction level, sample index, correction value, and elapsed time.
      Keep the full `SampleCorrection` available for serial calls, debugging,
      and optional solver diagnostics.

- [ ] Extract the lightweight summary inside the worker and discard fine and
      coarse solution arrays before returning the result to the main process.

- [ ] Verify that serial and parallel runs assign identical correction values
      to every `(level, sample_index)` task. Timing values are not expected to
      match.

- [ ] Benchmark worker initialization, duplicated model memory, task
      serialization, and result-transfer overhead before selecting a process,
      thread, or batching strategy.
