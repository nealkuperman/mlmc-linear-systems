# Understanding MLMC statistics and results

The statistics and result classes represent two different stages of an MLMC run.

During execution, the runner needs mutable objects that can be updated after every correction sample. These are `RunningStatistics` and `LevelStatistics`.

After a run or additional batch finishes, the user needs a stable object that can be inspected, stored, compared, or passed to another part of a program without changing later. These are `LevelResult` and `MLMCResult`.

The complete data flow is:

<pre>
SampleCorrection
      |
      | scalar correction value and elapsed time
      v
LevelStatistics
      |
      | owns two RunningStatistics accumulators
      | and is updated after every sample
      v
MLMCRunner
      |
      | creates an immutable snapshot
      v
LevelResult for every correction level
      |
      | collected in level order
      v
MLMCResult for the complete run
</pre>

This separation is important. The runner can continue accumulating samples without unexpectedly changing a result that was returned earlier.

## 1. Statistics versus results

In this package, a **statistics object** is a mutable accumulator. It changes whenever a new observation is added.

A **result object** is an immutable snapshot. It records the scalar state of the accumulators at one point in time.

| Object | Mutable? | Scope | Primary user |
|---|---:|---|---|
| `RunningStatistics` | Yes | One sequence of scalar observations | Statistics and runner internals |
| `LevelStatistics` | Yes | Correction and cost observations at one level | `MLMCRunner` |
| `LevelResult` | No | Snapshot of one correction level | User and reporting code |
| `MLMCResult` | No | Snapshot of the complete multilevel run | User and reporting code |

Users will normally receive an `MLMCResult` from the runner. They do not need to update `RunningStatistics` or `LevelStatistics` themselves when using `MLMCRunner`.

## 2. `RunningStatistics`: one scalar data stream

`RunningStatistics` accumulates statistics for one sequence of scalar values. It can represent correction values, elapsed-time costs, or any other scalar observations.

Suppose the observed values are

$$
x_1,x_2,\ldots,x_N.
$$

The object tracks three pieces of state:

| State | Meaning |
|---|---|
| `count` | Number $N$ of observations received |
| `mean` | Current sample mean $\overline{x}$ |
| `_sum_squared_deviations` | Internal sum of squared deviations, commonly denoted $M_2$ |

The mean is

$$
\overline{x}
=\frac{1}{N}\sum_{i=1}^{N}x_i.
$$

The internal quantity is

$$
M_2
=\sum_{i=1}^{N}(x_i-\overline{x})^2.
$$

The underscore in `_sum_squared_deviations` indicates that it is internal state. Users should normally access the public statistical properties rather than use $M_2$ directly.

### 2.1 Why statistics are updated online

The package uses Welford's algorithm to update the mean and $M_2$ when each observation arrives. It does not retain a list of all previous observations.

This has two advantages:

1. Memory use remains constant even if millions of corrections are sampled.
2. The variance calculation is more numerically stable than subtracting two large, nearly equal accumulated sums.

If the current count is $n$, the current mean is $\overline{x}_n$, and a new value $x_{n+1}$ arrives, the updated mean is

$$
\overline{x}_{n+1}
=\overline{x}_n
+\frac{x_{n+1}-\overline{x}_n}{n+1}.
$$

The corresponding $M_2$ update uses the difference from both the old and new means. This lets the accumulator recover the unbiased sample variance without storing the individual observations.

Only finite values are accepted. `NaN`, positive infinity, and negative infinity are rejected before the accumulator is changed.

### 2.2 Sample variance

The unbiased sample variance is

$$
s^2
=\frac{1}{N-1}
\sum_{i=1}^{N}(x_i-\overline{x})^2
=\frac{M_2}{N-1}.
$$

This quantity describes the variation of the individual observations.

At least two observations are required. When $N<2$, `sample_variance` is `NaN` because an unbiased variance cannot be estimated from fewer than two values.

### 2.3 Variance of the sample mean

The estimated variance of the sample mean is

$$
\widehat{\operatorname{Var}}[\overline{x}]
=\frac{s^2}{N}.
$$

This is exposed as `variance_of_mean`.

It is different from `sample_variance`:

- `sample_variance` estimates how much individual observations vary.
- `variance_of_mean` estimates the uncertainty in their calculated mean.

As more independent observations are added, the variance of the mean generally decreases because it is divided by $N$.

### 2.4 Total

The total of all observations is

$$
\sum_{i=1}^{N}x_i
=N\overline{x}.
$$

The `total` property therefore calculates `mean` multiplied by `count`. The total is not stored as an additional piece of state. This avoids retaining information that can already be recovered exactly from existing fields.

### 2.5 Empty and one-observation states

Before any observations arrive:

| Property | Value |
|---|---:|
| `count` | 0 |
| `mean` | 0.0 |
| `total` | 0.0 |
| `sample_variance` | `NaN` |
| `variance_of_mean` | `NaN` |

The initial mean of 0.0 is accumulator state, not an estimate based on data.

After one observation, the mean and total are defined, but both variance properties remain `NaN`. The fixed-sample runner requires at least two initial samples per active level so normally returned run results have defined correction variances.

## 3. `LevelStatistics`: mutable statistics for one correction level

`LevelStatistics` represents one MLMC correction level $\ell$. It owns two independent `RunningStatistics` objects:

| Accumulator | Observations received |
|---|---|
| `correction` | Correction values $Y_\ell^{(i)}$ |
| `cost` | Total elapsed time for each corresponding correction |

For level zero,

$$
Y_0=Q_0.
$$

For a positive level,

$$
Y_\ell=Q_\ell-Q_{\ell-1}.
$$

One `LevelStatistics` object never combines different correction levels. Its `level` must be nonnegative, and it rejects a `SampleCorrection` whose `fine_level` does not match.

### 3.1 Consuming a `SampleCorrection`

When a completed `SampleCorrection` is passed to `LevelStatistics`, the accumulator extracts only two scalar values:

1. `sample.value`, which is the correction $Y_\ell$;
2. `sample.elapsed_time`, which is the measured cost of obtaining that correction.

The elapsed time must be finite and nonnegative. The correction value must be finite because it is passed through `RunningStatistics.update()`.

The full `SampleCorrection` also contains fine and optional coarse `LinearSolveResult` objects, including solution arrays. `LevelStatistics` does not retain them. Once the runner finishes the statistics update and releases the temporary sample, those large arrays can be reclaimed.

This is what makes online sampling memory-efficient: stored memory depends primarily on the number of levels, not on the number or dimension of completed solves.

### 3.2 Public level statistics

`LevelStatistics` exposes correction and cost information through level-specific properties:

| Property | Formula or meaning |
|---|---|
| `sample_count` | $N_\ell$ |
| `mean_correction` | $\overline{Y}_\ell$ |
| `sample_variance` | $s_\ell^2$ |
| `variance_of_mean` | $s_\ell^2/N_\ell$ |
| `mean_sample_cost` | Mean time required for one $Y_\ell$ sample |
| `total_sample_cost` | Sum of all measured correction times at level $\ell$ |

The mean correction estimates

$$
\mathbb{E}[Y_\ell].
$$

The variance of the mean estimates the sampling uncertainty contributed by that level to the complete MLMC estimator.

The cost statistics measure the complete correction calculation described in `mlmc_correction.md`, not only the numerical linear-solver time. At a positive level this includes both fine and coarse model evaluations.

### 3.3 Why `LevelStatistics` remains mutable

The runner may add samples in several batches. If level $\ell$ currently contains $N_\ell$ observations, a later call to `add_samples()` continues with new observations and updates the same `LevelStatistics` object.

This mutable state is useful inside the runner, but it should not be exposed as a supposedly permanent run result. If a user held a direct reference to it, values previously inspected could change after more samples were added.

That is why the runner converts each active `LevelStatistics` into an immutable `LevelResult` before returning data to the user.

## 4. `LevelResult`: an immutable level snapshot

`LevelResult` contains the scalar state of one correction level at the moment a result is requested. It is a frozen dataclass.

| Field | Meaning |
|---|---|
| `level` | Correction level $\ell$ |
| `sample_count` | Completed correction count $N_\ell$ |
| `mean_correction` | Sample mean $\overline{Y}_\ell$ |
| `sample_variance` | Unbiased correction variance $s_\ell^2$ |
| `variance_of_mean` | Estimated mean variance $s_\ell^2/N_\ell$ |
| `mean_sample_cost` | Mean measured time per correction |
| `total_sample_cost` | Total measured time for this correction level |

Unlike `LevelStatistics`, a `LevelResult` has no update method. It contains only scalar values and cannot receive another `SampleCorrection`.

It also does not contain:

- either internal `RunningStatistics` object;
- individual correction observations;
- the sample variance of the measured correction costs;
- fine or coarse quantities from individual samples;
- solution vectors;
- solver residual or iteration histories; or
- the RNG used for any sample.

It is intended for inspection, reporting, comparison, and transfer—not for continuing accumulation.

The mutable `LevelStatistics.cost` accumulator internally has the general `RunningStatistics` properties, but the current `LevelResult` copies only `mean_sample_cost` and `total_sample_cost`. Cost variance is therefore not part of the returned public snapshot.

## 5. Creating the level snapshots

When the runner creates a result, it reads the current public properties from each mutable `LevelStatistics` object and copies those scalar values into a new `LevelResult`.

Conceptually, the transformation is:

<pre>
mutable LevelStatistics at level l
    correction RunningStatistics
    cost RunningStatistics
              |
              | copy current scalar properties
              v
immutable LevelResult at level l
</pre>

The runner performs this transformation for every active correction level from 0 through the selected finest level.

The snapshots are newly created whenever:

- `run_fixed()` completes;
- `add_samples()` completes; or
- the runner's `result` property is accessed.

Requesting a result does not clear or transfer ownership of the runner's internal statistics. The runner keeps its mutable accumulators so it can continue the same run later.

## 6. `MLMCResult`: the complete run snapshot

`MLMCResult` represents the complete multilevel run at one point in time. It is also a frozen dataclass.

It stores two fields:

| Field | Meaning |
|---|---|
| `finest_level` | Finest included correction level $L$ and therefore the finite-level target $Q_L$ |
| `level_results` | Ordered tuple of `LevelResult` objects for levels $0,1,\ldots,L$ |

If `finest_level` is 3, the tuple contains four results in the order

$$
(\text{level 0},\text{level 1},\text{level 2},\text{level 3}).
$$

The tuple preserves the complete per-level breakdown while preventing callers from adding, removing, or reordering entries in the returned snapshot.

### 6.1 MLMC estimate

The `estimate` property adds the correction-level sample means:

$$
\widehat Q_{\mathrm{ML}}
=\sum_{\ell=0}^{L}\overline{Y}_\ell.
$$

This estimates

$$
\mathbb{E}[Q_L],
$$

not automatically the expectation of the limiting quantity $Q$. Any remaining difference between $Q_L$ and $Q$ is discretization bias.

### 6.2 Estimator variance

The `estimator_variance` property adds the variance-of-mean contribution from every correction level:

$$
\widehat{\operatorname{Var}}
\left[\widehat Q_{\mathrm{ML}}\right]
=\sum_{\ell=0}^{L}
\frac{s_\ell^2}{N_\ell}.
$$

This formula describes sampling uncertainty under the runner's independent correction-sample design. It does not include discretization bias.

It is also not the sum of the raw correction variances $s_\ell^2$. Each contribution is the variance of a correction mean, so the sample variance is divided by the number of samples at that level.

### 6.3 Standard error

The `standard_error` property is

$$
\widehat{\operatorname{SE}}
=\sqrt{
\widehat{\operatorname{Var}}
\left[\widehat Q_{\mathrm{ML}}\right]
}.
$$

It measures the estimated scale of sampling error in the complete MLMC mean.

A small standard error does not prove that the finest level is sufficiently accurate. A run can have very small sampling uncertainty while still having substantial discretization bias.

The result does not currently construct a confidence interval. If confidence intervals are added later, their interpretation will require distributional or asymptotic assumptions in addition to the stored standard error.

### 6.4 Total cost

The `total_cost` property adds the per-level total correction costs:

$$
C_{\mathrm{total}}
=\sum_{\ell=0}^{L}C_\ell.
$$

This value is measured wall-clock time, not a deterministic model output. Repeating the same seeded run should reproduce correction values and statistical estimates, but it will not generally reproduce timing values exactly.

### 6.5 Why aggregates are properties

The estimate, estimator variance, standard error, and total cost are calculated from `level_results` whenever they are accessed. They are not stored as separate dataclass fields.

This avoids redundant state. If both the per-level values and their sums were independently stored, a bug or manual construction could allow them to disagree. Deriving the aggregates ensures that the run-level values always match the level snapshots contained in the same result.

## 7. Worked interpretation of a result

Consider a two-level result with the following level snapshots:

| Level | $N_\ell$ | $\overline{Y}_\ell$ | $s_\ell^2$ | $s_\ell^2/N_\ell$ | Total cost |
|---:|---:|---:|---:|---:|---:|
| 0 | 100 | 1.20 | 4.00 | 0.04 | 10 seconds |
| 1 | 25 | -0.15 | 0.25 | 0.01 | 5 seconds |

The complete estimate is

$$
\widehat Q_{\mathrm{ML}}
=1.20-0.15
=1.05.
$$

The estimated sampling variance is

$$
0.04+0.01=0.05.
$$

The estimated standard error is

$$
\sqrt{0.05}\approx 0.2236.
$$

The total measured cost is

$$
10+5=15\text{ seconds}.
$$

The positive level-zero mean and negative level-one correction combine into the estimate. The variance calculation uses the uncertainty in each mean, while the cost calculation uses the total work performed at each level.

## 8. Accessing results from the runner

Three runner operations return an `MLMCResult` snapshot:

| Runner operation | Result behavior |
|---|---|
| `run_fixed()` | Starts the run and returns its initial completed snapshot |
| `add_samples()` | Updates active statistics and returns a new snapshot |
| `result` | Returns a new snapshot of the current active statistics without adding samples |

Accessing `result` before a run has started raises an error. The runner does not return an empty `MLMCResult` that could be mistaken for a completed estimator.

The most common run-level values to inspect are:

- `result.finest_level` for the finite-level target;
- `result.estimate` for the MLMC mean estimate;
- `result.estimator_variance` for estimated sampling variance;
- `result.standard_error` for estimated sampling uncertainty;
- `result.total_cost` for measured total correction time; and
- `result.level_results` for the complete level-by-level breakdown.

Within `level_results`, the runner orders entries from level 0 through `finest_level`. A level can therefore be inspected by iterating over the tuple or, for runner-created results, by using its level index. Reading the stored `level` field is still useful in reporting because it makes the identity explicit.

## 9. What happens after `add_samples()`

Suppose an initial result contains sample counts

$$
(N_0,N_1,N_2)=(100,20,5).
$$

The runner then receives an additional request

$$
(\Delta N_0,\Delta N_1,\Delta N_2)=(0,10,5).
$$

Its mutable `LevelStatistics` counts become

$$
(100,30,10).
$$

The new `MLMCResult` reflects those updated counts and statistics. The earlier result still contains

$$
(100,20,5).
$$

The earlier snapshot does not refer back to the mutable accumulators, so later updates cannot alter it.

This behavior makes it safe to retain intermediate results for convergence tables, plotting, comparisons, or progress reports.

## 10. Passing results to other code

`MLMCResult` and `LevelResult` are the appropriate objects to pass to plotting, reporting, persistence, or analysis code. They contain only scalar statistical values and an immutable tuple of level snapshots.

Passing result snapshots instead of the runner has several benefits:

- downstream code cannot accidentally launch more model solves;
- it cannot mutate the runner's active accumulators;
- it does not need access to the model or solver;
- it does not retain large solution arrays; and
- the result remains stable if the runner later adds samples.

The package does not yet define a dedicated file format or serialization API for results. If a result is saved externally, the chosen format should preserve the level ordering, integer counts, floating-point values, and the possibility of `NaN` variance values in manually created or incomplete statistics.

### 10.1 Metadata that is not stored in `MLMCResult`

The current result records statistical outputs, but it is not a complete experiment manifest. It does not store:

- the base seed;
- the model class or model parameters;
- the solver or solver configuration;
- software versions;
- the date, machine, or execution backend;
- individual random samples;
- individual correction values; or
- sample indices.

If results must be independently reproducible or audited later, this metadata should be recorded alongside the `MLMCResult` by the surrounding experiment or documentation layer.

The sample counts can be recovered from the individual `LevelResult.sample_count` fields, and the selected target is recorded by `finest_level`. The remaining experiment configuration must currently be stored separately.

## 11. Information that cannot be recovered from a result

Because the package intentionally stores online summaries rather than raw observations, an `MLMCResult` cannot reconstruct:

- the original sequence of correction samples;
- histograms or empirical quantiles of those samples;
- correlations not represented by the stored statistics;
- individual fine and coarse quantities of interest;
- individual solver failures or successful solver diagnostics;
- solution vectors; or
- exact per-sample timing values.

This is a deliberate memory tradeoff. The stored statistics are sufficient for the current MLMC estimate, sampling variance, standard error, cost summary, and later variance-and-cost-based allocation.

Applications that require raw samples or detailed diagnostics will need a separate, explicitly designed logging or persistence mechanism. The runner should not begin retaining every solution array merely to support that use case.

## 12. Validation and interpretation cautions

### Fewer than two samples

An unbiased sample variance is undefined with fewer than two observations. `RunningStatistics` reports `NaN` in that case. The current fixed runner avoids this in successful initial runs by requiring at least two samples at every active level.

### Nonfinite values

Nonfinite correction observations are rejected by `RunningStatistics`. Nonfinite or negative elapsed times are rejected by `LevelStatistics`. Invalid values should not silently enter the returned result.

### Sampling error is not total error

`standard_error` describes estimated sampling uncertainty in $\widehat Q_{\mathrm{ML}}$. It does not include the finite-level bias

$$
\mathbb{E}[Q-Q_L].
$$

A complete root-mean-square error analysis must account for both sampling variance and discretization bias.

### Timing is observational

Cost values describe what was measured during this run. They are useful for profiling and future sample allocation, but they can change with machine load, hardware, process scheduling, caching, and the execution backend.

### A snapshot is not an accumulator

`LevelResult` and `MLMCResult` cannot be updated with new samples. To continue a run, the original active `MLMCRunner` must still exist and receive `add_samples()`.

## 13. Summary

The four classes form a deliberate progression:

1. `RunningStatistics` updates one scalar mean and variance online.
2. `LevelStatistics` owns correction and cost accumulators for one $Y_\ell$.
3. `LevelResult` freezes one correction level into a lightweight scalar snapshot.
4. `MLMCResult` combines all level snapshots and derives complete run-level quantities.

The key distinction is:

$$
\text{mutable execution state}
\quad\longrightarrow\quad
\text{immutable returned result}.
$$

The statistics classes make long runs memory-efficient. The result classes make completed states safe to inspect and pass around. Together they let the runner continue sampling without changing previously returned estimates.
