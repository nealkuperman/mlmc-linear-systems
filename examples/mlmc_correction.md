# Understanding one MLMC correction

The module [`mlmc_correction.py`](../src/mlmc_linear_systems/mlmc_correction.py) performs the complete calculation of one sampled MLMC correction. It is the bridge between a user-defined multilevel model and the repeated sampling performed by `MLMCRunner`.

The correction module is not itself a multi-sample MLMC runner. It does not choose sample counts, manage sample indices, accumulate means or variances, or form the final MLMC estimate. Instead, each call to `compute_sample_correction()` performs one self-contained random experiment at one correction level.

The relationship is:

<pre>
MLMCRunner
    |
    | chooses one (level, sample_index) task
    | and constructs its deterministic RNG
    v
compute_sample_correction()
    |
    | evaluates one coupled correction
    v
SampleCorrection
    |
    | correction value and elapsed time
    v
LevelStatistics
</pre>

## 1. What is one correction?

Suppose a multilevel model provides scalar approximations

$$
Q_0,Q_1,\ldots,Q_L,
$$

where level 0 is the coarsest level and level $L$ is the selected finest level.

The MLMC correction variables are defined by

$$
Y_0=Q_0
$$

and

$$
Y_\ell=Q_\ell-Q_{\ell-1},
\qquad \ell>0.
$$

These definitions produce the telescoping identity

$$
Q_L
=Q_0+(Q_1-Q_0)+\cdots+(Q_L-Q_{L-1})
=\sum_{\ell=0}^{L}Y_\ell.
$$

The function `compute_sample_correction()` calculates one realization of one $Y_\ell$. It does not calculate the mean $\mathbb{E}[Y_\ell]$. Estimating that mean requires many independent calls at the same correction level.

For example, if the runner requests 100 samples at correction level 2, it calls the correction calculation 100 times. Each call returns one realization of

$$
Y_2=Q_2-Q_1.
$$

The runner then uses `LevelStatistics` to estimate the mean and variance of those 100 correction values.

## 2. Inputs to the correction calculation

`compute_sample_correction()` receives four inputs:

| Input | Responsibility |
|---|---|
| `model` | Defines how randomness is sampled, how adjacent inputs are coupled, how each linear system is built, and how each solution becomes a scalar quantity of interest |
| `fine_level` | Identifies the requested correction $Y_\ell$ |
| `rng` | Supplies the random stream for this one correction sample |
| `solver` | Solves each `LinearSystem` constructed by the model |

The name `fine_level` is important. For a positive correction, it is not the only level that will be evaluated. It identifies the finer member of an adjacent pair:

$$
\text{fine level}=\ell,
\qquad
\text{coarse level}=\ell-1.
$$

Thus:

| `fine_level` | Correction | Systems solved |
|---:|---|---|
| 0 | $Y_0=Q_0$ | Level 0 only |
| 1 | $Y_1=Q_1-Q_0$ | Levels 1 and 0 |
| 2 | $Y_2=Q_2-Q_1$ | Levels 2 and 1 |
| $\ell$ | $Y_\ell=Q_\ell-Q_{\ell-1}$ | Levels $\ell$ and $\ell-1$ |

The function first verifies that `fine_level` is one of the levels provided by the model. No randomness is drawn and no system is built if the requested level is unavailable.

## 3. One shared random realization

The most important mathematical responsibility of the correction calculation is preserving the coupling between adjacent levels.

For one correction sample, the function asks the model to draw randomness exactly once. Conceptually, this produces one random object

$$
\omega^{(i)}_\ell,
$$

where $\ell$ is the correction level and $i$ is the sample index assigned by the runner.

The model then uses this one random object to construct the fine and coarse model inputs. The two inputs may have different dimensions and representations, but they must describe the same underlying random event.

For example, the shared randomness might be:

- one scalar random variable;
- a vector of standard-normal coefficients;
- one white-noise realization;
- a random field represented on a common background grid; or
- a custom object containing several coupled random quantities.

The model decides how to turn that randomness into level-specific inputs. The correction function only enforces the required fine/coarse structure:

- At level 0, the coarse input must be `None`.
- At every positive level, a coarse input must be present.

The correction function never asks the model to sample separate fine and coarse randomness. If it did, the difference would contain unrelated random variation and would generally have much larger variance.

## 4. Level-zero execution path

At level zero, the correction is

$$
Y_0=Q_0.
$$

The complete calculation follows these steps:

1. The model draws one random realization using the supplied RNG.
2. The model constructs the level-zero fine input.
3. The correction function verifies that there is no coarse input.
4. The model builds the level-zero `LinearSystem`.
5. The configured solver solves that system.
6. The correction function verifies that the solve succeeded.
7. The model evaluates $Q_0$ from the solution and the matching model input.
8. A `SampleCorrection` is returned with no coarse result.

The returned correction value is simply

$$
\texttt{sample.value}=Q_0.
$$

Only one system solve is required because there is no level below zero.

## 5. Positive-level execution path

For a positive `fine_level` $\ell$, the correction is

$$
Y_\ell=Q_\ell-Q_{\ell-1}.
$$

The complete calculation is:

1. The model draws one shared random realization using the supplied RNG.
2. The model constructs a fine input for level $\ell$ and a coupled coarse input for level $\ell-1$.
3. The correction function verifies that the coarse input exists.
4. The model builds the fine linear system at level $\ell$.
5. The configured solver solves the fine system.
6. The correction function verifies that the fine solve succeeded.
7. The model evaluates $Q_\ell$ from the fine solution and fine input.
8. The model builds the coarse linear system at level $\ell-1$.
9. The same configured solver solves the coarse system.
10. The correction function verifies that the coarse solve succeeded.
11. The model evaluates $Q_{\ell-1}$ from the coarse solution and coarse input.
12. A `SampleCorrection` containing both evaluations is returned.

The returned correction value is calculated as

$$
\texttt{sample.value}
=Q_\ell-Q_{\ell-1}.
$$

The fine and coarse systems are allowed to have different matrix sizes. Their relationship comes from their coupled random inputs, not from having identical algebraic representations.

## 6. Division of responsibilities during one correction

Several package components participate in one correction, but each owns a different decision.

| Component | Owns |
|---|---|
| `MultilevelModel` | Random sampling, fine/coarse input coupling, linear-system construction, and quantity-of-interest evaluation |
| `compute_sample_correction()` | The order of one correction calculation, fine/coarse structural checks, solver calls, failure propagation, and correction timing |
| `SystemSolver` | The numerical algorithm used to solve one supplied system, including tolerances, preconditioning, and iteration limits |
| `SampleCorrection` | The returned data from one completed correction |
| `LevelStatistics` | Online mean, variance, and cost accumulation for repeated corrections at one level |
| `MLMCRunner` | Base seed, sample identity, sample counts, repeated execution, and final estimator construction |

This separation allows the same correction calculation to be used with different models, solvers, sampling policies, and future execution backends.

## 7. Solver behavior and failure handling

The correction function treats the solver as a callable that transforms one `LinearSystem` into one `LinearSolveResult`.

The same configured solver is used for the fine and coarse systems. This does not require both systems to take the same number of iterations or have the same solve time. Each `LinearSolveResult` records diagnostics for its own system.

After every solve, the correction function checks the result's `success` flag.

If the fine solve fails:

- the fine quantity of interest is not evaluated;
- the coarse system is not solved; and
- a `RuntimeError` is raised with the fine level and solver message.

If the coarse solve fails:

- the fine solve and fine quantity may already have been completed;
- the coarse quantity of interest is not evaluated; and
- a `RuntimeError` is raised with the coarse level and solver message.

An unsuccessful correction is never returned as though it were a valid sample. This prevents the runner from silently inserting a failed solve into the MLMC statistics.

Invalid fine/coarse coupling raises `ValueError`. This includes a model returning a coarse input at level zero or failing to return a coarse input at a positive level.

## 8. What `SampleCorrection` contains

`compute_sample_correction()` returns an immutable `SampleCorrection` with the following fields:

| Field | Meaning |
|---|---|
| `fine_level` | Correction level $\ell$ |
| `fine_qoi` | Fine quantity $Q_\ell$ |
| `coarse_qoi` | Coarse quantity $Q_{\ell-1}$, or `None` at level zero |
| `fine_solve_result` | Fine solution and solver diagnostics |
| `coarse_solve_result` | Coarse solution and solver diagnostics, or `None` at level zero |
| `elapsed_time` | Total measured time for the complete correction calculation |
| `value` | Derived value $Q_0$ or $Q_\ell-Q_{\ell-1}$ |

The correction value is a property rather than separately stored state. At level zero it returns `fine_qoi`. At a positive level it subtracts `coarse_qoi` from `fine_qoi`.

This avoids storing a redundant value that could disagree with the two quantities from which it is calculated.

The complete fine and coarse `LinearSolveResult` objects are retained in `SampleCorrection`. This is useful for direct calls, testing, debugging, and inspecting solver behavior. It allows a caller to examine information such as convergence, iteration counts, residuals, and solution vectors.

The multi-sample runner does not retain these large objects. It passes each completed correction to `LevelStatistics`, which extracts only the scalar correction value and elapsed time. After that update, the temporary `SampleCorrection` and its solution arrays can be released.

## 9. What the elapsed time measures

The correction timer begins after the requested level has been validated and before the model samples its randomness. It ends after all required quantities of interest have been evaluated.

Therefore, `elapsed_time` includes:

- sampling the shared randomness;
- constructing coupled model inputs;
- building the fine system;
- solving the fine system;
- evaluating the fine quantity of interest;
- building and solving the coarse system when required; and
- evaluating the coarse quantity of interest when required.

It does not include work performed by the runner before or after the call, such as constructing the task's RNG, updating online statistics, or assembling the final `MLMCResult`.

The correction's `elapsed_time` is also different from the `solve_time` stored inside each `LinearSolveResult`. A solve time measures only one linear solve, while the correction time measures the complete one-sample model evaluation.

This total correction cost is the quantity needed by later MLMC allocation policies. Those policies must account for the entire cost of obtaining one $Y_\ell$, not only the time spent inside the numerical solver.

## 10. Conceptual example at correction level 2

Consider one call with `fine_level=2`.

The requested random variable is

$$
Y_2=Q_2-Q_1.
$$

The supplied RNG is used once to produce one underlying random realization. The model converts it into a level-2 fine input and a level-1 coarse input. It then builds two systems:

$$
A_2u_2=b_2
$$

and

$$
A_1u_1=b_1.
$$

The configured solver solves both systems. The model reduces the solutions to

$$
Q_2=Q_2(u_2)
$$

and

$$
Q_1=Q_1(u_1).
$$

The returned sample exposes

$$
Y_2=Q_2-Q_1.
$$

This is still only one observation of $Y_2$. The correction function does not know whether the runner will request 2, 20, or 20,000 level-2 samples. It also does not know how this correction will be combined with corrections from levels 0, 1, or any finer level.

## 11. How the runner uses the correction function

The runner assigns each task a correction level and sample index. Together with the base seed, these values identify a deterministic RNG:

$$
(\text{base seed},\ell,i)
\longrightarrow
\text{sample RNG}.
$$

The runner then supplies that RNG to `compute_sample_correction()`. The correction function uses it for exactly one coupled sample and returns a `SampleCorrection`.

For a fixed run with sample counts

$$
(N_0,N_1,\ldots,N_L),
$$

the runner requests $N_\ell$ independent corrections at each level $\ell$. It accumulates their sample means

$$
\overline{Y}_\ell
=\frac{1}{N_\ell}
\sum_{i=0}^{N_\ell-1}Y_\ell^{(i)}
$$

and constructs the estimator

$$
\widehat Q_{\mathrm{ML}}
=\sum_{\ell=0}^{L}\overline{Y}_\ell.
$$

The correction function is therefore the complete engine for one sample, while `MLMCRunner` is the orchestration layer for the full collection of samples.

## 12. Summary

One call to `compute_sample_correction()`:

- evaluates exactly one MLMC correction sample;
- draws model randomness exactly once;
- preserves fine/coarse coupling through one shared realization;
- solves one system at level zero or two adjacent systems at a positive level;
- evaluates the matching scalar quantities of interest;
- stops immediately if a required solve fails;
- measures the complete cost of obtaining the correction; and
- returns the correction together with detailed solver diagnostics.

It deliberately does not:

- own the base seed;
- assign sample indices;
- decide how many samples to run;
- accumulate correction statistics;
- calculate the final MLMC estimator; or
- decide when an accuracy tolerance has been reached.

Those repeated-sampling and estimator responsibilities belong to `MLMCRunner` and the statistics layer.
