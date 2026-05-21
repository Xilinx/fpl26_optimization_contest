# Scorecard Reference

After each evaluation run, the contest validation harness produces a `scorecard.json` containing the per-run and per-benchmark results that determined a team's score. Scorecards are distributed to teams alongside their results to detail how their results were interpreted and score calculated.

This page documents every field that may appear on a scorecard, organized into three groups:

1. [Run-Level Fields](#run-level-fields) &mdash; metadata describing the validation run as a whole.
2. [Benchmark-Level Fields](#benchmark-level-fields) &mdash; per-benchmark measurements and outcomes.
3. [Validation Check Fields](#validation-check-fields) &mdash; the individual pass/fail checks that gate scoring.

A short summary of the [scoring formula](#scoring-formula) is included at the end for convenience; full scoring rules live on the [Scoring Criteria](score.html) page.

> ℹ️ **NOTE**  
> Some fields are diagnostic and may be `null` when not applicable (for example, `failure_reason` on a successful run). Cost and runtime fields are reported with the raw measured values; any caps or saturations are indicated by companion boolean fields such as `gamma_capped`.

---

## Run-Level Fields

These fields describe the validation run as a whole and are emitted once per scorecard.

| Field | Description |
|---|---|
| `validation_id` | Unique ID for this validation attempt. |
| `round` | Contest round being validated, for example `alpha`. |
| `validator_git_sha` | Git commit of the trusted official validation tools bundle used for functional validation. |
| `total_score` | Sum of all benchmark scores. Failed benchmarks contribute `0`. |
| `failure_reason` | Run-level failure reason if validation failed before completing benchmark evaluation. Usually `null` for completed runs. |
| `status` | Overall validation status, such as `completed` or `failed`. |

---

## Benchmark-Level Fields

Each entry in the `benchmarks` array describes the result for one benchmark.

### Identity & Status

| Field | Description |
|---|---|
| `name` | Benchmark name. |
| `benchmark_sha256` | Composite SHA256 checksum of the benchmark manifest. |
| `input_dcp_sha256` | SHA256 of the benchmark `input.dcp` file, equivalent to running `sha256sum input.dcp`. |
| `status` | Benchmark outcome. `scored` means the benchmark completed validation and was scored; `failed` means a required step failed. |
| `produced_output_dcp` | `true` if the validation harness found an output DCP for this benchmark. |
| `failure_reason` | Benchmark-level failure reason if the benchmark did not receive a positive score. May also be `no_improvement` when validation passed but Fmax did not improve. |

### Timing Measurements

| Field | Description |
|---|---|
| `fmax_input_mhz` | Baseline Fmax of the input DCP, in MHz. |
| `fmax_output_mhz` | Measured Fmax of the submitted output DCP, in MHz. This is `0.0` if no valid Fmax measurement was produced. |
| `wns_ns` | Worst negative slack from timing analysis, in nanoseconds. |
| `whs_ns` | Worst hold slack, in nanoseconds. |

### Scoring Inputs

| Field | Description |
|---|---|
| `alpha_fmax_improvement_mhz` | Fmax improvement in MHz. Computed as `fmax_output_mhz - fmax_input_mhz` when a valid Fmax measurement exists. |
| `beta_openrouter_cost_usd` | OpenRouter API spend for this benchmark, in USD. Used as a score penalty. |
| `gamma_runtime_hours` | Runtime of `make run_optimizer` for this benchmark, in hours. Capped at `1.0` for scoring. |
| `score` | Final score for this benchmark after applying improvement, API-cost penalty, and runtime penalty. |

### Diagnostics

| Field | Description |
|---|---|
| `validation` | Object containing detailed pass/fail checks for routing, DRC, timing, and functional validation. See [Validation Check Fields](#validation-check-fields). |
| `wall_time_seconds` | Actual wall-clock runtime of `make run_optimizer`, in seconds. |
| `gamma_capped` | `true` if runtime reached the per-benchmark cap and the runtime penalty was saturated. |
| `openrouter_metering_failed` | `true` if OpenRouter usage could not be measured reliably. |

---

## Validation Check Fields

The `validation` object on each benchmark contains the gating pass/fail checks. A benchmark must pass all required checks to be eligible for a positive score.

| Field | Description |
|---|---|
| `par_routed` | `true` if the output DCP is fully routed according to Vivado `report_route_status`. |
| `par_drc_clean` | `true` if Vivado DRC checks passed. |
| `hold_passed` | `true` if hold timing checks passed. |
| `pulse_width_passed` | `true` if pulse-width checks passed. |
| `sim_passed` | `true` if trusted functional validation passed using the official `validate_dcps.py` flow. |

---

## Scoring Formula

For a benchmark with `status: "scored"`, the score is computed as:

```
score = max(0, alpha - 0.1 * alpha * beta - 0.1 * alpha * gamma)
```

where:

| Symbol | Source field | Meaning |
|---|---|---|
| `alpha` | `alpha_fmax_improvement_mhz` | `fmax_output_mhz - fmax_input_mhz` |
| `beta`  | `beta_openrouter_cost_usd`    | OpenRouter spend, in USD |
| `gamma` | `gamma_runtime_hours`         | Optimizer runtime, in hours (capped at `1.0`) |

A benchmark receives a score of `0` if any of the following are true:

- It fails a required validation check (see [Validation Check Fields](#validation-check-fields)).
- It does not produce an output DCP.
- It does not improve Fmax over the input.

For full scoring rules, ranking methodology, and tie-breaking, see the [Scoring Criteria](score.html) page.
