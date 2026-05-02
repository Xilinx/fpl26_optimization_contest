# Beta Submission

The beta submission round will evaluate team submissions who successfully passed the alpha submission.
The scores of the beta submission will be used to determine the finalists who will progress onto making a final submission;
those teams will receive additional AWS and OpenRouter credits in preparation for the final submission.
Contestants will receive private feedback from the organizers assessing the
performance of just their submission on the released benchmark suite (plus a hidden
benchmark) when run on the contest [runtime environment](runtime.html).

## Key Details

* Beta submission is mandatory for continued participation in the contest
* Performance of beta submissions will also be shared privately with contestants and will be used to determine the finalists
* Beta submissions will be evaluated on the contest [runtime environment](runtime.html)

## Submission Format

Contestants are required to submit an archive containing a clone of the contest
repository which has been modified to run their submission.  Zip files are
preferred, but `.tar.gz` archives are also accepted.  Submission archives are
limited to a maximum size of 4 GB (2^32 bytes).  The following code block
illustrates the commands that will be run on the verification instance to
evaluate a submission:

```
unzip submission.zip   # or: tar -xzf submission.tar.gz
cd fpl26_optimization_contest
make setup
make run_optimizer DCP=benchmark1.dcp
make run_optimizer DCP=benchmark2.dcp
make run_optimizer DCP=...
```

The `make setup` target is one of the initial steps that teams can update to
install any additional packages or perform any other one-time preparation
required before their submission is run.  The `make run_optimizer` target
will then be invoked once per benchmark DCP in the evaluation suite.

### Output DCP Location

For each `make run_optimizer DCP=<input>.dcp` invocation, the evaluation
harness will look for the optimized output in the **same directory as the
input DCP**, using the filename pattern:

```
<input_stem>_optimized*.dcp
```

This matches the default location produced by the example `dcp_optimizer.py`
(e.g. given `fpl26_contest_benchmarks/benchmark1.dcp`, the default output is
`fpl26_contest_benchmarks/benchmark1_optimized-<YYYYMMDD_HHMMSS>.dcp`).  A
fixed filename without a timestamp (e.g. `benchmark1_optimized.dcp`) is
also accepted.

If multiple files matching this pattern exist in the input directory at the
end of the run, the **most recently modified** file (by mtime) will be the
one validated and scored; any others are ignored.  As noted on the
[Scoring Criteria](score.html) page, the per-benchmark wall-clock runtime
is capped (see the [runtime environment](runtime.html)), so teams are
encouraged to overwrite or refresh this output file each time their agent
finds an improved solution — the last best result on disk is what will be
scored.

### OpenRouter API Key

When `make run_optimizer` is invoked by the organizers, the `OPENROUTER_API_KEY`
environment variable will be set to a key provisioned by the contest organizers
with a **$1.00 (USD) spending limit per benchmark**.  Submissions must read
this environment variable to access OpenRouter and must not bundle, hard-code,
or otherwise rely on a different API key.

The exact instructions for uploading the submission archive will be emailed
directly to teams.

### Closed-Source Submissions

While contestants are strongly encouraged to open-source their solutions at the
conclusion of the contest, there is no requirement to do so. In such cases,
it is still necessary to use the flow described above to produce a binary only
submission.
