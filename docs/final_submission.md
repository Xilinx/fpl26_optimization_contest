# Final Submission

## Key Details

* Final submission process/requirements are largely the same as for the [beta submission](beta_submission.html),
  but will be more strictly enforced
  * Please follow this process closely! Although contest organizers will use their best efforts to run all
    submissions, failures may lead to disqualification
* Final submissions will be evaluated on the contest [AWS instance configuration hardware](runtime.html)
  * Contest organizers will use their best efforts to run all submissions one or more (equal) number of times
    and accept their best result
* Submissions will be evaluated solely on a set of hidden benchmarks
* Final results will be made public at the [FPL 2026 conference](https://2026.fpl.org/)

## Runtime Environment

Please see the development and [runtime validation environment](runtime.html) page for details.

## Submission Process

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
