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
preferred, but `.tar.gz` archives are also accepted.  The following code block
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

### Closed-Source Submissions

While contestants are strongly encouraged to open-source their solutions at the
conclusion of the contest, there is no requirement to do so. In such cases,
it is still necessary to use the flow described above to produce a binary only
submission.
