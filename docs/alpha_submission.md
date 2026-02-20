# Alpha Submission

In order to ensure that the contest environment is able to support all 
entries ahead of the beta and final submission deadline, a mandatory step for continued
participation in the contest is the submission of an early "alpha" release.
The performance of this alpha submission will have **zero** effect on the
final submission score; instead, the organizers will endeavour to work with
contestants to ensure that the runtime environment is as desired.
Contestants will receive private feedback from the organizers assessing the
performance of just their submission on the released benchmark suite (plus a hidden
benchmark) when run on the contest [runtime environment](runtime.html).

## Key Details

* Alpha submission is mandatory for continued participation in the contest
* Performance of alpha submissions will be shared privately with contestants and will not impact the final score
* Alpha submissions will be evaluated on the contest [runtime environment](runtime.html)

## Submission Format

Contestants are required to submit a zip file containing a clone of the contest
repository which has been modified to run their submission.
Specifically, organizers must be able to run the submission by calling only
the `make run-submission` target.  More details to follow.

### Closed-Source Submissions

While contestants are strongly encouraged to open-source their solutions at the
conclusion of the contest, there is no requirement to do so. In such cases,
it is still necessary to use the flow described above to produce a binary only
submission.
