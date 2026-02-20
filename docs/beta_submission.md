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

Contestants are required to submit a zip file containing a clone of the contest
repository which has been modified to run their submission.
Specifically, organizers must be able to run the submission by calling only
the `make run-submission` target.  More details to follow.

### Closed-Source Submissions

While contestants are strongly encouraged to open-source their solutions at the
conclusion of the contest, there is no requirement to do so. In such cases,
it is still necessary to use the flow described above to produce a binary only
submission.
