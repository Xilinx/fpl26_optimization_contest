# Frequently Asked Questions


## Team Questions

### Can undergraduate students not in our research lab be part of a team?

Yes.

### Can recent graduates qualify to be members of a team?

Yes, being a student is not a requirement.

### Is there a constraint on the number of team members?

Currently, the team limit is 6 members (not counting the advisor(s)).

### Can the team have two advisors?

Yes.

### Can we have more than one team?

Yes, as long as team members belong exclusively to a single team.  However, advisors can advise more than one team.

### Can we collaborate with another University as a single team for the contest?

Yes.

## Contest Objective Questions

### Can we use other tools beyond RapidWright and Vivado to create our solutions?

Yes, any existing solutions or solutions built from scratch and/or derived from prior work are welcome.

### How are we going to be evaluated and ranked against other contestants?

All team solutions will be measured using the same criteria, hardware platform, and constraints.  Detailed information about how solutions will be scored and how teams will be ranked is available on the [Scoring Criteria](score.html) webpage.

### Can teams change the placement solution provided in the benchmarks to improve Fmax?

Yes. Teams are encouraged to explore all implementation optimizations techniques as long as the design maintains logical equivalence with the original design.

## Approach and Technique Questions

### Is the use of LLMs (OpenRouter) required?

No. Although the contest is designed with supporting the use of LLMs and an agentic workflow, contestants can develop solutions that do not rely on LLMs. One of the major challenges without LLMs is handling error conditions or recovering from unexpected issues.  

### Transformative technologies, like ML and DL, require large amounts of training data. Will you be providing training data to enable such approaches, or are you only interested in traditional algorithmic approaches?

We welcome and are interested in any ML and DL approaches.  We recognize the need for large amounts of training data and will provide ways of generating many more benchmark designs beyond the examples that are provided.  For example, Vivado can be used to synthesize and place any compatible design onto the contest device, and [RapidWright](https://github.com/Xilinx/RapidWright) used to analyze those results and/or convert that into the FPGA Interchange Format to serve as training data.  


## Submission Questions

### How many submission variants will be permitted from each team?

For the final submission, only the last submission made before the [final submission deadline](index.html#important-dates) will be accepted.
Prior to this, as part of the alpha/beta submission processes we intend to work with contestants to ensure that their submission runs as expected on the validation platform.

## More Questions?

Please post questions in our [Discussion](https://github.com/Xilinx/fpl26_optimization_contest/discussions) forum.
