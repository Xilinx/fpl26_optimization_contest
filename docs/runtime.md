# Development, Test, and Validation Runtime Environment

## Hardware and Software 

Teams are encouraged to develop as much as possible on their own hardware and environments.  However, to provide a fair playing field for testing and validating team submissions, contest organizers have selected the following platform upon which team submissions are evaluated:

AWS Instance [m7a.2xlarge](https://aws.amazon.com/ec2/instance-types/m7a/):
 * 8 vCPUs: 4th Gen AMD EPYC Processors
 * 32 GB RAM

AWS [Vivado ML 2025.1 Developer AMI](https://aws.amazon.com/marketplace/pp/prodview-evssv7ysyt6h4):
 * Ubuntu 22.02 Operating System
 * Pre-installed and licensed Vivado 2025.1

This platform should comfortably run a sequential workflow without any memory pressure issues. A larger platform with more memory and more CPU cores was avoided to enable teams to focus on optimization innovation rather than brute force parallelized exploration.  

At various stages of the contest, teams may be provided with AWS credit to enable them to validate with little or no cost on the contest validation platform.  Successful alpha and beta submissions may enable teams to earn additional credits.  More details to follow.

## Submission Setup and Time Limit

When a submission is evaluated, it is run on a **freshly provisioned instance** of the platform above that contains **only your submitted archive**.  The evaluator unpacks your submission and runs `make setup` **non-interactively** (as a normal user, through a login shell) to install and build everything your tool needs, before running your optimizer on each benchmark.  Anything you installed or configured by hand on a development instance (extra packages, files outside the archive, environment variables set in your shell, pre-built artifacts) will **not** be present, so `make setup` must be fully self-contained.

`make setup` must complete within **90 minutes**.  If it does not finish in time — or exits with an error — the submission is marked `setup_failed` and receives no score.  This limit is identical for preview and final evaluation.  If your setup builds large components (for example, compiling RapidWright), make sure the full `make setup` completes within this window starting from a clean instance.

## LLM Access

If teams will be using LLMs for their workflow (most will), they will be required to use [OpenRouter](https://openrouter.ai/) as demonstrated in the example `dcp_optimizer.py` agent.  OpenRouter is a unified API interface for a wide range of LLMs.  It provides a unified way to switch between many different LLMs using a single API.  Teams may also be provided with OpenRouter credit ahead of submission checkpoints to help develop and validate their solutions.  More details to follow.

Submissions must read from the environment variable `OPENROUTER_API_KEY` for the access key and use the OpenRouter API to access the models.  No other LLM service will be supported. 

