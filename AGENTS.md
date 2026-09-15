# Project Agent Instructions

Before working in this repository:

1. Read `AGENT_HANDOFF.md` completely.
2. Check `git status` and preserve changes you did not create.
3. Verify Slurm job state and outputs before reporting completion or resubmitting.
4. Keep model weights, datasets, logs, credentials, and large generated outputs
   out of Git. Never commit `.env`.
5. Update `AGENT_HANDOFF.md` with durable results, failures, decisions, and the
   first unfinished next step before committing substantial work.

Research claims must distinguish implemented code, measured results, diagnostic
or oracle analysis, and proposed mechanisms. Do not claim novelty solely from a
parser, mask, VLM verifier, hard composite, or generic constrained optimization.
