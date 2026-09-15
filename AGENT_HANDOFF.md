# ICEdit Research Handoff

Last updated: 2026-09-15 (Europe/London)

This file is the canonical cross-machine context for the project. Read it before
making changes. Chat transcripts are not the source of truth: update this file
when a decision, experiment, result, job state, or blocker changes.

## Objective

Develop a defensible CVPR/NeurIPS-level method around reliable image editing.
The current research scope is intentionally narrowed to object removal: remove
the specified instance and relevant effects while avoiding collateral changes to
nearby people, text, structure, other instances, and their effects.

Do not claim novelty or acceptance. The parser and orchestration are engineering
infrastructure. The paper contribution still requires a mechanism that improves
strong removal baselines under controlled evaluation.

## Current Research Position

- ICEdit (normal LoRA on FLUX.1 Fill) is the current engineering backbone and a
  required baseline, not the SOTA target by itself.
- Direct removal targets: ObjectClear (CVPR 2026) and OmniPaint (ICCV 2025).
- Mechanism comparisons: Attentive Eraser (AAAI 2025) and AdaEraser (2026
  preprint). AdaEraser already monitors residual target presence and adapts
  attention suppression, so that idea cannot be claimed as ours.
- ObjectClear already combines target-aware attention, attention-guided fusion,
  and spatially varying denoising strength. Generic object-plus-effect removal
  and background preservation are not open novelty claims.
- A better-motivated gap is instance-specific collateral-damage control when
  target and protected content overlap, occlude one another, or have entangled
  effects. This is a hypothesis to test, not an established contribution.
- Final hard compositing guarantees exact pixels only outside the chosen mask.
  It does not prove better generation and cannot protect content mistakenly
  included in the editable mask.

Read `edit_topology/experiments/2026-09-14_sota_removal_targets_zh.md` for the
literature and target matrix. Read `edit_topology/experiments/2026-09-14_removal_pilot.md`
for the first removal experiment protocol.

## Implemented Components

- `edit_topology/parser.py`: converts an instruction into a structured edit
  contract. Treat parser output as fallible and validate schema/contracts.
- `edit_topology/executor.py`: compiles contracts into prompts.
- `edit_topology/spatial_mask.py`: source-only box masks, protected-mask
  subtraction, diptych construction, compositing, and diagnostic reference
  difference masks. Reference-difference masks are oracle diagnostics only.
- `edit_topology/trajectory_probe.py`: captures and analyzes FLUX trajectories.
  Separate trajectories are not same-state causal interventions.
- `scripts/research_removal_pilot.py`: 3 seeds x 5 removal conditions; retains
  raw and composited outputs and labels semantic success as unevaluated.
- `scripts/research_contract_trajectory.py`: trajectory diagnostics. It is not a
  completed active constraint projection method.
- `submit_slurm_icedit_removal_pilot.slurm`: submitted as NCC job 1040171.
- `submit_slurm_objectclear_smoke.slurm`: submitted as NCC job 1040182.

## ObjectClear Setup

The official repository is expected at `ObjectClear/` as an upstream checkout:

- Upstream: `https://github.com/zjx0101/ObjectClear.git`
- Pinned code commit used here: `c052d91ecd8772744a5ab97527441c813ab83009`
- Inference model: `jixin0101/ObjectClear`
- Pinned model revision: `c73af80888dbd519819d0a521b5c8d0f3cda6859`
- NCC fp16 checkpoint size: approximately 7.3 GB; never commit it to Git.
- Upstream NTU S-Lab non-commercial research terms apply.

ObjectClear uses SDXL-Inpainting plus a CLIP visual object encoder. The smoke
test uses official sample 1, fp16, 20 steps, seed 42, guidance 1.0, and AGF on.
Submission is not completion: require a successful Slurm exit, summary JSON, and
manual inspection of the output before saying it ran successfully. Job 1040182
failed before pipeline construction because the shared environment imported an
incompatible system `pyOpenSSL` through `boto3/accelerate`. Build a clean
ObjectClear environment or install a compatible Python-level dependency before
retrying; do not report ObjectClear as having run successfully yet.

The NCC environment `/home3/dnrx52/.venvs/objectclear` references the existing
ICEdit site-packages through a `.pth` file. This shortcut exposed incompatible
system packages and is not a validated environment; treat it as failed setup,
not successful isolation.

## Existing Evidence

- Proxy InstructPix2Pix diagnostics: 36 paired removal runs.
- Proxy VAE locality diagnostics: 24 local latent perturbations.
- Proxy decoder pathway diagnostics: 36 ablations with no-edit anchor checks.
- Local actual ICEdit: one 8-step masked smoke run and one 28-step full output.
- Some visible degradation on the proxy case arises from VAE reconstruction
  alone. This does not establish that all ICEdit face damage has the same cause.
- Stronger source preservation can oppose deletion; aggressive source removal can
  produce flat blocks. Lower reference pixel error did not guarantee acceptable
  semantic removal.

Local research artifacts and large outputs are deliberately not Git content.
Historical artifact location: `/home/hetu/MY project/icedit_10h_research/`.
NCC experiment outputs are under `/home3/dnrx52/ICEdit/research_outputs/`.

## Active Jobs and Verification

Job IDs below are historical identifiers and must be queried before reporting:

```bash
ssh ncc1.clients.dur.ac.uk \
  'squeue -j 1040171,1040182 -o "%.18i %.24j %.10T %.10M %.30R"'
ssh ncc1.clients.dur.ac.uk \
  'sacct -j 1040171,1040182 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS'
```

- 1040171: FAILED before generation. Offline LoRA loading requires an explicit
  `weight_name`; the 15 intended generations did not run.
- 1040182: FAILED before pipeline construction due to incompatible system
  `pyOpenSSL` (`module 'lib' has no attribute 'GEN_EMAIL'`).

Do not resubmit automatically after failure. Inspect the `.out` and `.err` logs,
record the cause, patch the smallest issue, and use a new output directory.
Never cancel or modify unrelated jobs.

## Immediate Next Steps

1. Verify replacement ICEdit job 1042473, its logs and outputs. Historical jobs
   1040171/1040182 were reconfirmed FAILED on September 15; do not rerun them.
2. For ObjectClear, compare AGF on/off to separate generation from final fusion.
3. Run ICEdit, native Fill, ObjectClear, OmniPaint, and Attentive Eraser on the
   same small, fixed removal set with matched information budgets.
4. Evaluate target absence, replacement hallucination, protected-region damage,
   object-effect ownership, boundary quality, compute, and raw versus composite.
5. Only after repeated strong-baseline failures, implement the smallest
   instance-protection intervention. Compare with fixed mask expansion,
   fixed attention suppression, crop/inpaint, and equal-compute resampling.

No new benchmark should be built first. Use existing cases and official samples
to determine whether the proposed problem remains real on strong methods.

## Research Integrity Rules

- Record model/checkpoint revisions, code commits, seeds, masks, prompts, steps,
  guidance, precision, GPU, elapsed time, and failures.
- Do not use target/reference images to construct deployable masks or tune the
  method unless explicitly labeled oracle analysis.
- Do not call pixel L1, CLIP, or a VLM score semantic success by itself.
- Compare methods with the same input information. If one receives protected or
  object-effect masks, give compatible baselines the same data or label the
  comparison as a different information budget.
- Report all fixed seeds, negative results, raw outputs, and postprocessing.
- Never commit `.env`, tokens, model weights, datasets, Slurm logs, or large
  generated outputs. Store small summaries/figures only when licensing permits.

## Cross-Machine Start Procedure

On a new machine, clone the personal research repository and tell the agent:

> Read `AGENT_HANDOFF.md`, the two linked experiment notes, and `git status`.
> Verify active job states before changing or resubmitting anything. Continue
> from the first unfinished next step, preserve unrelated edits, and update the
> handoff before committing.

Then configure secrets locally using `.env.example`; never copy `.env` through
Git. Model weights and datasets must be downloaded separately according to their
licenses. Update `AGENT_HANDOFF.md` at the end of every substantial session.

## Git State

The original upstream remote is `https://github.com/River-Zhang/ICEdit.git`.
Keep it as `upstream`. Push research work only to the user's personal repository.
The personal repository is `https://github.com/luhetu/ICEDIT.git`; `origin` uses
SSH and the initial handoff was pushed at commit `96f2468`. The repository is
public. Keep checkpoints, private data, secrets and large outputs excluded.

## September 15: CVPR Progress Session

- User requested progress toward the September 16 tutor report, targeting CVPR.
- Read current logs and Slurm accounting: both old jobs failed before generation;
  no active ICEdit/ObjectClear job existed at the beginning of this session.
- Fixed the pilot's offline LoRA loading with explicit `--lora-weight-name`
  (default `pytorch_lora_weights.safetensors`) and `local_files_only=True`.
  Confirmed that the checkpoint file exists before making the change.
- Dry-run validated all 15 configurations in
  `research_outputs/removal_preflight_20260915`; this validates inputs, not GPU
  execution or generation quality. Python compilation and Slurm syntax passed.
- Submitted replacement job **1042473** with the existing 28-step, three-seed,
  five-condition protocol. Last checked PENDING (Priority). Output directory:
  `research_outputs/removal_pilot_1042473`; logs use `icedit_removal_pilot1042473`.
- Ran local unittest modules `test_spatial_mask`, `test_executor`, and
  `test_contract_schema`: **30 tests passed**. Full test suite not run this turn.
- NCC-only environment change: ObjectClear `pyvenv.cfg` now sets
  `include-system-site-packages = false`, retaining the ICEdit `.pth` path.
  A 120-second pipeline import check timed out; import timing shows slow Torch
  imports. This is still shared-package setup, not a newly validated clean env.
  Do not submit ObjectClear GPU jobs until the complete pipeline import passes.
- Added `edit_topology/experiments/2026-09-15_cvpr_progress_zh.md`: six-slide
  outline, evidence boundaries, proposed mechanism and two-week decision plan.
  Local report copy: `/home/hetu/MY project/ICEdit_CVPR_Progress_2026-09-15.md`.
- Rechecked primary ObjectClear v2, Consistency Critic and AdaEraser pages.
  Instance-ownership control remains a research hypothesis; no new method
  improvement or strong-baseline comparison has been established this session.

## Latest Sync Verification

- Python compilation passed for the new ObjectClear helpers, removal pilot, and parser.
- Bash syntax checks passed for the relevant activation and Slurm scripts.
- `git diff --cached --check` passed after formatting cleanup.
- A staged secret scan found only documented placeholders; a real OpenAI key was
  removed from `.env.example` before the first commit and was never pushed. The
  exposed key should nevertheless be revoked because it existed on NCC storage.
- Full tests are not verified in this sync: the NCC ICEdit environment lacks
  `pytest`, and `unittest discover` stalled during imports and was interrupted.
