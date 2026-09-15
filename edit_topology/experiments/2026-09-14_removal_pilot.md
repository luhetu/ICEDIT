# Removal Pilot: Reference Evidence and Edit Locality

## Scope

The user explicitly authorized NCC execution on 2026-09-14, superseding the
earlier local-only restriction. Existing unrelated jobs are not modified.
This is a one-case diagnostic pilot, not a benchmark or a novel method claim.

Submitted Slurm job: `1040171` (FAILED before generation).
Output: `research_outputs/removal_pilot_1040171/`.
Logs: `logs/icedit_removal_pilot1040171.out` and `.err`.
Submission script: `submit_slurm_icedit_removal_pilot.slurm`.
Runner: `scripts/research_removal_pilot.py`.

## Hypotheses

1. Full-frame regeneration contributes to unintended background/person changes.
2. A positive desired-state prompt may help removal compared with the original
   object-naming removal instruction.
3. Source target content can interfere with removal. Occluding it is only an
   OOD input diagnostic: it changes evidence and cannot isolate a semantic
   reference pathway or establish a same-state causal mechanism.

## Design

Existing Coca-Cola sign removal case; three seeds (731001, 731002, 731003),
28 steps, guidance 50, five conditions (15 generations):

| Condition | Generation support | Instruction | Reference |
|---|---|---|---|
| icedit_full | Entire right image | Original removal | Original left image |
| icedit_masked | Coarse target boxes | Original removal | Original left image |
| icedit_masked_desired | Coarse target boxes | Desired facade state | Original left image |
| icedit_reference_occluded | Coarse target boxes | Original removal | Left target boxes gray-filled |
| native_fill | Coarse target boxes, single image | Desired facade state | No diptych; LoRA disabled |

Masks use existing manually specified boxes, not reference-image differences.
The target reference is loaded only for evaluation and display. It does not
influence masks, prompts, or intervention selection. The native Fill condition
has a different latent shape; equal integer seeds do not imply identical noise
across native and diptych configurations.

## Outputs and Acceptance

Every run saves raw generation and final hard composite separately. The runner
asserts exact source equality outside the composite mask. This is a property of
compositing, not a guarantee that all protected objects are outside the mask.
The mask can include foreground people; manual inspection must check that risk.

Pixel metrics report raw outside-source L1, inside-reference L1, and inside-source
L1. None is labeled semantic removal success. `semantic_success` remains
`not_evaluated` until visual/tool assessment is performed. Completion must be
checked from per-run JSONs, scheduler exit status, and the final COMPLETE log.

Review all seeds for target absence, new logos/objects, background plausibility,
person/face damage, and mask-boundary artifacts. Do not select a single favorable
seed as proof. If the pilot is informative, expand to the existing key, person,
and hand-object removal cases after source-only target/protection annotation.

## Preflight

- Remote Python and model environment available: torch 2.7.0+cu126, diffusers 0.33.0.
- All 15 condition/seed inputs validated on NCC without loading the GPU model.
- Local condition construction and occlusion-support assertions passed.
- Hard-composite inside/outside invariants passed.
- Bash syntax check passed.
- Explicit 64 GB memory request rejected by QOS; reverted to existing project's
  default-memory submission convention and passed Slurm test-only validation.
- PEFT/offload inference-tensor issue avoided with `torch.no_grad()`.

## Contribution Boundary

This experiment diagnoses removal failures. It does not implement ACP, a learned
reference router, property disentanglement, or a new verifier. Any subsequent
controller must be compared with fixed schedules and existing source-suppression
methods at matched compute and must improve outputs before hard compositing.

## Scheduler Result

Job 1040171 exited with status 1 after loading Flux components and before any
generation. With Hugging Face offline mode enabled, Diffusers could not infer the
LoRA filename from a directory and required an explicit `weight_name`. Therefore
this job produced zero valid experiment generations and must not be counted as a
completed 15-run pilot.
