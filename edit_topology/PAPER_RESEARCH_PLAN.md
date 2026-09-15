# ICEdit Paper Plan: Active-Set Contract Projection

Literature snapshot: 2026-07-18.

This document replaces the earlier router-first proposal. It is a research
hypothesis and implementation plan, not a claim that the method or results
already exist.

## One-Minute Version

### We are here

The project already has:

- a conservative edit-contract parser;
- a validated JSON contract with target, dependent, context, and protected roles;
- optional LLM/VLM grounding adapters;
- prompt compilation, spatial masks, exact outside-mask compositing, and a demo;
- saved add/remove failure cases, including the Coca-Cola reappearance case;
- 87 passing local tests as of 2026-07-18.

These components make the system usable, but they are not yet a CVPR/NeurIPS
method. The missing piece is a generation mechanism that actually executes the
contract inside denoising.

### The proposed idea

> **An edit is a constrained state change: the target must change, while every
> protected contract clause should not get worse.**

Current editors use one global instruction signal. That signal can remove the
target while damaging the background, or preserve the source so strongly that
the removed object reappears. We propose **Active-Set Contract Projection
(ACP)**. At selected denoising steps, ACP detects which contract clauses are in
conflict with the proposed edit update and projects the update into a locally
feasible direction.

In simple terms:

```text
normal editor says: move this way
contract says: this move will break preserved text and building structure
ACP says: keep the useful part of the move and remove the harmful part
```

### Three paper contributions

1. **Contract conflict diagnosis.** Show that many editing failures are caused
   by a measurable conflict between target progress and invariant preservation
   inside the denoising trajectory.
2. **Active-Set Contract Projection.** Compile open-vocabulary edit clauses into
   spatial constraints and perform small, training-free constrained updates at
   selected denoising steps.
3. **Evidence-driven correction.** A VLM returns clause-level failure evidence,
   not one overall score. Only failed constraints are activated in one correction
   pass, while passed hard invariants remain protected.

The parser, VLM, mask, verifier, and texture loss support this idea. None should
be presented as the main novelty by itself.

## Why the Previous Direction Was Too Weak

The earlier plan centered on a contract-conditioned counterfactual router. The
diagnosis remains useful, but routing is now a crowded claim:

- AttnRouter already selects attention operations by edit category and localizes
  useful layer/time bands.
- CARE-Edit dynamically routes Text, Mask, Reference, and Base experts.
- OrionEdit uses orthogonal subspaces and causal information-flow masks.
- BindEdit studies source-dominance leakage and attention rebalancing.
- Vision-Language Binding traces direct and indirect reference paths.

Therefore, `contract + router` could look like an incremental combination. ACP
asks a different question: not "which path should be enabled?", but "what is the
closest edit update that still satisfies the currently active obligations?"

Two other tempting claims are also too weak alone:

- A VLM planner/verifier/reflection loop overlaps with DICE, Edit-R1,
  VisionDirector, MIRA, and IMAGAgent.
- Texture or reference-detail correction overlaps with Consistency Critic.

We can use both, but they must feed a new constrained generation mechanism.

## Core Scientific Hypothesis

Let `z_t` be the current latent, `d_edit` the update proposed by the normal
editor, `L_E` the target edit loss, and `L_i` an invariant-violation loss.

An edit update conflicts with invariant `i` when

```text
conflict_i = <grad L_i(z_t), d_edit> > 0
```

because moving along `d_edit` is predicted to increase the violation. The main
hypothesis is:

> Failed edits contain repeatable, spatially localized conflict events, and
> resolving only the active conflicts produces a better target/preservation
> trade-off than global guidance scaling, masks, compositing, or static routing.

This is falsifiable. If conflict scores do not correlate with visible failures,
or projection cannot improve the Pareto frontier, the paper idea should stop.

## Method

### 1. Compile an executable contract

The existing parser produces the semantic contract. A runtime compiler converts
it into clause records:

```json
{
  "clause_id": "remove_coca_cola_sign",
  "type": "change",
  "predicate": "coca-cola sign is absent",
  "role": "target",
  "hard": true,
  "mask_uri": "...",
  "proxy": "localized_concept_absence"
}
```

For the same example, separate invariant clauses could be:

```text
preserve building geometry       -> DINO/edge feature constraint
preserve people and identity     -> source feature constraint
preserve all other visible text  -> OCR/text-region constraint
allow exposed wall reconstruction -> editable dependent region
```

This separation matters. A single binary mask cannot say that the sign texture
must disappear while the wall geometry and nearby text must remain.

### 2. Build clause losses and evidence

Use the simplest reliable proxy for each clause:

| Clause | Differentiable proxy used during denoising | Final evidence |
| --- | --- | --- |
| add/remove/replace concept | localized SigLIP/CLIP or detector feature | detector plus VLM |
| preserve identity/semantics | masked DINO feature distance | VLM/identity model |
| preserve geometry | low-frequency latent, edge, or depth distance | pixel/edge/depth tool |
| preserve text | text-region feature and source glyph structure | OCR exact/soft match |
| preserve texture/detail | high-frequency or patch feature distance | patch similarity/VLM |
| count/relation | localized object-slot surrogate | detector/VLM count/relation |

The VLM is not assumed to be differentiable or perfectly reliable. It grounds
clauses, proposes evidence regions, and checks the generated result. Dense,
differentiable proxies steer the inner denoising loop.

### 3. Detect the active set

At a small number of denoising checkpoints, predict the clean image `x0_hat` and
evaluate target and invariant residuals. Activate clause `i` only when:

- it is already violated;
- the proposed update is predicted to worsen it; or
- the outer verifier failed it in the previous pass.

Inactive constraints add no preservation pressure. This avoids the common
failure where a globally strong preservation signal blocks the requested edit.

### 4. Project the update

Let `A_t` be the active hard-invariant set. Solve:

```text
minimize_d    1/2 ||d - d_edit||^2 + lambda ||slack||_1

subject to    <grad L_i, d> <= epsilon_i + slack_i,  i in A_t
              <grad L_E, d> <= -margin + slack_E
              ||d|| <= trust_radius
              slack >= 0
```

This finds the closest useful update that makes target progress without locally
worsening active invariants. The slack variables keep the problem feasible when
two clauses genuinely conflict.

For the first prototype, do not solve a huge pixel-space program. Project the
latent update with sequential half-space projections or solve for a few
coefficients over these candidate fields:

```text
edit field
source-reconstruction field
target-suppression field
context-reconstruction field
```

The coefficient problem is tiny and can be solved per role and timestep band.

### 5. Use roles, not only target masks

Apply different permissions to the existing topology roles:

| Role | Permission |
| --- | --- |
| target | large semantic change is allowed |
| dependent | local shadow, contact, hole filling, and boundary change are allowed |
| context | supplies geometry, lighting, and texture evidence |
| protected | hard or high-penalty no-regression constraints |

This is especially important for removal. The source target may help locate the
sign early, but target appearance must not survive in the final output. ACP does
not assume that source information is always helpful or harmful; it activates a
constraint when the proposed update produces a measurable violation.

### 6. Add one verifier-guided correction pass

After the first output, ask the VLM and deterministic tools to return:

```json
{
  "clause_id": "remove_coca_cola_sign",
  "status": "fail",
  "confidence": 0.94,
  "failure_type": "removed_concept_reappeared",
  "evidence_boxes": [[0.18, 0.03, 0.73, 0.31]]
}
```

The second pass does not receive a vague prompt such as "try again". The failed
clause and evidence box become an active constraint. Passed hard invariants are
kept in the feasible set. Accept the repair only when the failed clause improves
and no passed hard invariant regresses beyond tolerance.

### 7. Treat texture correction as one specialist constraint

Consistency Critic is useful for reference-specific texture and detail. In this
paper, texture preservation should be one clause proxy, not the headline:

- preserve source texture in protected regions;
- align external-reference detail only where the contract permits it;
- suppress old target texture for removal/replacement;
- reconstruct exposed surface texture from context.

This gives the paper a principled place for attentive alignment without copying
Consistency Critic's contribution.

## ICEdit Implementation

Use ICEdit's FLUX Fill diptych path for the first mechanism study because source
and edited panels coexist in one model input. Keep the current production rule
unchanged: native FluxFill remains the removal backend in
`scripts/inference_dataset.py`.

Create a separate research runner with the following stages:

1. Parse and validate the contract.
2. Ground target/dependent/protected regions.
3. Run the normal ICEdit trajectory and save predicted-clean checkpoints.
4. At selected steps, compute clause residuals, gradients, conflict scores, and
   the projected update.
5. Save the raw output before any exact compositing.
6. Run clause verification and optionally one correction pass.
7. Save a full trace: seed, prompt, masks, active clauses, conflict matrix,
   projection coefficients, raw output, composited output, and verification.

The first code should expose these research objects:

```text
ContractClause
ClauseProxy
ActiveSetState
ProjectionTrace
VerificationEvidence
```

Do not change the stable parser schema first. Store research-only fields in a
runtime sidecar until the mechanism survives the pilot.

## The First Convincing Experiment

Use the Coca-Cola failure as the opening figure:

```text
source -> "remove Coca-Cola signs; keep building and all other text"
```

Show five columns:

1. source plus role masks;
2. raw ICEdit failure;
3. target/invariant conflict heat map over denoising time;
4. the update component removed by projection;
5. ACP raw output and clause results.

Then show a matched operation quartet using the same concept:

```text
add sign | remove sign | replace sign | preserve sign while changing wall color
```

The point is not that "Coca-Cola" is difficult. The point is that the same visual
evidence has different legal effects under different contracts, and ACP enforces
those effects without a fixed operation table.

## Minimal Falsification Study

Do not create a benchmark. Start with 24 diagnostic cases already available or
easy to assemble:

- 8 removal cases;
- 6 addition/attachment cases;
- 6 replacement/attribute cases;
- 4 text or texture-preservation stress cases.

Use fixed seeds and inspect raw outputs. Compare:

1. base ICEdit;
2. contract-compiled prompt;
3. spatial mask plus exact compositing;
4. global guidance/LoRA-scale search;
5. static edit-category routing or source attenuation;
6. PCGrad or simple gradient clipping;
7. ACP without the VLM correction pass;
8. full ACP with one evidence-driven correction pass.

Report separate axes rather than one score:

- target-clause success;
- protected-clause regression;
- raw outside-target drift;
- edit footprint size;
- verifier false-pass/false-fail on the 24 cases;
- runtime and memory overhead.

### Go/no-go criteria

Continue toward a paper only if all of these are observed:

1. Conflict scores predict visible clause failures better than random or raw
   attention magnitude.
2. ACP moves the target-success/preservation Pareto frontier, rather than merely
   choosing a different fixed guidance scale.
3. The gain appears in at least three edit topologies, not removal alone.
4. Raw outputs improve before exact compositing.
5. Shuffled masks, shuffled clauses, and random projection directions lose the
   gain.

Stop or change the claim if conflict is not reproducible, if masks/compositing
explain all gains, or if a static category rule matches ACP.

## Four-Week Plan Before Scaling

### Week 1: prove the phenomenon

- Freeze 12 cases and fixed seeds.
- Add predicted-clean checkpoints and per-role feature losses.
- Plot target progress, invariant residuals, and gradient conflict over time.
- Determine whether the Coca-Cola failure has a measurable conflict signature.

Deliverable: one trajectory figure and a yes/no decision on contract conflict.

### Week 2: build the smallest ACP

- Implement one target loss and one DINO/pixel invariant loss.
- Add half-space projection at 3-5 denoising checkpoints.
- Compare against guidance-scale search and PCGrad.
- Test raw removal, addition, and attribute-edit outputs.

Deliverable: a 12-case comparison grid and Pareto plot.

### Week 3: make clauses real

- Add protected text/OCR and context-texture proxies.
- Use the existing target/dependent/context/protected roles.
- Add VLM clause evidence with `pass/fail/uncertain` and boxes.
- Implement one correction pass with no-regression acceptance.

Deliverable: full trace JSON and interactive demo view for one case.

### Week 4: falsify and present

- Expand to 24 cases and three or four topologies.
- Run shuffled-clause, shuffled-mask, and static-routing controls.
- Measure runtime and verifier errors.
- Prepare the tutor deck and decide whether the idea deserves full evaluation.

Deliverable: method figure, main result table, ablation table, and failure page.

Only after this decision should the project run public benchmarks. No new
benchmark is needed for the method paper unless existing sets cannot test a
specific contract property.

## Tutor Pitch

### Six-slide structure

1. **Problem:** editors understand the noun but violate the edit contract.
2. **Evidence:** Coca-Cola removal reintroduces the forbidden sign and changes
   protected text/building appearance.
3. **Gap:** prompts, masks, routers, and VLM retries do not enforce simultaneous
   change and preservation obligations inside denoising.
4. **Idea:** compile clauses into an active constrained update and project only
   when a proposed step would worsen an invariant.
5. **Experiment:** show conflict over time, projected component, raw output, and
   the target/preservation Pareto plot.
6. **Risk-controlled plan:** four weeks, 24 cases, explicit kill criteria, no new
   benchmark construction.

### One-sentence contribution claim

> We introduce a training-free active-set projection method that compiles
> open-vocabulary edit contracts into denoising-time constraints, improving the
> requested change while preventing localized invariant regression.

This wording should remain provisional until the literature audit and pilot are
complete.

## CVPR and NeurIPS Positioning

For a CVPR submission, emphasize:

- a strong visual failure that masks and prompt rewriting do not fix;
- localized role-aware constraints;
- raw-output improvements on multiple editors and edit types;
- compelling trajectory and conflict visualizations.

For a NeurIPS submission, add:

- a precise constrained-flow formulation;
- analysis of feasibility, slack, and active-set stability;
- a first-order no-regression result under smooth proxy losses and a trust
  region;
- generalization across backbones and clause proxies.

The realistic first target is a CVPR-style method paper. A NeurIPS-level claim
requires the optimization formulation to be more than an engineering wrapper.

## Novelty Boundary and Required Reading

The following papers are closest and must be compared directly:

| Work | What it already covers | Our required difference |
| --- | --- | --- |
| [AttnRouter](https://arxiv.org/abs/2605.01480) | category-specific attention operations | clause-level active constraints, not a routing table |
| [CARE-Edit](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_CARE-Edit_Condition-Aware_Routing_of_Experts_for_Contextual_Image_Editing_CVPR_2026_paper.html) | dynamic multimodal expert routing | project an update against explicit invariants |
| [OrionEdit](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_OrionEdit_Bridging_Reference_and_Source_Images_for_Generalized_Cross-Image_Editing_CVPR_2026_paper.html) | orthogonal reference/source spaces and flow masks | optimization over open-vocabulary edit obligations |
| [CogniEdit](https://openaccess.thecvf.com/content/CVPR2026/html/Li_CogniEdit_Dense_Gradient_Flow_Optimization_for_Fine-Grained_Image_Editing_CVPR_2026_paper.html) | dense trajectory-level reward optimization | hard/soft invariant constraints and active-set projection |
| [PROUD](https://arxiv.org/abs/2407.04493) | Pareto-guided multi-objective diffusion | image-edit contracts, spatial roles, and source-state invariants |
| [AugCLIP](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_Preserve_or_Modify_Context-Aware_Evaluation_for_Balancing_Preservation_and_Modification_CVPR_2025_paper.html) | context-aware edit/preserve evaluation | use separate clause constraints to control generation |
| [LUSD](https://openaccess.thecvf.com/content/ICCV2025/html/Chinchuthakun_LUSD_Localized_Update_Score_Distillation_for_Text-Guided_Image_Editing_ICCV_2025_paper.html) | localized gradient filtering and normalization | contract-driven multi-constraint conflict resolution |
| [Consistency Critic](https://openaccess.thecvf.com/content/CVPR2026/html/Ouyang_The_Consistency_Critic_Correcting_Inconsistencies_in_Generated_Images_via_Reference-Guided_CVPR_2026_paper.html) | reference detail alignment and local correction | texture is one constraint, not the method claim |
| [DICE](https://openaccess.thecvf.com/content/ICCV2025/html/Baraldi_What_Changed_Detecting_and_Evaluating_Instruction-Guided_Image_Edits_with_Multimodal_ICCV_2025_paper.html) | localized source/output difference evaluation | evidence activates an inner constrained update |
| [VisionDirector](https://openaccess.thecvf.com/content/CVPR2026/html/Chu_VisionDirector_Vision-Language_Guided_Closed-Loop_Refinement_for_Generative_Image_Synthesis_CVPR_2026_paper.html) | VLM-guided stepwise refinement and rollback | typed no-regression constraints inside denoising |

Also read CausalSliders and Path-independent Flow Matching before making any
claim about edit algebra, causal ordering, or commuting edit paths. Those ideas
can be future extensions, but they should not dilute the first paper.

## Main Risks

### Proxy mismatch

A DINO/CLIP/OCR proxy may disagree with human judgment. Report this disagreement,
use several typed tools, and do not call the constraint a semantic guarantee.

### Over-constrained editing

Hard preservation can block the desired edit. Use active sets, slack, a trust
region, and explicitly editable dependent regions.

### Compute cost

Gradient-based clean-image checks can be expensive. Start at 3-5 checkpoints,
cache source features, and project over a low-dimensional field basis.

### VLM unreliability

Use `uncertain`, require localized evidence, combine deterministic tools, and do
not use the same VLM family for planning and final judging.

### ICEdit-specific artifact

The Coca-Cola problem may come from the diptych LoRA. After the pilot, repeat the
core experiment on native FluxFill or a second MMDiT editor before claiming a
general mechanism.

## Immediate Next Work

The next repository work is a research harness, not more parser coverage:

1. Add a runtime contract-clause sidecar and typed trace schema.
2. Expose predicted-clean outputs at selected FLUX denoising steps.
3. Implement one target proxy, one invariant proxy, and conflict logging.
4. Run the 12-case Week 1 diagnosis without changing production inference.
5. Only implement projection after the conflict plot supports the hypothesis.
