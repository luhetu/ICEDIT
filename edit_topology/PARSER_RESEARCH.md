# Parser Completion and Research Plan

This document defines what it means to finish the first parser before building a
benchmark or adding more image-generation stages.

## Decision

The conservative v0.2 parser supports exactly one atomic edit and asks for
clarification instead of silently changing the meaning of an instruction. Do
not use image quality as evidence that the parser is correct.

The parser contract is complete for the current OmniEdit task taxonomy when it can:

- represent all nine current topologies: insert entity, attach entity, modify
  attribute, remove entity, remove attribute, replace entity, replace
  background, modify environment, or apply style;
- separate the positive edit from `keep`, `leave untouched`, `without changing`,
  and negated edit clauses;
- extract the target, count, selectors, anchor, relation, and explicit
  protections for supported instructions;
- mark unclear counts, pronouns, multiple targets, multiple operations, and
  unsupported operations as unresolved;
- produce a schema-valid contract for inspection while preventing execution
  whenever `clarification.needed` is true;
- avoid claiming that a text-only rule has visually located an object or mask.

The rule parser separates selector scope across source target, anchor, placement,
and replacement; it also blocks negated-only, multi-operation,
coordinated-target, unsupported-operation, under-specified count, and ambiguous
replacement-cardinality cases. These are development acceptance tests, not a
benchmark.

## System Boundary

```text
instruction
    |
    v
rule parser: deterministic draft and safety gate
    |
    v
LLM/VLM planner: language normalization and image-dependent evidence
    |
    v
semantic validator: cross-field and execution-safety checks
    |
    v
visual grounder: referring expression -> boxes -> masks
    |
    v
executor: ICEdit or masked removal backend
    |
    v
verifier: requested change, protected content, OCR, and boundary checks
    |
    v
optional ImageCritic/local retry: reference-guided detail repair
```

The parser should describe an edit. It should not generate the image, invent a
mask, or declare the output successful.

## LLM, VLM, and Grounding Roles

| Component | Good at | Must not be trusted to do alone |
| --- | --- | --- |
| Rule parser | Stable operation words, clause polarity, contract defaults, safe refusal | Resolve which visually similar instance is meant |
| Text LLM | Paraphrase normalization, clause decomposition, operation taxonomy | Count or locate objects that are only visible in the image |
| VLM | Scene evidence, referring expressions, visible attributes, candidate anchors | Produce a pixel-accurate mask or bypass contract validation |
| Grounding model | Convert a target query into boxes and segmentation masks | Decide the user's complete editing intent |
| Verifier VLM/detector | Compare source and output and explain a failure | Replace deterministic pixel, OCR, or mask checks where those are available |

The recommended model path is **draft plus restricted patch**:

1. The local parser creates a deterministic draft and detects unsafe language.
2. A VLM may patch only image-dependent grounding fields such as instance IDs,
   region roles, ambiguity evidence, and provenance. It cannot rewrite topology,
   selector scope, target semantics, desired state, or relation predicates.
3. Local schema and semantic validators check the merged contract.
4. The executor refuses unresolved contracts.

This is safer and easier to compare than asking each provider to invent the
entire contract independently.

## What Description We Need

A long general image caption is optional. The useful descriptions are local and
structured:

- `referring_expression`: the user's literal target phrase;
- `source_state`: what is visible before the edit;
- `desired_state`: the smallest requested difference;
- `anchor` and `relation`: where the target is or should be;
- `protected`: explicit content that must remain unchanged;
- `grounding_evidence`: candidate box or mask, confidence, and model provenance.

`source_state` and `desired_state` are v0.2 fields. Grounding evidence currently
lives in region roles, ambiguities, and provenance. A verbose scene description
should not be appended blindly to the generation prompt because it can
reintroduce a removed object as a positive token.

## Relevant Research

| Work | Useful idea for this project | Where it belongs |
| --- | --- | --- |
| [SGEdit](https://arxiv.org/abs/2410.11815) | Represent objects, attributes, relations, masks, and controlled scene-graph edits | Contract design and graph-delta planning |
| [InstructEdit](https://arxiv.org/abs/2305.18047) and [official code](https://github.com/QianWangX/InstructEdit) | Separate the language processor, segmentation query, source/target description, mask model, and editor | Parser-grounder-executor separation |
| [AnyEdit](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_AnyEdit_Mastering_Unified_High-Quality_Image_Editing_for_Any_Idea_CVPR_2025_paper.html) | A broad edit taxonomy beyond the nine current topologies | post-v0.2 operation backlog |
| [Planning, Reasoning, and Generation](https://openaccess.thecvf.com/content/ICCV2025/html/Ji_Instruction-based_Image_Editing_with_Planning_Reasoning_and_Generation_ICCV_2025_paper.html) | Decompose complex instructions, reason about a region, then generate each step | Future `operations[]` and dependencies |
| [Focus on Your Instruction](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_Focus_on_Your_Instruction_Fine-grained_and_Multi-instruction_Image_Editing_by_CVPR_2024_paper.html) | Associate each instruction with its own region of interest | Target/dependent/protected regions |
| [RefEdit](https://openaccess.thecvf.com/content/ICCV2025/html/Pathiraja_RefEdit_A_Benchmark_and_Method_for_Improving_Instruction-based_Image_Editing_ICCV_2025_paper.html) | Referring expressions are a distinct hard problem in scenes with similar entities | Grounding acceptance cases |
| [Grounded SAM](https://arxiv.org/abs/2401.14159) and [official code](https://github.com/IDEA-Research/Grounded-Segment-Anything) | Open-vocabulary text grounding followed by segmentation | Automatic box and mask stage |
| [DICE](https://openaccess.thecvf.com/content/ICCV2025/html/Baraldi_What_Changed_Detecting_and_Evaluating_Instruction-Guided_Image_Edits_with_Multimodal_ICCV_2025_paper.html) | Detect localized differences and judge whether they match the instruction | Post-execution add/remove/change verification |
| [ImageCritic](https://arxiv.org/abs/2511.20614) | Reference-guided local correction for inconsistent fine details | Optional post-verification repair, not parsing |

The papers agree on a useful design principle: complex editing works better when
intent planning, spatial localization, generation, and evaluation are explicit
stages. The exact models can change without changing the contract boundary.

## Implementation Order

### Milestone 1: v0.2 language and contract safety (complete)

- Keep the new clause polarity and atomicity gate.
- Keep semantic validation beyond JSON Schema, including count consistency,
  unresolved ambiguity consistency, required mask fields, and topology-specific
  relations.
- Turn the existing failure examples into small development acceptance cases.
- Keep one contract per edit; do not add `operations[]` yet.

### Milestone 2: restricted VLM grounding (adapter complete, grounding evaluation pending)

- Send the source image, original instruction, and deterministic draft.
- Require evidence for every image-dependent patch.
- Record provider, model, prompt version, image resize, latency, and request ID.
- Expose candidate target and anchor boxes in the demo for user confirmation.

### Milestone 3: automatic mask proposal

- Convert the contract's literal referring expression into a Grounded-SAM query.
- Preserve multiple candidates instead of selecting silently.
- Let the user accept or correct the mask before execution.

### Milestone 4: verification

- Verify the operation type: removal must not become addition.
- Compare changes inside and outside the allowed region.
- Add OCR checks for protected text and deterministic pixel checks where exact
  compositing is expected.
- Use a VLM or DICE-like model for semantic failures that fixed metrics cannot
  describe.

### Milestone 5: post-v0.2 complex planning

Add `operations[]`, operation IDs, `depends_on`, and one grounding record per
operation. Then support move, spatial-relation-only edits, resize, rotation, and
multi-step instructions. Replacement and material/attribute changes already fit
the single-operation v0.2 contract.

## Later Provider Comparison

Do this only after the v0.2 contract and acceptance cases are frozen. Compare:

1. local rules;
2. text LLM;
3. direct VLM contract generation;
4. local draft plus restricted VLM patch.

Measure contract validity, field accuracy, unsafe-execution rate, clarification
quality, latency, and cost. Image aesthetics are a separate executor evaluation.
