# Edit Topology Contract specification

This directory contains the training-free specification, parser, planner adapters,
and spatial execution helpers for topology-aware minimal-change image editing.
Start with [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for installation, API-key setup,
architecture, and end-to-end demo instructions.

## Layout

- `schema/edit_topology_contract.schema.json`: JSON Schema Draft 2020-12 contract.
- `examples/`: 20 hand-reviewable contracts derived from the documented ICEdit
  addition and removal failure cases.
- `parser.py`: conservative v0.2 rule parser for nine single-edit topologies. It
  records image paths but does not visually parse images.
- `model_parser.py`: optional one-call OpenAI or Gemini image-aware planner.
- `executor.py`: topology-specific prompt compilation and execution checks.
- `spatial_mask.py`: painted-mask extraction, diptych construction, and exact
  source compositing outside removal masks.
- `IMAGECRITIC_INTEGRATION.md`: planned reference-guided post-correction for
  texture, text, logo, and other fine-detail inconsistencies.
- `PARSER_RESEARCH.md`: v0.2 completion criteria, LLM/VLM roles, paper comparison,
  and staged implementation plan.
- `PAPER_RESEARCH_PLAN.md`: Active-Set Contract Projection research thesis,
  tutor pitch, four-week falsification study, ablations, and go/no-go criteria.
- `TUTOR_PITCH_ZH.md`: concise Chinese explanation of the proposed method,
  novelty boundary, first experiment, and tutor presentation story.
- `trajectory_probe.py`: matched FLUX trajectory masks and edit-energy metrics.
- `experiments/2026-07-23_contract_trajectory_zh.md`: first Coca-Cola
  trajectory-probe results, limitations, and next experiment decision.
- `OMNIEDIT_TASK_MAPPING.md`: dataset-label to contract-topology mapping, local
  shard coverage, and task-aware runner policy.
- `tests/test_contract_schema.py`: schema and example validation tests.

The examples are annotations of intended edits, not claims about facts visible in
an image. Image-dependent claims use `provenance.source = "image"`; unresolved
facts are represented in `ambiguities` and `clarification`.

## Contract trajectory probe

The first research runner compares full-panel and target-masked ICEdit
trajectories using a matched seed:

```bash
sbatch submit_slurm_icedit_contract_trajectory.slurm
```

It saves final images, predicted-clean checkpoints, FLUX velocity tensors,
role masks, pixel metrics, a CSV trace, and comparison figures under
`research_outputs/contract_trajectory_coca_cola_v3/`.

The current edit-energy metric measures where and how strongly a condition
changes the trajectory relative to matched reconstruction. It does not determine
whether that change is semantically correct; a clause-specific semantic proxy is
the next research step.

## Validate

Install the test-only dependency and run:

```bash
python3 -m pip install -r edit_topology/requirements-test.txt
python3 -m unittest discover -s edit_topology/tests -v
```

## Trial parser

Single instruction:

```bash
python3 -m edit_topology.parser \
  --image examples/images/hat_people.png \
  --instruction "Add a hat to the man on the left."
```

Batch sample plus review table:

```bash
mkdir -p edit_topology/examples/generated
python3 -m edit_topology.parser \
  --input-jsonl edit_topology/examples/parser_trial_inputs.jsonl \
  --output edit_topology/examples/generated/parser_trial_contracts.jsonl \
  --review-csv edit_topology/examples/generated/parser_trial_review.csv
```

This parser is a deterministic baseline. It supports insert, attach, attribute
change, entity/attribute removal, object/background replacement, environment
change, and style application. Every output passes JSON Schema and cross-field
semantic validation; the review CSV makes topology/target/anchor errors visible.

## Interactive planner demo

The existing ICEdit Gradio app contains an experimental **Edit Contract Planner**
accordion. The offline rule parser is the default. Put one provider key in the
ignored root `.env` file before submitting the Gradio job:

```bash
OPENAI_API_KEY=...   # OpenAI VLM option
# or
GEMINI_API_KEY=...   # Gemini VLM option
```

The Slurm script loads `.env` automatically:

```bash
sbatch submit_slurm_icedit_gradio.slurm
```

The optional model override is blank by default. Current defaults live in
`edit_topology/model_parser.py`, so model changes do not require UI edits. The
provider output is rejected unless it preserves language-derived contract fields
and passes the local schema and semantic validators. The VLM may ground instance
references and regions; it cannot rewrite topology, target semantics, selector
scope, desired state, or relation predicates. There are no automatic retries or
hidden verifier calls.

The demo exposes and executes the contract. Entity and attribute removal require
a user-painted target mask and use native FluxFill inpainting; output pixels
outside the mask are restored exactly from the source. Object/background
replacement and regional style edits also require a mask and use masked ICEdit.
Global environment/style and local insert/attach/modify paths use ICEdit with
concise contract guidance. Automatic grounding, segmentation, RAD-style
scheduling, and verifier-driven correction remain future work.

## Lightweight parser demo

To inspect the parser without loading FLUX or reserving a GPU, run:

```bash
python3 -m edit_topology.parser_demo --server-name 0.0.0.0 --port 7861
```

Open `http://127.0.0.1:7861` locally, or forward port 7861 when running on a
remote node. The first tab parses one instruction and exposes the contract,
planned region roles, and schema result. The second tab runs a small set of
deterministic development smoke cases. These checks are not a benchmark.

Semantic masks remain `pending` until the user paints a spatial target in the full
demo. Automatic visual grounding and segmentation are not connected yet.

The full GPU-backed Gradio app keeps **Run original instruction** for the upstream
behavior. **Proceed: run planned rounds** validates the editable contract, blocks
unresolved clarification, enforces topology-specific mask requirements, selects
the execution backend, and saves the contract, compiled prompt, backend, and seed
beside each output.

The planned path also supports 1--6 consecutive seeds. Set **Planned rounds**,
plan or edit the contract, then choose **Proceed: run planned rounds**. Every
round appears in a Gallery and its contract, compiled prompt, and seed are
saved in `gradio_results/`.

## OmniEdit dataset planning

The dataset runner preserves `raw` as its default baseline. `annotate` runs the
same raw ICEdit prompt and adds a validated contract to metadata. `plan-only`
parses and validates contracts without loading Flux or using a GPU. `planned`
compiles the validated contract and uses that prompt for generation:

```bash
python scripts/inference_dataset.py \
  --parquet train/parquet/train-00000-of-00105.parquet \
  --output-dir dataset_outputs/addition_plans \
  --pipeline-mode plan-only \
  --planner rule \
  --task addition \
  --max-samples 20 \
  --prompt-index 1
```

Use `--planner openai` or `--planner gemini` after loading the corresponding
key. Each metadata file records the raw/normalized task, contract, validation,
clarification, and whether the parsed topology is compatible with the dataset
label. Unknown, mismatched, invalid, and unresolved rows are marked not ready;
`planned` blocks them, while `annotate` deliberately retains raw baseline
generation. The task label is a hint and never proof of instruction semantics.

Every row also records a separate `execution` object. Its status is one of
`not_requested`, `blocked_by_plan`, `blocked_contract_compile`,
`blocked_missing_mask`, `blocked_invalid_mask`, `running`,
`generated_unverified`, or `failed`. A `running` record may remain after an
interrupted job. A generated image remains unverified until a critic is run;
generation alone is never reported as a semantic pass.

Planned removals, replacements, background replacements, and regional style
edits require a lossless PNG spatial mask. Supply masks with `--mask-dir`; filenames may be
either `<omni_edit_id>.png` or `<row_index>_<omni_edit_id>.png`. Missing masks
produce `blocked_missing_mask` metadata and do not call the image model. Empty
and whole-image masks produce `blocked_invalid_mask`. Masks
must come from user annotation or an independent grounding system, not from the
reference output image.

```bash
python scripts/inference_dataset.py \
  --parquet train/parquet/train-00000-of-00105.parquet \
  --output-dir dataset_outputs/addition_planned \
  --pipeline-mode planned \
  --planner rule \
  --task addition \
  --flux-path models/flux.1-fill-dev \
  --lora-path models/ICEdit-normal-LoRA \
  --max-samples 10 \
  --prompt-index 1
```

The Slurm wrapper exposes the same controls without changing its positional
arguments:

```bash
PIPELINE_MODE=plan-only TASK_FILTER=removal PLANNER=rule \
  sbatch submit_slurm_icedit_dataset.slurm \
  train/parquet/train-00115-of-00571.parquet dataset_outputs/removal_plans 20
```

For mask-gated planned jobs, add `MASK_DIR=/path/to/masks`. Summary fields keep
planning, generation, blocking, and metadata-only counts separate, so
`completed` no longer needs to be interpreted as an image-quality result.

## Old failure-case rounds

Five source/reference/old-output cases are listed in
`edit_topology/examples/failure_cases.json`. Run matching original and
contract-guided seeds on the cluster with:

```bash
sbatch submit_slurm_icedit_failure_rounds.slurm 3
```

The job writes all images and metadata under `failure_rounds/` and builds
`failure_rounds/comparison_gallery.html`. Successful generation does not imply
that the planned result is better; this page is intended for direct visual
inspection of edit success and preservation failures.

The schema version is `0.2.0`. Before `1.0.0`, a minor version may include a
documented contract migration; patch releases remain annotation/documentation
corrections. After `1.0.0`, incompatible changes require a major version.
