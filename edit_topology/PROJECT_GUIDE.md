# Edit Topology: Project and Build Guide

This extension turns a free-form image-edit instruction into a small, inspectable
contract before executing the edit. Its purpose is minimal change: identify what
may change, what must stay fixed, and which execution path fits the requested
topology.

## What the system does

The current path is:

```text
source image + instruction
        |
        v
rule parser or one-call VLM planner
        |
        v
validated Edit Topology Contract
        |
        +-- insert / attach / modify --> ICEdit diptych execution
        |
        +-- replace / regional style -> masked ICEdit diptych execution
        |
        +-- remove entity/attribute -> native FluxFill masked inpainting
                                            |
                                            v
                                 exact source compositing outside mask
        |
        +-- global env/style -------> full-image ICEdit execution
```

The contract separates four region roles:

- `target`: the object or attribute directly changed;
- `dependent`: the smallest boundary, contact, shadow, or occlusion area allowed
  to change;
- `context`: visual evidence used to make the edit, but not automatically editable;
- `protected`: people, text, layout, identity, and other content that must remain.

The rule parser is deterministic and works without an API. The OpenAI and Gemini
options make one image-aware call, then pass the result through the same local
schema and semantic validation. Provider output is not executed when it changes
language-derived fields, fails validation, or leaves a required clarification
unresolved.

## Repository map

- `parser.py`: deterministic instruction-to-contract baseline.
- `model_parser.py`: optional OpenAI or Gemini visual grounding.
- `schema/edit_topology_contract.schema.json`: contract boundary.
- `executor.py`: converts validated contracts into backend instructions.
- `spatial_mask.py`: mask extraction, diptych masks, and exact outside-mask
  compositing.
- `IMAGECRITIC_INTEGRATION.md`: concrete reference-guided texture, text, and logo
  correction design based on ImageCritic.
- `PARSER_RESEARCH.md`: parser completion standard, model roles, related research,
  and the post-v0.2 roadmap.
- `parser_demo.py`: lightweight parser-only Gradio app; no GPU model required.
- `scripts/gradio_demo.py`: full parser, mask editor, and image execution app.
- `tests/`: schema, parser, model adapter, executor, and spatial-mask tests.

## 1. Install

From the repository root:

```bash
python3.10 -m venv ~/.venvs/icedit
source ~/.venvs/icedit/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The full image demo also needs these local model directories:

```text
models/flux.1-fill-dev/
models/ICEdit-normal-LoRA/
```

On the current cluster workspace, `bash scripts/finish_setup.sh` prepares the
configured model paths.

## 2. Add an API key

The API is optional. Local rules work without one. A private file already exists
at the repository root:

```text
.env
```

Edit it and set one provider:

```bash
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=
```

or:

```bash
OPENAI_API_KEY=
GEMINI_API_KEY=your_key_here
```

`.env` is ignored by Git. `.env.example` is the safe template that may be
committed. Do not put a real key in source code, JSON contracts, screenshots, or
Slurm submission arguments.

Load the environment locally with:

```bash
source activate_icedit.sh
```

The main Slurm demo script also loads `.env` automatically. Changes to `.env`
take effect when a new process or Slurm job starts; restart an already-running
demo after changing a key.

## 3. Run tests

```bash
python3 -m unittest discover -s edit_topology/tests -v
```

These are development tests, not an image-quality benchmark. They verify schema
and parser behavior, one-call provider adapters, prompt compilation, and spatial
mask invariants.

## 4. Run the lightweight planner

This path does not load Flux or reserve a GPU:

```bash
python3 -m edit_topology.parser_demo --server-name 0.0.0.0 --port 7861
```

Use `Local rules (no image/API)` first. Select `OpenAI VLM` or `Gemini VLM` only
after the corresponding key is loaded.

## 5. Run the full editing demo

On the configured cluster:

```bash
sbatch submit_slurm_icedit_gradio.slurm 7860
```

Find the assigned node:

```bash
squeue -u "$USER" -o "%.18i %.20j %.8T %.20R"
```

If the node is `gpu8`, create the tunnel from your local computer:

```bash
ssh -L 7860:gpu8:7860 dnrx52@ncc1.clients.dur.ac.uk
```

Then open `http://127.0.0.1:7860`.

## 6. Use the planned workflow

1. Upload the source image.
2. Write one literal edit instruction.
3. Choose local rules, OpenAI VLM, or Gemini VLM.
4. Click **Plan / inspect contract**.
5. Inspect `topology`, `target`, `anchor`, clarification, and protected regions.
6. Paint only the target plus the minimum necessary boundary in the mask editor.
7. Choose 3 planned rounds and click **Proceed: run planned rounds**.
8. Inspect every result rather than treating a completed generation as a pass.

Removal, replacement, background replacement, and regional style changes require
a painted mask. Both removal topologies use base FluxFill inpainting because
sending removed content through the ICEdit LoRA can act like a positive
generation token. After removal inpainting, the source image is copied back
everywhere outside the mask, so unrelated pixels remain exact.

## 7. Plan OmniEdit rows without a GPU

Use `plan-only` to inspect dataset contracts before scheduling generation:

```bash
python scripts/inference_dataset.py \
  --parquet train/parquet/train-00000-of-00105.parquet \
  --output-dir dataset_outputs/addition_plans \
  --pipeline-mode plan-only \
  --planner rule \
  --max-samples 20 \
  --prompt-index 1
```

Use `annotate` to keep the current raw ICEdit generation while writing the
contract and planning status beside each output. The released OmniEdit `task`
label is normalized and checked against the parsed topology, but it is only a
hint: rows with mismatched semantics or unresolved clarification remain marked
as blocked in metadata.

Use `planned` when the validated contract should control the generation prompt.
Mask-gated topologies also need `--mask-dir`. Each row then records an explicit
execution status; `generated_unverified` means an image exists but no critic has
yet established that it follows the instruction.

## How to write instructions

Prefer one operation, a grounded target, and explicit preservation:

```text
Remove only the two red signs on the building. Keep the windows, people, and all
other text unchanged.
```

```text
Add exactly one black hat to the woman on the left. Keep her face, hair, clothes,
and the other people unchanged.
```

Avoid combining multiple edits in one instruction. If several similar targets
exist, identify position, count, carrier, or relation. When that is still
ambiguous, the contract should request clarification instead of guessing.

## Current limits and next work

- Masks are painted by the user; automatic VLM grounding and segmentation are
  not connected yet.
- Removal is stochastic. Multiple rounds help, but completion is not verification.
- OCR/text preservation is not yet automatically scored.
- There is no automatic Consistency Critic loop yet. The planned ImageCritic
  backend, reference policy, environment boundary, and correction sidecar are
  specified in [IMAGECRITIC_INTEGRATION.md](IMAGECRITIC_INTEGRATION.md).
- The next useful component is the Stage 1 manual adapter, followed by a verifier
  that checks target removal inside the mask, OCR preservation outside it, and
  boundary consistency before ranking or retrying rounds.
