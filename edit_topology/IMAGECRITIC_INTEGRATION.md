# ImageCritic Integration for Texture and Text Consistency

This document maps *The Consistency Critic: Correcting Inconsistencies in
Generated Images via Reference-Guided Attentive Alignment* to the Edit Topology
pipeline. It is an implementation plan, not a claim that ImageCritic is already
running in the main demo.

## What ImageCritic contributes

ImageCritic is a reference-guided local post-editor. Its correction model uses a
Detail Encoder and attention alignment on a FLUX.1-Kontext-based pipeline. A VLM
can detect or select inconsistent regions in an agent loop, but the VLM is not
the component that transfers the fine texture, logo, or text detail.

The official interface takes:

- a reference image and reference bounding box;
- the generated image and correction bounding box;
- a short object caption;
- a seed.

It returns a corrected local crop that is composed back into the generated image.
The official guidance recommends the smallest useful crop, enough surrounding
context, and matching aspect ratios for the two boxes. Low-resolution details may
need a larger contextual crop or multiple correction rounds.

## Position in our pipeline

```text
instruction + source
        |
        v
Edit Topology Contract
        |
        v
ICEdit or FluxFill first-pass output
        |
        v
Consistency detector
  - VLM semantic check
  - OCR comparison
  - local appearance score
        |
        v
Reference/correction crop builder
        |
        v
ImageCritic local correction
        |
        v
protected-region verification + bounded compositing
```

This stage should run only when the first-pass edit is semantically correct but a
fine detail is inconsistent. It is not the right backend for changing the edit
topology or replacing a substantially different whole object.

## Contract-role mapping

| Contract role | ImageCritic use |
| --- | --- |
| `target` | May supply the correction box when the intended target exists but its detail is wrong. |
| `dependent` | Defines the maximum correction/compositing boundary around seams, contact, or local texture transition. |
| `context` | Supplies nearby lighting, scale, material, and repeating-pattern evidence. |
| `protected` | Supplies reference details that must survive, and defines pixels that correction must not overwrite. |

The correction box may include `target + dependent`, but final compositing must
remain inside that allowed union. Context is readable evidence, not blanket edit
permission.

## Reference policy by issue type

### Texture, material, logo, or identity preservation

Use an undamaged source/reference crop containing the correct detail. Examples:

- source garment texture -> edited garment texture;
- original face identity -> generated face;
- clean product logo -> distorted generated logo;
- a correct repeating facade panel -> damaged neighboring panel.

The reference and correction crops should cover corresponding semantic units and
use similar aspect ratios.

### Text preservation

Use ImageCritic for visual glyph/style transfer only after OCR establishes the
expected string. Verification must compare OCR text after correction; visual
similarity alone cannot prove that spelling is correct.

### Entity removal

Do **not** use the original target crop as the reference. That would encourage the
removed object to reappear. For removal, use one of:

1. a clean neighboring instance of the exposed surface;
2. an external clean reference of the same surface;
3. the best FluxFill removal round as the correction input, with surrounding
   non-target texture as reference.

If no trustworthy clean surface exists, keep FluxFill as the reconstruction
backend and use the critic only to rank boundary consistency.

## Sidecar correction task

Keep ImageCritic runtime fields outside contract schema `0.2.0` until the backend
works end to end. Store a sidecar JSON beside each generated result:

```json
{
  "issue_type": "texture",
  "reference_image": "source.png",
  "generated_image": "planned_round_01.png",
  "reference_bbox": [120, 80, 260, 220],
  "correction_bbox": [118, 82, 258, 222],
  "object_caption": "gray stone facade panel",
  "allowed_region": "target_plus_dependent",
  "seed": 0
}
```

Required validation before inference:

- both boxes are non-empty and within image bounds;
- aspect-ratio difference is below a declared tolerance;
- the correction box intersects the contract's failed region;
- the correction box does not exceed `target + dependent` without approval;
- removal tasks reject references that contain the removed target.

## Implementation stages

### Stage 1: Manual research adapter

1. Install the official ImageCritic repository in a separate environment.
2. Download FLUX.1-Kontext, `detail_encoder.safetensors`, and `lora.safetensors`.
3. Accept two images, two boxes, caption, and seed through a small CLI.
4. Save the corrected crop, composed image, and sidecar metadata.
5. Test texture/logo cases before adding automatic detection.

### Stage 2: Contract-aware local correction

1. Read target, dependent, context, and protected roles from the contract.
2. Convert the painted mask to a correction bounding box with controlled context
   expansion.
3. Let the user select a reference box from the source or a separate reference.
4. Composite only inside the approved mask, restoring protected pixels exactly.

### Stage 3: Detection and verification

1. Use OCR for text-string preservation.
2. Use a VLM for semantic mismatch and reference/correction box proposals.
3. Use local appearance metrics for texture/reference ranking.
4. Run ImageCritic only on failed regions.
5. Re-check the corrected result and stop after a small bounded retry count.

## Environment boundary

Do not install the official requirements directly into the current ICEdit
environment. The official repository pins a Git revision of Diffusers and a
specific Gradio version, while this repository currently uses Diffusers `0.33.0`
and its own ICEdit pipeline. Use a separate environment such as:

```bash
python3.10 -m venv ~/.venvs/imagecritic
source ~/.venvs/imagecritic/bin/activate
git clone https://github.com/HVision-NKU/ImageCritic external/ImageCritic
python -m pip install -r external/ImageCritic/requirements.txt
```

The first integration should call that environment through a CLI/subprocess and
exchange image paths plus JSON. Merge Python environments only after dependency
compatibility is demonstrated.

## Licensing

The official ImageCritic repository is licensed under CC BY-NC 4.0 for
non-commercial use. Record this constraint in experiment and release notes, and
obtain permission before any commercial use.

## Primary sources

- Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Ouyang_The_Consistency_Critic_Correcting_Inconsistencies_in_Generated_Images_via_Reference-Guided_CVPR_2026_paper.html
- Official code: https://github.com/HVision-NKU/ImageCritic
- Project page: https://ouyangziheng.github.io/ImageCritic-Page/
