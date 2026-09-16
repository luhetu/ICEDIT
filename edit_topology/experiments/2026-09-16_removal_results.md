# Removal pilot: verified September 16

NCC job 1042473 completed successfully (exit 0, elapsed 00:16:19).
All 15 per-run records and raw/composite images are present under
`research_outputs/removal_pilot_1042473/`; `gallery.html` is the visual index.
One scene, five conditions, seeds 731001/731002/731003, 28 steps, guidance 50.

## Measurements

Mean raw-output L1 outside the coarse editable mask, RGB normalized to [0,1]:

| Condition | Mean L1 |
|---|---:|
| ICEdit full | 0.05176 |
| ICEdit masked | 0.02246 |
| ICEdit masked, desired-state prompt | 0.02264 |
| ICEdit masked, source target occluded | 0.02258 |
| Native Fill | 0.03567 |

Masked ICEdit reduces this pixel-change measure by approximately 57% relative
to full ICEdit on this scene. This is not a face-identity metric or a removal
success score. All composites report zero changed pixels outside the mask by
construction; this is not an improvement in the generative model.

## Visual inspection

Agent inspected all 15 raw outputs at their saved resolution on September 16.
This is a qualitative, non-blinded review, not a benchmark success annotation.

- Full ICEdit: all three seeds generate new/replacement signs or lettering;
  substantial changes to people are also visible.
- Masked ICEdit: all three retain or regenerate prominent rooftop signage.
- Desired-state prompt: rooftop lettering persists across all three seeds;
  the lower sign region sometimes becomes a visibly different texture.
- Reference occlusion: all three still produce prominent replacement signs.
  Occluding source pixels is an out-of-distribution diagnostic, not causal
  proof that a particular attention pathway is responsible.
- Native Fill: rooftop signage is removed in all three inspected images, but
  background reconstruction varies and people outside the mask visibly lose
  detail. Seed 731002 also has residual/synthetic lettering in the lower region.

These results support testing deletion and preservation jointly. They do not
show superiority over ObjectClear or other specialized removal models.
The coarse lower box overlaps visible content near raised hands/faces, so
hard compositing alone cannot guarantee preservation inside that box.

## Next controlled run

Run ObjectClear official sample 1 with AGF on/off, seed 42, 20 steps, guidance
1.0. Use separate output folders keyed by Slurm job ID. Code inspection shows
the AGF flag changes both first-step latent blending and final pixel fusion:
this comparison must not be called a pure postprocessing ablation.
Only submit after complete pipeline import passes in the intended environment.
