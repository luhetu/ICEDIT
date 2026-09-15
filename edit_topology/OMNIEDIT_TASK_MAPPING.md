# OmniEdit task mapping

The released OmniEdit dataset uses six task labels, while the contract uses nine
topologies. Dataset labels are routing hints, not parser ground truth.

| Dataset task | Allowed contract topologies | Runtime note |
| --- | --- | --- |
| `addition` | `insert_entity`, `attach_entity`, `modify_attribute` | An instruction that adds hair/text changes an existing carrier rather than inserting an independent object. |
| `removal` | `remove_entity`, `remove_attribute` | Both require a target mask and use native fill without repeating the removed content in the prompt. |
| `attribute_modification` | `modify_attribute` | `desired_state` is required; desired values must not become source selectors. |
| `swap` | `replace_entity`, `replace_background` | The released label mixes object and background swaps, so instruction semantics choose the topology. |
| `env` | `modify_environment` | Global weather, atmosphere, time, or lighting change; preserve entities, geometry, layout, and text. |
| `style` | `apply_style` | Global style needs no mask; a regional style edit requires a mask. |

The local workspace currently contains 201 addition rows in
`train-00000-of-00105.parquet` and 2,107 removal rows in
`train-00115-of-00571.parquet`. Attribute, swap, environment, and style coverage
must be exercised with other official shards or curated contract fixtures. The
v0.2 acceptance suite therefore includes explicit examples for all nine
topologies instead of pretending the two local shards cover the full taxonomy.

## Selection policy

1. Parse the literal instruction first.
2. Use `task` only to resolve an otherwise plausible interpretation.
3. Validate the parsed topology against the normalized task label.
4. Mark an unknown label, mismatch, invalid contract, or unresolved ambiguity as
   blocked in metadata.
5. Never change the raw ICEdit baseline because a planner row is blocked.

This policy matters for compound and mislabeled rows. It also prevents a `swap`
label from turning a color-only instruction into an object replacement.

## Dataset modes

- `raw`: existing ICEdit generation, with no parser dependency.
- `annotate`: existing raw generation plus contract and planning metadata.
- `plan-only`: contract and status metadata only; no Flux load and no GPU.
- `planned`: validated contract prompt drives image generation.

Mask-free planned rows can execute automatically. Removal, replacement,
background replacement, and regional style edits need a grounded or
user-confirmed mask from `--mask-dir`; otherwise their execution status is
`blocked_missing_mask`. Empty or whole-image masks are `blocked_invalid_mask`.
A task label cannot supply that mask.

## Sources

- OmniEdit project: <https://tiger-ai-lab.github.io/OmniEdit/>
- OmniEdit filtered dataset: <https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M>
- Maintainer note on mixed `swap` examples: <https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M/discussions/4>
