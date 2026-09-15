# Edit Topology annotation guide (v0.2)

Annotate the smallest valid transformation licensed by the instruction. Do not
encode a visually plausible redesign as a required edit.

## Decision order

1. Identify the grammatical operation and the entity or attribute it acts on.
2. Separate target selectors from targets. In “remove two men in white shirts,”
   the men are targets; white shirts are grounding constraints.
3. Record the quantifier. Singular articles normally mean exactly one; explicit
   numbers are exact; “all” and definite plurals may use `quantifier: all`.
4. Choose topology:
   - independent new object: `insert_entity`;
   - new object attached to a carrier/holder: `attach_entity`;
   - change to an existing carrier: `modify_attribute`;
   - deletion of a complete object/person: `remove_entity`;
   - deletion of a local signal while retaining its carrier: `remove_attribute`;
   - one object replaced by another: `replace_entity`;
   - scenery/background replacement with foreground protection:
     `replace_background`;
   - whole-scene weather, time, atmosphere, or lighting change:
     `modify_environment`;
   - whole-scene or explicitly masked rendering-style change: `apply_style`.
5. Add only relations required for validity, attachment, support, or instruction
   satisfaction.
6. Assign region roles and protected invariants.
7. Record where every consequential claim came from.
8. Expose unresolved target, anchor, count, placement, or appearance choices.

## Region-role boundary

- **Target:** the entity/attribute directly created, changed, or deleted.
- **Dependent:** the smallest collateral area allowed to change for coherence,
  such as fingers, a contact shadow, or an object boundary.
- **Context:** evidence read to perform the edit; context is not automatically
  editable.
- **Protected:** content outside the target/dependent budget, plus named fragile
  content such as identity, logos, and OCR text.

For removal, the hidden surface is context-guided reconstruction, not permission
to regenerate the carrier or scene. For attachment, the anchor is not itself the
target; only its necessary contact parts belong in `dependent`.

Every referring constraint must declare `scope`. Use `target` only for selectors
of existing source content, `anchor` for carrier/reference selectors,
`placement` for a requested insertion location, and `replacement` for properties
of the desired new object. Never let a replacement color select the source
object. For example, in "replace the red cup beside the blue plate with a green
glass mug," red is target-scoped, blue is anchor-scoped, and green/glass are
replacement-scoped.

## Ambiguity and provenance

Use `unresolved` when multiple materially different valid contracts remain and
the image/instruction does not select one. Set `clarification.needed` when acting
on one choice would be a user-visible commitment (for example, choosing one of
several people). Use `inferred` only when a stated assumption permits review.

Provenance values mean:

- `instruction`: explicit language or a direct linguistic implication;
- `image`: observable source-image evidence (including OCR);
- `user_input`: masks, clicks, selections, or follow-up answers;
- `default`: a declared system policy, never disguised as user intent;
- `annotation_rule`: an ontology/minimal-change consequence.

Image-derived fields in these examples are hypotheses for manual review until
linked to an actual image and grounding output.

## Review checklist

- Does entity/attribute scope match the instruction?
- Is target count explicit and internally consistent?
- Are selectors stored in `referring_constraints`, not mistaken for targets?
- Does every selector have the correct target/anchor/placement/replacement scope?
- Do modify/replace/style/environment contracts have a literal `desired_state`?
- Is an attachment or attribute carrier represented by `anchor`?
- Are dependent changes necessary rather than merely convenient?
- Are identities, unrelated entities, source style, and visible text protected?
- Are invalid outcomes concrete enough to test later?
- Are unresolved choices visible, with a clarification question when needed?
