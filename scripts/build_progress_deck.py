"""Build the dated, evidence-qualified tutor progress presentation."""
from pathlib import Path
from io import BytesIO
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
INK, GREEN, MUTED = "20282C", "16776B", "647176"
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)


def text(slide, x, y, w, h, value, size=22, color=INK, bold=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(value.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = "Calibri"
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.fill.solid()
        p.font.fill.fore_color.rgb = RGBColor.from_string(color)
        p.space_after = Pt(14)
    return box


def slide(title, subtitle):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, .55, .28, 12, .35, "ICEDIT / RESEARCH PROGRESS / 15 SEP 2026", 12, GREEN, True)
    text(s, .55, .87, 12.2, .85, title, 32, bold=True)
    text(s, .55, 1.77, 12, .7, subtitle, 18, MUTED)
    text(s, .55, 7.04, 11.6, .25, "CVPR-oriented research | Implemented components and hypotheses are distinguished", 11, MUTED)
    text(s, 12.15, 7.02, .6, .3, str(len(prs.slides)), 12, GREEN)
    return s


def panel(s, x, y, w, h, title, body, fill="EDF4F2"):
    shape = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    shape.line.fill.background()
    text(s, x+.2, y+.15, w-.4, .6, title, 22, GREEN, True)
    text(s, x+.2, y+.88, w-.4, h-.95, body, 18)


s = slide("Remove the Target. Preserve the Others.", "Research focus: selective object removal when target and protected content interact.")
panel(s, .6, 2.65, 5.9, 3.6, "The practical problem", "Remove the specified instance and its effects.\nPreserve nearby faces, text, structure and other instances.\nA plausible image can still be an incorrect edit.")
panel(s, 6.8, 2.65, 5.9, 3.6, "Our current stage", "Editing infrastructure is implemented.\nStrong-baseline validation is in progress.\nA new method contribution remains to be demonstrated.", "F1F2F5")

s = slide("What We Have Built", "Implemented pipeline; active denoising intervention remains a proposal.")
labels = ["Instruction", "Edit contract", "Spatial controls", "ICEdit / Fill", "Raw + composite"]
for i, label in enumerate(labels):
    x = .6 + i*2.57
    panel(s, x, 2.8, 2.25, 1.35, label, "")
    if i < 4:
        arrow = s.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x+2.29), Inches(3.25), Inches(.22), Inches(.3))
        arrow.fill.solid(); arrow.fill.fore_color.rgb = RGBColor.from_string(GREEN)
        arrow.line.fill.background()
text(s, .75, 4.6, 11.8, 1.6, "Contract: target, operation and preservation requirements.\nControls: source-only masks and protected-region subtraction.\nFinal compositing preserves pixels outside the mask; it does not prove better generation.", 23)

s = slide("Progress and Evidence", "Status snapshot from the September 15 progress session; live jobs must be checked separately.")
panel(s, .6, 2.6, 5.9, 3.9, "Verified progress", "30 unit tests passed across spatial masks, executor and contract schema.\nOffline LoRA loading fixed.\n15 pilot configurations passed input preflight.")
panel(s, 6.8, 2.6, 5.9, 3.9, "Execution status", "Job 1042473 submitted; last session check: pending.\nObjectClear: code and weights ready, pipeline import unresolved.\nNo new generated results were available at that check.", "F1F2F5")

s = slide("Where a Contribution Must Go Further", "A contract, verifier or final composite alone is not the proposed contribution.")
text(s, .75, 2.65, 11.8, 2.45, "ObjectClear: target/effect localization and background fusion.\nConsistency Critic: reference-guided consistency correction.\nAdaEraser: adaptive attention suppression for removal.", 25)
text(s, .75, 5.0, 11.8, .85, "Hypothesis: explicit instance ownership can reduce collateral damage when effects or boundaries overlap.", 25, GREEN, True)
text(s, .75, 6.18, 11.8, .55, "Sources: arxiv.org/abs/2505.22636v2 | arxiv.org/abs/2511.20614 | arxiv.org/abs/2605.15921", 12, MUTED)

s = slide("Candidate Mechanism: Test Before Building", "Proposed diagnostic, not a validated method or physical causal model.")
text(s, .75, 2.6, 11.8, 3.5, "1. Identify target, protected instances and uncertain regions.\n2. At the same denoising state, vary target and protected reference evidence separately.\n3. Test whether prediction responses distinguish target persistence from protected-content damage.\n4. Build a regional controller only if this signal beats simple spatial heuristics.", 24)
text(s, .75, 6.2, 11.8, .55, "Compare equal information and compute budgets. Report raw outputs before compositing.", 18, GREEN, True)

s = slide("Next Experiments and Decision Criteria", "Use existing cases first; no new benchmark is required for this diagnostic phase.")
panel(s, .6, 2.65, 5.9, 3.8, "Experimental sequence", "Run ICEdit: 5 conditions x 3 fixed seeds.\nValidate ObjectClear, including fusion on/off.\nAdd strong removal comparisons and held-out cases.\nThen test the smallest intervention module.")
panel(s, 6.8, 2.65, 5.9, 3.8, "Continue only if...", "The failure persists on strong baselines.\nRemoval improves at comparable protection damage, or vice versa.\nGains survive raw-output and held-out evaluation.\nSimple masks and equal-cost resampling cannot explain the gain.", "F1F2F5")

OUT.mkdir(exist_ok=True)
path = OUT / "ICEdit_CVPR_Progress_2026-09-15.pptx"
buffer = BytesIO()
prs.save(buffer)
path.write_bytes(buffer.getvalue())
assert len(Presentation(path).slides) == 6
print(path)
