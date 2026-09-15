"""Small, inspectable prompt set for the interactive contract demo."""

PROMPT_TEMPLATES = {
    "Choose a template...": "",
    "Flower image · man on right stone": (
        "Add a small man standing on the flat gray stone on the right. "
        "Keep the pink flower, all leaves, flower pot, lighting, and background unchanged."
    ),
    "Flower image · bottle on right stone": (
        "Add one small glass bottle on the flat gray stone on the right. "
        "Keep the pink flower, all leaves, flower pot, lighting, and background unchanged."
    ),
    "Attachment · hat on left person": (
        "Add one small hat to the person on the left. "
        "Keep the person's face, identity, pose, clothes, other people, and background unchanged."
    ),
    "Attachment · sword in selected hand": (
        "Add one sword to the selected person's right hand. "
        "Keep the face, identity, pose, clothes, other objects, and background unchanged."
    ),
    "Removal · flower only": (
        "Remove only the pink flower and reconstruct only the newly exposed local area. "
        "Keep all leaves, the flower pot, gray stone, lighting, and background unchanged."
    ),
    "Attribute · red car to blue": (
        "Change the red car to blue. "
        "Keep its shape, position, reflections, surroundings, and all text unchanged."
    ),
    "Replacement · cup to ceramic mug": (
        "Replace the red cup with a blue ceramic mug. "
        "Keep the table, nearby objects, lighting, shadows, and background unchanged."
    ),
    "Background · snowy mountains": (
        "Replace the background with a snowy mountain landscape. "
        "Keep every foreground person, object, face, pose, and readable text unchanged."
    ),
    "Environment · rainy scene": (
        "Make the whole scene rainy. "
        "Keep all people, objects, geometry, layout, and readable text unchanged."
    ),
    "Style · watercolor image": (
        "Apply watercolor style to the whole image. "
        "Keep faces, object identities, geometry, composition, and readable text unchanged."
    ),
}


def template_prompt(label: str | None) -> str:
    return PROMPT_TEMPLATES.get(label or "", "")
