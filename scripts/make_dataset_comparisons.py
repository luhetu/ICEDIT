import argparse
import html
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args():
    parser = argparse.ArgumentParser(description="Create source/generated/reference comparison images.")
    parser.add_argument("dataset_output", help="Dataset output directory")
    parser.add_argument("--force", action="store_true", help="Rebuild existing comparisons")
    return parser.parse_args()


def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def resample_filter():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def place_contained(canvas, image_path, box):
    left, top, right, bottom = box
    image = Image.open(image_path).convert("RGB")
    max_size = (right - left - 16, bottom - top - 16)
    if hasattr(ImageOps, "contain"):
        image = ImageOps.contain(image, max_size, resample_filter())
    else:
        image.thumbnail(max_size, resample_filter())
    x = left + (right - left - image.width) // 2
    y = top + (bottom - top - image.height) // 2
    canvas.paste(image, (x, y))


def render_comparison(root, metadata_path, output_path, force=False):
    if output_path.is_file() and not force:
        return

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stem = metadata_path.stem
    width, image_height = 1440, 420
    title_height, label_height = 76, 38
    canvas = Image.new("RGB", (width, title_height + label_height + image_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(19)
    label_font = load_font(18)

    title = "#{index} [{task}] {prompt}".format(**metadata)
    title_lines = textwrap.wrap(title, width=115)[:2]
    draw.rectangle((0, 0, width, title_height), fill=(232, 236, 242))
    draw.multiline_text((16, 12), "\n".join(title_lines), fill="black", font=title_font, spacing=4)

    column_width = width // 3
    labels = ["SOURCE", "ICEDIT GENERATED", "DATASET REFERENCE"]
    folders = ["source", "generated", "reference"]
    for column, (label, folder) in enumerate(zip(labels, folders)):
        left = column * column_width
        draw.rectangle(
            (left, title_height, left + column_width, title_height + label_height),
            fill=(247, 248, 250),
        )
        draw.text((left + 14, title_height + 8), label, fill="black", font=label_font)
        place_contained(
            canvas,
            root / folder / f"{stem}.png",
            (left, title_height + label_height, left + column_width, canvas.height),
        )

    canvas.save(output_path, quality=90, optimize=True)


def write_gallery(root, entries):
    cards = []
    for metadata, comparison_name in entries:
        title = "#{index} [{task}] {prompt}".format(**metadata)
        cards.append(
            '<article><h2>{}</h2><img loading="lazy" src="comparisons/{}" alt="{}"></article>'.format(
                html.escape(title),
                html.escape(comparison_name),
                html.escape(title),
            )
        )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ICEdit dataset comparisons</title>
<style>
body {{ margin: 0 auto; max-width: 1500px; padding: 24px; background: #eef1f5; font-family: sans-serif; }}
h1 {{ margin: 0 0 24px; }} article {{ background: white; margin: 0 0 24px; padding: 16px; border-radius: 10px; }}
h2 {{ font-size: 18px; font-weight: 500; margin: 0 0 12px; }} img {{ width: 100%; height: auto; display: block; }}
</style></head><body>
<h1>ICEdit: source → generated → reference ({} samples)</h1>
{}
</body></html>
""".format(len(entries), "\n".join(cards))
    (root / "comparison_gallery.html").write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.dataset_output)
    metadata_paths = sorted((root / "metadata").glob("*.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No metadata found below {root}")

    comparisons = root / "comparisons"
    comparisons.mkdir(parents=True, exist_ok=True)
    entries = []
    for number, metadata_path in enumerate(metadata_paths, start=1):
        output_path = comparisons / f"{metadata_path.stem}.jpg"
        render_comparison(root, metadata_path, output_path, args.force)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        entries.append((metadata, output_path.name))
        if number % 20 == 0 or number == len(metadata_paths):
            print(f"Rendered {number}/{len(metadata_paths)}")

    write_gallery(root, entries)
    print(f"Gallery: {root / 'comparison_gallery.html'}")


if __name__ == "__main__":
    main()
