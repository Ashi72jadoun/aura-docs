"""Make equal-size, uncropped dashboard images for the README."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"
OUTPUT_SIZE = (1200, 675)

IMAGES = (
    (
        "aura-peach-sunrise.png",
        "aura-peach-sunrise-readme.png",
        (255, 247, 241),
    ),
    (
        "aura-midnight-bloom.png",
        "aura-midnight-bloom-readme.png",
        (28, 27, 37),
    ),
)


def _resampling_filter():
    """Support both current and older Pillow versions."""
    resampling = getattr(Image, "Resampling", Image)
    return resampling.LANCZOS


def prepare_image(
    source_name: str,
    output_name: str,
    background: tuple[int, int, int],
) -> None:
    source_path = ASSETS_DIR / source_name
    output_path = ASSETS_DIR / output_name

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Screenshot not found: {source_path}\n"
            "Save the original dashboard screenshot at that path and run "
            "this script again."
        )

    with Image.open(source_path) as opened_image:
        image = ImageOps.exif_transpose(opened_image).convert("RGB")

    # Fit the full screenshot inside the shared canvas. This keeps the whole
    # dashboard visible instead of stretching or cropping it.
    fitted_image = ImageOps.contain(
        image,
        OUTPUT_SIZE,
        method=_resampling_filter(),
    )

    canvas = Image.new("RGB", OUTPUT_SIZE, background)
    left = (OUTPUT_SIZE[0] - fitted_image.width) // 2
    top = (OUTPUT_SIZE[1] - fitted_image.height) // 2
    canvas.paste(fitted_image, (left, top))
    canvas.save(output_path, format="PNG", optimize=True)

    print(
        f"Created {output_path.relative_to(PROJECT_ROOT)} "
        f"({OUTPUT_SIZE[0]} x {OUTPUT_SIZE[1]})"
    )


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    for source_name, output_name, background in IMAGES:
        prepare_image(source_name, output_name, background)


if __name__ == "__main__":
    main()