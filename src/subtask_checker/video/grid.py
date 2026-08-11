"""Tile extracted frames into one contact-sheet JPEG with burned-in time badges.

The layout follows the representation the previous evaluation project converged on
for VLM input (4 columns, 336 px tiles, JPEG q95, black time badge in the top-left
of every tile): timestamps as pixels inside the image proved far more reliable than
timestamps as prose next to it, and 336 px x <=12 tiles is the geometry that ran
cleanly against a shared vLLM engine. Implemented directly with Pillow — the
``macrodata-refiner`` dependency the old code wrapped is not worth its asyncio and
caching complexity for this pipeline.
"""

import io
import math

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from subtask_checker.video.frames import ExtractedFrame

COLUMNS = 4
TILE_WIDTH = 336
JPEG_QUALITY = 95
BADGE_TEXT_SIZE = 20


class GridMeta(BaseModel):
    columns: int
    rows: int
    tile_width: int
    tile_height: int
    width: int
    height: int
    n_tiles: int
    badge_times_sec: list[float]


def default_badge_text_size(tile_width: int) -> int:
    """Scale the badge with the tile (20 px at the reference 336 px tile) so it
    stays readable on small tiles without covering a large fraction of them."""
    return max(10, round(tile_width * BADGE_TEXT_SIZE / TILE_WIDTH))


def _badge_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _badge(tile: Image.Image, text: str, font: ImageFont.ImageFont) -> None:
    draw = ImageDraw.Draw(tile)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    pad = 4
    draw.rectangle((0, 0, right - left + 2 * pad, bottom - top + 2 * pad), fill="black")
    draw.text((pad - left, pad - top), text, fill="white", font=font)


def build_grid(
    frames: list[ExtractedFrame],
    columns: int = COLUMNS,
    tile_width: int = TILE_WIDTH,
    quality: int = JPEG_QUALITY,
    badge_text_size: int | None = None,
) -> tuple[bytes, GridMeta]:
    """One JPEG contact sheet from the frames, badge text ``SSS.SSs`` counting from
    the sampling-window start. Raises ValueError on an empty frame list."""
    if not frames:
        raise ValueError("no frames to tile")

    font = _badge_font(badge_text_size or default_badge_text_size(tile_width))
    tiles = []
    for fr in frames:
        img = Image.open(io.BytesIO(fr.jpeg)).convert("RGB")
        tile_height = round(img.height * tile_width / img.width)
        tile = img.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        _badge(tile, f"{fr.badge_time_sec:06.2f}s", font)
        tiles.append(tile)

    tile_height = tiles[0].height
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "black")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * tile_width, (i // columns) * tile_height))

    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=quality)
    meta = GridMeta(
        columns=columns,
        rows=rows,
        tile_width=tile_width,
        tile_height=tile_height,
        width=sheet.width,
        height=sheet.height,
        n_tiles=len(tiles),
        badge_times_sec=[fr.badge_time_sec for fr in frames],
    )
    return buf.getvalue(), meta
