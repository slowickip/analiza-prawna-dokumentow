"""PDF raster inspection and Tesseract OCR."""

from __future__ import annotations

import csv
import io
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pypdfium2 as pdfium  # type: ignore[import-untyped]
from PIL import Image

from contract_analyzer.config import RunConfig
from contract_analyzer.domain import DocumentPayload, ReadMode, SourceAnchor

from .errors import IngestError
from .payload import _append_word, _payload, _record_blank_page

if TYPE_CHECKING:
    from PIL.Image import Image as PilImage

logger = logging.getLogger(__name__)

_INK_DOWNSAMPLE_LONGEST_SIDE = 400


def _page_ink_share(image: PilImage) -> float:
    grey = image.convert("L")
    width, height = grey.size
    longest = max(width, height)
    if longest > _INK_DOWNSAMPLE_LONGEST_SIDE:
        scale = _INK_DOWNSAMPLE_LONGEST_SIDE / longest
        grey = grey.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.BOX,
        )
        width, height = grey.size
    total = width * height
    if total == 0:
        return 0.0
    histogram = grey.histogram()
    dark = sum(histogram[:128])
    return dark / total


def _run_tesseract(image_path: Path, config: RunConfig) -> list[dict[str, str]]:
    tesseract = shutil.which("tesseract")
    if tesseract is None:
        logger.error("OCR failed: tesseract binary not found")
        raise IngestError("missing_tool", "missing_tool: tesseract")
    try:
        completed = subprocess.run(
            [
                tesseract,
                str(image_path),
                "stdout",
                "-l",
                config.ocr_language,
                "tsv",
            ],
            check=True,
            timeout=config.ocr_timeout_seconds,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        logger.error("OCR tesseract timed out for image %s", image_path.name)
        raise IngestError("ocr_failed", "ocr_failed: timeout") from None
    except subprocess.CalledProcessError as exc:
        logger.error(
            "OCR tesseract failed with exit code %s for image %s",
            exc.returncode,
            image_path.name,
        )
        raise IngestError("ocr_failed", f"ocr_failed: exit {exc.returncode}") from None
    rows = list(csv.DictReader(io.StringIO(completed.stdout), delimiter="\t"))
    return [
        row for row in rows if row.get("level") == "5" and row.get("text", "").strip()
    ]


def _read_pdf_ocr(
    data: bytes, read_mode: ReadMode, config: RunConfig
) -> DocumentPayload:
    parts: list[str] = []
    anchors: list[SourceAnchor] = []
    offset = 0
    last_page: int | None = None
    last_block: str | None = None
    last_par: str | None = None
    last_line: str | None = None
    try:
        pdf = pdfium.PdfDocument(data)
    except Exception:
        logger.warning("PDF OCR failed: corrupt input")
        raise IngestError("corrupt_input", "corrupt_input") from None
    try:
        for page_index in range(len(pdf)):
            page_number = page_index + 1
            image = pdf[page_index].render(scale=config.pdf_ocr_render_scale).to_pil()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                image_path = Path(handle.name)
            try:
                try:
                    image.save(image_path)
                except OSError:
                    logger.error(
                        "PDF OCR failed to save image for page %d", page_number
                    )
                    raise IngestError("ocr_failed", "ocr_failed") from None
                words = _run_tesseract(image_path, config)
                if not words:
                    if _page_ink_share(image) <= config.blank_page_ink_share:
                        _record_blank_page(
                            anchors=anchors,
                            page_number=page_number,
                            read_mode=read_mode,
                            offset=offset,
                        )
                        continue
                    logger.error("PDF OCR page %d yielded no words", page_number)
                    raise IngestError(
                        "ocr_failed",
                        f"ocr_failed: page {page_number} yielded no words",
                    )
                scale = config.pdf_ocr_render_scale
                for row in words:
                    word = row["text"]
                    block_num = row.get("block_num")
                    par_num = row.get("par_num")
                    line_num = row.get("line_num")
                    if not parts:
                        separator = " "
                    elif last_page is not None and page_number != last_page:
                        separator = "\n"
                    elif (
                        (last_block is not None and block_num != last_block)
                        or (last_par is not None and par_num != last_par)
                        or (last_line is not None and line_num != last_line)
                    ):
                        separator = "\n"
                    else:
                        separator = " "
                    left = float(row["left"]) / scale
                    top = float(row["top"]) / scale
                    width = float(row["width"]) / scale
                    height = float(row["height"]) / scale
                    bbox = (left, top, left + width, top + height)
                    offset = _append_word(
                        parts=parts,
                        anchors=anchors,
                        offset=offset,
                        word=word,
                        page_number=page_number,
                        bbox=bbox,
                        read_mode=read_mode,
                        separator=separator,
                    )
                    last_page = page_number
                    last_block = block_num
                    last_par = par_num
                    last_line = line_num
            finally:
                image_path.unlink(missing_ok=True)
    finally:
        pdf.close()
    text = "".join(parts)
    if not text.strip() and not anchors:
        logger.error("PDF OCR yielded empty text and no anchors")
        raise IngestError("ocr_failed", "ocr_failed")
    return _payload(text, tuple(anchors), read_mode)
