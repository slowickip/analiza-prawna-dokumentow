"""PDF mode selection and native reading."""

from __future__ import annotations

import io
import logging

import pdfplumber
import pypdfium2 as pdfium  # type: ignore[import-untyped]

from contract_analyzer.config import RunConfig
from contract_analyzer.domain import DocumentPayload, ReadMode, SourceAnchor

from .errors import IngestError, _raise_limit_breach
from .ocr import _page_ink_share, _read_pdf_ocr
from .payload import _append_word, _payload, _record_blank_page

logger = logging.getLogger(__name__)

# Chosen by pdfplumber's default y_tolerance; not measured.
_PDF_LINE_TOLERANCE_POINTS = 3.0


def _pdf_page_count(data: bytes) -> int:
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return len(pdf.pages)
    except Exception:
        logger.warning("PDF page count failed: corrupt input")
        raise IngestError("corrupt_input", "corrupt_input") from None


def _select_pdf_mode(data: bytes, config: RunConfig) -> ReadMode:
    try:
        pdf = pdfium.PdfDocument(data)
    except Exception:
        logger.warning("PDF mode selection failed: corrupt input")
        raise IngestError("corrupt_input", "corrupt_input") from None
    try:
        with pdfplumber.open(io.BytesIO(data)) as plumber_pdf:
            text_bearing = 0
            coverage_pages = 0
            for page_index in range(len(pdf)):
                native_text = ""
                if page_index < len(plumber_pdf.pages):
                    native_text = plumber_pdf.pages[page_index].extract_text() or ""
                native_chars = sum(1 for char in native_text if not char.isspace())
                if native_chars >= config.pdf_min_native_chars_per_page:
                    text_bearing += 1
                    coverage_pages += 1
                    continue
                image = (
                    pdf[page_index].render(scale=config.pdf_ocr_render_scale).to_pil()
                )
                if _page_ink_share(image) <= config.blank_page_ink_share:
                    continue
                coverage_pages += 1
            if coverage_pages == 0:
                logger.warning("PDF mode selection failed: no content pages")
                raise IngestError("empty_input", "empty_input")
            if text_bearing / coverage_pages >= config.pdf_native_coverage_share:
                return ReadMode.NATIVE_PDF
            return ReadMode.OCR_PDF
    finally:
        pdf.close()


def _read_pdf_native(data: bytes, read_mode: ReadMode) -> DocumentPayload:
    parts: list[str] = []
    anchors: list[SourceAnchor] = []
    offset = 0
    last_page: int | None = None
    last_top: float | None = None
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                words = page.extract_words() or []
                page_had_content = False
                for word_info in words:
                    word = word_info.get("text", "")
                    if not word.strip():
                        continue
                    page_had_content = True
                    top = float(word_info["top"])
                    if not parts:
                        separator = " "
                    elif last_page is not None and page_number != last_page:
                        separator = "\n"
                    elif (
                        last_top is not None
                        and abs(top - last_top) > _PDF_LINE_TOLERANCE_POINTS
                    ):
                        separator = "\n"
                    else:
                        separator = " "
                    bbox = (
                        float(word_info["x0"]),
                        top,
                        float(word_info["x1"]),
                        float(word_info["bottom"]),
                    )
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
                    last_top = top
                if not page_had_content:
                    _record_blank_page(
                        anchors=anchors,
                        page_number=page_number,
                        read_mode=read_mode,
                        offset=offset,
                    )
    except Exception:
        logger.warning("PDF native read failed: corrupt input")
        raise IngestError("corrupt_input", "corrupt_input") from None
    text = "".join(parts)
    if not text.strip():
        logger.warning("PDF native read failed: empty text")
        raise IngestError("empty_input", "empty_input")
    return _payload(text, tuple(anchors), read_mode)


def _read_pdf(data: bytes, config: RunConfig) -> DocumentPayload:
    if not data:
        logger.warning("PDF read failed: empty input")
        raise IngestError("empty_input", "empty_input")
    page_count = _pdf_page_count(data)
    if page_count > config.max_pdf_pages:
        _raise_limit_breach("max_pdf_pages", config.max_pdf_pages, page_count)
    read_mode = _select_pdf_mode(data, config)
    if read_mode is ReadMode.NATIVE_PDF:
        return _read_pdf_native(data, read_mode)
    return _read_pdf_ocr(data, read_mode, config)
