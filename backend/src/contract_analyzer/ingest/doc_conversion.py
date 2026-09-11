"""Legacy DOC conversion through LibreOffice."""

import hashlib
import logging
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

from contract_analyzer.config import RunConfig
from contract_analyzer.domain import ConversionProvenance, DocumentPayload

from .docx import _read_docx
from .errors import IngestError

logger = logging.getLogger(__name__)

_OLE2_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
# Fail-loud ceiling chosen by argument, not measured.
_MAX_REPLACEMENT_CHARACTER_SHARE = 0.1


def _libreoffice_version(soffice: str, config: RunConfig) -> str:
    try:
        completed = subprocess.run(
            [soffice, "--version"],
            check=True,
            timeout=config.conversion_timeout_seconds,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        logger.warning("LibreOffice version check timed out")
        raise IngestError(
            "conversion_failed", "conversion_failed: version timeout"
        ) from None
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "LibreOffice version check failed with exit code %s", exc.returncode
        )
        raise IngestError(
            "conversion_failed",
            f"conversion_failed: version exit {exc.returncode}",
        ) from None
    version = completed.stdout.strip()
    if not version:
        logger.warning("LibreOffice version check returned empty output")
        raise IngestError("conversion_failed", "conversion_failed: empty version")
    return version


def _validate_doc_conversion_output(text: str) -> None:
    has_disallowed_control = any(
        unicodedata.category(char) == "Cc" and char not in "\t\n\r" for char in text
    )
    replacement_share_exceeded = (
        text.count("\ufffd") > len(text) * _MAX_REPLACEMENT_CHARACTER_SHARE
    )
    if has_disallowed_control or replacement_share_exceeded:
        logger.warning("DOC conversion output validation failed: corrupt output")
        raise IngestError("conversion_failed", "conversion_failed: corrupt output")


def _read_doc(data: bytes, config: RunConfig) -> DocumentPayload:
    if not data:
        logger.warning("DOC conversion failed: empty input")
        raise IngestError("empty_input", "empty_input")
    if not data.startswith(_OLE2_SIGNATURE):
        logger.warning("DOC conversion failed: corrupt input")
        raise IngestError("corrupt_input", "corrupt_input")
    soffice = shutil.which("soffice")
    if soffice is None:
        logger.error("DOC conversion failed: soffice binary not found")
        raise IngestError("missing_tool", "missing_tool: soffice")
    converter_version = _libreoffice_version(soffice, config)
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input.doc"
        input_path.write_bytes(data)
        try:
            subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--convert-to",
                    "docx",
                    "--outdir",
                    temp_dir,
                    str(input_path),
                ],
                check=True,
                timeout=config.conversion_timeout_seconds,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            logger.error("DOC conversion timed out")
            raise IngestError(
                "conversion_failed", "conversion_failed: timeout"
            ) from None
        except subprocess.CalledProcessError as exc:
            logger.error("DOC conversion failed with exit code %s", exc.returncode)
            raise IngestError(
                "conversion_failed",
                f"conversion_failed: exit {exc.returncode}",
            ) from None
        converted = Path(temp_dir) / "input.docx"
        if not converted.is_file():
            logger.error("DOC conversion produced no output file")
            raise IngestError("conversion_failed", "conversion_failed")
        converted_bytes = converted.read_bytes()
        provenance = ConversionProvenance(
            converter="libreoffice",
            converter_version=converter_version,
            output_hash=hashlib.sha256(converted_bytes).hexdigest(),
        )
        payload = _read_docx(converted_bytes, conversion=provenance)
        _validate_doc_conversion_output(payload.text)
        return payload
