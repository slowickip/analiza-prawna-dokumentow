from __future__ import annotations

import http.client
import os
import subprocess
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "contract-analyzer-backend:test"
WEB_IMAGE = "contract-analyzer-web:test"
FIXTURES = ROOT / "tests" / "fixtures"
LINUX_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
POLISH_OCR_TEXT = "Umowa najmu lokalu użytkowego w Warszawie"


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=30,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="docker unavailable",
)


def _run(
    cmd: list[str], *, timeout: int = 600, env: dict[str, str] | None = None
) -> None:
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=False,
        timeout=timeout,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )


def _run_output(cmd: list[str], *, timeout: int = 600) -> str:
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        timeout=timeout,
        capture_output=True,
        text=True,
    )
    return result.stdout


# A hang detector, not a budget. Measured 2026-09-02 on an Apple Silicon Mac under
# Docker Desktop: 29s with --no-cache, 0.35s for a no-op rebuild. 900s leaves room for
# a cold cache on a slow link while still failing a build that has stopped making
# progress rather than one that is merely slow.
BUILD_TIMEOUT_SECONDS = 900


def _image_exists(image_name: str = IMAGE) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
        check=False,
        timeout=60,
    )
    return result.returncode == 0


@pytest.fixture(scope="session")
def app_image() -> str:
    """The image `docker compose up backend` would run, built only when absent."""
    if not _image_exists(IMAGE):
        _run(
            ["docker", "compose", "build", "backend"],
            timeout=BUILD_TIMEOUT_SECONDS,
        )
    return IMAGE


@pytest.fixture(scope="session")
def web_image() -> str:
    """The image `docker compose up frontend` would run, built only when absent."""
    if not _image_exists(WEB_IMAGE):
        _run(
            ["docker", "compose", "build", "frontend"],
            timeout=BUILD_TIMEOUT_SECONDS,
        )
    return WEB_IMAGE


def test_tesseract_lists_polish(app_image: str) -> None:
    output = _run_output(
        ["docker", "run", "--rm", app_image, "tesseract", "--list-langs"],
        timeout=120,
    )
    langs = {line.strip() for line in output.splitlines()}
    assert "pol" in langs


def test_libreoffice_converts_doc_end_to_end(app_image: str) -> None:
    script = textwrap.dedent(
        """
        import subprocess
        import tempfile
        from pathlib import Path

        from contract_analyzer.ingest import IngestService

        docx = Path("/fixtures/sample.docx")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    "doc",
                    "--outdir",
                    str(out),
                    str(docx),
                ],
                check=True,
                timeout=120,
            )
            doc_path = out / "sample.doc"
            if not doc_path.is_file():
                raise SystemExit(f"expected converted .doc at {doc_path}")
            payload = IngestService().ingest("legacy.doc", doc_path.read_bytes())
            if not payload.text.strip():
                raise SystemExit("converted .doc produced empty text")
            print("DOC_OK")
        """
    )
    output = _run_output(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{FIXTURES}:/fixtures:ro",
            app_image,
            "python",
            "-c",
            script,
        ],
        timeout=300,
    )
    assert "DOC_OK" in output


def test_polish_ocr_uses_linux_font(app_image: str) -> None:
    script = textwrap.dedent(
        f"""
        import io
        from pathlib import Path

        from PIL import Image, ImageDraw, ImageFont

        from contract_analyzer.ingest import IngestService

        font_path = Path({LINUX_FONT!r})
        if not font_path.is_file():
            raise SystemExit(f"missing Linux font: {{font_path}}")

        font = ImageFont.truetype(str(font_path), 24)
        image = Image.new("RGB", (1200, 500), "white")
        ImageDraw.Draw(image).text(
            (40, 40), {POLISH_OCR_TEXT!r}, fill="black", font=font
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PDF", resolution=150.0)

        payload = IngestService().ingest("scan.pdf", buffer.getvalue())
        text = payload.text.lower()
        if "najmu" not in text and "umowa" not in text:
            raise SystemExit(f"OCR missed Polish text, got: {{payload.text!r}}")
        print("OCR_OK")
        """
    )
    output = _run_output(
        ["docker", "run", "--rm", app_image, "python", "-c", script],
        timeout=300,
    )
    assert "OCR_OK" in output


def test_compiled_ui_is_present(web_image: str) -> None:
    output = _run_output(
        [
            "docker",
            "run",
            "--rm",
            web_image,
            "sh",
            "-c",
            "test -f /usr/share/nginx/html/index.html && echo UI_OK",
        ],
        timeout=60,
    )
    assert "UI_OK" in output


def _readiness() -> tuple[int, str] | None:
    """Readiness as the local deployment reports it, or None if nothing answers."""
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8000/api/v1/health/ready", timeout=5
        ) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, http.client.RemoteDisconnected, TimeoutError):
        return None


def test_local_deployment_is_ready(app_image: str) -> None:
    """Assert against the deployment a developer actually runs, and start nothing.

    This used to stand up an isolated compose project with its own corpus volume, so
    every run rebuilt a corpus -- roughly four minutes of live ELI fetching and
    eighteen of embedding -- and then asserted against a stack nobody was using.

    It now observes the default project on port 8000: the one `docker compose up -d
    app` gives you. It does not start it. A test that starts a service it will not
    stop has taken ownership of something it does not own -- it would leave a stack
    running after a bare `pytest`, and the developer would not know which of their
    containers were theirs. An absent deployment is a setup error with an obvious
    remedy, so it is reported as one.
    """
    if not os.environ.get("MODEL_API_KEY"):
        pytest.skip(
            "MODEL_API_KEY is not set; readiness reports whether a credential is "
            "configured, so inventing one here would assert readiness on a fake"
        )

    result = _readiness()
    if result is None:
        pytest.fail(
            "nothing is serving http://127.0.0.1:8000 -- this suite observes the "
            "local deployment, it does not start one.\n"
            "    docker compose up -d\n"
            "If containers exit at startup, check that qdrant and postgre are healthy "
            "and seeder has finished:\n"
            "    docker compose logs"
        )

    # Readiness may report not-ready briefly while the container settles.
    deadline = time.monotonic() + 60
    last = ""
    while True:
        status, body = result
        if status == 200 and '"ready":true' in body.replace(" ", ""):
            return
        last = f"HTTP {status}: {body}"
        if time.monotonic() >= deadline:
            break
        time.sleep(3)
        polled = _readiness()
        if polled is None:
            pytest.fail(f"the deployment stopped responding mid-check. Last: {last}")
        result = polled

    pytest.fail(
        f"the local deployment is up but not ready: {last}\n"
        "Check that qdrant is populated and EMBEDDINGS_BASE_URL is reachable:\n"
        "    docker compose run --rm seeder"
    )
