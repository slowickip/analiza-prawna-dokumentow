from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
COMPOSE_PATH = ROOT / "compose.yaml"
DOCKERFILE_PATH = ROOT / "backend" / "Dockerfile"


def _load_compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def test_compose_has_no_secret_literal() -> None:
    text = COMPOSE_PATH.read_text(encoding="utf-8")
    assert "MODEL_API_KEY:" in text
    assert "${MODEL_API_KEY:?" in text
    assert not re.search(r"sk-[A-Za-z0-9]", text)


def test_backend_tmpfs_uploads() -> None:
    compose = _load_compose()
    backend = compose["services"]["backend"]
    # The tmpfs is charged to the backend's memory limit, so it carries a size cap
    # strictly below that limit: an upload must not be able to spend the whole
    # allowance and get the container killed.
    assert backend.get("tmpfs") == ["/data/uploads:size=512m"]


def test_backend_can_be_sealed_for_a_measured_batch() -> None:
    """The evaluation runner refuses an unsealed server, so compose must pass it.

    Without this variable reaching the container the flag can only ever read
    false, and a batch could not be run against the documented deployment at all.
    """
    compose = _load_compose()
    environment = compose["services"]["backend"]["environment"]
    assert environment["EVALUATION_BATCH_OPEN"] == "${EVALUATION_BATCH_OPEN:-false}"


def test_every_service_bounds_its_memory() -> None:
    compose = _load_compose()
    for name, service in compose["services"].items():
        assert "mem_limit" in service, f"{name} has no mem_limit"


def test_service_dependencies() -> None:
    compose = _load_compose()
    services = compose["services"]

    assert "backend" in services
    assert "frontend" in services
    assert "postgre" in services
    assert "qdrant" in services
    assert "seeder" in services

    assert services["seeder"]["depends_on"]["qdrant"]["condition"] == "service_healthy"

    backend_deps = services["backend"]["depends_on"]
    assert backend_deps["postgre"]["condition"] == "service_healthy"
    assert backend_deps["qdrant"]["condition"] == "service_healthy"
    assert backend_deps["seeder"]["condition"] == "service_completed_successfully"

    assert "backend" in services["frontend"]["depends_on"]


def test_image_is_pinned_not_latest() -> None:
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "latest" not in dockerfile
    assert "@sha256:" in dockerfile


_PINNED = re.compile(r"[^\s:@]+:(?!latest@)[^\s:@]+@sha256:[0-9a-f]{64}")


def test_every_pulled_image_is_version_and_digest_pinned() -> None:
    """A floating tag can be repointed upstream, so a rebuild is not the same build.

    A service the artefact builds is exempt because there is nothing to pull.
    Everything else carries a tag naming the version a reader expects and a
    digest naming the bytes they get.
    """
    compose = _load_compose()
    pulled = {
        name: service["image"]
        for name, service in compose["services"].items()
        if "image" in service and "build" not in service
    }
    assert pulled, "no pulled image found in compose.yaml"
    unpinned = {n: i for n, i in pulled.items() if not _PINNED.fullmatch(i)}
    assert unpinned == {}, f"compose images are not digest-pinned: {unpinned}"

    dockerfiles = sorted(ROOT.rglob("Dockerfile"))
    assert dockerfiles, "no Dockerfile found under the artefact"
    for dockerfile in dockerfiles:
        for line in dockerfile.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("ARG ") or "_IMAGE=" not in stripped:
                continue
            image = stripped.split("=", 1)[1].strip()
            assert _PINNED.fullmatch(image), (
                f"{dockerfile} base image is not digest-pinned: {image!r}"
            )
