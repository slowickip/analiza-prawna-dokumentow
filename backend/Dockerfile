# No `# syntax=` directive on purpose. It pins the BuildKit *frontend* to an image
# resolved from Docker Hub on every build, which is the one remote dependency this
# file did not pin by digest -- and the one that stalls the build for 60s and then
# fails when the registry or the credential helper is unavailable. The built-in
# frontend already provides everything used here (cache mounts, heredocs, COPY
# --chmod/--chown); those need Docker 23.0+, which the pinned base images predate
# needing anyway. Builds are hermetic once the base image digests are cached.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:0f5b26b9518d002b6173fd61daad821fa340635ebfec5bba471013f9ca114579
ARG UV_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58

FROM ${UV_IMAGE} AS python-deps
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
# --no-group evaluation: the judge SDK and its pinned Codex runtime score saved
# answers and are never imported by the served application.
# Third-party packages first, keyed on the lockfile alone. The project itself is
# installed in a second pass so that editing src/ does not re-resolve and
# re-download the dependency set.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group evaluation --no-install-project
COPY backend/src ./backend/src
COPY seeder/src ./seeder/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group evaluation

FROM ${PYTHON_IMAGE} AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        fonts-liberation \
        libreoffice-writer \
        tesseract-ocr \
        tesseract-ocr-pol \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

# Created before the copies so each one can land with its final ownership. A
# trailing `chown -R /app` would instead rewrite every file into a second copy
# of the virtual environment.
RUN useradd --create-home --uid 1000 appuser \
    && install -d -o appuser -g appuser /app

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

COPY --from=python-deps --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
COPY --chown=appuser:appuser backend/src ./backend/src
COPY --chown=appuser:appuser seeder/src ./seeder/src
COPY --chown=appuser:appuser corpus/manifest.example.json corpus/manifest.smoke.json ./corpus/

COPY --chmod=755 <<'ENTRY' /entrypoint.sh
#!/bin/sh
set -e
mkdir -p /data/metadata /data/uploads
chown -R appuser:appuser /data/metadata /data/uploads
if [ -w /data/corpus ]; then
    chown -R appuser:appuser /data/corpus
fi
exec runuser -u appuser -- "$@"
ENTRY

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "contract_analyzer.serve:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
