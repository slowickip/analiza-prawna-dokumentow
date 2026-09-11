ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:0f5b26b9518d002b6173fd61daad821fa340635ebfec5bba471013f9ca114579
ARG UV_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58

FROM ${UV_IMAGE} AS python-deps
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
# --no-group backend: this is a one-shot corpus build job. It reads ELI, cuts
# acts into units, encodes them and writes them to the vector store, and imports
# nothing from the analysis backend's HTTP surface, agent runtime or metadata
# store, and nothing from the answer-scoring judge either. It used to install
# all of them anyway.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group backend --no-group evaluation --no-install-project
COPY backend/src ./backend/src
COPY seeder/src ./seeder/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group backend --no-group evaluation

FROM ${PYTHON_IMAGE} AS runtime
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

USER appuser

ENTRYPOINT ["python", "-m", "seeder"]
CMD ["--manifest", "/app/corpus/manifest.example.json", "--qdrant-url", "http://qdrant:6333"]
