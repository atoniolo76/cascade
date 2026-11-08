# Cascade app image: main mode (`cascade serve`) and the disk that sim forks boot from.
# Builds only the app. Accel-Sim, its configs, the post-processor, and the simulator entrypoint
# come from BASE_IMAGE (see "Simulator image" in docs/api.md).
ARG BASE_IMAGE=ubuntu:22.04

FROM node:22-bookworm-slim AS web
WORKDIR /src/web
COPY web/ ./
# vite.config.ts writes the build to ../cascade/static.
RUN mkdir -p /src/cascade/static \
    && if [ -f package.json ]; then npm ci && npm run build; fi

FROM ${BASE_IMAGE}
USER root
ARG RCLONE_VERSION=1.76.0
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*
# rclone 1.76 writes PUTs and multipart uploads to a temp object and renames it on success; ingest
# relies on that and ignores .rclone_temp_put_* files. Check `rclone serve s3 --help` before changing.
RUN arch="${TARGETARCH:-amd64}" \
    && curl -fsSL -o /tmp/rclone.zip \
        "https://downloads.rclone.org/v${RCLONE_VERSION}/rclone-v${RCLONE_VERSION}-linux-${arch}.zip" \
    && unzip -q /tmp/rclone.zip -d /tmp \
    && install -m 0755 "/tmp/rclone-v${RCLONE_VERSION}-linux-${arch}/rclone" /usr/local/bin/rclone \
    && rm -rf /tmp/rclone*

COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /uvx /usr/local/bin/
ENV UV_PYTHON=3.12 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
RUN uv python install 3.12

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra sail --no-install-project
COPY cascade ./cascade
COPY scripts ./scripts
# Real Accel-Sim configs and logs, so a Sail deploy can run end to end with `cascade worker --replay`.
COPY tests/fixtures ./tests/fixtures
# Alessio's real OPT-125m output files, which the stand-in replays with --replay-outputs.
COPY results/20261108-021007-opt125m-h100-accelsim/outputs \
    ./results/20261108-021007-opt125m-h100-accelsim/outputs
COPY --from=web /src/cascade/static ./cascade/static
RUN uv sync --frozen --no-dev --extra sail

ENV PATH=/app/.venv/bin:$PATH \
    CASCADE_DATA_DIR=/data
EXPOSE 8000 9000
# Sail does not run CMD; scripts/deploy_sail.py starts `cascade serve` with exec.
CMD ["cascade", "serve"]
