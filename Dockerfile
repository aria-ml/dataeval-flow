# DataEval Application Container - GPU variant
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Copy uv binary from official image (pin version for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

# Install Python 3.12 to /opt/python (accessible by all users)
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.12

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies only (--no-install-project skips installing the project itself)
RUN uv sync --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY src/ ./src/

# Copy entrypoint script for GPU validation
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash dataeval

# Create mount points with correct ownership
RUN mkdir -p /data/dataset /data/model /data/incoming /output \
    && touch /data/dataset/.not_mounted /data/model/.not_mounted /data/incoming/.not_mounted \
    && chown -R dataeval:dataeval /data /output /app

# Container labels
LABEL org.opencontainers.image.title="DataEval Application" \
      org.opencontainers.image.description="DataEval Application container with GPU support" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://gitlab.jatic.net/jatic/aria/dataeval-app"

# Switch to non-root user
USER dataeval

ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "src/workflows/inspect_dataset.py"]
