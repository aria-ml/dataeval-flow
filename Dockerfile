# DataEval Application Container - GPU variant
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Copy uv binary from official image (pin version for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

# Install Python 3.12 to /opt/python (accessible by all users)
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.12

WORKDIR /app

# 1. Copy config files (and touch README to prevent cache busting)
COPY pyproject.toml uv.lock noxfile.py ./
RUN touch README.md

# 2. Install PROD + DEV dependencies (needed for testing)
RUN uv sync --frozen --group dev --extra cu118 --extra onnx-gpu --extra opencv --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

# 3. Copy source and tests
COPY src/ ./src/
COPY tests/ ./tests/

# -------------------------------------------------------
# CI GATE: Run Tests & Linting
# If this fails, the Docker build fails.
# -------------------------------------------------------
RUN nox

# 4. Re-sync only prod dependencies (removes dev tools from final layer)
RUN uv sync --frozen --no-dev --extra cu118 --extra onnx-gpu --extra opencv --no-install-project

# 5. Set PYTHONPATH for src layout (required since project not installed)
ENV PYTHONPATH="/app/src"

# 6. Setup Runtime
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash dataeval

# Create mount points with correct ownership
RUN mkdir -p /data/config /data/dataset /data/model /output \
    && touch /data/config/.not_mounted /data/dataset/.not_mounted /data/model/.not_mounted \
    && chown -R dataeval:dataeval /data /output /app

# Container labels
LABEL org.opencontainers.image.title="DataEval Application" \
      org.opencontainers.image.description="DataEval Application container with GPU support" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://gitlab.jatic.net/jatic/aria/dataeval-app"

# Switch to non-root user
USER dataeval

ENV UV_EXTRAS_OVERRIDE=cu118

ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "src/container_run.py"]