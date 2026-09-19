# Iron Bank hardened container for DataEval Workflows
# DISA Container Hardening SRG compliant
# Variant: cu126

ARG BASE_REGISTRY="registry1.dso.mil"
ARG BASE_IMAGE="ironbank/redhat/ubi/ubi9-minimal"
ARG BASE_TAG="9.5"

FROM ${BASE_REGISTRY}/${BASE_IMAGE}:${BASE_TAG}

ARG DATAEVAL_FLOW_VERSION="0.2.2"
ARG WHEELHOUSE_ARCHIVE="dataeval-flow-0.2.2-wheels-cu126.tar.gz"

USER root

# OS update and required runtime packages; clean caches in same layer to minimize image size and CVE surface
RUN microdnf update -y \
    && microdnf install -y --nodocs \
        bash \
        ca-certificates \
        coreutils-single \
        findutils \
        glibc-langpack-en \
        python3.12 \
        python3.12-pip \
        shadow-utils \
        tar \
        gzip \
    && microdnf clean all \
    && rm -rf /var/cache/yum /var/cache/dnf

# Create non-root user with explicit numeric UID/GID >= 10001 (DISA STIG requirement)
ARG USER_NAME="dataeval"
ARG USER_UID="10001"
ARG USER_GID="10001"

RUN groupadd -g ${USER_GID} ${USER_NAME} \
    && useradd -u ${USER_UID} -g ${USER_GID} -d /home/${USER_NAME} -m -s /sbin/nologin ${USER_NAME}

# Stage application artifacts from pre-packaged wheelhouse
COPY ${WHEELHOUSE_ARCHIVE} /tmp/
COPY requirements.txt /tmp/requirements.txt
COPY entrypoint.sh /opt/dataeval-flow/entrypoint.sh
COPY LICENSE /opt/dataeval-flow/LICENSE

# dataeval-flow's own [project].dependencies carries an unpinned
# `dataeval @ git+...@main` reference, which pip's resolver refuses to
# satisfy from --find-links (a direct URL requirement is only ever
# satisfied by refetching that exact URL). requirements.txt pins that same
# dependency to the exact commit the wheelhouse's `dataeval` wheel was built
# from, but pip's hash-checking mode also refuses to process a git+ line at
# all (it has no way to hash a VCS checkout) -- so that line has to be
# stripped and its prebuilt wheel installed by name instead, same as
# build-wheelhouse.sh already does when it builds the wheelhouse itself.
# --no-deps on dataeval and dataeval-flow then skips re-resolving either one
# and keeps the whole install hermetic (offline, hash-verified).
RUN mkdir -p /tmp/wheelhouse /opt/dataeval-flow/venv \
    && tar -xzf /tmp/${WHEELHOUSE_ARCHIVE} -C /tmp/wheelhouse \
    && python3.12 -m venv /opt/dataeval-flow/venv \
    && grep -v -E '^[a-zA-Z0-9_-]+ @ git\+' /tmp/requirements.txt > /tmp/requirements.pinned.txt \
    && /opt/dataeval-flow/venv/bin/pip install --no-cache-dir --no-index --find-links=/tmp/wheelhouse -r /tmp/requirements.pinned.txt \
    && /opt/dataeval-flow/venv/bin/pip install --no-cache-dir --no-index --no-deps --find-links=/tmp/wheelhouse dataeval \
    && /opt/dataeval-flow/venv/bin/pip install --no-cache-dir --no-index --no-deps --find-links=/tmp/wheelhouse dataeval-flow \
    && rm -rf /tmp/wheelhouse /tmp/${WHEELHOUSE_ARCHIVE} /tmp/requirements.txt /tmp/requirements.pinned.txt \
    && chmod 0755 /opt/dataeval-flow/entrypoint.sh

# Mount points and workspace directories with least-privilege permissions
RUN mkdir -p /dataeval /output /cache \
    && touch /dataeval/.not_mounted /output/.not_mounted \
    && chown -R ${USER_UID}:${USER_GID} /dataeval /output /cache /home/${USER_NAME} \
    && chmod 0750 /dataeval /output /cache /home/${USER_NAME} \
    && chown -R root:root /opt/dataeval-flow \
    && chmod -R 0755 /opt/dataeval-flow

# STIG: Remove SetUID / SetGID bits across filesystem to prevent privilege escalation
RUN find / -perm /6000 -type f -exec chmod a-s {} + 2>/dev/null || true

# Container metadata labels
LABEL org.opencontainers.image.title="DataEval Workflows (CUDA 12.6)" \
      org.opencontainers.image.description="DataEval Workflows container - CUDA 12.6 variant" \
      org.opencontainers.image.version="${DATAEVAL_FLOW_VERSION}" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="ARiA" \
      org.opencontainers.image.url="https://github.com/aria-ml/dataeval-flow" \
      mil.dso.ironbank.image.type="opensource" \
      mil.dso.ironbank.image.parent="ironbank/redhat/ubi/ubi9-minimal"

ENV PATH="/opt/dataeval-flow/venv/bin:$PATH"
ENV DATAEVAL_DATA=/dataeval
ENV DATAEVAL_OUTPUT=/output
ENV DATAEVAL_CACHE=/cache
ENV UV_EXTRAS_OVERRIDE=cu126
ENV CONTAINER_MODE=gpu

WORKDIR /dataeval

# STIG: Container Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /opt/dataeval-flow/venv/bin/python -m dataeval_flow --help || exit 1

# Run as non-privileged numeric UID/GID
USER 10001:10001

ENTRYPOINT ["/opt/dataeval-flow/entrypoint.sh"]
CMD ["python", "-m", "dataeval_flow"]
