#!/usr/bin/env bash
# Build an offline wheelhouse and package Iron Bank bundle for a specified variant.
#
# Usage:
#   ./ironbank/build-wheelhouse.sh [VARIANT] [VERSION] [OUTPUT_DIR]
#
# Examples:
#   ./ironbank/build-wheelhouse.sh cpu
#   ./ironbank/build-wheelhouse.sh cu126 0.2.3 dist/ironbank/cu126

set -euo pipefail

VARIANT="${1:-cpu}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve version
if [[ -z "${2:-}" ]]; then
  eval "$("${ROOT_DIR}/docker/resolve-version.sh")"
  VERSION="${DATAEVAL_FLOW_VERSION}"
else
  VERSION="$2"
fi

OUTPUT_DIR="${3:-${ROOT_DIR}/dist/ironbank/${VARIANT}}"
mkdir -p "${OUTPUT_DIR}"

echo "======================================================================"
echo "Building Iron Bank wheelhouse: variant='${VARIANT}', version='${VERSION}'"
echo "Output directory: ${OUTPUT_DIR}"
echo "======================================================================"

# Determine variant extras
EXTRAS=""
if command -v python3 &>/dev/null; then
  EXTRAS=$(python3 -c "
import yaml
try:
    with open('${ROOT_DIR}/ironbank/variants.yaml') as f:
        c = yaml.safe_load(f)
    print(' '.join(c['variants']['${VARIANT}']['extras']))
except Exception:
    pass
" 2>/dev/null || true)
fi

if [[ -z "${EXTRAS}" ]]; then
  case "${VARIANT}" in
    cpu)   EXTRAS="cpu onnx opencv app" ;;
    cu126) EXTRAS="cu126 onnx-cu126 opencv app" ;;
    cu130) EXTRAS="cu130 onnx-cu130 opencv app" ;;
    *)     EXTRAS="${VARIANT} opencv app" ;;
  esac
fi

echo "Resolved extras: ${EXTRAS}"

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

WHEELS_DIR="${TEMP_DIR}/wheels"
mkdir -p "${WHEELS_DIR}"

# Build dataeval-flow wheel
echo "==> Building dataeval-flow wheel..."
if command -v uv &>/dev/null; then
  uv build --wheel --out-dir "${WHEELS_DIR}" "${ROOT_DIR}"
elif command -v pip &>/dev/null; then
  pip wheel --no-deps "${ROOT_DIR}" -w "${WHEELS_DIR}"
else
  echo "ERROR: Neither 'uv' nor 'pip' command found." >&2
  exit 1
fi

# Download/build all dependency wheels
echo "==> Resolving dependencies for variant: ${VARIANT}..."
REQ_FILE=""

if [[ -f "${ROOT_DIR}/requirements.${VARIANT}.txt" ]]; then
  echo "==> Found committed requirements file: requirements.${VARIANT}.txt"
  REQ_FILE="${ROOT_DIR}/requirements.${VARIANT}.txt"
elif command -v uv &>/dev/null; then
  echo "==> Exporting frozen requirements for extras: ${EXTRAS}..."
  EXTRA_FLAGS=()
  for e in ${EXTRAS}; do
    EXTRA_FLAGS+=(--extra "$e")
  done
  uv export --frozen --no-dev "${EXTRA_FLAGS[@]}" -o "${TEMP_DIR}/requirements.txt"
  REQ_FILE="${TEMP_DIR}/requirements.txt"
fi

if [[ -n "${REQ_FILE}" && -f "${REQ_FILE}" ]]; then
  # 1. Build VCS / git dependencies first into wheels (pip cannot hash git URLs in requirements files)
  echo "==> Checking for VCS/git dependencies in ${REQ_FILE}..."
  (grep -E '^[a-zA-Z0-9_-]+ @ git\+' "${REQ_FILE}" || true) | while read -r line; do
    if [[ -n "${line}" ]]; then
      pkg_spec=$(echo "$line" | sed 's/^[a-zA-Z0-9_-]* @ //')
      echo "==> Building wheel for VCS dependency: ${pkg_spec}"
      pip wheel --no-deps "${pkg_spec}" -w "${WHEELS_DIR}"
    fi
  done

  # 2. Filter out VCS dependencies so pip can run in strict hash-checking mode
  grep -v -E '^[a-zA-Z0-9_-]+ @ git\+' "${REQ_FILE}" > "${TEMP_DIR}/pypi_requirements.txt"

  # 3. Download remaining hashed wheels using --no-deps (all transitive deps are locked)
  echo "==> Downloading hashed dependency wheels into wheelhouse..."
  pip download --dest "${WHEELS_DIR}" --no-deps -r "${TEMP_DIR}/pypi_requirements.txt"

  # 4. If any source distribution (.tar.gz / .zip) was downloaded, convert it to a wheel
  shopt -s nullglob
  for sdist in "${WHEELS_DIR}"/*.tar.gz "${WHEELS_DIR}"/*.zip; do
    if [[ -f "$sdist" ]]; then
      echo "==> Building binary wheel from sdist: $(basename "$sdist")"
      pip wheel --no-deps "$sdist" -w "${WHEELS_DIR}"
      rm -f "$sdist"
    fi
  done
  shopt -u nullglob
else
  pip wheel --wheel-dir "${WHEELS_DIR}" "${ROOT_DIR}[${VARIANT}]"
fi

# Create tarball archive
ARCHIVE_NAME="dataeval-flow-${VERSION}-wheels-${VARIANT}.tar.gz"
echo "==> Archiving wheels into ${ARCHIVE_NAME}..."
tar -czf "${OUTPUT_DIR}/${ARCHIVE_NAME}" -C "${WHEELS_DIR}" .

# Compute SHA256 checksum
SHA256=$(sha256sum "${OUTPUT_DIR}/${ARCHIVE_NAME}" | awk '{print $1}')
echo "${SHA256}" > "${OUTPUT_DIR}/${ARCHIVE_NAME}.sha256"

echo "==> Archive created successfully:"
echo "    File:   ${OUTPUT_DIR}/${ARCHIVE_NAME}"
echo "    SHA256: ${SHA256}"

# Ensure Iron Bank templates have been generated
if [[ ! -d "${ROOT_DIR}/ironbank/${VARIANT}" ]]; then
  echo "==> Generating Iron Bank template files..."
  if command -v python3 &>/dev/null; then
    python3 "${ROOT_DIR}/ironbank/generate.py"
  fi
fi

# Stage files into output directory
if [[ -d "${ROOT_DIR}/ironbank/${VARIANT}" ]]; then
  echo "==> Staging Iron Bank repository files into ${OUTPUT_DIR}..."
  cp "${ROOT_DIR}/ironbank/${VARIANT}/README.md" "${OUTPUT_DIR}/README.md"
  cp "${ROOT_DIR}/ironbank/${VARIANT}/LICENSE" "${OUTPUT_DIR}/LICENSE"
  cp "${ROOT_DIR}/ironbank/${VARIANT}/entrypoint.sh" "${OUTPUT_DIR}/entrypoint.sh"
  cp "${ROOT_DIR}/ironbank/${VARIANT}/requirements.txt" "${OUTPUT_DIR}/requirements.txt"

  # Write Dockerfile with matching default version and archive name
  sed -e "s/ARG DATAEVAL_FLOW_VERSION=.*/ARG DATAEVAL_FLOW_VERSION=\"${VERSION}\"/" \
      -e "s/ARG WHEELHOUSE_ARCHIVE=.*/ARG WHEELHOUSE_ARCHIVE=\"${ARCHIVE_NAME}\"/" \
    "${ROOT_DIR}/ironbank/${VARIANT}/Dockerfile" > "${OUTPUT_DIR}/Dockerfile"

  # Write hardening_manifest.yaml with actual archive name and SHA256 checksum
  sed -e "s/value: .*/value: \"${SHA256}\"/" \
      -e "s/filename: .*/filename: \"${ARCHIVE_NAME}\"/" \
      -e "s|download/[^/]*/dataeval-flow-.*|download/v${VERSION}/${ARCHIVE_NAME}\"|" \
    "${ROOT_DIR}/ironbank/${VARIANT}/hardening_manifest.yaml" > "${OUTPUT_DIR}/hardening_manifest.yaml"
  
  # Also update the in-repo manifest
  sed -i "s/value: .*/value: \"${SHA256}\"/" "${ROOT_DIR}/ironbank/${VARIANT}/hardening_manifest.yaml" 2>/dev/null || true
fi

echo "======================================================================"
echo "Iron Bank bundle ready in: ${OUTPUT_DIR}"
echo "To test locally (with access to registry1.dso.mil):"
echo "  docker build -t dataeval-flow:${VERSION}-${VARIANT}-ironbank \\"
echo "    --build-arg WHEELHOUSE_ARCHIVE=\"${ARCHIVE_NAME}\" \\"
echo "    \"${OUTPUT_DIR}\""
echo "======================================================================"
