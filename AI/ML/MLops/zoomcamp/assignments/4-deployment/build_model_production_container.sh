#!/usr/bin/env bash
set -euo pipefail

# Match the Docker base image: agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim
PYTHON_VERSION="3.10"
SKLEARN_VERSION="1.5.0"
IMAGE_NAME="mlops-zoomcamp-hw4"
CONDA_ENV_NAME="py310"

echo "==> Cleaning previous Pipfile / Pipfile.lock"
rm -f Pipfile Pipfile.lock

# Find or create a Python 3.10 interpreter
if command -v python3.10 &>/dev/null; then
    PYTHON_BIN="$(command -v python3.10)"
elif command -v conda &>/dev/null; then
    if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "==> Creating conda env ${CONDA_ENV_NAME} with Python ${PYTHON_VERSION}"
        conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y
    fi
    PYTHON_BIN="$(conda run -n ${CONDA_ENV_NAME} which python)"
else
    echo "ERROR: no Python ${PYTHON_VERSION} and no conda available" >&2
    exit 1
fi

echo "==> Creating pipenv with ${PYTHON_BIN}"
pipenv --python "${PYTHON_BIN}"

echo "==> Installing dependencies"
pipenv install \
    "scikit-learn==${SKLEARN_VERSION}" \
    pandas \
    pyarrow

echo "==> Normalizing Pipfile python_version to ${PYTHON_VERSION}"
sed -i "s/python_version = \"[0-9.]*\"/python_version = \"${PYTHON_VERSION}\"/" Pipfile
sed -i '/python_full_version/d' Pipfile

echo "==> Regenerating Pipfile.lock"
pipenv lock

echo ""
echo "==> Q4 answer — first scikit-learn hash:"
python3 -c "import json; print(json.load(open('Pipfile.lock'))['default']['scikit-learn']['hashes'][0])"

echo ""
echo "==> Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .

echo ""
echo "==> Done."
echo ""
echo "      docker run --rm ${IMAGE_NAME} 2023 5"