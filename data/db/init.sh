#!/bin/bash

set -e

SCRIPT_DIR_MINGW="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$SCRIPT_DIR_MINGW" == /c/* ]]; then
    SCRIPT_DIR_WINDOWS="C:${SCRIPT_DIR_MINGW#/c}"
else
    echo "ERROR: Script is not on the C: drive or path conversion failed."
    echo "SCRIPT_DIR_MINGW was: $SCRIPT_DIR_MINGW"
    exit 1
fi

cd "$SCRIPT_DIR_MINGW"


IMAGE_NAME="quant-sqlite-db"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
HOST_VOLUME_SUBDIR="volume"

HOST_VOLUME_PATH_WINDOWS="${SCRIPT_DIR_WINDOWS}/${HOST_VOLUME_SUBDIR}"
CONTAINER_DB_PATH="/db"

check_image_exists() {
    if docker image inspect "$FULL_IMAGE_NAME" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

build_image() {
    echo "--- Building Docker image: $FULL_IMAGE_NAME ---"
    echo "Using build context: $(pwd)" # $(pwd) here is fine for build context
    if docker build -t "$FULL_IMAGE_NAME" .; then
        echo "Docker image built successfully."
    else
        echo "ERROR: Docker build failed."
        exit 1
    fi
}

run_container() {
    echo "--- Running Docker container from image: $FULL_IMAGE_NAME ---"
    echo "Mounting host path '${HOST_VOLUME_PATH_WINDOWS}' to container path '${CONTAINER_DB_PATH}'"
    if docker run --rm \
        -v "${HOST_VOLUME_PATH_WINDOWS}:${CONTAINER_DB_PATH}" \
        "$FULL_IMAGE_NAME"; then
        echo "Docker container ran successfully."
        echo "Database should be initialized at host path corresponding to: ${HOST_VOLUME_PATH_WINDOWS}/quant.db"
    else
        echo "ERROR: Docker container failed to run."
        exit 1
    fi
}

echo "--- Initializing Quant DB Setup ---"
echo "Script directory (MINGW): $SCRIPT_DIR_MINGW"
echo "Script directory (Windows): $SCRIPT_DIR_WINDOWS"
echo "Target image: $FULL_IMAGE_NAME"

echo "Ensuring host volume directory exists: ${SCRIPT_DIR_MINGW}/${HOST_VOLUME_SUBDIR}"
mkdir -p "${SCRIPT_DIR_MINGW}/${HOST_VOLUME_SUBDIR}"

if check_image_exists; then
    echo "Image '$FULL_IMAGE_NAME' already exists. Skipping build."
else
    echo "Image '$FULL_IMAGE_NAME' not found."
    build_image
fi

run_container

echo "--- Quant DB Setup Finished ---"
