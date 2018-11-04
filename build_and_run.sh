#!/usr/bin/env bash

[ -z "${RSNA_BONEAGE_DATASET_ROOT}" ] && echo "You need to specify the RSNA_BONEAGE_DATASET_ROOT environment variable" && exit 1
[ ! -d "${RSNA_BONEAGE_DATASET_ROOT}" ] && echo "Invalid RSNA_BONEAGE_DATASET_ROOT path?" && exit 1

DOCKER_NAME=rsna-boneage-ossification-roi-detection
OUTPUT_DIR=$(pwd)/output
echo $OUTPUT_DIR

mkdir -p ${OUTPUT_DIR}

docker build -t ${DOCKER_NAME} .

nvidia-docker run \
    -it \
    --rm \
    -v=/etc/passwd:/etc/passwd:ro \
    -v=/etc/group:/etc/group:ro \
    -v=${RSNA_BONEAGE_DATASET_ROOT}:/data:ro \
    -v=${OUTPUT_DIR}:/output \
    --user=$(id -u) \
    ${DOCKER_NAME}:latest

