#!/usr/bin/env bash

MODEL_DIR=/output/benchmark-models

#rm -rf ${MODEL_DIR}
mkdir -p ${MODEL_DIR}/output/expert/default
mkdir -p ${MODEL_DIR}/output/expert/centers
mkdir -p ${MODEL_DIR}/output/nonexpert/default
mkdir -p ${MODEL_DIR}/output/nonexpert/centers

for SEED in 4 8 15 16 23 42 48 1516 2342 4815
do
    for SAMPLE_RATE in 0.2 0.4 0.6 0.8 1.0
    do
        TRAIN_TFRECORD_FILE=/output/data-benchmark/train_s${SEED}_sr${SAMPLE_RATE}.tfrecord
        TRAIN_DIR=${MODEL_DIR}/model_s${SEED}_sr${SAMPLE_RATE}

        mkdir -p ${TRAIN_DIR}

        # create pipeline config from template
        PIPELINE_CONFIG=${TRAIN_DIR}/pipeline.config
        sed -e "s#\${TRAIN_TFRECORD_FILE}#${TRAIN_TFRECORD_FILE}#" \
            -e "s#\${EVAL_TFRECORD_FILE}#${EVAL_TFRECORD_FILE}#" \
            faster_rcnn_inception_resnet_v2_atrous_boneage.config.template \
        > ${PIPELINE_CONFIG}

        # train
        python -m object_detection.train \
            --train_dir ${TRAIN_DIR} \
            --pipeline_config_path=${PIPELINE_CONFIG}

        # freeze
        CHECKPOINT_PATH=$(head -n 1 ${TRAIN_DIR}/checkpoint | cut -d \" -f2)
        python -m object_detection.export_inference_graph \
            --input_type=image_tensor \
            --pipeline_config_path=${PIPELINE_CONFIG} \
            --trained_checkpoint_prefix=${CHECKPOINT_PATH} \
            --output_directory=${TRAIN_DIR}/exported

        # infer
        python -m object_detection.inference.infer_detections \
            --input_tfrecord_paths=/output/data/eval_nonexpert.tfrecord \
            --output_tfrecord_path=${TRAIN_DIR}/detection_nonexpert.tfrecord \
            --inference_graph=${TRAIN_DIR}/exported/frozen_inference_graph.pb \
            --discard_image_pixels

        python -m object_detection.inference.infer_detections \
            --input_tfrecord_paths=/output/data/eval_expert.tfrecord \
            --output_tfrecord_path=${TRAIN_DIR}/detection_expert.tfrecord \
            --inference_graph=${TRAIN_DIR}/exported/frozen_inference_graph.pb \
            --discard_image_pixels

        # evaluate
        echo "
        label_map_path: '/source/label_map.pbtxt'
        tf_record_input_reader: {input_path: '${TRAIN_DIR}/detection_expert.tfrecord' }
        " > ${TRAIN_DIR}/eval_input_config.pbtxt

        echo "
        metrics_set: 'open_images_metrics'
        " > ${TRAIN_DIR}/eval_config.pbtxt

        python -m object_detection.metrics.offline_eval_map_corloc \
            --eval_dir=${TRAIN_DIR} \
            --eval_config_path=${TRAIN_DIR}/eval_config.pbtxt \
            --input_config_path=${TRAIN_DIR}/eval_input_config.pbtxt

        cp ${TRAIN_DIR}/metrics.csv ${MODEL_DIR}/output/expert/default/metrics_s${SEED}_sr${SAMPLE_RATE}.csv

        echo "
        label_map_path: '/source/label_map.pbtxt'
        tf_record_input_reader: {input_path: '${TRAIN_DIR}/detection_nonexpert.tfrecord' }
        " > ${TRAIN_DIR}/eval_input_config.pbtxt

        python -m object_detection.metrics.offline_eval_map_corloc \
            --eval_dir=${TRAIN_DIR} \
            --eval_config_path=${TRAIN_DIR}/eval_config.pbtxt \
            --input_config_path=${TRAIN_DIR}/eval_input_config.pbtxt

        cp ${TRAIN_DIR}/metrics.csv ${MODEL_DIR}/output/nonexpert/default/metrics_s${SEED}_sr${SAMPLE_RATE}.csv

        # evaluate center points
        python helper/eval_detection_centers.py \
            --input_detections=${TRAIN_DIR}/detection_expert.tfrecord \
            --output_path=${MODEL_DIR}/output/expert/centers/metrics_s${SEED}_sr${SAMPLE_RATE}.csv

        python helper/eval_detection_centers.py \
            --input_detections=${TRAIN_DIR}/detection_nonexpert.tfrecord \
            --output_path=${MODEL_DIR}/output/nonexpert/centers/metrics_s${SEED}_sr${SAMPLE_RATE}.csv
    done
done