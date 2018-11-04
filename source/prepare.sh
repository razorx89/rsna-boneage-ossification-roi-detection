#!/usr/bin/env bash

mkdir -p /output/pretrained

if [ ! -d "/workspace/pretrained/faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08/" ]; then
    wget -qO- http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08.tar.gz | tar vxz -C /output/pretrained
fi

# Generate subsampled training tfrecords
mkdir -p /output/data-benchmark

python helper/xml_to_csv.py \
    --xml_dir=/data/train/annotations \
    --output_path=/output/data-benchmark/train_labels.csv

for SEED in 4 8 15 16 23 42 48 1516 2342 4815
do
    for SAMPLE_RATE in 0.2 0.4 0.6 0.8 1.0
    do
        python helper/generate_tfrecord.py \
            --image_root /data/train/images \
            --csv_input /output/data-benchmark/train_labels.csv \
            --output_path /output/data-benchmark/train_s${SEED}_sr${SAMPLE_RATE}.tfrecord \
            --seed=${SEED} \
            --sample=${SAMPLE_RATE}
    done
done

# Generate training tfrecord
mkdir -p /output/data

python helper/xml_to_csv.py \
    --xml_dir=/annotations/train/ \
    --output_path=/output/data/train_labels.csv

python helper/generate_tfrecord.py \
    --image_root /data/train/images \
    --csv_input /output/data/train_labels.csv \
    --output_path /output/data/train.tfrecord

# Generate nonexpert evaluation tfrecord
python helper/xml_to_csv.py \
    --xml_dir=/annotations/eval/nonexpert/ \
    --output_path=/output/data/eval_nonexpert_labels.csv

python helper/generate_tfrecord.py \
    --image_root /data/train/images \
    --csv_input /output/data/eval_nonexpert_labels.csv \
    --output_path /output/data/eval_nonexpert.tfrecord

# Generate expert evaluation tfrecord
python helper/xml_to_csv.py \
    --xml_dir=/annotations/eval/expert/ \
    --output_path=/output/data/eval_expert_labels.csv

python helper/generate_tfrecord.py \
    --image_root /data/train/images \
    --csv_input /output/data/eval_expert_labels.csv \
    --output_path /output/data/eval_expert.tfrecord