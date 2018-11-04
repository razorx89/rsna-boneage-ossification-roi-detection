"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import cv2
import numpy as np
import os
import io
import pandas as pd
import tensorflow as tf

from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

from image_preprocessing import preprocess_image


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'DIP':
        return 1
    elif row_label == 'PIP':
        return 2
    elif row_label == 'MCP':
        return 3
    elif row_label == 'Radius':
        return 4
    elif row_label == 'Ulna':
        return 5
    elif row_label == 'Wrist':
        return 6
    else:
        print(row_label)
        return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    print(os.path.join(path, group.filename))
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = cv2.imdecode(np.fromstring(encoded_jpg_io.read(), np.uint8), 0)
    height, width = image.shape
    image = preprocess_image(image)
    #image = np.expand_dims(image[:, :, 0], axis=0)
    _, encoded = cv2.imencode('.jpg', image)

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded.tostring()),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/group_of': dataset_util.int64_list_feature(0),
        'image/object/difficult': dataset_util.int64_list_feature(0),
    }))
    return tf_example


def main(args):
    writer = tf.python_io.TFRecordWriter(args.output_path)
    examples = pd.read_csv(args.csv_input)
    grouped = split(examples, 'filename')

    if args.seed:
        print('Using seed for shuffling: %d' % args.seed)
        np.random.seed(args.seed)
        perm = np.random.permutation(range(len(grouped)))
        grouped = [grouped[x] for x in perm]

    if args.sample and args.sample > 0.0 and args.sample < 1.0:
        print('Sample rate: %f' % args.sample)
        count = int(args.sample * len(grouped))
        grouped = grouped[:count]

    for group in grouped:
        tf_example = create_tf_example(group, args.image_root)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), args.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', required=True, help='Path to the images')
    parser.add_argument('--csv_input', required=True, help='Path to the CSV input')
    parser.add_argument('--output_path', required=True, help='Path to output TFRecord')
    parser.add_argument('--seed', type=int, default=None, help='Seed for shuffling')
    parser.add_argument('--sample', type=float, default=None, help='Value between 0 and 1 for sampling the images')
    args = parser.parse_args()

    main(args)
