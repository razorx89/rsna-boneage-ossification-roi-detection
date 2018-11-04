import argparse
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_detections', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()

    accuracy = {
        i: {
            'num_match': 0,
            'num_total': 0,
        } for i in range(1, 7)
    }

    for record in tf.python_io.tf_record_iterator(args.input_detections):
        example = tf.train.Example()
        example.ParseFromString(record)

        # Image info
        image_width = example.features.feature['image/width'].int64_list.value[0]
        image_height = example.features.feature['image/height'].int64_list.value[0]
        shape = np.asarray([image_height, image_width])

        # Groundtruth
        gt_xmin = np.asarray(example.features.feature['image/object/bbox/xmin'].float_list.value)
        gt_xmax = np.asarray(example.features.feature['image/object/bbox/xmax'].float_list.value)
        gt_ymin = np.asarray(example.features.feature['image/object/bbox/ymin'].float_list.value)
        gt_ymax = np.asarray(example.features.feature['image/object/bbox/ymax'].float_list.value)
        gt_xc = gt_xmin + (gt_xmax - gt_xmin) / 2.0
        gt_yc = gt_ymin + (gt_ymax - gt_ymin) / 2.0
        gt_center = np.round(np.stack([gt_yc, gt_xc], axis=-1) * shape)
        gt_label = np.asarray(example.features.feature['image/object/class/label'].int64_list.value)

        # Detection
        det_xmin = np.asarray(example.features.feature['image/detection/bbox/xmin'].float_list.value)
        det_xmax = np.asarray(example.features.feature['image/detection/bbox/xmax'].float_list.value)
        det_ymin = np.asarray(example.features.feature['image/detection/bbox/ymin'].float_list.value)
        det_ymax = np.asarray(example.features.feature['image/detection/bbox/ymax'].float_list.value)
        det_xc = det_xmin + (det_xmax - det_xmin) / 2.0
        det_yc = det_ymin + (det_ymax - det_ymin) / 2.0
        det_center = np.round(np.stack([det_yc, det_xc], axis=-1) * shape)
        det_label = np.asarray(example.features.feature['image/detection/label'].int64_list.value)
        det_score = np.asarray(example.features.feature['image/detection/score'].float_list.value)

        # Filter detections for at least 50% confidence
        valid_detections = np.where(det_score >= 0.5)
        det_center = det_center[valid_detections]
        det_label = det_label[valid_detections]
        det_score = det_score[valid_detections]

        # Calculate threshold based on paper metrics
        threshold = image_height * (6.0 / 256.0)

        #print('threshold: %f' % threshold)

        # Compute distances
        for cls in range(1, 7):
            #print('cls: %d' % cls)
            distances = np.sqrt(
                np.sum(
                    np.power(
                        gt_center[gt_label == cls, np.newaxis, :] - det_center[det_label == cls],
                        2.0
                    ),
                    axis=-1
                )
            )

            accuracy[cls]['num_total'] += sum(det_label == cls)  # sum(gt_label == cls)
            try:
                matches = np.min(distances, axis=1) < threshold
                if sum(det_label == cls) < sum(matches):
                    print(distances)
                    print(np.argmin(distances, axis=1))
                    print(sum(det_label == cls), sum(matches))
                accuracy[cls]['num_match'] += sum(matches)
            except Exception:
                pass

            #print(distances)
            #print('')

    class_names = ['Background', 'DIP', 'PIP', 'MCP', 'Radius', 'Ulna', 'Wrist']

    with open(args.output_path, 'w') as ofile:
        map = 0.0
        for cls in range(1, 7):
            ap = accuracy[cls]['num_match'] / float(accuracy[cls]['num_total'])
            ofile.write('AveragePrecision/%s,%f\n' % (class_names[cls], ap))
            map += ap
        map /= 6.0
        ofile.write('mAP,%f\n' % map)
