import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def image_to_np_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def load_model_to_session(sess, pb_model_path):
    od_graph_def = sess.graph_def
    with tf.gfile.GFile(pb_model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def inference_one_image(sess, image):
    graph = sess.graph
    image_tensor = graph.get_tensor_by_name('image_tensor:0')  # input
    feed_dict = {image_tensor: image}
    ops = graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = dict()
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = graph.get_tensor_by_name(tensor_name)
    output_dict = sess.run(tensor_dict, feed_dict=feed_dict)
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict


def draw_bounding_boxes_on_image_array(image_np, boxes, classes, scores, min_score_threshold=0.5): 
    height = image_np.shape[0]
    width = image_np.shape[1]

    # Create figure and axes
    fig, ax = plt.subplots()
    ax.imshow(image_np)
    for i in range(boxes.shape[0]): 
        if scores[i] < min_score_threshold:
            continue
        box = tuple(boxes[i].tolist())
        y0 = int(box[0]*height)
        x0 = int(box[1]*width)
        y1 = int(box[2]*height)
        x1 = int(box[3]*width)
        # display_str = str(classes[i]) + ', ' + str(round(scores[i]*100))
        # Create a Rectangle patch
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor='r', facecolor='none', fill=False)
        ax.add_patch(rect)

    plt.show()
