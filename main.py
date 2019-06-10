# a minimal version of tensorflow object detection API tutorial:
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

from utils import *
from PIL import Image


frozen_model_path = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'

label_file_path = 'mscoco_label_map.pbtxt'
label_dict = load_pbtxt_to_label_dict(label_file_path)

sess = tf.Session()

load_model_to_session(sess, frozen_model_path)

image_path = 'image1.jpg'
image = Image.open(image_path)
image_np = image_to_np_array(image)
image_in = np.expand_dims(image_np, axis=0)  # input ready

output_dict = inference_one_image(sess, image_in)

boxes = output_dict['detection_boxes']
classes = output_dict['detection_classes']
scores = output_dict['detection_scores']

draw_bounding_boxes_on_image_array(image_np, boxes, classes, scores, 0.5, label_dict)
