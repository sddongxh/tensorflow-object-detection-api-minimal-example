# a minimal version of tensorflow object detection API tutorial:
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

from utils import *
from PIL import Image


#PATH_TO_FROZEN_GRAPH = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
frozen_model_path = '../frozen_models/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb'

sess = tf.Session()
detection_graph = sess.graph

load_model_to_session(sess, frozen_model_path)

image_path = 'test_images/image1.jpg'
image = Image.open(image_path)
image_np = image_to_np_array(image)
image_in = np.expand_dims(image_np, axis=0)  # input ready

output_dict = inference_one_image(sess, image_in)

boxes = output_dict['detection_boxes']
classes = output_dict['detection_classes']
scores = output_dict['detection_scores']

draw_bounding_boxes_on_image_array(image_np, boxes, classes, scores)
