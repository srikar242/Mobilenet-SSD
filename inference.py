import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
import json
from flask import Flask, request, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/dev/shm/Images/'

app = Flask(__name__)
#print("App name ", app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "1234"


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image):
  # Get handles to input and output tensors
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)
  if 'detection_masks' in tensor_dict:
    # The following processing is only for single image
    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(
        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    # Follow the convention by adding back the batch dimension
    tensor_dict['detection_masks'] = tf.expand_dims(
        detection_masks_reframed, 0)
  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

  # Run inference
  sess=tf.Session()
  #with detection_graph.as_default():
    #with tf.Session() as sess:
  output_dict = sess.run(tensor_dict,
                          feed_dict={image_tensor: np.expand_dims(image, 0)})
  #t2=time.time()
  #time4=(t2-t1)
  

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def infer(image_path):
    with detection_graph.as_default():
        with tf.Session() as sess:
            #for image_path in TEST_IMAGE_PATHS:
            t1=time.time()
            t2=time.time()
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            print("*** For image {} ***".format(image_path)) 
            print("FPS for preprocessing:",1/(time.time()-t2))
            # Actual detection.
            t3=time.time()
            output_dict = run_inference_for_single_image(image_np)
            print("FPS for inference:",1/(time.time()-t3))
            
            # Visualization of the results of a detection.
            results = []
            for i,j in enumerate(output_dict.get('detection_scores')):
                dets = {}
                if j > 0.6:
                    dets['score'] = j
                    dets['rectangular_box'] = output_dict['detection_boxes'][i]
                    #dets['class'] = output_dict['detection_classes'][i]
                    dets['class'] = category_index[output_dict['detection_classes'][i]]['name']
                    results.append(dets)

            
            print("FPS for whole process:",1/(time.time()-t1))
            print("========================================================")
            return(results)
            #plt.figure(figsize=IMAGE_SIZE)
            #plt.imshow(image_np)


@app.route('/api/v1/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if file part is present
        if 'file' not in request.files:
            return app.response_class(
                response=json.dumps({'Error': 'No file part'}),
                status=400,
                mimetype='application/json')
        file = request.files['file']
        if file.filename == '':
            return app.response_class(
                reponse=json.dumps({'Error': 'No file selected'}),
                status=400,
                mimetype='application/json')               
        if file:
            filename = secure_filename(file.filename)
            try:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                full_name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                print("I am here.....")
                detections =   infer(full_name)
                print(detections)
                return app.response_class(
                    response=json.dumps(str(detections)),
                    status=200,
                    mimetype='application/json')
            except:
                print("Error I didn't reciiievee any file!")

if __name__ == "__main__":

    # patch tf1 into `utils.ops`
    utils_ops.tf = tf.compat.v1

    # Patch the location of gfile
    tf.gfile = tf.io.gfile

    # What model to download.
    MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT ='/home/srikar/Music/models/fine_tuned_model_100626/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = '/home/srikar/Music/models/annotations/label_map.pbtxt'

    NUM_CLASSES = 5

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    PATH_TO_TEST_IMAGES_DIR = '/dev/shm/Images/'
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 11) ]

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)
   
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

