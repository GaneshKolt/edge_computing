import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from imageio import imread
import numpy as np
import tensorflow as tf
from io import BytesIO

from PIL import Image
import re, time, base64

from random import randint

app = Flask(__name__)

# Adding Cross Origin Resource Sharing to allow requests made from the front-end
# to be successful.
CORS(app)

# Defining the model configuration files.
# Change these files to add your own model!
dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_DETECT_PATH = dir_path + '/model/frozen_inference_graph.pb'

##########################################################
# Menu of loaded groceries, and their respective prices. #
##########################################################


menu = {'item' : 
    { '1':'colgatetoothpaste big',
    '2' : 'colgatetoothpaste small',
    '3': 'gokulsantol small',
    '4': 'gokulsantol big',
    '5': 'darkfantasy small',
    '6': 'darkfantasy big',
    '7': 'boost small',
    '8': 'boost big',
    '9': 'lgasafoetidapowder small',
    '10': 'lgasafoetidapowder large'
    },
    'price' : { '1':25.00,
    '2': 10.00,
    '3': 15.00,
    '4': 35.00,
    '5': 25.00,
    '6': 45.00,
    '7': 40.00,
    '8': 70.00,
    '9': 80.00,
    '10': 120.00 }
    }

##################################################
# Utilities
##################################################


def getI420FromBase64(codec):
    """ Convert image from a base64 bytes stream to an image. """
    base64_data = re.sub(b'^data:image/.+;base64,', b'', codec)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img


def readLabels():
    # Read each line of label file and strip \n
    labels = [label.rstrip('\n') for label in open(LABEL_PATH)]
    return labels


def apiResponseCreator_det(inputs, outputs):
	return dict(list(zip(inputs,outputs)))


def apiResponseCreator(labels, classifications):
    return dict(zip(labels, classifications))


def printTensors(model_file):
    # read protobuf into graph_def
    with tf.gfile.GFile(model_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for operation in graph.get_operations():
        print(operation.name)


##################################################
# REST API Endpoints For Web App
##################################################


@app.route('/')
def homepage():
    return 'This backend serves as a REST API for the React front end. Try running npm start from the self-checkout folder.'


@app.route('/detection', methods=['POST'])
def detection():
    request.get_data()
    
    # Load in an image to object detect and preprocess it
    img_data = getI420FromBase64(request.data)
    x_input = np.expand_dims(img_data, axis=0) # add an extra dimention.

    # Setting initial detection time, so execution time can be calculated.    
    t_det = time.time()

    # Get the predictions (output of the softmax) for this image
    tf_results_det = sess_det.run([output_tensor_det,detection_boxes,detection_scores,detection_num], {input_tensor_det : x_input})

    dt_det = time.time() - t_det
    app.logger.info("Execution time: %0.2f" % (dt_det * 1000.))

    # Different results arrays
    predictions_det = tf_results_det[0]
    prediction_scores_det=tf_results_det[2]
    prediction_boxes_det=tf_results_det[1]
    prediction_num_det=tf_results_det[3]

    # print("----------------------------")
    # print(predictions_det)
    # print(prediction_scores_det)
    # print("----------------------------")

    threshold = 0.85

    num=int(prediction_num_det)
    predict_list=predictions_det[0].astype(int).tolist()
    scores = prediction_scores_det[0]
    label=[]

    for i in range(num):
        new_item = {}
        if scores[i] > threshold:
            prediction_label = str(predict_list[i])
            obj_name = menu['item'][prediction_label]
            obj_price = menu['price'][prediction_label]

            new_item = {'id': randint(0, 100000),
                        'name': obj_name,
                        'quantity': 1,
                        'price': obj_price}

            label.append(new_item)
           
    print("number and list of items that above the threshold")
    print(len(label))
    print(label)
    return jsonify(label)


##################################################
# Starting the server
##################################################


if __name__ == '__main__':
    print('Starting TensorFlow Server')

    print('Configuring TensorFlow Graph..')
    # Create the session that we'll use to execute the model
    sess_config = tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement = True
    )

    print('Loading Model...')
    #G Read the graph definition file
    with open(MODEL_DETECT_PATH, 'rb') as k:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(k.read())

    #G Load the graph stored in `graph_def` into `graph`
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    #G Enforce that no new nodes are added
    graph.finalize()

    sess_det = tf.Session(graph=graph, config=sess_config)

    print('Done.')

    input_op_det = graph.get_operation_by_name('image_tensor') # mport/Preprocessor/map/Shape
    input_tensor_det = input_op_det.outputs[0]
    output_op_det = graph.get_operation_by_name('detection_classes')#or num_detections
    output_tensor_det = output_op_det.outputs[0]

    detection_boxes_op=graph.get_operation_by_name('detection_boxes')
    detection_boxes=detection_boxes_op.outputs[0]
    detection_scores_op=graph.get_operation_by_name('detection_scores')
    detection_scores=detection_scores_op.outputs[0]
    detection_num_op=graph.get_operation_by_name('num_detections')
    detection_num=detection_num_op.outputs[0]

    app.run(debug=False, host='0.0.0.0')
