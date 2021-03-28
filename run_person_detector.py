# -*- coding: utf-8 -*-


# %%
import os.path as osp 


import cv2 
import numpy as np 
from PIL import Image
from absl import app, flags, logging  # argparse 대용인가?; (ref) https://github.com/abseil/abseil-py
from absl.flags import FLAGS
import tensorflow as tf 
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import core.utils as utils
from core.yolov4 import filter_boxes


#%% argparse 
flags.DEFINE_string('framework', 'tflite', 'TF lite' )
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416.tflite', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo-tiny moodel')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/persons.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.75, 'score threshold')

# ================================================================= #
#                         1. Set device                             #
# ================================================================= #
# %% 01. 프로세스 장비 설정 
physical_devices = tf.config.list_physical_devices('GPU')   # GPU 장치 목록 출력; 
                                                            # (ref) https://stackoverflow.com/questions/58956619/tensorflow-2-0-list-physical-devices-doesnt-detect-my-gpu

if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

else: 
    print("No GPU")




# ================================================================= #
#                             Functions                             #
# ================================================================= #
#%% 
def model_inference(image_list):

    interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

#    print(input_details)
#    print(output_details)

    interpreter.set_tensor(input_details[0]['index'], image_list)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([FLAGS.size, FLAGS.size]))

    return  boxes, pred_conf 



#%% 
def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True    
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    image_path = FLAGS.image


    """ 이미지 불러오기 
    """
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) # convert BGR to RGB for 'PIL'

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.


    """ 모델 입력을 위해 데이터를 리스트에 담음 
    """
    image_list = []

    for i in range(1):
        image_list.append(image_data)
    image_list = np.asarray(image_list).astype(np.float32)    


    """ Model inference 
    """
    boxes, pred_conf = model_inference(image_list)



    """ bbox 얻기 
    """
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape( pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
        )    

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = utils.draw_bbox(original_image,  pred_bbox)


    """ 결과 출력 
    """
    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(FLAGS.output, image)








# ================================================================= #
#                               Main                                #
# ================================================================= #
# %%

if __name__ == '__main__':
    try:
        app.run(main)

    except SystemExit:
        pass 