# -*- coding: utf-8 -*-


# %%
import os.path as osp 
import time 


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
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.45, 'score threshold')
flags.DEFINE_boolean('dis_cv2_window', False, 'disable cv2 window during the process') # this is good for the .ipynb


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

def model_inference(image_input, interpreter, input_details, output_details ):

    interpreter.set_tensor(input_details[0]['index'], image_input)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([FLAGS.size, FLAGS.size]))

    return  boxes, pred_conf 



def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True    
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size


    """ init. Webcam object 
    """
    vid  = cv2.VideoCapture(0)   # (ref) https://076923.github.io/posts/Python-opencv-2/
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    """ Setup model 
    """
    interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    """ Video frame input 
    """
    frame_id = 0

    while True:
        return_value, frame  = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

#            cv2.imshow("VideoFrame_original", frame)

        else: 
            raise ValueError("No image! Try with another video format")

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()        


        """ Inference 
        """
        boxes, pred_conf = model_inference(image_data, interpreter, input_details, output_details)



        """ Post Processing 
        """
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
            )        


        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
#        print(info)           


        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not FLAGS.dis_cv2_window:
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break        









    vid.release()
    cv2.destroyAllWindows()





# ================================================================= #
#                               Main                                #
# ================================================================= #
# %%

if __name__ == '__main__':
    try:
        app.run(main)

    except SystemExit:
        pass 
