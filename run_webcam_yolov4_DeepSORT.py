# -*- coding: utf-8 -*-


# %%
import os.path as osp 
import time 


import cv2 
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from absl import app, flags, logging  # argparse 대용인가?; (ref) https://github.com/abseil/abseil-py
from absl.flags import FLAGS
import tensorflow as tf 
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


#%% argparse 
flags.DEFINE_string('framework', 'tflite', 'TF lite' )
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416.tflite', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo-tiny moodel')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.45, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')


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


    """ init. Deep SORT object  
    """
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    # initialize tracker
    tracker = Tracker(metric)




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


        """ Convert data to numpy arrays and slice out unused elements
        """
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores_npy = scores.numpy()[0]
        scores_npy = scores_npy[0:int(num_objects)]
        classes_npy = classes.numpy()[0]
        classes_npy = classes_npy[0:int(num_objects)]

#        print(f"The number of objects: {num_objects}")

        """ format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        """
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)


        """ store all predictions in one parameter for simplicity when calling functions
        """
        pred_bbox = [bboxes, scores_npy, classes_npy , num_objects]


        """ read in all class names from config
        """
        class_names = utils.read_class_names(cfg.YOLO.CLASSES) 
        

        """loop through objects and use class index to get class name, allow only classes in allowed_classes list
        """
        names = []

        for i in range(num_objects):
            cls_indx = int(classes_npy[i])
            cls_name = class_names[cls_indx]

            names.append(cls_name)

        names = np.array(names)
        count = len(names)

        

        """ Tracking 
        """
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))


        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores_npy, names, features)]


        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()


            """ Draw bbox on screen 
            """
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)


            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))


        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - prev_time)
        print("FPS: %.2f" % fps)



        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not FLAGS.dont_show:
            cv2.imshow("Output", result)

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
