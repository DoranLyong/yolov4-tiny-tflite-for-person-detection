# Yolov4-tiny-tflite for Person Detection
This repository is about a person detection using yolov4-tiny-tflite. <br/>
It doesn't include how to train your custom dataset, but only the ```pretrained weights``` for detecting ```person class``` only.


<br/>

## Inference with a image
* run the code like below 
* the input arguments can be changed by yourself
```bash
~$ python run_person_detector.py --image ./data/persons.jpg --score 0.75
```

<br/>


## Inference with your webcam device 
* run the code like below 
* the input arguments can be changed by yourself
```bash
~$ python run_video_person_detector.py --score 0.75
```

<br/>




***
## Reference 

[1] [TRAIN A CUSTOM YOLOv4-tiny OBJECT DETECTOR USING GOOGLE COLAB](https://medium.com/analytics-vidhya/train-a-custom-yolov4-tiny-object-detector-using-google-colab-b58be08c9593#a70f) / 학습 방법은 여기를 참고.<br/>
[2] [How to Train a Custom Mobile Object Detection Model (with YOLOv4 Tiny and TensorFlow Lite)](https://blog.roboflow.com/how-to-train-a-custom-mobile-object-detection-model/) / YOLOv4-Tiny 로 학습한 모델을 TensorFlow Lite 로 바꿔보기. <br/>
[3] [tensorflow-yolov4-tflite, github](https://github.com/hunglc007/tensorflow-yolov4-tflite) / 이걸 참고해서 Darknet 모델을 tensorflow lite 모델로 바꿈.<br/>
[4] [AVA Dataset Processing for Person Detection, github](https://github.com/DoranLyong/AVA-Dataset-Processing-for-Person-Detection) / Person detection 용 학습데이터 얻는 방법 (이걸로 DB를 구축함).  <br/>