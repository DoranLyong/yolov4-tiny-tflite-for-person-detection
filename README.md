# Yolov4-tiny-tflite for Person Detection
This repository is about a person detection using yolov4-tiny-tflite. <br/>
It doesn't include how to train your custom dataset, but only the ```pretrained weights``` for detecting ```person class``` only.


<br/>

## Requirements 
Check the ```requirements.txt``` to install requested packages, or run like below:
``` bash
~$ pip install -r requirements.txt
```


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

## How did I do?
Even though there is not a code to train yolov4-tiny, let me describe how I get the trained weights with my custom dataset:
1. Achieve custom dataset from YouTube videos (using ```AVA``` dataset)
2. Train yolov4-tiny to detect only person class using the ```Darknet``` with the custom dataset
3. Convert the trained ```Darknet``` model to ```tensorflow lite``` model
4. Make the inference codes like this repository for my own application

<br/>

### 1. Achieve custom dataset from YouTube videos (using ```AVA``` dataset)
Go to [[ref5]](https://github.com/DoranLyong/AVA-Dataset-Processing-for-Person-Detection), then you can get to know how to prepare a custom dataset, which consists of only persons. ```AVA``` dataset has been used for multiple application tasks, but I only used this for getting ```person images``` with bounding box labels.


### 2. Train yolov4-tiny to detect only person class using the ```Darknet``` with the custom dataset
I got the idea that training person detector and uploading on my edge device from [roboflow blog, here](https://blog.roboflow.com/how-to-train-a-custom-mobile-object-detection-model/). They supply various and useful tools for data preprocessing, however, it's not free for massive datasets. Furthermore, I couldn't know ```how to set up my dataset``` for training ```yolov4-tiny``` just reading the blog.

So, I found out another awesome post in medium ([hear](https://medium.com/analytics-vidhya/train-a-custom-yolov4-tiny-object-detector-using-google-colab-b58be08c9593#a70f)). I followed the whole instruction of the post in order to train my model. Consequently, I trained my model using ```google colab```.


### 3. Convert the trained ```Darknet``` model to ```tensorflow lite``` model
After training own ```darknet yolov4-tiny``` model, we finally need to convert the darknet model to Tensorflow one. This is pretty much easy if you following [this github](https://github.com/hunglc007/tensorflow-yolov4-tflite) well. 
* [```Darknet.weights```] → [```.tflite```]

If you succeed to take the weight file in ```.tflite``` format, we're ready to build an inference code for person detection using Tensorflow lite.

### 4. Make the inference codes like this repository for my own application
I built two inference codes; one for image input and another for webcam video input. These codes are referred to the original inference codes [here](https://github.com/hunglc007/tensorflow-yolov4-tflite);  ```detect.py``` and ```detectvideo.py```.

***
## Reference 

[1] [TRAIN A CUSTOM YOLOv4-tiny OBJECT DETECTOR USING GOOGLE COLAB](https://medium.com/analytics-vidhya/train-a-custom-yolov4-tiny-object-detector-using-google-colab-b58be08c9593#a70f) / yolov4-tiny 학습 방법은 여기를 참고.<br/>
[2] [How to Train a Custom Mobile Object Detection Model (with YOLOv4 Tiny and TensorFlow Lite)](https://blog.roboflow.com/how-to-train-a-custom-mobile-object-detection-model/) / YOLOv4-Tiny 로 학습한 모델을 TensorFlow Lite 로 바꿔보기 (블로그). <br/>
[3] [tensorflow-yolov4-tflite, github](https://github.com/hunglc007/tensorflow-yolov4-tflite) / 이걸 참고해서 Darknet 모델을 tensorflow lite 모델로 바꿈.<br/>
[4] [Yolo v4, v3 and v2 for Windows and Linux, github](https://github.com/AlexeyAB/darknet) / yolov4-tiny 학습 코드는 여기걸 참고 <br/>
[5] [AVA Dataset Processing for Person Detection, github](https://github.com/DoranLyong/AVA-Dataset-Processing-for-Person-Detection) / Person detection용 학습데이터 얻는 방법 (이걸로 DB를 구축함).  <br/>

## Further helpable readings 
[a] [theAIGuysCode, tensorflow-yolov4-tflite, github](https://github.com/theAIGuysCode/tensorflow-yolov4-tflite) / (ref3)을 참고한 레포지토리 <br/>
[b] [Object Detection and Tracking in 2020, netcetera blog](https://blog.netcetera.com/object-detection-and-tracking-in-2020-f10fb6ff9af3) / Detection 모듈과 Tracking 모듈을 조합할 때 참고하자 <br/>
[c] [Real-time Human Detection in Computer Vision - Part2, medium](https://medium.com/@madhawavidanapathirana/real-time-human-detection-in-computer-vision-part-2-c7eda27115c6) / 사람 감지 모듈을 만들 때 생각해볼 수 있는 파이프라인 소개 <br/>
[d] [Object detection and tracking in PyTorch, towarddatascience](https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98) / 심플한 튜토리얼 코드 제공 <br/>
[e] [Object Detection and Tracking, MediaPipe](https://google.github.io/mediapipe/solutions/box_tracking.html) / Google MediaPipe의 경우에도 Object detection을 위한 ML inference 모듈과 Box Tracking 모듈을 조합해서 프로세싱에 효율성을 더함. 이렇게 구성하면 매 프레임마다 detection inference를 할 필요가 없어진다. <br/> 