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
~$ python run_webcam_person_detector.py --score 0.75
```
* with DeepSORT tracker 
``` bash
~$ python run_webcam_yolov4_DeepSORT.py --score 0.75
```

<br/>

## How did I do?
Even though there is not a code to train yolov4-tiny, let me describe how I got the trained weights with my custom dataset:
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
* [```Darknet.weights```] ??? [```.tflite```]

If you succeed to take the weight file in ```.tflite``` format, we're ready to build an inference code for person detection using Tensorflow lite.

### 4. Make the inference codes like this repository for my own application
I built two inference codes; one for image input and another for webcam video input. These codes are referred to the original inference codes [here](https://github.com/hunglc007/tensorflow-yolov4-tflite);  ```detect.py``` and ```detectvideo.py```.


### (New)The above training processes is uploaded in [this repository](https://github.com/DoranLyong/Darknet-YOLOv4-Tensorflow_Lite-Tutorial)

***
## Reference 

[1] [TRAIN A CUSTOM YOLOv4-tiny OBJECT DETECTOR USING GOOGLE COLAB](https://medium.com/analytics-vidhya/train-a-custom-yolov4-tiny-object-detector-using-google-colab-b58be08c9593#a70f) / yolov4-tiny ?????? ????????? ????????? ??????.<br/>
[2] [How to Train a Custom Mobile Object Detection Model (with YOLOv4 Tiny and TensorFlow Lite)](https://blog.roboflow.com/how-to-train-a-custom-mobile-object-detection-model/) / YOLOv4-Tiny ??? ????????? ????????? TensorFlow Lite ??? ???????????? (?????????). <br/>
[3] [tensorflow-yolov4-tflite, github](https://github.com/hunglc007/tensorflow-yolov4-tflite) / ?????? ???????????? Darknet ????????? tensorflow lite ????????? ??????.<br/>
[4] [Yolo v4, v3 and v2 for Windows and Linux, github](https://github.com/AlexeyAB/darknet) / yolov4-tiny ?????? ????????? ????????? ?????? <br/>
[5] [AVA Dataset Processing for Person Detection, github](https://github.com/DoranLyong/AVA-Dataset-Processing-for-Person-Detection) / Person detection??? ??????????????? ?????? ?????? (????????? DB??? ?????????).  <br/>

## Further helpable readings 
[a] [theAIGuysCode, tensorflow-yolov4-tflite, github](https://github.com/theAIGuysCode/tensorflow-yolov4-tflite) / (ref3)??? ????????? ??????????????? <br/>
[b] [Object Detection and Tracking in 2020, netcetera blog](https://blog.netcetera.com/object-detection-and-tracking-in-2020-f10fb6ff9af3) / Detection ????????? Tracking ????????? ????????? ??? ???????????? <br/>
[c] [Real-time Human Detection in Computer Vision - Part2, medium](https://medium.com/@madhawavidanapathirana/real-time-human-detection-in-computer-vision-part-2-c7eda27115c6) / ?????? ?????? ????????? ?????? ??? ???????????? ??? ?????? ??????????????? ?????? <br/>
[d] [Object detection and tracking in PyTorch, towarddatascience](https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98) / ????????? ???????????? ?????? ?????? <br/>
[e] [Object Detection and Tracking, MediaPipe](https://google.github.io/mediapipe/solutions/box_tracking.html) / Google MediaPipe??? ???????????? Object detection??? ?????? ML inference ????????? Box Tracking ????????? ???????????? ??????????????? ???????????? ??????. ????????? ???????????? ??? ??????????????? detection inference??? ??? ????????? ????????????. <br/> 
[f] [Object-Detection-and-Tracking, github](https://github.com/yehengchen/Object-Detection-and-Tracking/tree/master/OneStage/yolo) / YOLOv4 + Deep_SORT ??? ?????? ???????????? ?????? (?????? ??????) <br/>
