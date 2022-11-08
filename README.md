# A Face Recognition Program Based on facenet and MTCNN
* Using the C++programming language.
* Only depend on opencv library.
* Real time face detection and recognition.

## Dependencies
* opencv4.4.0

The opencv version must be greater than 4.0.0, otherwise the model may not be loaded.
___
## Offical pre-trained faceNet models
| Model name                                                                    | LFW accuracy | Training dataset | Architecture |
|-------------------------------------------------------------------------------|--------------|------------------|-------------|
| [Facenet](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

The facenet model cannot be directly loaded by opencv. [Here](https://github.com/TanFluent/facenet_opencv_dnn) is the processing method.

|Model name| Download  |
|---|-----------|
|MTCNN|[caffe](https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code/codes/MTCNNv1/model)|

Fortunately, mtcnn can be directly loaded by opencv.
We provide the processed model and executable program. \
[Baidu Disk](https://pan.baidu.com/s/1FrDNs3ijDBYUfgjHVzcfUw) code:fd46 [Google Disk](https://drive.google.com/file/d/1lXTJV8bcvONNGn1PnVdkTYFfcES2Z0eU/view?usp=sharing)

## Implement
### [STEP 1] Put the model file in the model folder
### [STEP 2] Prepare Datasets and Labels
Put the image(.jpg) to be recognized into test/dataset.
Set the picture label in test/dataset/label.txt.
format:imagename-label
### [STEP 3] Run program
Wait for the model to load. Enter the path of the picture or video to be detected and recognized. Enter the detection mode. (0: camera, 1: video, 2: picture).
format:path mode \
![result](test/result.jpg)

Because opencv overwrites the image, it may cause an antivirus software alarm.
___
## Reference 
https://github.com/TanFluent/facenet_opencv_dnn \
https://blog.csdn.net/qq_33221533/article/details/125270236
