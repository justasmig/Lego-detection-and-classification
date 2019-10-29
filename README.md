# Lego-detection-and-classification
Lego parts detection and classification. Developed using OpenCV and Tensorflow

## Description of folders structure and usage
- [croppedImages](croppedImages) - collection of automatically cropped Lego images, used to retrain Tensorflow model (model used in retraining: (https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/3))
- [images](images) - photo collection of Lego pieces
- [retrained](retrained) - location of retrained [graph](retrained/retrainedLego_graph.pb), graph's [labels.txt](retrained/labels.txt) and graph's [labels.pbtxt](retrained/retrainedLegoMap.pbtxt) files.
- [tmp](tmp) - location of temporary files created in retraining process.
- [Detect Objects Script](DetectObject.py) - main script used to detect and classify Lego objects using camera.
- [Command for retraining](Command_for_retraining.txt) - command I used to retrain Tensorflow model (model used in retraining: (https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/3))

## Requirements and usage of [Detect Objects Script](DetectObject.py)
### Tools:
[Anaconda with Python 3.7.x](https://www.anaconda.com/distribution/)

### Dependencies:

To meet all dependencies, execute these commands in your Anaconda environment:
```
conda install -c conda-forge opencv
pip install tensorflow==1.15.0
            ^^^^^^^^^^ tensorflow-gpu can be used, but setupping will be different
pip install numpy keyboard
```

### Usage:

Execute DetectObject.py script in Anaconda environment and in root directory of repository:

```
python DetectObject.py
```
