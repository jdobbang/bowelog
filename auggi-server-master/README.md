This package contains a Flask server to take an image file path and return a bristol scale prediction.


Download and put the following models into the "saved_models" directory of this repo, and the model will be ready.

Segmentation model .pth from drive:

1.  https://drive.google.com/open?id=1YUtcrwEOY99keN-GhVUBvP11_D9BgPjU

Classifier model .pth from drive:

2.  https://drive.google.com/open?id=1t2Lbwx3-GbxdK0ONOnBn8gRYuVo-cBg1


# To run
run auggi_server.py - decides when to run an inference with an image.  Creates an Inference object and passes an image.


About each script:

full_inference.py - contains the main Inference class that creates all the necessary module objects to classify an image.

segmentor.py - produces a binary mask from an image

bbox_from_mask.py - produces bounding box coordinates from a binary mask

classifier_bristol.py - produces a bristol scale prediction from an image path and bounding box coordinates


This model requires the following dependancies (use a virtual environment):

Python 3.6

Pytorch

imutils==0.5.1

Flask==1.0.2

numpy==1.14.5

opencv-python==3.4.2.17

Pillow==5.2.0

torchvision==0.2.1


