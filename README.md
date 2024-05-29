# Real-time Object Instance Segmentation
## Introduction
YOLOv8 or You Only Look Once (version 8) is a state-of-the-art model in the YOLO family of object detection model.\
It is renowned for its speed and accuracy in real-time detection task and other tasks such as segmentation, classification, etc.
This repository aims to build an video segmentation webapp on top of the YOLOv8 model provided by [Ultralytics](https://docs.ultralytics.com/) and quick deploying with the help of [Streamlit](https://streamlit.io/).
## Technical Overview
+ Python 3.11.5.
+ Ultralytics library, YOLOv8 segmentation model.
+ OpenCV 4.9.0
## Installation
+ First clone the repo:
```
git clone https://github.com/ndtduy/object-seg-track-streamlit.git
```
+ On Windows, in the main directory of the repo, run the command:
```
pip install -r requirements.txt
```
to install all dependencies for the webapp. On Linux, we might have to install the *libgl1* external depedency for OpenCV: ```sudo apt install libgl1``` or ```apt-get install libgl1``` for example.
+ To run the webapp with Streamlit, use command:
```
streamlit run app.py
```
+ Quick deployment of Streamlit Community Cloud: [https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).
## Features and Usages
+ User can upload a video consists of objects to the webapp and see the segmentation result.
+ Live camera object segmentation (Sadly, only works on local machine). Segmentation of objects captured by the local machine camera are shown in real-time.
## References
+ [Ultralytics's example](https://docs.ultralytics.com/guides/instance-segmentation-and-tracking/).
+ [Streamlit demo app](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py).
+ [OpenCV VideoCapture Example](https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/).