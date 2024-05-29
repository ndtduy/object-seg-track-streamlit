import os
import uuid
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import streamlit as st

def detectOnFrame(model, labels, frame):
    # https://docs.ultralytics.com/guides/instance-segmentation-and-tracking/
    result = model.predict(frame)
    annotator = Annotator(frame, line_width=2)

    if result[0].masks is not None:
        classes = result[0].boxes.cls.cpu().tolist()
        masks = result[0].masks.xy
        for C, mask in zip(classes, masks):
            annotator.seg_bbox(mask=mask, mask_color=colors(int(C), True), det_label=labels[int(C)])

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def detectOnUploadedVideo(model):
    uploaded_file = st.file_uploader("Upload video from machine...", type=["mp4", "mov"])
    if uploaded_file is not None:
        if not os.path.exists("upload_videos"):
            os.makedirs("upload_videos")
        uploaded_file_path = os.path.join("upload_videos", uuid.uuid4().hex + uploaded_file.name)
        uploaded_file_bytes = uploaded_file.read()
        with open(uploaded_file_path, 'wb') as f:
            f.write(uploaded_file_bytes)
        cap = cv2.VideoCapture(uploaded_file_path)
        cap.set(cv2.CAP_PROP_FPS, 120)
    else:
        return
    
    labels = model.model.names

    col1, col2 = st.columns(2)
    with col1:
        video_file = open(uploaded_file_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
    
    with col2:
        container = st.empty()
        stop_button = st.button('Stop')
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                # st.write("Video Ended!")
                cap = cv2.VideoCapture(uploaded_file_path)
                continue

            frame = detectOnFrame(model, labels, frame)
            container.image(frame, channels="RGB", use_column_width=True)
            
            if stop_button:
                break
        
        cap.release()

def detectOnLiveCamera(model):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    labels = model.model.names
    container = st.empty()
    stop_button = st.button("Stop")
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended!")
            break

        frame = detectOnFrame(model, labels, frame)
        container.image(frame, channels="RGB", use_column_width=True)
        
        if stop_button:
            break
    
    cap.release()

def runTask(model, choice):
    if choice == 'Object segmentation on uploaded video':
        detectOnUploadedVideo(model)
    elif choice == 'Live camera object segmentation':
        detectOnLiveCamera(model)
    elif choice == None:
        st.warning("Please select a task from the sidebar.")

def main():
    st.set_page_config(page_title="Instance Segmentation YOLOv8", layout="wide")
    st.title("Real-time Object Instance Segmentation")
    st.caption("Powered by Ultralytics, OpenCV, Streamlit")

    with st.sidebar:
        st.title("Real-time Object Instance Segmentation")

        task_choice = st.selectbox(
            'Choose an option',
            ('Object segmentation on uploaded video', 'Live camera object segmentation'),
            index=None,
            placeholder='Select a task',
        )
        "[GitHub Repository](https://github.com/ndtduy/object-seg-track-streamlit)"

    model = YOLO("yolov8n-seg.pt")
    runTask(model, task_choice)

if __name__ == "__main__":
    main()