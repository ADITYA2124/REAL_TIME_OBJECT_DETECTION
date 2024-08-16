import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from tracker import Tracker
import os
import datetime
import math
from PIL import Image, ImageTk
import numpy as np
import tempfile
import threading
import tkinter as tk
from tkinter import Label
from queue import Queue

# Load the trained YOLO model
try:
    model = YOLO('yolov8s.pt')
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()

# Construct the relative path to coco.txt
current_directory = os.path.dirname(__file__)
coco_file_path = os.path.join(current_directory, "yolo_tracker", "coco.txt")

if not os.path.exists(coco_file_path):
    st.error(f"File not found: {coco_file_path}")
    st.stop()

with open(coco_file_path, "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

# Tracker instance
tracker = Tracker()
reported_ids = []

def update_frame(frame):
    confidence = []
    dt = datetime.datetime.now()
    text = str(dt)
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(frame, text, (10, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (168, 50, 60), 2, cv2.LINE_AA)
            conf = math.ceil((box.conf[0]) * 100) / 100
            confidence.append(conf)
            cls = box.cls[0]
            txt = f'{conf} {class_list[int(cls)]}'
            cv2.putText(frame, txt, (max(0, x1), max(35, y1)), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    list = []
    for result in results:
        a = result.boxes.data
        px = pd.DataFrame(a).astype("float")
        for index, row in px.iterrows():
            x1, y1, x2, y2, d = map(int, row[:4])
            d = int(row[5])
            if d >= len(class_list):
                st.warning(f"Detected class ID {d} is out of range for class_list.")
                continue
            list.append([x1, y1, x2, y2, d])

    bbox_id = tracker.update([bbox[:4] for bbox in list])
    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        class_name = class_list[obj_id]
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)
        cv2.circle(frame, (cx, cy), 3, (255, 0, 255), -1)
        cv2.putText(frame, f"{class_name} {obj_id}", (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        if obj_id not in reported_ids:
            send_location(obj_id)
            reported_ids.append(obj_id)
        else:
            st.write(f"{obj_id} - already sent")
    return frame

def send_location(obj_id):
    st.write(f"Location sent - {obj_id}")

def process_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    processed_frame = update_frame(image)
    return processed_frame

def process_video(video_file, frame_queue, stop_event):
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        if stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = update_frame(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize the frame to fit within the Tkinter window
        #frame = cv2.resize(frame, (600, 500))
        frame = Image.fromarray(frame)
        # Put the frame in the queue
        frame_queue.put(frame)
    cap.release()

def process_webcam(frame_queue, stop_event):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        if stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = update_frame(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize the frame to fit within the Tkinter window
        #frame = cv2.resize(frame, (600, 500))
        frame = Image.fromarray(frame)
        # Put the frame in the queue
        frame_queue.put(frame)
    cap.release()

def update_label(root, label, frame_queue):
    if not frame_queue.empty():
        frame = frame_queue.get()
        frame = ImageTk.PhotoImage(frame)
        label.config(image=frame)
        label.image = frame
    root.after(10, update_label, root, label, frame_queue)

def run_tkinter(video_file=None, use_webcam=False):
    root = tk.Tk()
    root.title("YOLO Object Detection")
    root.geometry("600x500")
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (600 // 2)
    y = (root.winfo_screenheight() // 2) - (500 // 2)
    root.geometry(f'{600}x{500}+{x}+{y}')
    label = Label(root)
    label.pack()
    
    frame_queue = Queue()
    stop_event = threading.Event()
    
    if use_webcam:
        thread = threading.Thread(target=process_webcam, args=(frame_queue, stop_event))
    else:
        thread = threading.Thread(target=process_video, args=(video_file, frame_queue, stop_event))
    
    thread.start()
    
    root.after(10, update_label, root, label, frame_queue)

    def on_closing():
        stop_event.set()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

# Streamlit App
st.title("Object Detection and Tracking - Developed by Aditya Arora")

st.write("""
    ## Welcome to the YOLO Object Detection App!
    This project leverages the YOLOv8 (You Only Look Once) model to perform real-time object
    detection on various media inputs, including images, videos, and live webcam feeds. 
    YOLO is a state-of-the-art object detection algorithm known for its speed and accuracy.
    The application allows users to upload images or videos, or use their webcam to detect objects
    in real-time. It highlights detected objects with bounding boxes and labels, and displays the
    confidence scores for each detection. Additionally, the app integrates a tracking feature to maintain
    object identities across frames in videos. This project demonstrates the power of deep learning and
    computer vision, making it accessible for various practical applications such as surveillance, 
    traffic monitoring, and more.
""")

option = st.sidebar.selectbox("Select Input Type", ("Image", "Video", "Webcam"))

if option == "Image":
    st.write("""
        ### Image Upload
        Upload an image file (JPG, JPEG, PNG) and the application will detect objects in the image using the YOLO model.
    """)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Detecting...")
        processed_frame = process_image(image)
        st.image(processed_frame, caption='Processed Image.', use_column_width=True)

elif option == "Video":
    st.write("""
        ### Video Upload
        Upload a video file (MP4, MOV, AVI, MKV) and the application will detect objects in the video using the YOLO model.
    """)
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        st.write("Detecting...")
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        threading.Thread(target=run_tkinter, args=(tfile.name,)).start()

elif option == "Webcam":
    st.write("""
        ### Live Webcam
        Use your webcam to perform real-time object detection. Click the "Start Webcam" button to begin.
    """)
    if st.button("Start Webcam"):
        st.write("Webcam started.")
        threading.Thread(target=run_tkinter, kwargs={'use_webcam': True}).start()
