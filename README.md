
# YOLO Object Detection with Streamlit

This project is a YOLO (You Only Look Once) object detection application built with Streamlit. It supports processing images, videos, and live webcam feeds.

## Features

- **Image Processing**: Upload an image and detect objects within it.
- **Video Processing**: Upload a video file to perform object detection on each frame.
- **Live Webcam Feed**: Use your webcam for real-time object detection.

## Requirements

- Python 3.x
- Required Python libraries:
  - `streamlit`
  - `opencv-python`
  - `numpy`

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the pre-trained YOLO model weights and configuration files:**

    - YOLOv3: [Download the YOLOv3 weights](https://github.com/ultralytics/yolov5.git) 

## Usage

1. **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2. **Access the application:**

    Open your web browser and go to `http://localhost:8501`.

3. **Choose the input type:**

    - **Image**: Upload an image to detect objects.
    - **Video**: Upload a video to perform detection on each frame.
    - **Webcam**: Use your webcam for real-time object detection.
  
## SAMPLE IMAGE
![image](https://github.com/user-attachments/assets/8740a7f5-6548-416e-9f90-641e6d8762d5)


## Project Structure

- `app.py`: Main Streamlit application file.
- `requirements.txt`: Python dependencies.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions or improvements.

## License

This project is licensed under the MIT License.
