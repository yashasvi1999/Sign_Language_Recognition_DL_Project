# Sign Language Recognition

This project provides real-time sign language gesture recognition using MediaPipe and a Keras-based ResNet50 model (default). There are two ways to use it: directly via a Python script or through a web app using FastAPI.

## Prerequisites

- Python 3.10 or later
- Install dependencies: `pip install -r requirements.txt`
- Ensure your Keras model (e.g., `resnet50_best.keras`) is available. For optimized inference on Intel CPUs, convert the model using OpenVINO as described in the conversion guide.

## Usage Options

### 1. Direct Script Usage (both_hand_class.py)

This option runs gesture recognition directly from your webcam.

- Add your Keras model to the `model` folder (create it if it doesn't exist).
- Update `model_config.txt` with the path to your model (e.g., `model/your_model.keras`).
- Run the script: `python both_hand_class.py`

The script will open a webcam feed, detect hands, crop and classify gestures, and display results in real-time.

### 2. Web App Usage (FastAPI + WebSocket)

This option runs a web server for gesture recognition via a browser interface.

- Model selection is handled via `model_config.txt` (update it with your desired model path as in option 1).
- Start the server: `uvicorn app2:app --host 0.0.0.0 --port 8000`
- Open `index.html` in your browser to access the real-time recognition interface.

The web app uses WebSocket for live updates and supports camera input for gesture detection.
Note: FastAPI + WS has some latency due to network overhead.