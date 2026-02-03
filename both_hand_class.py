import cv2
import mediapipe as mp
import numpy as np
import keras
import time
from keras.applications.resnet50 import preprocess_input

# 1. Load Model and Labels
with open('model_config.txt', 'r') as f:
    model_path = f.read().strip()
    print(f"loaded: {model_path}")
my_model = keras.models.load_model(model_path)
custom_labels = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]

# 2. Setup Gesture Recognizer
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

with GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(time.time() * 1000)
        result = recognizer.recognize_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            h, w, _ = frame.shape
            all_x, all_y = [], []
            
            # Collect all points from all detected hands (1 or 2)
            for hand_landmarks in result.hand_landmarks:
                for lm in hand_landmarks:
                    all_x.append(int(lm.x * w))
                    all_y.append(int(lm.y * h))

            # Unified Bounding Box for ALL hands
            x_min, x_max = max(min(all_x)-40, 0), min(max(all_x)+40, w)
            y_min, y_max = max(min(all_y)-40, 0), min(max(all_y)+40, h)
            
            crop = frame[y_min:y_max, x_min:x_max]
            
            if crop.size > 0:
                # Square Padding to 224x224 (prevents squishing hands)
                cw, ch = x_max - x_min, y_max - y_min
                size = max(cw, ch)
                square_canvas = np.zeros((size, size, 3), dtype=np.uint8)
                
                # Center the crop on the canvas
                dx, dy = (size - cw) // 2, (size - ch) // 2
                square_canvas[dy:dy+ch, dx:dx+cw] = crop
                
                # Final Resize for ResNet50
                input_img = cv2.resize(square_canvas, (224, 224))
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                input_img = np.expand_dims(input_img, axis=0)
                input_img = preprocess_input(input_img)

                # Predict
                preds = my_model.predict(input_img, verbose=0)[0]
                pred_idx = np.argmax(preds)
                label = f"ISL: {custom_labels[pred_idx]} ({preds[pred_idx]:.2f})"

                # Feedback
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, label, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Unified ISL Recognition (2026)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
