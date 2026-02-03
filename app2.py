import base64
import cv2
import numpy as np
import tensorflow as tf
import keras
import asyncio
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from keras.applications.resnet50 import preprocess_input

# -------------------- APP --------------------
app = FastAPI()

# -------------------- GPU CONFIG --------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

with open('model_config.txt', 'r') as f:
    model_path = f.read().strip()

with tf.device("/GPU:0"):
    model = keras.models.load_model(
        # "resnet50_best.keras",
        model_path,
        compile=False
    )

custom_labels = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]

# -------------------- WEBSOCKET --------------------
@app.websocket("/ws")
async def ws_classifier(ws: WebSocket):
    await ws.accept()
    print("Client connected")

    try:
        while True:
            message = await ws.receive()

            # ---------- HEARTBEAT ----------
            if message.get("text") == "ping":
                await ws.send_text("pong")
                continue

            # ---------- EXPECT BASE64 IMAGE ----------
            if "text" not in message:
                continue

            text = message["text"]
            if "," not in text:
                continue

            try:
                base64_img = text.split(",", 1)[1]
                img_bytes = base64.b64decode(base64_img)
                if not img_bytes:
                    continue
            except Exception:
                continue

            np_img = np.frombuffer(img_bytes, np.uint8)
            if np_img.size == 0:
                continue

            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # ---------- PREPROCESS ----------
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_input(np.expand_dims(img, axis=0))

            # ---------- GPU INFERENCE ----------
            with tf.device("/GPU:0"):
                preds = model.predict(img, verbose=0)[0]

            idx = int(np.argmax(preds))

            await ws.send_json({
                "label": custom_labels[idx],
                "confidence": float(preds[idx])
            })

            # 🔑 Yield control (prevents WS starvation)
            await asyncio.sleep(0)

    except WebSocketDisconnect as e:
        print(f"Client disconnected (code={e.code})")

    except Exception as e:
        print("Backend error:", e)

    finally:
        print("Connection closed")
