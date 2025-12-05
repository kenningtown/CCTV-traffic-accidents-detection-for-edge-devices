import os
import time

import cv2
import torch
from ultralytics import YOLO

# ====== CONFIG ======
# Путь к обученной YOLOv8n-cls модели на Raspberry Pi
# поправь, если имя юзера другое
MODEL_PATH = "/home/nurobot427/traffic_accident_edge/best.pt"

# Если True — используем веб-камеру (USB / CSI),
# если False — читаем видеофайл
USE_CAMERA = False
# если USE_CAMERA = False
VIDEO_PATH = "/home/nurobot427/traffic_accident_edge/crash.mp4"

# Классы: при обучении было [accident, normal] -> 0 = accident, 1 = normal
ACCIDENT_CLASS = 0
NORMAL_CLASS = 1

IMG_SIZE = 224               # размер, с которым обучали YOLOv8n-cls
ACCIDENT_THRESHOLD = 0.8     # порог для "ALERT"
# =====================


def main():
    # ----- устройство -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ----- загрузка модели -----
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    print("Model loaded from:", MODEL_PATH)

    # Проверим имена классов
    print("Model class names:", model.names)

    # ----- источник видео -----
    if USE_CAMERA:
        cap = cv2.VideoCapture(0)  # /dev/video0
        if not cap.isOpened():
            raise RuntimeError("Could not open camera (index 0)")
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {VIDEO_PATH}")

    fps = 0.0
    print("Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received, exiting.")
            break

        start = time.time()

        # Запускаем классификацию YOLO на этом кадре
        results = model(
            frame,
            imgsz=IMG_SIZE,
            device=device,
            verbose=False
        )

        r = results[0]
        probs = r.probs            # вероятности классов
        pred_class = int(probs.top1)      # 0 или 1
        pred_conf = float(probs.top1conf)  # уверенность top1-класса

        # Вероятность "accident" (класс 0)
        accident_prob = float(probs.data[ACCIDENT_CLASS])

        # Текст и цвет для вывода
        if pred_class == ACCIDENT_CLASS:
            label = "ACCIDENT"
            color = (0, 0, 255)  # красный (BGR)
        else:
            label = "NORMAL"
            color = (0, 255, 0)  # зелёный (BGR)

        end = time.time()
        dt = end - start
        current_fps = 1.0 / dt if dt > 0 else 0.0
        fps = 0.9 * fps + 0.1 * current_fps  # сглаживание

        # ----- рисуем оверлеи -----
        h, w, _ = frame.shape

        text = f"{label}  p_acc={accident_prob:.2f}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        # ALERT, если вероятность аварии высокая
        if accident_prob >= ACCIDENT_THRESHOLD:
            alert_text = "!!! ACCIDENT ALERT !!!"
            cv2.putText(frame, alert_text, (10, int(h * 0.9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("YOLO Accident Detector (Pi)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Finished.")


if __name__ == "__main__":
    main()
