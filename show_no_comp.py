import cv2
import socket
import numpy as np
import pickle
import torch
from options.test_options import TestOptions
from models import create_model
from util import util
from ultralytics import YOLO
import time

# === CONFIG ===
model_name = 'derain'
model_type = 'test'
no_dropout = True
results_dir = './results/'
TARGET_CLASS = 0  # person class

# === YOLO SETUP ===
yolo_model = YOLO("yolov8n.pt")

# === Desmoke Model Setup ===
opt = TestOptions().parse()
opt.num_threads = 0
opt.batch_size = 1
opt.serial_batches = True
opt.no_flip = True
opt.display_id = -1
opt.results_dir = results_dir
opt.name = model_name
opt.model = model_type
opt.no_dropout = no_dropout
opt.eval = True

model = create_model(opt)
model.setup(opt)
model.eval()

# === Server Setup ===
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = "192.168.127.114"
port = 2323
s.bind((ip, port))

# === Fullscreen Window ===
cv2.namedWindow("Processed View", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Processed View", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_time = 0

while True:
    x = s.recvfrom(100000000)
    data = pickle.loads(x[0])
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if frame is None:
        continue

    # === Desmoke Processing ===
    input_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = np.transpose(np.expand_dims(input_rgb, 0), (0, 3, 1, 2)).astype(np.float32) / 255.0
    data = {"A": torch.FloatTensor(input_tensor), "A_paths": ["frame.jpg"]}

    with torch.no_grad():
        model.set_input(data)
        model.test()
        result_image = model.get_current_visuals()['fake']
        processed = cv2.cvtColor(np.array(util.tensor2im(result_image)), cv2.COLOR_RGB2BGR)

    # Resize match
    if processed.shape[:2] != frame.shape[:2]:
        processed = cv2.resize(processed, (frame.shape[1], frame.shape[0]))

    # === YOLOv8 Detection ===
    results = yolo_model(processed, verbose=False)[0]
    for box in results.boxes:
        if int(box.cls[0]) == TARGET_CLASS:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(processed, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # === FPS ===
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(processed, f"FPS: {fps:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # === Display ===
    cv2.imshow("Processed View", processed)

    # Exit on Enter key
    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
s.close()
