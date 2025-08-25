import cv2
import numpy as np
import torch
from collections import OrderedDict
import time
import imutils
from ultralytics import YOLO
from scipy.spatial import distance as dist

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "drone", "military drone", "quadcopter", "helicopter", "fighter jet", "tank", "armored vehicle",
    "missile", "rocket", "artillery", "radar", "satellite dish", "antenna", "soldier", "military truck"
]

class AdvancedObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.class_names = CLASS_NAMES
        self.frame_width = 640
        self.frame_height = 480
        self.focal_length = 1000
        self.tracked_objects = OrderedDict()
        self.next_object_id = 0
        self.max_disappeared = 50
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.3
        self.cap = self.initialize_camera()
        if self.cap is None:
            raise RuntimeError("Failed to initialize camera")

    def initialize_camera(self):
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Camera found at index {i}")
                self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return cap
        print("Warning: No camera found, using sample video")
        cap = cv2.VideoCapture("sample.mp4")
        if cap.isOpened():
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return cap
        return None

    def calibrate_camera(self, known_width, known_distance, measured_pixels):
        self.focal_length = (measured_pixels * known_distance) / known_width
        print(f"Calibrated focal length: {self.focal_length}")

    def estimate_distance(self, box_width, class_name):
        known_widths = {
            "person": 0.5, "car": 1.8, "truck": 2.5, "drone": 0.5,
            "laptop": 0.35, "cell phone": 0.075
        }
        known_width = known_widths.get(class_name.lower(), 0.5)
        return round((known_width * self.focal_length) / box_width, 2)

    def update_tracked_objects(self, detections):
        if len(detections) == 0:
            for object_id in list(self.tracked_objects.keys()):
                self.tracked_objects[object_id]["disappeared"] += 1
                if self.tracked_objects[object_id]["disappeared"] > self.max_disappeared:
                    self.tracked_objects.pop(object_id)
            return self.tracked_objects

        current_centers = np.zeros((len(detections), 2), dtype=int)
        for i, (x1, y1, x2, y2, _, class_id) in enumerate(detections):
            current_centers[i] = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        if len(self.tracked_objects) == 0:
            for i in range(len(detections)):
                self.tracked_objects[self.next_object_id] = {
                    "center": current_centers[i],
                    "bbox": detections[i][:4],
                    "class_id": detections[i][5],
                    "disappeared": 0
                }
                self.next_object_id += 1
        else:
            object_ids = list(self.tracked_objects.keys())
            previous_centers = np.array([obj["center"] for obj in self.tracked_objects.values()])
            D = dist.cdist(previous_centers, current_centers)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.tracked_objects[object_id] = {
                    "center": current_centers[col],
                    "bbox": detections[col][:4],
                    "class_id": detections[col][5],
                    "disappeared": 0
                }
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.tracked_objects[object_id]["disappeared"] += 1
                if self.tracked_objects[object_id]["disappeared"] > self.max_disappeared:
                    self.tracked_objects.pop(object_id)

            for col in unused_cols:
                self.tracked_objects[self.next_object_id] = {
                    "center": current_centers[col],
                    "bbox": detections[col][:4],
                    "class_id": detections[col][5],
                    "disappeared": 0
                }
                self.next_object_id += 1

        return self.tracked_objects

    def detect_objects(self, frame):
        results = self.model(frame)
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                if conf > self.confidence_threshold:
                    detections.append([*box, conf, class_id])
        return detections

    def draw_detections(self, frame, tracked_objects):
        fps_text = f"FPS: {self.fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for object_id, obj_info in tracked_objects.items():
            x1, y1, x2, y2 = map(int, obj_info["bbox"])
            class_id = obj_info["class_id"]
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            box_width = x2 - x1
            distance = self.estimate_distance(box_width, class_name)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{object_id} {class_name} {distance}m"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            center_x, center_y = obj_info["center"]
            cv2.circle(frame, (center_x, center_y), 4, color, -1)
        return frame

    def calculate_fps(self):
        self.frame_count += 1
        if self.frame_count >= 10:
            now = time.time()
            self.fps = self.frame_count / (now - self.last_time)
            self.frame_count = 0
            self.last_time = now

    def run(self):
        try:
            self.calibrate_camera(known_width=0.5, known_distance=2.0, measured_pixels=200)
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                frame = imutils.resize(frame, width=1000)
                detections = self.detect_objects(frame)
                tracked_objects = self.update_tracked_objects(detections)
                output_frame = self.draw_detections(frame, tracked_objects)
                self.calculate_fps()
                cv2.imshow("Advanced Object Detection", output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Initializing Advanced Object Detection System...")
    try:
        detector = AdvancedObjectDetector()
        print("Starting detection...")
        detector.run()
    except Exception as e:
        print(f"Failed to initialize detector: {e}")