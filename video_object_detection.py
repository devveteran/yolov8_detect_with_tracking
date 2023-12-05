import cv2
import os
from yolov8 import YOLOv8

# # Initialize video
# cap = cv2.VideoCapture("input.mp4")
filename = "AdobeStock_491869850_Video_HD_Preview.mov"
tup_file = os.path.splitext(filename)
filename_without_extension = tup_file[0]

videoUrl = '../../Dataset/pet_vid/%s'%filename

cap = cv2.VideoCapture(videoUrl)
start_time = 5 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not os.path.exists("../output"):
    os.makedirs("../output")
out = cv2.VideoWriter('../output/output_%s.avi'%filename_without_extension, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (width, height))

model_path = "../../models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)
    out.write(combined_img)

out.release()
