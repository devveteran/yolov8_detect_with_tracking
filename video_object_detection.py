from cmath import sqrt
import cv2
import os
import numpy as np
from yolov8 import YOLOv8
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks
import dlib
from trackable_object import TrackableObject, UpdateTrackObjectsFromDetection

SKIP_FRAME_COUNT = 10
SKIP_SECONDS = 1

trackableObjects = []

classnames=["normal", "poop"]

def main(tracker):
    global trackableObjects
    
    filename = "istockphoto-1486058168-640_adpp_is.mp4"
    tup_file = os.path.splitext(filename)
    filename_without_extension = tup_file[0]

    videoUrl = '../../Dataset/new/%s'%filename

    cap = cv2.VideoCapture(videoUrl)
    start_time = SKIP_SECONDS # skip first {start_time} seconds
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not os.path.exists("../output"):
        os.makedirs("../output")
    out = cv2.VideoWriter('../output/output_%s.avi'%filename_without_extension, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (width, height))

    model_path = "../../models/best.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.1)

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

        boxes, scores, class_ids = yolov8_detector(frame)

        # combined_img = yolov8_detector.draw_detections(frame)
        combined_img = frame

        trackableObjects = UpdateTrackObjectsFromDetection(trackableObjects, boxes, class_ids)

        for t in trackableObjects:
            startX = int(t.GetBBox()[0])
            startY = int(t.GetBBox()[1])
            endX = int(t.GetBBox()[2])
            endY = int(t.GetBBox()[3])

            #cv2.circle(combined_img, (startX + (endX - startX)//2, startY + (endY - startY) // 2), 3, color, 3)
            if classnames[t.GetClassId()] == "normal":
                color = (255, 0, 0)
            elif classnames[t.GetClassId()] == "poop":
                color = (0, 0, 255)

            cv2.rectangle(combined_img, (startX, startY), (endX, endY), color)
            cv2.putText(combined_img, str(t.GetId())+":"+classnames[t.GetClassId()], (startX + (endX - startX)//2, startY + (endY - startY) // 2), cv2.FONT_HERSHEY_COMPLEX, 0.5, color)
            
        cv2.imshow("Detected Objects", combined_img)
        out.write(combined_img)
    out.release()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Object detections in input video using YOLOv8 trained on custom dataset.'
    )

    parser.add_argument(
        '--tracker', type=str, default='SORT',
        help="Tracker used to track objects. Options include ['CentroidTracker', 'CentroidKF_Tracker', 'SORT']")
    args = parser.parse_args()

    if args.tracker == 'CentroidTracker':
        tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    elif args.tracker == 'CentroidKF_Tracker':
        tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')
    elif args.tracker == 'SORT':
        tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
    elif args.tracker == 'IOUTracker':
        tracker = IOUTracker(max_lost=2, iou_threshold=0.5, min_detection_confidence=0.4, max_detection_confidence=0.7,
                             tracker_output_format='mot_challenge')
    else:
        raise NotImplementedError
    
    main(tracker)