from cmath import sqrt
import cv2
import os
import numpy as np
from yolov8 import YOLOv8
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks
import dlib

SKIP_FRAME_COUNT = 30

trackers = []
labels = []
classes = []
classnames=["normal", "poop"]

def main(tracker):
    global trackers, labels, classes, classnames

    filename = "istockphoto-1486058168-640_adpp_is.mp4"
    tup_file = os.path.splitext(filename)
    filename_without_extension = tup_file[0]

    videoUrl = '../../Dataset/new/%s'%filename

    cap = cv2.VideoCapture(videoUrl)
    start_time = 1 # skip first {start_time} seconds
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not os.path.exists("../output"):
        os.makedirs("../output")
    out = cv2.VideoWriter('../output/output_%s.avi'%filename_without_extension, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (width, height))

    model_path = "../../models/best.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

    skippedFrameCount = 0
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
        combined_img = yolov8_detector.draw_detections(frame)        
        
        if len(trackers) == 0 or skippedFrameCount == 0:
            trackers = []
            labels = []
            classes = []

            index = 1
            for (box, class_id) in zip(boxes, class_ids):
                t = dlib.correlation_tracker()
                rect = dlib.rectangle(box[0], box[1], box[2], box[3])
                t.start_track(frame, rect)
                
                labels.append(index)
                trackers.append(t)
                classes.append(class_id)
                index += 1
        else:
            print("----->")
            print(classes)
            new_classes = []
            for (t, l, c) in zip(trackers, labels, classes):
                t.update(frame)
                pos = t.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                centerX = startX + (endX - startX) // 2
                centerY = startY + (endY - startY) // 2
                max_dist = 999999
                new_class = 0

                for (box, class_id) in zip(boxes, class_ids):
                    box_x1 = box[0]
                    box_y1 = box[1]
                    box_x2 = box[2]
                    box_y2 = box[3]
                    box_CenterX = box_x1 + (box_x2 - box_x1) // 2
                    box_CenterY = box_y1 + (box_y2 - box_y1) // 2
                    dist = int(np.real(sqrt((centerX-box_CenterX)*(centerX-box_CenterX) + (centerY-box_CenterY)*(centerY-box_CenterY))))
                    if (dist < max_dist):
                        new_class = class_id
                new_classes.append(new_class)
            classes = new_classes
            print(classes)

        for (t, l, c) in zip(trackers, labels, classes):
            pos = t.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            if classnames[c] == "poop":
                color = (255, 0, 0)
            elif classnames[c] == "normal":
                color = (0, 0, 255)
            else:
                color = (0, 0, 0)
            #cv2.circle(combined_img, (startX + (endX - startX)//2, startY + (endY - startY) // 2), 3, color, 3)
            cv2.putText(combined_img, str(l)+":"+classnames[c], (startX + (endX - startX)//2, startY + (endY - startY) // 2), cv2.FONT_HERSHEY_COMPLEX, 2, color)
            
        cv2.imshow("Detected Objects", combined_img)
        skippedFrameCount = (skippedFrameCount + 1) % SKIP_FRAME_COUNT

        # tracks = tracker.update(boxes, scores, class_ids)
        # tracked_img = draw_tracks(combined_img, tracks)
        # cv2.imshow("Detected Objects", tracked_img)
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