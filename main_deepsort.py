import cv2
import os, time
from matplotlib import pyplot as plt
import numpy as np
from yolov8 import YOLOv8

SKIP_SECONDS = 0
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3

ROI_RECT = [800, 100, 1400, 800]

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tracking_helpers import create_box_encoder

reID_model_path = "./deep_sort/model_weights/mars-small128.pb"

encoder = create_box_encoder(reID_model_path, batch_size=1)
max_cosine_distance = 0.4
nn_budget: float = None
nms_max_overlap: float = 1.0
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance,
                                                   nn_budget)  # calculate cosine distance metric
tracker = Tracker(metric)  # initialize tracker

class_names = ["normal", "poop", "pee"]

def main():
    filename = "13.12.2023_01.21.51_REC.mp4"

    tup_file = os.path.splitext(filename)
    filename_without_extension = tup_file[0]

    videoUrl = 'videos/%s' % filename

    cap = cv2.VideoCapture(videoUrl)
    start_time = SKIP_SECONDS
    
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if cam_fps == 0:
        cam_fps = 30
    print("FPS: ", cam_fps)
    
    # TrackableObject.SetVariables(cam_fps, classes=classnames)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cam_fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # if not os.path.exists("videos/output"):
    #     os.makedirs("videos/output")
    # out = cv2.VideoWriter('videos/output/output_%s.avi' % filename_without_extension,
    #                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (width, height))

    model_path = "models/best_320.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=CONFIDENCE_THRESHOLD, iou_thres=IOU_THRESHOLD)

    # cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        if cv2.waitKey(1) == ord('q'):
            break

        try:
            ret, frame = cap.read()
            if not ret:
                break
        except Exception as e:
            continue

        start_time = time.time()

        boxes, scores, class_ids = yolov8_detector(frame)

        detect_time = time.time()

        if len(boxes) != 0:
            boxes = np.asarray(boxes, dtype='int')
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # convert from xyxy to xywh
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        # --- DETECTION PART COMPLETED ---
        
        if boxes.__len__() > 0:
            num_objects = boxes.shape[0]
        else:
            num_objects = 0
        names = []

        # loop through objects and use class index to get class name
        for i in range(num_objects):  
            class_indx = class_ids[i]
            class_name = class_names[class_indx]
            names.append(class_name)
        names = np.array(names)

        # --- DeepSORT tacker work starts here ---
        # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
        features = encoder(frame, boxes)
        detections = [Detection(bbox, score, class_name, feature) \
                      for bbox, score, class_name, feature in
                      zip(boxes, scores, names, features)]  # [No of BB per frame] deep_sort.detection.Detection object

        # cmap = plt.get_cmap('tab20b')  # initialize color map
        # colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()  # Call the tracker
        tracker.update(detections)  # updtate using Kalman Gain
        
        track_time = time.time()

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            class_name = track.get_class()
        
            startX = int(bbox[0])
            startY = int(bbox[1])
            endX = int(bbox[2])
            endY = int(bbox[3])

            id = track.track_id

            if class_name == "normal":
                color = (255, 0, 0)
            elif class_name == "poop" or class_name == "pee":
                color = (0, 0, 255)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
            cv2.putText(frame, "{}".format(id), \
                        (startX + 10, startY + 30), \
                        cv2.FONT_HERSHEY_COMPLEX, 0.4, color)

        resized_width = 1024
        resized_height = 768
        scale = max(resized_width/width, resized_height/height)
        resized_frame = cv2.resize(frame, (int(width*scale), int(height*scale)))
        cv2.imshow("Detected Objects", resized_frame)
        print("Detect: {:.3f},   Track: {:.3f}".format(detect_time - start_time, track_time - detect_time))
        # out.write(combined_img)
    # out.release()


if __name__ == '__main__':
    main()