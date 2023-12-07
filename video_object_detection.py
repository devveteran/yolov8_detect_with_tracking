import cv2
import os
import numpy as np
from yolov8 import YOLOv8
from trackable_object import IsRectOverlapping, UpdateTrackObjectsFromDetection

SKIP_SECONDS = 0
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.9
trackableObjects = []

classnames=["normal", "poop"]

ROI_RECT = [800, 100, 1400, 800]

def main():
    global trackableObjects
    
    filename = "AdobeStock_584253963_Video_HD_Preview.mov"
    # filename = "istockphoto-1486058168-640_adpp_is.mp4"

    tup_file = os.path.splitext(filename)
    filename_without_extension = tup_file[0]

    videoUrl = '../../Dataset/Pictures/pee_files/videos/%s'%filename
    # videoUrl = '../../Dataset/new/%s'%filename

    cap = cv2.VideoCapture(videoUrl)
    start_time = SKIP_SECONDS # skip first {start_time} seconds
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not os.path.exists("../output"):
        os.makedirs("../output")
    out = cv2.VideoWriter('../output/output_%s.avi'%filename_without_extension, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (width, height))

    model_path = "../../models/best.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=CONFIDENCE_THRESHOLD, iou_thres=IOU_THRESHOLD)

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
        cv2.rectangle(combined_img, (ROI_RECT[0], ROI_RECT[1]), (ROI_RECT[2], ROI_RECT[3]), (0, 255, 0), 3)

        trackableObjects = UpdateTrackObjectsFromDetection(trackableObjects, boxes, scores, class_ids)

        dogsIn = [(0, False) for i in range(trackableObjects.__len__())]

        iter  = 0
        for t in trackableObjects:
            startX = int(t.GetCurrentBox()[0])
            startY = int(t.GetCurrentBox()[1])
            endX = int(t.GetCurrentBox()[2])
            endY = int(t.GetCurrentBox()[3])

            dogsIn[iter] = (t.GetId(), t.IsObjectInRect(ROI_RECT))

            class_id = t.GetClassId()

            if classnames[class_id] == "normal":
                color = (255, 0, 0)
            elif classnames[class_id] == "poop":
                color = (0, 0, 255)

            #centerX = startX + (endX - startX) // 2
            #centerY = startY + (endY - startY) // 2
            # cv2.circle(combined_img, (centerX, centerY), 3, color, 3)
            cv2.rectangle(combined_img, (startX, startY), (endX, endY), color, 2)
            cv2.putText(combined_img, "{}:".format(t.GetId()) + classnames[class_id], (startX + 10, startY + 30), cv2.FONT_HERSHEY_COMPLEX, 0.9, color)
            cv2.putText(combined_img, "Score: {:.2f}".format(t.GetScore()), (startX + 10, startY + 60), cv2.FONT_HERSHEY_COMPLEX, 0.9, color)
            moving, mov_est = t.IsObjectMoving()
            posing, pos_est = t.IsObjectChangingPose()
            cv2.putText(combined_img, "Move: {:.2f}".format(mov_est), (startX + 10, startY + 90), cv2.FONT_HERSHEY_COMPLEX, 0.9, color)
            cv2.putText(combined_img, "Pose: {:.2f}".format(pos_est), (startX + 10, startY + 120), cv2.FONT_HERSHEY_COMPLEX, 0.9, color)
            
            iter += 1

        for i in range(dogsIn.__len__()):
            (id, is_in) = dogsIn[i]
            strMsg = "Out"
            if is_in == True:
                strMsg = "In"
            else:
                strMsg = "Out"
            cv2.putText(combined_img, "Dog{}:".format(id) + "{}".format(strMsg), (100, 100 + i * 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))

        cv2.imshow("Detected Objects", combined_img)
        out.write(combined_img)
    out.release()

if __name__ == '__main__':
    main()