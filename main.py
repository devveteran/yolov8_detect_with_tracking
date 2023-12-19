import cv2
import time
import os
import numpy as np
from yolov8 import YOLOv8
from xtracker.tracker import Tracker

SKIP_SECONDS = 0
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3
tracker = Tracker()
classnames = ["normal", "poop", "pee"]

KEY_FRAME_DURATION = 6 # seconds

ROI_RECT = [300, 100, 800, 600]

def read_timestamp(file):
    key_frames = []
    f = open(f"../videos/{file}", "r")
    lines = f.readlines()
    for line in lines:
        line_token = line.split(",")
        cls = line_token[0].strip()
        hour = line_token[1].strip()
        min = line_token[2].strip()
        sec = line_token[3].strip()
        key_frames.append((cls, hour, min, sec))
    return key_frames

def main():
    global tracker

    filename = "9418.mp4"
    timestamp_filename = "9418_timestamps.txt"
    
    key_frames = read_timestamp(timestamp_filename)
    tup_file = os.path.splitext(filename)
    filename_without_extension = tup_file[0]

    videoUrl = '../videos/%s'%filename
    
    cap = cv2.VideoCapture(videoUrl)
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if cam_fps == 0:
        cam_fps = 30
    print("FPS: ", cam_fps)
    
    tracker.SetVariables(cam_fps, classes=classnames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, SKIP_SECONDS * cam_fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not os.path.exists("../videos/output"):
        os.makedirs("../videos/output")
    if not os.path.exists("output/to_add"):
        os.makedirs("output/to_add")

    model_path = "models/best_320.onnx"
    yolov8_detector = YOLOv8(model_path, \
                            conf_thres=CONFIDENCE_THRESHOLD, \
                            iou_thres=IOU_THRESHOLD)

    iter = 0
    use_keyframes = False
    if key_frames.__len__() > 0:
        use_keyframes = True
    
    exit_flag = False
    while cap.isOpened() and (use_keyframes == False or iter < key_frames.__len__()):
        if exit_flag == True:
            break
        
        if use_keyframes == True:

            start_time = int(key_frames[iter][1]) * 3600 + \
                        int(key_frames[iter][2]) * 60 + \
                        int(key_frames[iter][3]) - round(KEY_FRAME_DURATION / 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cam_fps)
            total_frame_count = KEY_FRAME_DURATION * cam_fps

        else:
            total_frame_count = float("inf")

        currentframe = 0

        out = cv2.VideoWriter(f'../videos/output/output_{filename_without_extension}_{iter}.avi', \
                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
                        cap.get(cv2.CAP_PROP_FPS), \
                        (width, height))
        
        tracker.EmptyObjects()

        while(currentframe < total_frame_count):
            try:
                ret, frame = cap.read()
                if not ret:
                    break
            except Exception as e:
                continue
            
            key_pressed = cv2.waitKey(1)
            if key_pressed == ord('q'):
                exit_flag = True
                out.release()
                break
            elif key_pressed == ord('s'):
                cv2.imwrite(f"output/to_add/{key_frames[iter][0]}_{key_frames[iter][1]}_{key_frames[iter][2]}_{key_frames[iter][3]}_{currentframe}.jpg", frame)
            elif key_pressed == ord('d'):
                break
            elif key_pressed == ord('a'):
                iter -= 2
                if iter < 0: iter = 0
                break
            elif key_pressed == ord('r'):
                iter -=1
                if iter < 0: iter = 0
                break

            detection_start_time = time.time()
            boxes, scores, class_ids = yolov8_detector(frame)
            detection_end_time = time.time()

            cv2.rectangle(frame, (ROI_RECT[0], ROI_RECT[1]), (ROI_RECT[2], ROI_RECT[3]), \
                      (0, 255, 0), 1)
            tracker.Update(boxes, scores, class_ids)
            track_end_time = time.time()
        
            dogsIn = []

            for t in tracker.tracks:
                if t.IsActiveNow() == False:
                    continue

                cur_box = t.GetCurrentBox()
                
                startX, startY = int(cur_box[0]), int(cur_box[1])
                endX, endY = int(cur_box[2]), int(cur_box[3])

                dogsIn.append((t.GetId(), t.IsObjectInRect(ROI_RECT)))
                class_id = t.GetClassId()

                class_name = tracker.class_names[class_id]
                
                if class_name == "normal": color = (255, 0, 0)
                else: color = (0, 0, 255)

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)

                cv2.putText(frame, \
                            "{}:".format(t.GetId()) + class_name, \
                            (startX + 10, startY + 10), \
                            cv2.FONT_HERSHEY_COMPLEX, 0.4, \
                            color)
                # cv2.putText(frame, "Score: {:.2f}".format(t.GetScore()), \
                #             (startX + 10, startY + 60), \
                #             cv2.FONT_HERSHEY_COMPLEX, 0.9, \
                #             (0, 0, 255))

                _, mov_est = t.IsObjectMoving()
                _, pos_est = t.IsObjectChangingPose()
                # cv2.putText(frame, "Move: {:.3f}".format(mov_est), \
                #             (startX + 10, startY + 90), \
                #             cv2.FONT_HERSHEY_COMPLEX, 0.9, \
                #             (0, 0, 255))
                # cv2.putText(frame, "Pose: {:.3f}".format(pos_est), \
                #             (startX + 10, startY + 120), \
                #             cv2.FONT_HERSHEY_COMPLEX, 0.9, \
                #             (0, 0, 255))

            if use_keyframes == True:
                cv2.putText(frame, f"TimeStamp: {key_frames[iter][0]}-{key_frames[iter][1]}-{key_frames[iter][2]}-{key_frames[iter][3]}",\
                        (100, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255))
                
            for i in range(dogsIn.__len__()):
                (id, is_in) = dogsIn[i]
                strMsg = "Out"
                if is_in == True:
                    strMsg = "In"
                else:
                    strMsg = "Out"
                cv2.putText(frame, "Dog{}:".format(id) + "{}".format(strMsg), \
                            (100, 100 + i * 30), \
                            cv2.FONT_HERSHEY_DUPLEX, 0.4, \
                            (0, 255, 0))

            print("Detect: {:.3f},   Track: {:.3f}".format(detection_end_time - detection_start_time, track_end_time - detection_end_time))

            resized_width = 1024
            resized_height = 768
            scale = max(resized_width/width, resized_height/height)
            resized_frame = cv2.resize(frame, (int(width*scale), int(height*scale)))
            cv2.imshow("Detected Objects", resized_frame)
            out.write(frame)
            if use_keyframes == True:
                currentframe += 1
        out.release()

        if use_keyframes == True:
            iter += 1

if __name__ == '__main__':
    main()