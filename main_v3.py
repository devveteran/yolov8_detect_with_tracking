import cv2
import os
import numpy as np
import time
from trackable_object_v1 import UpdateTrackObjectsFromDetection, TrackableObject

SKIP_SECONDS = 0
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.9
trackableObjects = []

ROI_RECT = [800, 100, 1400, 800]

configPath_for_dog = "models/finger.cfg"
weightsPath_for_dog = "models/finger_last.weights"
dog_net = cv2.dnn_Net

confThreshold = 0.8  # Confidence threshold
nmsThreshold = 0.6 # Non-maximum suppression threshold

def load_model():
    global dog_net
    dog_net = cv2.dnn.readNetFromDarknet(configPath_for_dog, weightsPath_for_dog)

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def DetectionProcess(image, outs):
    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:

            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                width = int(detection[2] * image.shape[1])
                height = int(detection[3] * image.shape[0])
                left = int(center_x - width / 2)
                left += 10
                if (left < 0): left = 0
                top = int(center_y - height / 2)
                if (top < 0): top = 0
                right = left + int(width)
                if (right > image.shape[1]): right = image.shape[1] - 1
                bottom = top + int(height * 1.0)
                if (bottom > image.shape[0]): bottom = image.shape[0] - 1

                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, right, bottom])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    results = []
    for i in indices:
        results.append(boxes[i])
    return results,  confidences, classIds

def main():
    global trackableObjects    
    load_model()

    filename = "AdobeStock_584253963_Video_HD_Preview.mov"

    tup_file = os.path.splitext(filename)
    filename_without_extension = tup_file[0]

    videoUrl = 'videos/%s' % filename

    cap = cv2.VideoCapture(videoUrl)
    start_time = SKIP_SECONDS  # skip first {start_time} seconds
    cam_fps = cap.get(cv2.CAP_PROP_FPS)

    TrackableObject.SetVariables(cam_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cam_fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not os.path.exists("videos/output"):
        os.makedirs("videos/output")
    out = cv2.VideoWriter('videos/output/v3_output_%s.avi' % filename_without_extension,
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (width, height))

    while cap.isOpened():
        startTime = time.time()
        if cv2.waitKey(1) == ord('q'):
            break

        try:
            ret, frame = cap.read()
            if not ret:
                break
        except Exception as e:
            continue

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
        dog_net.setInput(blob, "data")
        
        outs = dog_net.forward(getOutputsNames(dog_net))
        boxes, scores, class_ids = DetectionProcess(frame, outs)

        # combined_img = frame
        cv2.rectangle(frame, (ROI_RECT[0], ROI_RECT[1]), (ROI_RECT[2], ROI_RECT[3]), (0, 255, 0), 3)

        trackableObjects = UpdateTrackObjectsFromDetection(trackableObjects, boxes, scores, class_ids)

        dogsIn = [(0, False) for i in range(trackableObjects.__len__())]

        iter = 0
        for t in trackableObjects:
            startX = int(t.GetCurrentBox()[0])
            startY = int(t.GetCurrentBox()[1])
            endX = int(t.GetCurrentBox()[2])
            endY = int(t.GetCurrentBox()[3])

            dogsIn[iter] = (t.GetId(), t.IsObjectInRect(ROI_RECT))

            class_id = t.GetClassId()

            class_name = TrackableObject.class_names[class_id]
            if class_name == "normal":
                color = (255, 0, 0)
            elif class_name == "poop":
                color = (0, 0, 255)

            # centerX = startX + (endX - startX) // 2
            # centerY = startY + (endY - startY) // 2
            # cv2.circle(combined_img, (centerX, centerY), 3, color, 3)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, "{}:".format(t.GetId()) + class_name, (startX + 10, startY + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9, color)
            cv2.putText(frame, "Score: {:.2f}".format(t.GetScore()), (startX + 10, startY + 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9, color)
            _, mov_est = t.IsObjectMoving()
            _, pos_est = t.IsObjectChangingPose()
            cv2.putText(frame, "Move: {:.2f}".format(mov_est), (startX + 10, startY + 90),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9, color)
            cv2.putText(frame, "Pose: {:.2f}".format(pos_est), (startX + 10, startY + 120),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9, color)

            # print("{id}:{state}:[{x1},{x2},{y1},{y2}]".format(id=t.GetId(), state=class_name, x1=startX,
            #                                                   x2=endX, y1=startY, y2=endY))
            iter += 1

        for i in range(dogsIn.__len__()):
            (id, is_in) = dogsIn[i]
            strMsg = "Out"
            if is_in == True:
                strMsg = "In"
            else:
                strMsg = "Out"
            cv2.putText(frame, "Dog{}:".format(id) + "{}".format(strMsg), (100, 100 + i * 50),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))

        cv2.imshow("Detected Objects", frame)
        endTime = time.time()
        print(f"Runtime of the program is {endTime - startTime}")
        out.write(frame)

    out.release()

if __name__ == '__main__':
    main()