from cmath import sqrt
import numpy as np

HISTORY_COUNT = 20
PERSIST_CLASS_OCCURANCE = 15

class TrackableObject:
    maxId = 0

    def __init__(self, box, class_id = 0):
        self.id = TrackableObject.maxId + 1
        self.class_history = [class_id]
        self.bbox_history = [box]
        self.last_class_decided = class_id
        TrackableObject.maxId += 1

    def Update(self, box, class_id):
        if self.last_class_decided == -1:
            self.last_class_decided = class_id

        self.class_history.append(class_id)
        if self.class_history.__len__() > HISTORY_COUNT:
            self.class_history = self.class_history[self.class_history.__len__()-HISTORY_COUNT : ]
        self.bbox_history.append(box)
        if self.bbox_history.__len__() > HISTORY_COUNT:
            self.bbox_history = self.bbox_history[self.bbox_history.__len__()-HISTORY_COUNT : ]

    def GetId(self):
        return self.id
    
    def SetId(self, i):
        self.id = i

    def GetClassId(self):
        last_class = self.class_history[self.class_history.__len__()-1]
        last_class_detected_count = 0
        # print(self.class_history)
        for i, cls in reversed(list(enumerate(self.class_history))):
            if cls == last_class:
                last_class_detected_count += 1
                if last_class_detected_count >= PERSIST_CLASS_OCCURANCE:
                    self.last_class_decided = last_class
                    return last_class
            else:
                return self.last_class_decided
        
        return self.class_history[self.class_history.__len__()-1]

    def GetBBox(self):
        return self.bbox_history[self.bbox_history.__len__()-1]

def UpdateTrackObjectsFromDetection(trackobjects, boxes, class_ids):
    newTrackableObjects = trackobjects
    
    if newTrackableObjects.__len__() == 0:
        for (box, class_id) in zip(boxes, class_ids):
            newTrackableObjects.append(TrackableObject(box, class_id))
    else:
        boxParsed = []
        for i in range(boxes.__len__()):
            boxParsed.append(False)

        for i in range(newTrackableObjects.__len__()):
            object = newTrackableObjects[i]
            startX = int(object.GetBBox()[0])
            startY = int(object.GetBBox()[1])
            endX = int(object.GetBBox()[2])
            endY = int(object.GetBBox()[3])
            
            centerX = startX + (endX - startX) // 2
            centerY = startY + (endY - startY) // 2
            max_dist = 999999
            min_j = 0

            for j in range(boxes.__len__()):
                box = boxes[j]

                if boxParsed[j] == True:
                    continue

                box_x1 = box[0]
                box_y1 = box[1]
                box_x2 = box[2]
                box_y2 = box[3]
                box_CenterX = box_x1 + (box_x2 - box_x1) // 2
                box_CenterY = box_y1 + (box_y2 - box_y1) // 2
                dist = int(np.real(sqrt((centerX-box_CenterX)*(centerX-box_CenterX) + (centerY-box_CenterY)*(centerY-box_CenterY))))
                if (dist < max_dist):
                    min_j = j
                    max_dist = dist
            boxParsed[min_j] = True

            newTrackableObjects[i].Update(boxes[min_j], class_ids[min_j])
        
        for i in range(boxes.__len__()):
            if boxParsed[i] == True:
                continue
            newTrackableObjects.append(TrackableObject(boxes[i], class_ids[i]))
        
    return newTrackableObjects