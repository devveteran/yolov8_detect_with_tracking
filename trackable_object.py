from cmath import sqrt
import numpy as np

CLASSES = dict()
CLASSES["normal"] = 0
CLASSES["poop"] = 1

HISTORY_COUNT = 40
PERSIST_CLASS_OCCURANCE = 30

FRAME_COUNT_FOR_MOTION_DETECT = 20
MOVE_THRESHOLD = 0.03
POSE_THRESHOLD = 0.03

class TrackableObject:
    maxId = 0

    def __init__(self, box, score, class_id = 0):
        self.id = TrackableObject.maxId + 1
        self.class_history = [class_id]
        self.bbox_history = [box]
        self.last_class_decided = class_id
        self.score_history = [score]
        TrackableObject.maxId += 1

    def Update(self, box, score, class_id):
        self.bbox_history.append(box)
        if self.bbox_history.__len__() > HISTORY_COUNT:
            self.bbox_history = self.bbox_history[self.bbox_history.__len__()-HISTORY_COUNT : ]
        
        moving, _ = self.IsObjectMoving()
        posing, _ = self.IsObjectChangingPose()
        if moving == True or posing == True:
            self.class_history.append(CLASSES["normal"]) # normal
            self.last_class_decided = CLASSES["normal"]
        else: 
            self.class_history.append(class_id)

        if self.class_history.__len__() > HISTORY_COUNT:
            self.class_history = self.class_history[self.class_history.__len__()-HISTORY_COUNT : ]
        
        self.score_history.append(score)
        if self.score_history.__len__() > HISTORY_COUNT:
            self.score_history = self.score_history[self.score_history.__len__()-HISTORY_COUNT : ]

    def GetId(self):
        return self.id
    
    def SetId(self, i):
        self.id = i
    
    def GetScore(self):
        return self.score_history[self.score_history.__len__() - 1]
    
    def GetClassId(self):
        moving, _ = self.IsObjectMoving()
        posing, _ = self.IsObjectChangingPose()
        if moving == True or posing == True:
            self.last_class_decided = CLASSES["normal"]
            return CLASSES["normal"]
        
        last_class = self.class_history[self.class_history.__len__() - 1]
        last_class_detected_count = 0
        for i, cls in reversed(list(enumerate(self.class_history))):
            if cls == last_class:
                last_class_detected_count += 1
                if last_class_detected_count >= PERSIST_CLASS_OCCURANCE:
                    self.last_class_decided = last_class
                    return last_class
            else:
                return self.last_class_decided
        
        return self.class_history[self.class_history.__len__() - 1]

    def IsObjectMoving(self, thres = MOVE_THRESHOLD):
        xoffset_sum = 0
        yoffset_sum = 0
        calc_cnt = min(self.GetBoxHistoryCount(), FRAME_COUNT_FOR_MOTION_DETECT)
        
        last_x = -1
        last_y = -1
        for i in range(calc_cnt):
            box = self.GetiBox(self.GetBoxHistoryCount() - i - 1)
            centerX = box[0] + (box[2] - box[0]) // 2
            centerY = box[1] + (box[3] - box[1]) // 2
            if last_x != -1 and last_y != -1:
                xoffset_sum += (centerX - last_x) / centerX
                yoffset_sum += (centerY - last_y) / centerY

            last_x = centerX
            last_y = centerY
            
        xoffset_sum = abs(xoffset_sum)
        yoffset_sum = abs(yoffset_sum)

        estimate_value = (xoffset_sum + yoffset_sum) / 2
        # print("Move estimation: {:.2f}".format(estimate_value))
        if estimate_value >= thres:
            return True, estimate_value
        else:
            return False, estimate_value

    def IsObjectChangingPose(self, thres = POSE_THRESHOLD):
        width_sum = 0
        height_sum = 0
        calc_cnt = min(self.GetBoxHistoryCount(), FRAME_COUNT_FOR_MOTION_DETECT)
        
        last_width = -1
        last_height = -1
        for i in range(calc_cnt):
            box = self.GetiBox(self.GetBoxHistoryCount() - i - 1)
            width = box[2] - box[0]
            height = box[3] - box[1]
            if last_width != -1 and last_height != -1:
                width_sum += (width - last_width) / width
                height_sum += (height - last_height) / height

            last_width = width
            last_height = height

        width_sum = abs(width_sum)
        height_sum = abs(height_sum)

        # estimate_value = (width_sum + height_sum) / 2
        estimate_value = width_sum
        # print("Pose estimation: {:.2f}".format(estimate_value))
        if estimate_value >= thres:
            return True, estimate_value
        else:
            return False, estimate_value
        
    def GetCurrentBox(self):
        return self.bbox_history[self.bbox_history.__len__() - 1]
    
    def GetBoxHistoryCount(self):
        return self.bbox_history.__len__()
    
    def GetBoxHistory(self):
        return self.bbox_history
    
    def GetiBox(self, i):
        if i >= 0 and i < self.bbox_history.__len__():
            return self.bbox_history[i]
        else:
            return []
    
    def IsObjectInRect(self, rect, enter_thres = 0.1, exit_thres = 0.1):
        current_box = self.GetCurrentBox()
        
        last_box = []
        if self.GetBoxHistoryCount() > 2:
            last_box = self.GetiBox(self.GetBoxHistoryCount() - 3)
        elif self.GetBoxHistoryCount() > 1:
            last_box = self.GetiBox(self.GetBoxHistoryCount() - 2)
        else:
            last_box = self.GetCurrentBox()

        last_overlap_rate = GetRectOverlap(last_box, rect)
        current_overlap_rate = GetRectOverlap(current_box, rect)

        objIn = True
        if last_overlap_rate > current_overlap_rate: # exiting
            objIn = IsRectOverlapping(current_box, rect, exit_thres)
        else: # entering
            objIn = IsRectOverlapping(current_box, rect, enter_thres)
        
        return objIn


def CleanTrackedObjects(trackobjects = []):
    # newTrackObjects = trackobjects
    # result = []

    # i = 0
    # for t1 in newTrackObjects:
    #     j = 0
    #     for t2 in newTrackObjects:
    #         if i != j and IsRectOverlapping(t1.GetCurrentBox(), t2.GetCurrentBox()):
    #             if t1.GetScore() > t2.GetScore():
    #                 # newTrackObjects.remove(t2)
    #                 result.append(t1)
    #             else:
    #                 # newTrackObjects.remove(t1)
    #                 result.append(t2)
    #         j += 1
    #     i += 1

    # return result
    new_objects = []
    passed_indicies = []

    i = 0
    for t1 in trackobjects:
        try: 
            i_exists = passed_indicies.index(i)
            if i_exists >= 0: continue
        except: pass 

        bFoundOverlap = False
        best_score = 0
        best_index = -1

        j = 0
        for t2 in trackobjects:
            try: 
                j_exists = passed_indicies.index(j)
                if j_exists >= 0: continue
            except: pass

            if i != j and IsRectOverlapping(t1.GetCurrentBox(), t2.GetCurrentBox()) :
                bFoundOverlap = True

                if t1.GetScore() > t2.GetScore():
                    if t1.GetScore() > best_score :
                        best_score = t1.GetScore()
                        best_index = i
                else:
                    if t2.GetScore() > best_score :
                        best_score = t2.GetScore()
                        best_index = j

                try: i_exists = passed_indicies.index(i)
                except: passed_indicies.append(i)

                try: j_exists = passed_indicies.index(j)
                except: passed_indicies.append(j)

            j += 1

        if bFoundOverlap == True:
            new_objects.append(trackobjects[best_index])
        else:
            new_objects.append(t1)

        i += 1

    return new_objects

def GetRectOverlap(box1 = [], box2 = []):
    box1_x1 = box1[0]
    box1_y1 = box1[1]
    box1_x2 = box1[2]
    box1_y2 = box1[3]
    area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)

    box2_x1 = box2[0]
    box2_y1 = box2[1]
    box2_x2 = box2[2]
    box2_y2 = box2[3]
    area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    dx = min(box1_x2, box2_x2) - max(box1_x1, box2_x1)
    dy = min(box1_y2, box2_y2) - max(box1_y1, box2_y1)
    
    area = 0
    if (dx >= 0) and (dy >= 0):
        area = dx * dy
    else:
        area = 0
    
    rate1 = area / area1
    rate2 = area / area2

    return max(rate1, rate2)

def IsRectOverlapping(box1 = [], box2 = [], diffThres = 0.95):
    rate = GetRectOverlap(box1, box2)

    if rate >= diffThres:
        return True
    else:
        return False

def GetBoxCenterDistance(box1 = [], box2 = []):
    box1_x1 = box1[0]
    box1_y1 = box1[1]
    box1_x2 = box1[2]
    box1_y2 = box1[3]
    centerX1 = box1_x1 + (box1_x2 - box1_x1) // 2
    centerY1 = box1_y1 + (box1_y2 - box1_y1) // 2

    box2_x1 = box2[0]
    box2_y1 = box2[1]
    box2_x2 = box2[2]
    box2_y2 = box2[3]
    centerX2 = box2_x1 + (box2_x2 - box2_x1) // 2
    centerY2 = box2_y1 + (box2_y2 - box2_y1) // 2

    dist = int(np.real(sqrt((centerX1-centerX2)*(centerX1-centerX2) + (centerY1-centerY2)*(centerY1-centerY2))))
    return dist

def CleanDetectionBoxes(boxes = [], scores = [], class_ids = []):
    new_boxes = []
    new_scores = []
    new_class_ids = []

    passed_indicies = []

    i = 0
    for (box1, score1, class_id1) in zip(boxes, scores, class_ids):
        try: 
            i_exists = passed_indicies.index(i)
            if i_exists >= 0: continue
        except: pass 

        bFoundOverlap = False
        best_score = 0
        best_box = []
        best_classid = 0

        j = 0
        for (box2, score2, class_id2) in zip(boxes, scores, class_ids):
            try: 
                j_exists = passed_indicies.index(j)
                if j_exists >= 0: continue
            except: pass

            if i != j and IsRectOverlapping(box1, box2) :
                bFoundOverlap = True

                if score1 > score2:
                    if score1 > best_score :
                        best_score = score1
                        best_box = box1
                        best_classid = class_id1
                else:
                    if score2 > best_score :
                        best_score = score2
                        best_box = box2
                        best_classid = class_id2

                try: i_exists = passed_indicies.index(i)
                except: passed_indicies.append(i)

                try: j_exists = passed_indicies.index(j)
                except: passed_indicies.append(j)

            j += 1

        if bFoundOverlap == True:
            new_boxes.append(best_box)
            new_scores.append(best_score)
            new_class_ids.append(best_classid)
        else:
            new_boxes.append(box1)
            new_scores.append(score1)
            new_class_ids.append(class_id1)

        i += 1

    return new_boxes, new_scores, new_class_ids

def UpdateTrackObjectsFromDetection(trackobjects = [], boxes = [], scores = [], class_ids = [], strict_mode = True):
    # boxes, scores, class_ids = CleanDetectionBoxes(boxes, scores, class_ids)

    newTrackableObjects = trackobjects
    objects_to_remove = []

    if newTrackableObjects.__len__() == 0:
        TrackableObject.maxId = 0
        for (box, score, class_id) in zip(boxes, scores, class_ids):
            newTrackableObjects.append(TrackableObject(box, score, class_id))
    else:
        if boxes.__len__() == 0:
            return []
        
        boxParsed = []
        for i in range(boxes.__len__()):
            boxParsed.append(False)

        for i in range(newTrackableObjects.__len__()):
            object = newTrackableObjects[i]

            max_dist = 999999
            min_j = 0
            bFoundedBox = False

            for j in range(boxes.__len__()):
                box = boxes[j]

                if boxParsed[j] == True:
                    continue

                dist = GetBoxCenterDistance(object.GetCurrentBox(), box)
                if (dist < max_dist):
                    min_j = j
                    max_dist = dist
                    bFoundedBox = True

            if bFoundedBox == True:
                newTrackableObjects[i].Update(boxes[min_j], scores[min_j], class_ids[min_j])
                boxParsed[min_j] = True
            else:
                objects_to_remove.append(newTrackableObjects[i])

        if strict_mode == True:
            for obj_remove in objects_to_remove:
                newTrackableObjects.remove(obj_remove)

        for i in range(boxes.__len__()):
            if boxParsed[i] == False:
                newTrackableObjects.append(TrackableObject(boxes[i], scores[i], class_ids[i]))

    newTrackableObjects = CleanTrackedObjects(newTrackableObjects)
    return newTrackableObjects