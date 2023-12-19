from cmath import sqrt
from math import dist
import numpy as np
from enum import Enum

CAM_FPS = 25
PARAM_L = 1 # m
PARAM_S = 2.5 # m/s

CONST_DIST_CONFUSION_THRES = 0.95

class TrackState(Enum):
    Tracked = 0
    Lost = 1

class TrackableObject:
    maxId = 0
    
    class_names = []

    cam_fps = CAM_FPS
    move_frame_count = CAM_FPS
    pose_frame_count = CAM_FPS

    calc_frame_count = CAM_FPS / 2
    history_count = CAM_FPS + 1
    
    move_threshold = (0.5 / cam_fps) / PARAM_L
    pose_threshold = 0.2

    lost_frame_limit = int(PARAM_L / (PARAM_S / CAM_FPS))

    @staticmethod
    def SetVariables(fps = CAM_FPS, classes = [], track_thres = 1):
        TrackableObject.cam_fps = round(fps)

        TrackableObject.move_frame_count = round(fps)
        TrackableObject.pose_frame_count = round(fps)
        TrackableObject.calc_frame_count = 1 # round(fps / 2) 

        t_val = max(TrackableObject.move_frame_count, TrackableObject.pose_frame_count)
        t_val = max(t_val, TrackableObject.calc_frame_count)
        TrackableObject.history_count = t_val + 1
        
        TrackableObject.move_threshold = 0.5 / (fps * PARAM_L)
        TrackableObject.lost_frame_limit = int((PARAM_L / (PARAM_S / fps)) / track_thres)
        TrackableObject.class_names = classes

        return
    
    def __init__(self, box, score, class_id = 0):
        self.id = TrackableObject.maxId + 1
        self.class_history = [class_id]
        self.bbox_history = [box]
        self.last_class_decided = class_id
        self.score_history = [score]
        self.track_history = [TrackState.Tracked]
        TrackableObject.maxId += 1

        self.last_predicted_box = []

    @staticmethod
    def GetClassIndex(classname):
        index = 0
        bFound = False
        for class_name in TrackableObject.class_names:
            if class_name == classname:
                bFound = True
                break
            index += 1
        if bFound == True:
            return index
        else:
            return 0 # -1
    
    def Update(self, tracked: TrackState, box = None, score = None, class_id = None):
        self.last_predicted_box = self.GetNextBox()

        if tracked == TrackState.Tracked:
            self.bbox_history.append(box)
            if self.bbox_history.__len__() > TrackableObject.history_count:
                self.bbox_history = self.bbox_history[self.bbox_history.__len__() - TrackableObject.history_count : ]
            
            moving, _ = self.IsObjectMoving()
            posing, _ = self.IsObjectChangingPose()

            if moving == True or posing == True:
                class_index = TrackableObject.GetClassIndex("normal")
                self.class_history.append(class_index)
                self.last_class_decided = class_index
            else: 
                self.class_history.append(class_id)

            if self.class_history.__len__() > TrackableObject.history_count:
                self.class_history = self.class_history[self.class_history.__len__()-TrackableObject.history_count : ]
            
            self.score_history.append(score)
            if self.score_history.__len__() > TrackableObject.history_count:
                self.score_history = self.score_history[self.score_history.__len__()-TrackableObject.history_count : ]
        else:

            self.bbox_history.append(self.last_predicted_box)
            if self.bbox_history.__len__() > TrackableObject.history_count:
                self.bbox_history = self.bbox_history[self.bbox_history.__len__() - TrackableObject.history_count : ]

        self.track_history.append(tracked)
        if self.track_history.__len__() > TrackableObject.history_count:
                self.track_history = self.track_history[self.track_history.__len__()-TrackableObject.lost_frame_limit : ]
        

    def GetId(self):
        return self.id
    
    def SetId(self, i):
        self.id = i
    
    def GetScore(self):
        return self.score_history[self.score_history.__len__() - 1]

    def GetCurrentTracked(self):
        return self.track_history[self.track_history.__len__() - 1]
    
    def IsActiveNow(self):
        if self.GetCurrentTracked() == TrackState.Tracked:
            return True
        else:
            return False
    
    def IsToKeepInList(self):
        if self.IsActiveNow() == True:
            return True
        else:
            lost_count = 0
            for i, tracked in reversed(list(enumerate(self.track_history))):
                if tracked == TrackState.Lost:
                    lost_count += 1
                else:
                    break

            if lost_count >= TrackableObject.lost_frame_limit:
                return False
            else:
                return True

    def GetClassId(self):
        # moving, _ = self.IsObjectMoving()
        # posing, _ = self.IsObjectChangingPose()
        # if moving == True or posing == True:
        #     class_index = TrackableObject.GetClassIndex("normal")
        #     self.last_class_decided = class_index
        #     return class_index
        
        last_class = self.class_history[self.class_history.__len__() - 1]
        last_class_detected_count = 0
        for i, cls in reversed(list(enumerate(self.class_history))):
            if cls == last_class:
                last_class_detected_count += 1
                if last_class_detected_count >= TrackableObject.calc_frame_count:
                    self.last_class_decided = last_class
                    return last_class
            else:
                return self.last_class_decided
        
        return self.class_history[self.class_history.__len__() - 1]

    def IsObjectMoving(self, thres = move_threshold):
        x1_offset_sum = 0
        x1_offset_sum_counter = 0

        x2_offset_sum = 0
        x2_offset_sum_counter = 0

        y1_offset_sum = 0
        y1_offset_sum_counter = 0

        y2_offset_sum = 0
        y2_offset_sum_counter = 0

        x_counter = 0
        y_counter = 0

        calc_cnt = min(self.GetBoxHistoryCount(), TrackableObject.move_frame_count)
        
        last_box = np.array([])
        for i in range(calc_cnt):
            box = self.GetiBox(self.GetBoxHistoryCount() - i - 1)
            if last_box.__len__() > 0:
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                width = x2 - x1
                height = y2 - y1
            
                last_x1 = last_box[0]
                last_y1 = last_box[1]
                last_x2 = last_box[2]
                last_y2 = last_box[3]

                if abs(x1 - last_x1) < width:
                    x1_offset_sum += (x1 - last_x1) / width
                    x1_offset_sum_counter += 1

                if abs(x2 - last_x2) < width:
                    x2_offset_sum += (x2 - last_x2) / width
                    x2_offset_sum_counter += 1

                if abs(y1 - last_y1) < height:
                    y1_offset_sum += (y1 - last_y1) / height
                    y1_offset_sum_counter += 1

                if abs(y2 - last_y2) < height:
                    y2_offset_sum += (y2 - last_y2) / height
                    y2_offset_sum_counter += 1
                
            last_box = box
        
        if abs(x1_offset_sum) > abs(x2_offset_sum):
            xoffset_sum = x2_offset_sum
            x_counter = x2_offset_sum_counter
        else:
            xoffset_sum = x1_offset_sum
            x_counter = x1_offset_sum_counter
        
        if abs(y1_offset_sum) > abs(y2_offset_sum):
            yoffset_sum = y2_offset_sum
            y_counter = y2_offset_sum_counter
        else:
            yoffset_sum = y1_offset_sum
            y_counter = y1_offset_sum_counter

        xoffset_sum = abs(xoffset_sum)
        yoffset_sum = abs(yoffset_sum)
        
        estimate_value = 0
        if xoffset_sum > yoffset_sum:
            if x_counter > 0:
                estimate_value = xoffset_sum / x_counter
        else:
            if y_counter > 0:
                estimate_value = yoffset_sum / y_counter

        if estimate_value >= thres:
            return True, estimate_value
        else:
            return False, estimate_value

    def IsObjectChangingPose(self, thres = pose_threshold):
        width_sum = 0
        height_sum = 0
        calc_cnt = min(self.GetBoxHistoryCount(), TrackableObject.pose_frame_count)
        
        last_width = -1
        last_height = -1
        for i in range(calc_cnt):
            box = self.GetiBox(self.GetBoxHistoryCount() - i - 1)
            width = box[2] - box[0]
            height = box[3] - box[1]
            if last_width != -1 and last_height != -1:
                if abs(width - last_width) < width:
                    width_sum += (width - last_width) / width
                if abs(height - last_height) < height:
                    height_sum += (height - last_height) / height

            last_width = width
            last_height = height

        width_sum = abs(width_sum)
        height_sum = abs(height_sum)

        estimate_value = max(width_sum, height_sum)
        if estimate_value >= thres:
            return True, estimate_value
        else:
            return False, estimate_value

    def GetLastBox(self):
        if self.GetBoxHistoryCount() < 2:
            return self.bbox_history[self.bbox_history.__len__() - 1]
        return self.bbox_history[self.bbox_history.__len__() - 2]
    
    def GetLastPredictedBox(self):
        if self.last_predicted_box.__len__() > 0:
            return self.last_predicted_box
        else:
            return self.GetCurrentBox()
        
    def GetCurrentBox(self):
        return self.bbox_history[self.bbox_history.__len__() - 1]
    
    def GetNextBox(self):
        current_box = self.bbox_history[self.bbox_history.__len__() - 1]
        if self.GetBoxHistoryCount() < 2:
            return current_box
        
        last_box = self.bbox_history[self.bbox_history.__len__() - 2]
        
        cur_centerX, cur_centerY = (current_box[2] + current_box[0]) / 2, (current_box[3] + current_box[1]) / 2
        last_centerX, last_centerY = (last_box[2] + last_box[0]) / 2, (last_box[3] + last_box[1]) / 2

        offx_rate, offy_rate = cur_centerX / last_centerX, cur_centerY / last_centerY
        wid_rate, hei_rate = ((current_box[2] - current_box[0]) / (last_box[2] - last_box[0])), \
                             ((current_box[3] - current_box[1]) / (last_box[3] - last_box[1]))
        
        next_centerX, next_centerY = cur_centerX * offx_rate, cur_centerY * offy_rate
        next_wid, next_hei = ((current_box[2] - current_box[0]) * wid_rate), \
                                ((current_box[3] - current_box[1]) * hei_rate)     

        return np.array([next_centerX - next_wid / 2, \
                         next_centerY - next_hei / 2, \
                         next_centerX + next_wid / 2, \
                         next_centerY + next_hei / 2])
    
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

        last_overlap_rate = GetBoxOverlap(last_box, rect)
        current_overlap_rate = GetBoxOverlap(current_box, rect)

        objIn = True
        if last_overlap_rate > current_overlap_rate: # exiting
            objIn = CheckBoxSimilarity(current_box, rect, exit_thres, False)
        else: # entering
            objIn = CheckBoxSimilarity(current_box, rect, enter_thres, False)
        
        return objIn
## class end

def CleanTrackedObjects(trackobjects = []):
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

            if i != j and CheckBoxSimilarity(t1.GetCurrentBox(), t2.GetCurrentBox()) :
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

def CleanLostObjects(trackobjects = []):
    new_objects = []
    for t1 in trackobjects:
        if t1.IsToKeepInList() == False:
            continue
        new_objects.append(t1)
    return new_objects

def GetBoxOverlap(box1 = [], box2 = [], get_min = True):
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[2], box1[3]
    area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)

    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[2], box2[3]
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

    if get_min == True:
        return min(rate1, rate2)
    else:
        return max(rate1, rate2)
    
def CheckBoxSimilarity(box1 = [], box2 = [], diffThres = 0.9, get_min = True):
    rate = GetBoxOverlap(box1, box2, get_min)

    if rate >= diffThres:
        return True
    else:
        return False

def GetBoxCenterDistance(box1 = [], box2 = []):
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[2], box1[3]
    
    centerX1 = box1_x1 + (box1_x2 - box1_x1) / 2
    centerY1 = box1_y1 + (box1_y2 - box1_y1) / 2

    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[2], box2[3]

    centerX2 = box2_x1 + (box2_x2 - box2_x1) / 2
    centerY2 = box2_y1 + (box2_y2 - box2_y1) / 2

    distance = dist((centerX1, centerY1), (centerX2, centerY2))
    return distance

def GetBoxDistance(box1, box2):
    x1, y1 = box1[0], box1[1]
    x1b, y1b = box1[2], box1[3]

    x2, y2 = box2[0], box2[1]
    x2b, y2b = box2[2], box2[3]
    
    dist_left_top = dist((x1, y1), (x2, y2))
    dist_left_bottom = dist((x1, y1b), (x2, y2b))
    dist_right_top = dist((x1b, y1), (x2b, y2))
    dist_right_bottom = dist((x1b, y1b), (x2b, y2b))

    average_distance = (dist_left_top + dist_left_bottom + dist_right_top + dist_right_bottom) / 4
    return min(dist_left_top, dist_left_bottom, dist_right_top, dist_right_bottom), \
        max(dist_left_top, dist_left_bottom, dist_right_top, dist_right_bottom), \
        average_distance

def GetBoxRatioSimilarity(box1, box2):
    width1 = box1[2] - box1[0]
    height1 = box1[3] - box1[1]

    width2 = box2[2] - box2[0]
    height2 = box2[3] - box2[1]

    ratio1 = width1 / height1
    ratio2 = width2 / height2

    return min(ratio1/ratio2, ratio2/ratio1)

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

            if i != j and CheckBoxSimilarity(box1, box2) :
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

def GetNearestDetectionBox(track_box = [], boxes = [], boxUsed = []):
    def sortDistance(val):
        return val[1]
    def sortRatio(val):
        return val[2]
    
    dist_array = []
    for i in range(boxes.__len__()):
        if boxUsed[i] == True: continue

        box = boxes[i]
        _, _, distance = GetBoxDistance(track_box, box)
        aspect_ratio = GetBoxRatioSimilarity(track_box, box)
        dist_array.append((i, distance, aspect_ratio))

    if dist_array.__len__() == 0:
        return float("inf"), -1
    
    if dist_array.__len__() == 1:
        return dist_array[0][1], dist_array[0][0]    
    
    dist_array.sort(key=sortDistance, reverse=False)
    
    filtered_array = []
    for (i, dist, ratio) in dist_array:
        if dist == 0 or dist_array[0][1] == 0:
            t = 1
        else:
            t = min(dist/dist_array[0][1], dist_array[0][1]/dist)
        
        if t > CONST_DIST_CONFUSION_THRES:
            filtered_array.append((i, dist, ratio))

    filtered_array.sort(key=sortRatio, reverse=True)
    return filtered_array[0][1], filtered_array[0][0]

def GetNearestTrackObject(box, objects = [], objUpdated = []):
    def sortDistance(val):
        return val[1]
    def sortRatio(val):
        return val[2]

    dist_array = []
    
    for i in range(objects.__len__()):
        if objUpdated[i] == True: continue

        obj_box = objects[i].GetCurrentBox()
        _, _, distance = GetBoxDistance(obj_box, box)
        aspect_ratio = GetBoxRatioSimilarity(obj_box, box)
        dist_array.append((i, distance, aspect_ratio))

    if dist_array.__len__() == 0:
        return float("inf"), -1

    if dist_array.__len__() == 1:
        return dist_array[0][1], dist_array[0][0]    
    
    dist_array.sort(key=sortDistance, reverse=False)
    
    filtered_array = []
    for (i, dist, ratio) in dist_array:
        if dist == 0 or dist_array[0][1] == 0:
            t = 1
        else:
            t = min(dist/dist_array[0][1], dist_array[0][1]/dist)
        
        if t > CONST_DIST_CONFUSION_THRES:
            filtered_array.append((i, dist, ratio))

    filtered_array.sort(key=sortRatio, reverse=True)
    return filtered_array[0][1], filtered_array[0][0]

def IsDetectionReasonable(obj, detect_box = []):
    cur_box = obj.GetCurrentBox()
    last_box = obj.GetLastBox()

    cur_width = cur_box[2] - cur_box[0]
    cur_height = cur_box[3] - cur_box[1]

    last_width = last_box[2] - last_box[0]
    last_height = last_box[3] - last_box[1]

    detect_width = detect_box[2] - detect_box[0]
    detect_height = detect_box[3] - detect_box[1]

    min_dist, _, _ = GetBoxDistance(cur_box, detect_box)
    last_min_dist, _, _ = GetBoxDistance(last_box, cur_box)

    if min_dist > last_min_dist + dist((0, 0), (cur_width, cur_height)):
        return False
    return True

def UpdateTrackObjectsFromDetection(trackobjects = [], boxes = [], scores = [], class_ids = []):
    # boxes, scores, class_ids = CleanDetectionBoxes(boxes, scores, class_ids)    
    newTrackableObjects = trackobjects
    
    if TrackableObject.maxId > 100000:
        TrackableObject.maxId = 0

    if newTrackableObjects.__len__() == 0:
        TrackableObject.maxId = 0
        for (box, score, class_id) in zip(boxes, scores, class_ids):
            newTrackableObjects.append(TrackableObject(box, score, class_id))
    else:
        if boxes.__len__() == 0:
            return []

        objectUpdated = []
        for i in range(newTrackableObjects.__len__()):
            objectUpdated.append(False)

        boxUsed = []
        for i in range(boxes.__len__()):
            boxUsed.append(False)
        
        objsAdd = []
        for i in range(newTrackableObjects.__len__()):
            if objectUpdated[i] == True: 
                continue

            obj = newTrackableObjects[i]
            
            box_dist, box_index = GetNearestDetectionBox(obj.GetNextBox(), boxes, boxUsed)
            
            if box_index == -1:
                newTrackableObjects[i].Update(TrackState.Lost)
                objectUpdated[i] = True
            else:
                obj_dist, obj_index = GetNearestTrackObject(boxes[box_index], \
                                                            newTrackableObjects, objectUpdated)
                if i == obj_index:
                    if IsDetectionReasonable(newTrackableObjects[i], boxes[box_index]) == True:
                        objectUpdated[i] = True
                        newTrackableObjects[i].Update(TrackState.Tracked, \
                                                        boxes[box_index], \
                                                        scores[box_index], \
                                                        class_ids[box_index])
                    else:
                        objectUpdated[i] = True
                        newTrackableObjects[i].Update(TrackState.Lost)
                elif box_dist > obj_dist:
                    if IsDetectionReasonable(newTrackableObjects[obj_index], boxes[box_index]) == True:
                        objectUpdated[obj_index] = True
                        newTrackableObjects[obj_index].Update(TrackState.Tracked, \
                                                            boxes[box_index], \
                                                            scores[box_index], \
                                                            class_ids[box_index])
                    else:
                        objectUpdated[i] = True
                        objectUpdated[obj_index] = True
                        newTrackableObjects[i].Update(TrackState.Lost)
                        newTrackableObjects[obj_index].Update(TrackState.Lost)
                        objsAdd.append(TrackableObject(boxes[box_index], scores[box_index], class_ids[box_index]))
                else:
                    if IsDetectionReasonable(newTrackableObjects[i], boxes[box_index]) == True:
                        newTrackableObjects[i].Update(TrackState.Tracked, \
                                                    boxes[box_index], \
                                                    scores[box_index], \
                                                    class_ids[box_index])
                    else:
                        newTrackableObjects[i].Update(TrackState.Lost)
                        objsAdd.append(TrackableObject(boxes[box_index], scores[box_index], class_ids[box_index]))
                    objectUpdated[i] = True
                
                boxUsed[box_index] = True

        for i in range(newTrackableObjects.__len__()):
            if objectUpdated[i] == False:
                newTrackableObjects[i].Update(TrackState.Lost)

        for obj_add in objsAdd:
            newTrackableObjects.append(obj_add)

        for i in range(boxes.__len__()):
            if boxUsed[i] == False:
                newTrackableObjects.append(TrackableObject(boxes[i], scores[i], class_ids[i]))
    
    newTrackableObjects = CleanLostObjects(newTrackableObjects)
    newTrackableObjects = CleanTrackedObjects(newTrackableObjects)
    return newTrackableObjects
