from xtracker.track import Track, TrackState
from xtracker.utility import *

CAM_FPS = 25
CONST_DIST_CONFUSION_THRES = 0.95

class Tracker:
    def __init__(self):
        self.tracks = []
        self.maxId = 0

    def SetVariables(self, fps = CAM_FPS, classes = [], track_thres = 1):
        self.cam_fps = round(fps)
        self.class_names = classes
        self.maxId = 0

    def EmptyObjects(self):
        self.tracks = []
        self.maxId = 0


    def CleanTrackedObjects(self):
        new_objects = []
        passed_indicies = []

        i = 0
        for t1 in self.tracks:
            try: 
                i_exists = passed_indicies.index(i)
                if i_exists >= 0: continue
            except: pass 

            bFoundOverlap = False
            best_score = 0
            best_index = -1

            j = 0
            for t2 in self.tracks:
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
                new_objects.append(self.tracks[best_index])
            else:
                new_objects.append(t1)

            i += 1

        self.tracks = new_objects

    def CleanLostObjects(self):
        new_objects = []
        for t1 in self.tracks:
            if t1.IsToKeepInList() == False:
                continue
            new_objects.append(t1)
        self.tracks = new_objects

    def CleanDetectionBoxes(self, boxes = [], scores = [], class_ids = []):
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

    def GetNearestDetectionBox(self, track_box = [], boxes = [], boxUsed = []):
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

    def GetNearestTrackObject(self, box, objects = [], objUpdated = []):
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

    def IsDetectionReasonable(self, obj, detect_box = []):
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

    def Update(self, boxes = [], scores = [], class_ids = []):
        # boxes, scores, class_ids = CleanDetectionBoxes(boxes, scores, class_ids)    
        newTrackableObjects = self.tracks
        
        if self.maxId > 100000:
            self.maxId = 0

        if newTrackableObjects.__len__() == 0:
            self.maxId = 0
            for (box, score, class_id) in zip(boxes, scores, class_ids):
                newTrackableObjects.append(Track(self.maxId + 1, self.cam_fps, self.class_names, box, score, class_id))
                self.maxId += 1
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
                
                box_dist, box_index = self.GetNearestDetectionBox(obj.GetNextBox(), boxes, boxUsed)
                
                if box_index == -1:
                    newTrackableObjects[i].Update(TrackState.Lost)
                    objectUpdated[i] = True
                else:
                    obj_dist, obj_index = self.GetNearestTrackObject(boxes[box_index], \
                                                                newTrackableObjects, objectUpdated)
                    if i == obj_index:
                        if self.IsDetectionReasonable(newTrackableObjects[i], boxes[box_index]) == True:
                            objectUpdated[i] = True
                            newTrackableObjects[i].Update(TrackState.Tracked, \
                                                            boxes[box_index], \
                                                            scores[box_index], \
                                                            class_ids[box_index])
                        else:
                            objectUpdated[i] = True
                            newTrackableObjects[i].Update(TrackState.Lost)
                    elif box_dist > obj_dist:
                        if self.IsDetectionReasonable(newTrackableObjects[obj_index], boxes[box_index]) == True:
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
                            objsAdd.append(Track(self.maxId + 1, self.cam_fps, self.class_names, boxes[box_index], scores[box_index], class_ids[box_index]))
                            self.maxId += 1
                    else:
                        if self.IsDetectionReasonable(newTrackableObjects[i], boxes[box_index]) == True:
                            newTrackableObjects[i].Update(TrackState.Tracked, \
                                                        boxes[box_index], \
                                                        scores[box_index], \
                                                        class_ids[box_index])
                        else:
                            newTrackableObjects[i].Update(TrackState.Lost)
                            objsAdd.append(Track(self.maxId + 1, self.cam_fps, self.class_names, boxes[box_index], scores[box_index], class_ids[box_index]))
                            self.maxId += 1

                        objectUpdated[i] = True
                    
                    boxUsed[box_index] = True

            for i in range(newTrackableObjects.__len__()):
                if objectUpdated[i] == False:
                    newTrackableObjects[i].Update(TrackState.Lost)

            for obj_add in objsAdd:
                newTrackableObjects.append(obj_add)

            for i in range(boxes.__len__()):
                if boxUsed[i] == False:
                    newTrackableObjects.append(Track(self.maxId + 1, self.cam_fps, self.class_names, boxes[i], scores[i], class_ids[i]))
                    self.maxId += 1
        
        self.tracks = newTrackableObjects
        self.CleanLostObjects()
        self.CleanTrackedObjects()