from enum import Enum
import numpy as np
from xtracker.utility import *

PARAM_L = 1 # m
PARAM_S = 2.5 # m/s

class TrackState(Enum):
    Tracked = 0
    Lost = 1

class Track:

    def SetVariables(self, fps, classes = [], track_thres = 1):
        self.cam_fps = round(fps)

        self.move_frame_count = round(fps)
        self.pose_frame_count = round(fps)
        self.calc_frame_count = 1 # round(fps / 2) 

        t_val = max(self.move_frame_count, self.pose_frame_count)
        t_val = max(t_val, self.calc_frame_count)
        self.history_count = t_val + 1
        self.pose_threshold = 0.2

        self.move_threshold = 0.5 / (fps * PARAM_L)
        self.lost_frame_limit = int((PARAM_L / (PARAM_S / fps)) / track_thres)
        self.class_names = classes

        return
    
    def __init__(self, id, fps, classes, box, score, class_id = 0):
        self.SetVariables(fps, classes)

        self.id = id
        self.class_history = [class_id]
        self.bbox_history = [box]
        self.last_class_decided = class_id
        self.score_history = [score]
        self.track_history = [TrackState.Tracked]

        self.last_predicted_box = []

    def GetClassIndex(self, classname):
        index = 0
        bFound = False
        for class_name in self.class_names:
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
            if self.bbox_history.__len__() > self.history_count:
                self.bbox_history = self.bbox_history[self.bbox_history.__len__() - self.history_count : ]
            
            moving, _ = self.IsObjectMoving()
            posing, _ = self.IsObjectChangingPose()

            if moving == True or posing == True:
                class_index = self.GetClassIndex("normal")
                self.class_history.append(class_index)
                self.last_class_decided = class_index
            else: 
                self.class_history.append(class_id)

            if self.class_history.__len__() > self.history_count:
                self.class_history = self.class_history[self.class_history.__len__()-self.history_count : ]
            
            self.score_history.append(score)
            if self.score_history.__len__() > self.history_count:
                self.score_history = self.score_history[self.score_history.__len__()-self.history_count : ]
        else:

            self.bbox_history.append(self.last_predicted_box)
            if self.bbox_history.__len__() > self.history_count:
                self.bbox_history = self.bbox_history[self.bbox_history.__len__() - self.history_count : ]

        self.track_history.append(tracked)
        if self.track_history.__len__() > self.history_count:
                self.track_history = self.track_history[self.track_history.__len__()-self.lost_frame_limit : ]
        

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

            if lost_count >= self.lost_frame_limit:
                return False
            else:
                return True

    def GetClassId(self):
        # moving, _ = self.IsObjectMoving()
        # posing, _ = self.IsObjectChangingPose()
        # if moving == True or posing == True:
        #     class_index = Track.GetClassIndex("normal")
        #     self.last_class_decided = class_index
        #     return class_index
        
        last_class = self.class_history[self.class_history.__len__() - 1]
        last_class_detected_count = 0
        for i, cls in reversed(list(enumerate(self.class_history))):
            if self.GetClassGroup(cls) == self.GetClassGroup(last_class):
                last_class_detected_count += 1
                if last_class_detected_count >= self.calc_frame_count:
                    self.last_class_decided = last_class
                    return last_class
            else:
                return self.last_class_decided
        
        return self.class_history[self.class_history.__len__() - 1]

    def GetClassGroup(self, cls):
        if self.class_names[cls] == "normal": return 1
        else: return 2
        
    def IsObjectMoving(self):
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

        calc_cnt = min(self.GetBoxHistoryCount(), self.move_frame_count)
        
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

        if estimate_value >= self.move_threshold:
            return True, estimate_value
        else:
            return False, estimate_value

    def IsObjectChangingPose(self):
        width_sum = 0
        height_sum = 0
        calc_cnt = min(self.GetBoxHistoryCount(), self.pose_frame_count)
        
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
        if estimate_value >= self.pose_threshold:
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