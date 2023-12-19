from math import dist

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
