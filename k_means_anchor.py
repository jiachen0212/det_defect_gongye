# coding=utf-8
import json
import os
import collections
import numpy as np



def compute_iou(box, anchors):
    # distance = 1 - iou
    # dis = []
    ious = []
    for anchor in anchors:
        w_min = np.min([box[0], anchor[0]])
        h_min = np.min([box[1], anchor[1]])
        intersection = w_min*h_min
        union = box[0]*box[1] + anchor[0]*anchor[1]
        iou = intersection/(union - intersection)
        # dis.append(1 - iou)
        ious.append(iou)

    return ious



def kmeans(boxes, k, dist=np.median):
    # number of boxes
    box_num = len(boxes)
    # store cluster center of each box
    nearest_id = np.zeros(box_num)
    np.random.seed(42)
    # initialize the cluster
    clusters = boxes[np.random.choice([i for i in range(box_num)], k, replace=False)]
    while True:
        # store iou distance between each pair of boxes and anchors
        distance = []
        for i in range(box_num):
            ious = compute_iou(boxes[i], clusters)
            dis = [1-iou for iou in ious]
            distance.append(dis)
        distance = np.array(distance)
        # calculate box cluster id
        new_nearest_id = np.argmin(distance, axis=1)
        # break condition
        if (new_nearest_id == nearest_id).all():
            break
        # update clusters using median strategy
        for j in range(k):
            clusters[j] = dist(boxes[new_nearest_id == j], axis=0)
        nearest_id = new_nearest_id

    return clusters


if __name__ == "__main__":

    boxes = []
    label_dir = './label'
    txts = os.listdir(label_dir)
    for txt in txts:
        path = './label/{}'.format(txt)
        lines = open(path, 'r').readlines()
        if len(lines) == 1 and 'P' not in lines[0]:
            label_info = lines[0].split(',')
            # print(label_info)
            hw = [int(label_info[3]), int(label_info[4])]
            boxes.append(hw)

    boxes = np.array(boxes)
    k = 9
    clusters = kmeans(boxes, 9)
    
    for box in clusters:
        # print(np.sqrt(box[0]*box[1]))
        print(box[0]/box[1])
