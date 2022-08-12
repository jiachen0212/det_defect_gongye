# coding=utf-8
import os
import numpy as np 
import cv2
import matplotlib.pyplot as plt


def read_label():
    data_path = r'D:\work\project\DL\kesen\data\dataset\dataset\train\label'
    datas = [os.path.join(data_path, a) for a in os.listdir(data_path) if '.txt' in a]
    defect_lines = []
    for data_ in datas:
        defect_lines += open(data_, 'r').readlines()

    cls1, cls2, cls3 = 0, 0, 0
    for line in defect_lines:
        if line[0] == '1':
            cls1 += 1
        elif line[0] == '2':
            cls2 += 1
        elif line[0] == '3':
            cls3 += 1
    text = 'dianhen: {}, huahen: {}, zangzi: {}'.format(cls1, cls2, cls3)
    print(text)


def index_times(xs, ys):
    # 取第一条样本: event中记录各个灰度值, xs, ys中分别是像素坐标(有重复记录)
    
    index_times = np.zeros((800, 1000))
    for i in range(xs.shape[0]):
        if xs[i] >= 700:
            xs[i] -= 700
        index_times[ys[i]][xs[i]] += 1

    # r, c = np.where(index_times == np.max(index_times))
    # print(r, c)
    # print(index_times[207][33])
    # cv2.circle(index_times, (c[0], r[0]), 10, (255,255,0), 0)
    # cv2.circle(index_times, (c[1], r[1]), 10, (255,255,0), 0)
    # cv2.circle(index_times, (c[2], r[2]), 10, (255,255,0), 0)
    # cv2.imwrite('./index_times.bmp', index_times)

    return index_times


def index_values(value, xs, ys):
    index_value = np.zeros((800, 1000))
    for i in range(xs.shape[0]):
        if xs[i] >= 700:
            xs[i] -= 700
        index_value[ys[i]][xs[i]] += value[i]/16
    
    return index_value
    

def read_data(data_):
    file = h5py.File(data_,"r") 
    event_gs = np.array(file['events']['event_gs'])
    t = np.array(file['events']['ts'])
    x = np.array(file['events']['xs'])
    y = np.array(file['events']['ys'])

    return event_gs, t, x, y


import h5py
data_path = r'D:\work\project\DL\kesen\data\dataset\dataset\train\data'
datas = [os.path.join(data_path, a) for a in os.listdir(data_path)]
for data in datas:
    basename = os.path.basename(data) # 1.h5
    # print(basename[:-2]+'jpg')
    event_gs, t, x, y = read_data(data)
    image = np.zeros([1300, 800])
    for i in range(1000000, 1500000):
        image[x[i], y[i]] += int(event_gs[i]//8)
        image[x[i], y[i]] = min(image[x[i], y[i]], 255) 
    # cv2.imshow('', image)
    # cv2.waitKey(1000)
    # plt.show(image)