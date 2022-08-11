# coding=utf-8
import os
import numpy as np 


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



# 灰度区间: 2^12, 
# HxW: 1280x800

import h5py
data_path = r'D:\work\project\DL\kesen\data\dataset\dataset\train\data'
datas = [os.path.join(data_path, a) for a in os.listdir(data_path)]
events, ts, xs, ys = [],[],[],[]
for data_ in datas:
    file = h5py.File(data_,"r") 
    # e=(x,y,event_g,t)。其中，(x,y)表示触发事件的像素的空间坐标；event_g表示事件绝对灰度值；t表示该事件发生的时间戳。
    for index, da in enumerate(file['events'].items()):
        # event_gs, ts, xs, ys
        for ind, d in enumerate(da):
            if index == 0 and ind == 1:
                events.append(d[:])
            elif index == 1 and ind == 1:
                ts.append(d[:])
            elif index == 2 and ind == 1:
                xs.append(d[:])
            elif index == 3 and ind == 1:
                ys.append(d[:])
    print(len(events), len(ts), len(xs), len(ys))
    file.close()   

events = np.array(events)
ts = np.array(ts)
xs = np.array(xs)
ys = np.array(ys)
np.save('./events.npy', events)
np.save('./ts.npy', ts)
np.save('./xs.npy', xs)
np.save('./yx.npy', ys)

