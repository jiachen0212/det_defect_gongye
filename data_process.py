# coding=utf-8
import h5py
import pandas as pd
import os
import numpy as np 


data_path = r'D:\work\project\DL\kesen\data\dataset\dataset\train\data'
datas = [os.path.join(data_path, a) for a in os.listdir(data_path)]
for data_ in datas:
    img = h5py.File(data_,"r") 
    for name in img:
        # e=(x,y,event_g,t)。其中，(x,y)表示触发事件的像素的空间坐标；event_g表示事件绝对灰度值；t表示该事件发生的时间戳。
        for da in img[name].items():
           for d in da:
               if len(d[:]) > 20:
                   print(d[:].shape)
    print('----')

'''
event_gs
[1041 2000  801 ... 1940 2048 1639]
ts
[1.65277125e+09 1.65277125e+09 1.65277125e+09 ... 1.65277125e+09
 1.65277125e+09 1.65277125e+09]
xs
[ 579  403  628 ...  801 1230  196]
ys
[  0   0   0 ... 443 443 444]

'''
 
 