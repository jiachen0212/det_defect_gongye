# coding=utf-8
import cv2
import os
import h5py
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange
from BM3D import bm3d

def hd52img_list(f, step):
    img_list = []
    event_g = np.array(f['events']['event_gs'])
    t = np.array(f['events']['ts'])
    x = np.array(f['events']['xs'])
    y = np.array(f['events']['ys'])
    for i in trange(math.ceil(len(event_g)/step)):
        start = i * step
        end = min(start + step, len(event_g))
        img = np.zeros([1280, 800])
        for p in range(start, end):
            img[x[p], y[p]] += int(event_g[p] / 16)
            img[x[p], y[p]] = min(img[x[p], y[p]], 255)
        img_list.append(img)
    
    return img_list


if __name__ == "__main__":

    #1. generate video
    # data_path = './dataset/train/data'
    # step = 100000
    # hd5s = [os.path.join(data_path, a) for a in os.listdir(data_path)]
    # for hd5 in hd5s:
    #     base_name = os.path.basename(hd5)
    #     f = h5py.File(hd5, 'r')
    #     img_list = hd52img_list(f, step)
    #     video = cv2.VideoWriter("./{}.mp4".format(base_name[:-3]), cv2.VideoWriter_fourcc('m','p','4','v'), 10, (1280, 800))
    #     for img in img_list:
    #         img = cv2.cvtColor(img.astype(np.uint8).transpose(), cv2.COLOR_GRAY2BGR)
    #         video.write(img)
    #     video.release()

    # 拆分video, 每一帧做噪声剔除
    for i in range(1, 27):
        im_name = './{}.jpg'.format(i)
        bm3d(im_name)





