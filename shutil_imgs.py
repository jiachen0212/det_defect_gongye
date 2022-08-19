# coding=utf-8
import shutil
import os

# val_path = '/newdata/jiachen/data/det/dataset/gy_data/val'

# vals = open('/newdata/jiachen/data/det/dataset/gy_data/annotations/val.txt', 'r').readlines()
# vals = [a[:-1] for a in vals]
# for val in vals:
#     new_path = os.path.join(val_path, val.split('/')[-1])
#     try:
#         os.remove(val)
#     except:
#         continue

val_dir = '/newdata/jiachen/data/det/dataset/gy_data/val'
all_dir = '/newdata/jiachen/data/det/dataset/gy_data/auged_frames_median'
ims = os.listdir(val_dir)
for im in ims:
    path = os.path.join(all_dir, im)
    print(path)
    try:
        os.unlink(path)
    except:
        continue 