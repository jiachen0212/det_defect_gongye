# coding=utf-8
import cv2
import os
import h5py
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange
from BM3D import bm3d
import json 

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


def generate_coco_data_json():
    # 制作coco格式数据集
    # dict_keys(['info', 'license', 'images', 'annotations', 'categories'])
    train_0818 = dict()
    train_0818["info"] = "spytensor created"
    train_0818["license"] = ["license"]
    train_0818["categories"] = [{"id": 1,  "name": "huahen"}, {"id": 2, "name": "dian"}, {"id": 3,  "name": "wuzi"}]
    
    # imgname_id = dict()
    imgid_imname = dict()
    imgname_imid = dict()
    img_dir = './auged_frames_median'
    ims = os.listdir(img_dir)
    for id_, im in enumerate(ims):
        imgid_imname[id_] = im 
        imgname_imid[im] = id_

    label_dir = './label'
    lab_txts = os.listdir(label_dir)
    videoname_labelinfo = dict()
    for txt in lab_txts:
        lab_txt_path = os.path.join(label_dir, txt)
        label_info = open(lab_txt_path, 'r').readlines()
        if label_info[0][0] == 'P' or len(label_info) == 2:
            # print(label_info, txt)
            pass
        else:
            box_lab = [int(a) for a in label_info[0].split(',')]
            box_lab = box_lab[:2] + [-1*box_lab[2]] + box_lab[3:]
            videoname_labelinfo[txt[:-4]] = box_lab

    center_json = json.load(open('./train_frames_center.json', 'r'))
    annotations = []
    images = []
    id_ = 0
    # 针对类别做box放宽, 比如dian缺陷, center定位不准就导致box内无缺陷. 上下左右放开若干个像素点.
    temp_dict = {'1':4, '2':6, '3': 4}
    for k, v in center_json.items():
        # k,v: 052_143.jpg [514, 398]
        single_ann = dict()
        sigle_img = dict()
        image_id = imgname_imid[k]
        single_ann['image_id'] = image_id
        sigle_img['id'] = image_id
        sigle_img['file_name'] = k 
        sigle_img['height'] = 800
        sigle_img['width'] = 1280
        single_ann['id'] = id_
        id_ += 1
        pre_txt = k.split('_')[0]
        try:
            # 使用lab_box信息 
            box_lab = videoname_labelinfo[pre_txt]
            single_ann['category_id'] = box_lab[0]
            single_ann['segmentation'] = [0,0]
            box_center = v[0]+box_lab[1], v[1]+box_lab[2]
            p1 = [box_center[0]-box_lab[3]//2-temp_dict[str(box_lab[0])], box_center[1]-box_lab[4]//2-temp_dict[str(box_lab[0])]]
            single_ann['bbox'] = p1 + box_lab[3:]
            single_ann['iscrowd'] = 0
            single_ann['area'] = int(box_lab[3])*int(box_lab[4])
            annotations.append(single_ann)
            images.append(sigle_img)
        except:
            pass 
    train_0818['annotations'] = annotations
    train_0818['images'] = images
    with open("./train.json", "w") as f:
        f.write(json.dumps(train_0818, indent=4))


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

    # 生成总的data-coco-json
    generate_coco_data_json()


    # split tarin val annotations
    # 按照类别, 分别拆分0.3给到test
    import random
    cls1_huahen, cls2_dian, cls3_wuzi = [],[],[]
    label_dir = './label'
    txts = os.listdir(label_dir)
    for txt in txts:
        path = './label/{}'.format(txt)
        lines = open(path, 'r').readlines()
        if len(lines) == 1 and 'P' not in lines[0]:
            lab_info = lines[0].split(',')
            if lab_info[0] == '1':
                cls1_huahen.append(txt)
            elif lab_info[0] == '2':
                cls2_dian.append(txt)
            else:
                cls3_wuzi.append(txt)
    random.shuffle(cls1_huahen)
    random.shuffle(cls2_dian)
    random.shuffle(cls3_wuzi)
    # 0.3比例划分给val
    all_test = cls1_huahen[:7] + cls2_dian[:13] + cls3_wuzi[:13] 
    print(cls1_huahen[:7])
    print(cls2_dian[:13])
    print(cls3_wuzi[:13])
    # 拆 train val json 
    all_data_js = json.load(open('./train.json', 'r'))
    train_dict = dict()
    val_dict = dict()

    infos = all_data_js['info']
    licenses = all_data_js['license']
    categories = all_data_js['categories']

    # dict_keys(['info', 'license', 'categories', 'annotations', 'images'])
    keys = list(all_data_js.keys())
    all_annotations = all_data_js['annotations'] 
    all_images = all_data_js['images']
    test_imgs = []
    test_anns = []
    for ind, img_dict in enumerate(all_images):
        if img_dict['file_name'].split('_')[0]+'.txt' in all_test:
            test_imgs.append(img_dict)
            all_images.remove(img_dict)
            test_anns.append(all_annotations[ind])
            all_annotations.remove(all_annotations[ind])
    
    train_dict['info'] = infos
    train_dict['license'] = licenses
    train_dict[ 'categories'] = categories
    train_dict['annotations'] = all_annotations
    train_dict['images'] = all_images

    val_dict['info'] = infos
    val_dict['license'] = licenses
    val_dict[ 'categories'] = categories
    val_dict['annotations'] = test_anns
    val_dict['images'] = test_imgs

    with open("./train.json", "w") as f:
        f.write(json.dumps(train_dict, indent=4))

    with open("./val.json", "w") as f:
        f.write(json.dumps(val_dict, indent=4))


    

    
            

    
 
        
    









