# coding=utf-8
'''
flag == 1
    data_process.py中处理得到了train{}.json: 
    然后在这个脚本转换成mmdet需要的coco_train_{}.json

flag == 2
    用single-video中有俩defect的那些数据做val(data_process.py里生成了val.json), 生成coco_val.json

flag == 3
    show val defect_box(multi_defect)

'''

import cv2
import os
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



def get_multi_defect_video_list():
    multi_defects = open('./multi_defects_video_list.txt', 'w')
    label_dir = './label'
    lab_txts = os.listdir(label_dir)
    videoname_labelinfo = dict()
    for txt in lab_txts:
        lab_txt_path = os.path.join(label_dir, txt)
        label_info = open(lab_txt_path, 'r').readlines()
        if len(label_info) == 2:
            # print(label_info, txt)
            multi_defects.write(txt[:-4]+',')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_coco_train_json(train_index):
    # 制作coco格式数据集
    # dict_keys(['info', 'license', 'images', 'annotations', 'categories'])
    train_0818 = dict()
    train_0818["info"] = "spytensor created"
    train_0818["license"] = ["license"]
    train_0818["categories"] = [{"id": 1,  "name": "huahen"}, {"id": 2, "name": "dian"}, {"id": 3,  "name": "wuzi"}]
    
    imgid_imname = dict()
    imgname_imid = dict()
    img_dir = './roi_train'
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
            pass
        else:
            box_lab = [int(a) for a in label_info[0].split(',')]
            box_lab = box_lab[:2] + [-1*box_lab[2]] + box_lab[3:]
            videoname_labelinfo[txt[:-4]] = box_lab

    # train_index_json: train{1,2,3}.json
    center_json = json.load(open('./train{}'.format(train_index), 'r'))
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
    with open("./coco_train_{}.json".format(train_index), "w") as f:
        f.write(json.dumps(train_0818, indent=4))


def h5_to_video(data_path):
    data_path = './dataset/train/data'
    step = 100000
    hd5s = [os.path.join(data_path, a) for a in os.listdir(data_path)]
    for hd5 in hd5s:
        base_name = os.path.basename(hd5)
        f = h5py.File(hd5, 'r')
        img_list = hd52img_list(f, step)
        video = cv2.VideoWriter("./{}.mp4".format(base_name[:-3]), cv2.VideoWriter_fourcc('m','p','4','v'), 10, (1280, 800))
        for img in img_list:
            img = cv2.cvtColor(img.astype(np.uint8).transpose(), cv2.COLOR_GRAY2BGR)
            video.write(img)
        video.release()

# def split_train_val():
#     import random
#     cls1_huahen, cls2_dian, cls3_wuzi = [],[],[]
#     label_dir = './label'
#     txts = os.listdir(label_dir)
#     for txt in txts:
#         path = './label/{}'.format(txt)
#         lines = open(path, 'r').readlines()
#         if len(lines) == 1 and 'P' not in lines[0]:
#             lab_info = lines[0].split(',')
#             if lab_info[0] == '1':
#                 cls1_huahen.append(txt)
#             elif lab_info[0] == '2':
#                 cls2_dian.append(txt)
#             else:
#                 cls3_wuzi.append(txt)
#     random.shuffle(cls1_huahen)
#     random.shuffle(cls2_dian)
#     random.shuffle(cls3_wuzi)
#     # 0.3比例划分给val
#     all_test = cls1_huahen[:7] + cls2_dian[:13] + cls3_wuzi[:13] 
#     # print(cls1_huahen[:7])
#     # print(cls2_dian[:13])
#     # print(cls3_wuzi[:13])
#     # 拆 train val json 
#     all_data_js = json.load(open('./train.json', 'r'))
#     train_dict = dict()
#     val_dict = dict()

#     infos = all_data_js['info']
#     licenses = all_data_js['license']
#     categories = all_data_js['categories']

#     # dict_keys(['info', 'license', 'categories', 'annotations', 'images'])
#     keys = list(all_data_js.keys())
#     all_annotations = all_data_js['annotations'] 
#     all_images = all_data_js['images']
#     test_imgs = []
#     test_anns = []
#     for ind, img_dict in enumerate(all_images):
#         if img_dict['file_name'].split('_')[0]+'.txt' in all_test:
#             test_imgs.append(img_dict)
#             all_images.remove(img_dict)
#             test_anns.append(all_annotations[ind])
#             all_annotations.remove(all_annotations[ind])
    
#     train_dict['info'] = infos
#     train_dict['license'] = licenses
#     train_dict[ 'categories'] = categories
#     train_dict['annotations'] = all_annotations
#     train_dict['images'] = all_images

#     val_dict['info'] = infos
#     val_dict['license'] = licenses
#     val_dict[ 'categories'] = categories
#     val_dict['annotations'] = test_anns
#     val_dict['images'] = test_imgs

#     with open("./coco_train.json", "w") as f:
#         f.write(json.dumps(train_dict, indent=4))

#     with open("./coco_val.json", "w") as f:
#         f.write(json.dumps(val_dict, indent=4))



def generate_coco_val_json():
    val_0825 = dict()
    val_0825["info"] = "spytensor created"
    val_0825["license"] = ["license"]
    val_0825["categories"] = [{"id": 1,  "name": "huahen"}, {"id": 2, "name": "dian"}, {"id": 3,  "name": "wuzi"}]
    
    imgid_imname = dict()
    imgname_imid = dict()
    img_dir = './roi_train'
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
        if len(label_info) == 2:
            box_labs = []
            for line_index in range(2):
                box_lab = [int(a) for a in label_info[line_index].split(',')]
                box_lab = box_lab[:2] + [-1*box_lab[2]] + box_lab[3:]
                box_labs.append(box_lab)
            videoname_labelinfo[txt[:-4]] = box_labs

    center_json = json.load(open('./val.json', 'r'))
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
            box_labs = videoname_labelinfo[pre_txt]
            for box_lab in box_labs:
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
    val_0825['annotations'] = annotations
    val_0825['images'] = images
    with open("./coco_val.json".format(train_index), "w") as f:
        f.write(json.dumps(val_0825, indent=4))



def show_video_defect_box(video_index, save_dir, center_json):
    
    # defect box上下左右添加一些冗余
    temp_dict = {'1':4, '2':6, '3': 4}

    # center_json = './val.json', 来自data_process.py
    center_json = json.load(open(center_json, 'r'))
    ims = os.listdir('./roi_train')

    ims = [a for a in ims if a.split('_')[0] == video_index]
    label = open('./label/{}.txt'.format(video_index), 'r').readlines()
    box_labs = []
    for lab in label:
        box_lab = [int(a) for a in lab.split(',')]
        box_lab = box_lab[:2] + [-1*box_lab[2]] + box_lab[3:]
        box_labs.append(box_lab)
        print('box_lab: {}'.format(box_lab))
    for im in ims:
        # 中值滤波优化视频帧
        im_path = './train_frames_median/{}'.format(im)
        img = cv2.imread(im_path)
        try:
            center = center_json[im]  
        except:
            continue
        # 圆心-3, 稍做微调, temp_dict也对应做了上下左右宽高补偿.
        center = [a-3 for a in center]
        for box_lab in box_labs:
            box_center = center[0]+box_lab[1], center[1]+box_lab[2]
            # box上下左右放宽些
            p1 = [box_center[0]-box_lab[3]//2-temp_dict[str(box_lab[0])], box_center[1]-box_lab[4]//2-temp_dict[str(box_lab[0])]]
            p2 = [box_center[0]+box_lab[3]//2+temp_dict[str(box_lab[0])], box_center[1]+box_lab[4]//2+temp_dict[str(box_lab[0])]]
            cv2.rectangle(img, p1, p2, (0, 0, 255), 1, 8)
            # cv2.circle(img,(center[0],center[1]),1,(0,255,0),2)
        cv2.imwrite(os.path.join(save_dir, im), img)


if __name__ == "__main__":
    
    '''
    # split tarin val annotations
    # 按照类别, 分别拆分0.3给到test
    # split_train_val()

    '''

    flag = 2

    if flag == 0:
        #1. generate video
        import h5py
        h5_to_video(data_path)

    elif flag == 1:
        #2. 生成coco_train_{1,2,3}.json
        for i in range(1, 4):
            generate_coco_train_json(i)

    elif flag == 2:
        generate_coco_val_json()

    # show: 单个video中有俩defect的那些数据
    elif flag == 3:
        # 统计有俩defect的video_list
        # get_multi_defect_video_list()
        
        # val_vis_box
        save_dir = './show_defect_box'
        mkdirs(save_dir)
        muliti_defects = open('./multi_defects_video_list.txt', 'r').readlines()[0]
        muliti_defects_txt = muliti_defects.split(',')[:-1]
        # 把所有有两个defect的video的帧都save下来
        for txt_ in muliti_defects_txt:
            show_video_defect_box(txt_, save_dir, center_json)
        
        # 写一个python交互工具, 点击键盘一个键就remove or save 当前帧啥的..
        # 这个后续也可以用在清洗train data上, 有些时序帧上没有明显缺陷但给了box label
        # clear_val_imgs = './clear_val_imgs'
        # mkdirs(clear_val_imgs)
        # ims = [os.path.join(save_dir, a) for a in os.listdir(save_dir)]
        # for im in ims:
        #     image = cv2.imread(im)
        #     cv2.imshow('', image)
        #     if 0xFF == ord('q'):
        #         continue
        #     # 按键 'b', 表示保存这张img作为 clear image
        #     elif cv2.waitKey(2500) & 0xFF == ord('b'):
        #         new_path = './clear_val_imgs/{}'.format(os.path.basename(im))
        #         copyfile(im, new_path)
        #         # os.remove(im)
        #         print('copyed {}'.format(new_path))

            







        

        
                

        
    
            
        









