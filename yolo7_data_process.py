# coding=utf-8

'''
data_process.py中处理得到了coco_train_{}.json, coco_val.json
然后在这个脚本转换成yolo7需要的: label/train/im_name.txt, label/val/im_name.txt

'''


import json
import os

def generate_yolo7_label_txts(flag, train_index=None):

    # images都全部放在train下.
    train_img = '/newdata/jiachen/data/det/dataset/median_gy_data/images/train'

    if flag == 'train':
        train_js = json.load(open('/newdata/jiachen/data/det/dataset/median_gy_data/annotations/coco_{}_{}.json'.format(flag, train_index), 'r'))
    else:
        train_js = json.load(open('/newdata/jiachen/data/det/dataset/median_gy_data/annotations/coco_{}.json'.format(flag), 'r'))

    train_ims = [a['file_name'] for a in train_js['images']]
    # dict_keys(['info', 'license', 'categories', 'annotations', 'images'])
    trains = train_js['images']
    anns = train_js['annotations']
    imname_imid = dict()
    for tr in trains:
        imname_imid[tr['file_name']] = tr['id']
    
    imid_ann = dict()
    imid_cls = dict()
    for ann_dict in anns:
        image_id = str(ann_dict['image_id'])
        if image_id not in imid_ann:
            imid_ann[image_id] = []
            imid_cls[image_id] = []
        imid_ann[image_id].append(ann_dict['bbox'])
        imid_cls[image_id].append(ann_dict['category_id'])
    for train_im in train_ims:
        if train_index:
            f = open('/newdata/jiachen/data/det/dataset/median_gy_data/labels/{}{}/{}.txt'.format(flag, train_index, train_im.split('.')[0]), 'w')
        else:
            f = open('/newdata/jiachen/data/det/dataset/median_gy_data/labels/{}/{}.txt'.format(flag, train_im.split('.')[0]), 'w')
        try:
            img_id = str(imname_imid[train_im])
            boxs = imid_ann[img_id]
            clss = imid_cls[img_id]
            for k in range(len(clss)):
                x, y, w, h = boxs[k][:4]
                center = [(x+w//2)/1280, (y+h//2)/800]
                cls_ = clss[k]-1
                yolo_box = '{} {} {} {} {}\n'.format(cls_, center[0], center[1], w/1280, h/800)
                f.write(yolo_box)
            f.close()
        except:
            continue


if __name__ == '__main__':

    # train-yolo7
    # for i in range(1, 4):
    #     generate_yolo7_label_txts('train', train_index=i)
    
    # # val-yolo7
    # generate_yolo7_label_txts('val')

    # for index in range(1, 4):
    #     f = open('/newdata/jiachen/data/det/dataset/median_gy_data/train{}.txt'.format(index), 'w')
    #     im_names = [a[:-3]+'jpg' for a in os.listdir('/newdata/jiachen/data/det/dataset/median_gy_data/labels/train{}'.format(index))]
    #     for im in im_names:
    #         f.write('/newdata/jiachen/data/det/dataset/median_gy_data/images/train/{}\n'.format(im))

    pass 

