# coding=utf-8

'''
data_process.py中处理得到了coco_train_{}.json, coco_val.json
然后在这个脚本转换成yolo7需要的: label/train/im_name.txt, label/val/im_name.txt

'''


import json
import os

def generate_yolo7_label_txts(flag, train_index=None):

    # train_img, val_img 都是全量的
    train_img = '/newdata/jiachen/data/det/dataset/median_gy_data/images/{}'.format(flag)
    train_ims = os.listdir(train_img)

    if flag == 'train':
        train_js = json.load(open('/newdata/jiachen/data/det/dataset/median_gy_data/annotations/coco_{}_{}.json'.format(flag, train_index), 'r'))
    else:
        train_js = json.load(open('/newdata/jiachen/data/det/dataset/median_gy_data/annotations/coco_{}.json'.format(flag), 'r'))

    # dict_keys(['info', 'license', 'categories', 'annotations', 'images'])
    trains = train_js['images']
    anns = train_js['annotations']
    img_id = dict()
    for ind, tr in enumerate(trains):
        img_id[tr['file_name']] = str(ind) 

    ind_ann = dict()
    ind_cls = dict()
    for ind, ann in enumerate(anns):
        ind_ann[str(ind)] = ann['bbox']
        ind_cls[str(ind)] = ann['category_id']


    for train_im in train_ims:
        f = open('/newdata/jiachen/data/det/dataset/median_gy_data/labels/{}/{}.txt'.format(flag, train_im.split('.')[0]), 'w')
        try:
            box = ind_ann[img_id[train_im]]
            x, y, w, h = box[:4]
            center = [(x+w//2)/1280, (y+h//2)/800]
            # 1 0.716797 0.395833 0.216406 0.147222
            cls_ = ind_cls[img_id[train_im]]-1
            yolo_box = '{} {} {} {} {}\n'.format(cls_-1, center[0], center[1], w/1280, h/800)
            print(yolo_box)
            f.write(yolo_box)
            f.close()
        except:
            continue


if __name__ == '__main__':

    # train-yolo7
    for i in range(1, 4):
        generate_yolo7_label_txts('train', train_index=i))
    
    # val-yolo7
    generate_yolo7_label_txts('val')
    