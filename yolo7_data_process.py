# coding=utf-8
import json
import os

def generate_yolo7_label_txts(flag):
    train_img = '/newdata/jiachen/data/det/dataset/median_gy_data/images/{}'.format(flag)
    train_ims = os.listdir(train_img)

    train_js = json.load(open('/newdata/jiachen/data/det/dataset/median_gy_data/annotations/{}.json'.format(flag), 'r'))

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
        f = open('/newdata/jiachen/data/det/dataset/median_gy_data/labels/train/{}.txt'.format(train_im.split('.')[0]), 'w')
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

    # generate_yolo7_label_txts('train')
    # generate_yolo7_label_txts('val')
    pass 