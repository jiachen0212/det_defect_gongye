# coding=utf-8
import cv2
import os 
import json 
from pattern_find_center import get_min_apple_pattern
from shutil import copyfile
from circle_alignment import circle_alin


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def Erode_Dilate(img):
    # img = cv2.imread(img_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dst_Otsu1 = cv2.erode(img, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dst_Otsu2 = cv2.dilate(dst_Otsu1, kernel, 1)  
    # cv2.imwrite('./auged/{}'.format(os.path.basename(img_path)), dst_Otsu1)
    
    return dst_Otsu2


def locate_circle(auged):
    gray_auged = auged[:,:,0]
    try:
        circles = cv2.HoughCircles(gray_auged,cv2.HOUGH_GRADIENT,1,100,param1=60,param2=50,minRadius=320,maxRadius=400)[0]
    except:
        return None, None, 0
    for circle in circles:
        circle = [int(a) for a in circle]
        # cv2.circle(auged,(circle[0],circle[1]),circle[2],(0,255,0),2)

    return auged, circles, 1


def video2frames_and_aug2detcircle(video_path, out_dir_path, train_frames):
    frame_circle = dict()
    videos = os.listdir(video_path)
    for video in videos:
        # out_path = os.path.join(out_dir_path, video[:-4])
        # mkdirs(out_path)
        times = 0
        # 间隔5save一次img
        frame_frequency = 5
        full_video_path = os.path.join(video_path, video)
        print(full_video_path)
        camera = cv2.VideoCapture(full_video_path)
        while True:
            times = times + 1
            res, image = camera.read()
            if res:
                if times % frame_frequency == 0:
                    # auged_img = Erode_Dilate(image)
                    auged_img = cv2.medianBlur(image, 5)
                    # 提取aug后图像的圆心
                    circled_img, circles, tag = locate_circle(auged_img)
                    if tag != 0:
                        auged_base_name = "{}_{}.jpg".format(video[:-4], times)
                        save_im_name = os.path.join(out_dir_path, auged_base_name)
                        train_im_name = os.path.join(train_frames, auged_base_name)
                        # 保存提取到了圆心的图像帧, 且做画圆可视化
                        cv2.imwrite(save_im_name, circled_img)
                        # 提取到原因的帧, 保存未经处理的原图(后续可做一些别的aug使得训练更easy..)
                        cv2.imwrite(train_im_name, image)
                        if auged_base_name not in frame_circle:
                            frame_circle[auged_base_name] = []
                        frame_circle[auged_base_name].extend(circles.tolist())
            else:
                print('read video fail')
                break 
        camera.release()

    return frame_circle



def show_video_defect_box(video_index, save_dir):
    # defect box上下左右添加一些荣誉
    temp_dict = {'1':4, '2':6, '3': 4}
    center_json = json.load(open('./train_frames_center.json', 'r'))
    ims = os.listdir('./train_frames_median')
    # 001.h5 test 
    ims = [a for a in ims if a.split('_')[0] == video_index]
    label = open('./label/{}.txt'.format(video_index), 'r').readlines()
    box_labs = []
    for lab in label:
        box_lab = [int(a) for a in lab.split(',')]
        box_lab = box_lab[:2] + [-1*box_lab[2]] + box_lab[3:]
        box_labs.append(box_lab)
        print(box_lab)
    for im in ims:
        # 中值滤波优化视频帧
        im_path = './train_frames_median/{}'.format(im)
        img = cv2.imread(im_path)
        try:
            center = center_json[im] # [0][:2]
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



def random_shuffle_0_6_for_train(index):
    video_ims = dict()
    train_frames = []
    train_data = json.load(open("./train.json", "r"))
    for im_name, center in train_data.items():
        video_name = im_name[:-4]
        if video_name not in video_ims:
            video_ims[video_name] = []
        video_ims[video_name].append(im_name)
    assert len(video_ims) == 99
    for k, v in video_ims.items():
        # 设置了seed(100), so不同次数运行这个.py,得到的train{1,2,3}.json都是一样的
        random.shuffle(v)
        # train_frames, 选择了0.6比例的数据tarin
        train_frames.extend(v[:int(len(v)*0.6)])
    train_json = dict()
    for train_name in train_frames:
        train_json[train_name] = train_data[train_data]
    print(len(train_data), len(train_json))
    with open("./train{}.json".format(index), "w") as f:
        f.write(json.dumps(train_json, indent=4))


if __name__ == "__main__":

    # 1. 提取各视频的帧s
    # 2. aug每一帧, 再用霍夫圆检测得到圆心. or 用模板匹配检测得到圆心. 
    # 3. 根据圆心和label.txt, 在未aug的原视频帧中定位box

    # img_aug:
        #1. 中值滤波效果不错
        #2. 先腐蚀再膨胀
        #3. 0825新的数据处理方式
        #4. 每个video选择0.6比例的数据train, 可多份数据训多个模型, 然后融合结果.
        
    flag = 4
    
    video_path = './mp4s'
    roi_train_img_dir = './roi_train'
    auged_frames = './auged_frames_median'
    org_frames = './train_frames_median'
    mkdirs(auged_frames)
    mkdirs(org_frames)
    mkdirs(roi_train_img_dir)
    
    if flag == 0:
        # mp4分帧提取imgs, 然后每帧做霍夫检测圆. 检测圆的目的是: 只有定位到了圆才可能基于圆心话label.
        frame_circle = video2frames_and_aug2detcircle(video_path, out_dir_path, train_frames)
        # with open("./video_frames_circle.json", "w") as f:
        #     f.write(json.dumps(frame_circle, indent=4))
        
        # 模板匹配得到圆心: 用的中值滤波后的图区匹配,
        # 也可试试直接用原视频帧匹配? 
        img_center = get_min_apple_pattern(out_dir_path)
        with open("./train_frames_center.json", "w") as f:
            f.write(json.dumps(img_center, indent=4))

    elif flag == 1:
        save_dir = './show_video_defect_box'
        mkdirs(save_dir)
        muliti_defects = open('./multi_defects.txt', 'r').readlines()[0]
        muliti_defects_txt = muliti_defects.split(',')[:-1]
        # print(muliti_defects_txt)
        # 把所有有两个defect的video的帧都save下来
        for txt_ in muliti_defects_txt:
            show_video_defect_box(txt_, save_dir)
    
    elif flag == 2:
        # 写一个python交互工具, 点击键盘一个键就remove or save 当前帧啥的..
        # 这个后续也可以用在清洗train data上, 有些时序帧上没有明显缺陷但给了box label
        
        save_dir = './show_video_defect_box'
        clear_val_imgs = './clear_val_imgs'
        mkdirs(clear_val_imgs)
        ims = [os.path.join(save_dir, a) for a in os.listdir(save_dir)]
        for im in ims:
            image = cv2.imread(im)
            cv2.imshow('', image)
            if 0xFF == ord('q'):
                continue
            # 按键 'b', 表示保存这张img作为 clear image
            elif cv2.waitKey(2500) & 0xFF == ord('b'):
                new_path = './clear_val_imgs/{}'.format(os.path.basename(im))
                copyfile(im, new_path)
                # os.remove(im)
                print('copyed {}'.format(new_path))

    elif flag == 3:
        # 0825新的数据处理代码
        # 1.霍夫圆检测设置圆的个数<=3上限, 剔除很多无效数据
        # 2. 检测到圆心后, 扣出roi部分, 剔除无效的可能干扰检测的image冗余部分
        # 3. roied_img再重新定位圆心, 然后可作为train-data输入网络
        roi_img_centers = circle_alin(video_path, auged_frames, org_frames, roi_train_img_dir)
        with open("./train.json", "w") as f:
            f.write(json.dumps(roi_img_centers, indent=4))
        
        # 这样处理后还存在一些fram其实没有明显的defect但被标注了defect. 
        # train-data融入网络时候, 针对没个.mp4做0.6的随机采样.
        # 甚至可构造出几批数据, 然后训出不同的几个模型. 
        # ↓flag==4
    
    elif flag == 4:
        # 对flag3中得到的train.json, 针对每个.mp4做0.6比例筛选, 构造出3份train{1,2,3}.json
        import random
        random.seed(100)
        for index in range(1, 4):
            random_shuffle_0_6_for_train(index)

    


        
           


            
