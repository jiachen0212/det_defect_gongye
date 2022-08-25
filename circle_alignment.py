# coding=utf-8
'''
全部帧 crop圆圈区域 然后对齐 通道上堆叠起来
然后所有通道上对每个pixel位置求均值 方差之类的统计量
以上, 就处理变成了新的图

'''

import os
import cv2
import json
from pattern_find_center import get_min_apple_pattern



def locate_circle(auged):
    gray_auged = auged[:,:,0]
    try:
        circles = cv2.HoughCircles(gray_auged,cv2.HOUGH_GRADIENT,1,100,param1=60,param2=50,minRadius=320,maxRadius=400)[0]
    except:
        return None, None, 0
    
    # 检出太多的圆也是不对的, 可能因为图像中的"雪花点"导致
    if len(circles) >= 3:
        return None, None, 0
    else:
        for circle in circles:
            circle = [int(a) for a in circle]
            
            # cv2.circle(auged,(circle[0],circle[1]),circle[2],(0,255,0),2)

        return auged, circles, 1


def video_2_frams(full_video_path, out_dir_path, train_frames, auged_or_org):
    # auged_or_org = True, auged后的图像做霍夫圆检测
                   # False, 原视频帧做霍夫圆检测
    # video_path, video = '/Users/chenjia/Downloads/Smartmore/2022/比赛-工业表面缺陷检测/mp4s', '001.mp4'
    video = os.path.basename(full_video_path)
    frame_circle = dict()
    times = 0
    frame_frequency = 1
    # full_video_path = os.path.join(video_path, video)
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
                if auged_or_org:
                    circled_img, circles, tag = locate_circle(auged_img)
                else:
                    circled_img, circles, tag = locate_circle(image)
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



def vis_defect_box(roi_train_img_dir, roi_img_centers, box_labs):
    temp_dict = {'1':4, '2':6, '3': 4}
    ims = [os.path.join(roi_train_img_dir, a) for a in os.listdir(roi_train_img_dir)]
    for im in ims:
        center = roi_img_centers[os.path.basename(im)]
        center = [a-3 for a in center]
        img = cv2.imread(im)
        for box_lab in box_labs:
            box_center = center[0]+box_lab[1], center[1]+box_lab[2]
            # box上下左右放宽些
            p1 = [box_center[0]-box_lab[3]//2-temp_dict[str(box_lab[0])], box_center[1]-box_lab[4]//2-temp_dict[str(box_lab[0])]]
            p2 = [box_center[0]+box_lab[3]//2+temp_dict[str(box_lab[0])], box_center[1]+box_lab[4]//2+temp_dict[str(box_lab[0])]]
            cv2.rectangle(img, p1, p2, (0, 0, 255), 1, 8)
            # cv2.circle(img,(center[0],center[1]),1,(0,255,0),2)
        cv2.imwrite(im, img)
    


def aline_img_roi(img_centers, org_frames, roi_train_img_dir):
    for k, v in img_centers.items():
        image = cv2.imread(os.path.join(org_frames, k))
        p1, p2 = [v[0]-375, v[1]-375], [v[0]+375, v[1]+375]
        p1 = [max(0,a) for a in p1]
        p2 = [min(1280, p2[0]), min(800, p2[1])]
        # 一些物料不完整的可通过以下条件过滤掉
        if (p2[0]-p1[0]) <= 700 or (p2[1]-p1[1])<=700:
            continue
        img_roi = image[p1[1]:p2[1], p1[0]:p2[0]]
        # cv2.imshow('', img_roi)
        # cv2.waitKey(2000)
        cv2.imwrite(os.path.join(roi_train_img_dir, k), img_roi) 



def circle_alin(video_path, auged_frames, org_frames, roi_train_img_dir):
    # video_path, video_name = '/Users/chenjia/Downloads/Smartmore/2022/比赛-工业表面缺陷检测/mp4s', '001.mp4'
    # auged_frames = './1'
    # org_frames = './2'
    # roi_train_img_dir = './roi_train_img_dir'

    video_paths = [os.path.join(video_path, a) for a in os.listdir(video_path)]
    for video_path in video_paths:

        #1. video拆分成frames, median_blur_aug后做霍夫圆检测.  
        video_2_frams(video_path, auged_frames, org_frames, True)

        #2. org_frame 做模板匹配定位出小圆心.
        img_centers = get_min_apple_pattern('./pattern/train', org_frames, 'train.jpg')
    
        #3. 根据原心坐标, 对齐矫正物料圆: 其实就是扣出整个物料圆的矩形img, 物料圆的半径是375. 根据扣出的rio_img大小, 可以剔除一些定位不准的物料, 直接丢弃.
        aline_img_roi(img_centers, org_frames, roi_train_img_dir)

        # 重新定位roi_img的圆心
        roi_img_centers = get_min_apple_pattern('./pattern/train', roi_train_img_dir, 'train_roi.jpg')

    return roi_img_centers


if __name__ == "__main__":
    #1. video拆分成frames, 并且做aug和霍夫圆检测. 筛选save下后续的train-image
    video_path, video_name = '/Users/chenjia/Downloads/Smartmore/2022/比赛-工业表面缺陷检测/mp4s', '001.mp4'
    auged_frames = './1'
    org_frames = './2'
    # video_2_frams(video_path, video_name, auged_frames, org_frames, True)
   
    #2. 分别用org_frame 和aug_frame做模板匹配定位出小圆心.
    # org
    # img_centers = get_min_apple_pattern('./pattern/train', org_frames)
    # with open("./image_centers.json", "w") as f:
    #     f.write(json.dumps(img_centers, indent=4))
    # print(len(img_centers))
    '''
    # auged
    img_centers = get_min_apple_pattern('./pattern_org/train', auged_frames)
    print(len(img_centers))
    模板匹配到的图像数量都是一样额, 个人感觉是用原图好一些? [防止滤波改变图像的一些粗细度]
    '''

    #3. 根据原心坐标, 对齐矫正物料圆: 其实就是扣出整个物料圆的矩形img, 物料圆的半径是375
       #另外根据扣出的rio_img大小, 可以剔除一些定位不准的物料. 直接丢弃即可.
    roi_train_img_dir = './roi_train_img_dir'
    img_centers = json.load(open("./image_centers.json", "r"))
    for k, v in img_centers.items():
        image = cv2.imread(os.path.join(org_frames, k))
        p1, p2 = [v[0]-375, v[1]-375], [v[0]+375, v[1]+375]
        p1 = [max(0,a) for a in p1]
        p2 = [min(1280, p2[0]), min(800, p2[1])]
        # 一些物料不完整的可通过以下条件过滤掉
        if (p2[0]-p1[0]) <= 700 or (p2[1]-p1[1])<=700:
            continue
        img_roi = image[p1[1]:p2[1], p1[0]:p2[0]]
        # cv2.imshow('', img_roi)
        # cv2.waitKey(2000)
        cv2.imwrite(os.path.join(roi_train_img_dir, k), img_roi)  

    # 重新定位roi_img的圆心
    roi_img_centers = get_min_apple_pattern('./pattern/train', roi_train_img_dir, 'train_roi.jpg')
    # 画个defect可视化看看准否 
    vis_defect_box(roi_train_img_dir, roi_img_centers, [[1,93,59,7,3]])

    
        

     






