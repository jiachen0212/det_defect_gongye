# coding=utf-8
import cv2
import os 
import json 

def mkdirs(path):
    if not os.path.existis(path):
        os.makedirs(path)


def Erode_Dilate(img):
    # img = cv2.imread(img_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dst_Otsu1 = cv2.erode(img, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dst_Otsu2 = cv2.dilate(dst_Otsu1, kernel, 2)  
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
        cv2.circle(auged,(circle[0],circle[1]),circle[2],(0,255,0),2)

    return auged, circles, 1


def video2frames_and_aug2detcircle(video_path, out_dir_path, train_frames):
    frame_circle = dict()
    videos = os.listdir(video_path)
    for video in videos:
        # out_path = os.path.join(out_dir_path, video[:-4])
        # mkdirs(out_path)
        times = 0
        frame_frequency = 1 
        full_video_path = os.path.join(video_path, video)
        print(full_video_path)
        camera = cv2.VideoCapture(full_video_path)
        while True:
            times = times + 1
            res, image = camera.read()
            if res:
                if times % frame_frequency == 0:
                    auged_img = Erode_Dilate(image)
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
                        frame_circle[auged_base_name].extend(circles)
            else:
                print('read video fail')
                break 
        camera.release()

    return frame_circle


if __name__ == "__main__":

    # 1. 提取各视频的帧s
    # 2. 形态学腐蚀+膨胀, aug下每一帧, 再用霍夫圆检测, 得到圆心.
    # 3. 根据圆心和label.txt, 在未aug的原视频帧中定位box
    video_path = '/Users/chenjia/Downloads/Smartmore/2022/比赛-工业表面缺陷检测/mp4s'
    out_dir_path = '/Users/chenjia/Downloads/Smartmore/2022/比赛-工业表面缺陷检测/auged_frames'
    train_frames = '/Users/chenjia/Downloads/Smartmore/2022/比赛-工业表面缺陷检测/train_frames'
    frame_circle = video2frames_and_aug2detcircle(video_path, out_dir_path, train_frames)
    with open("./video_frames_circle.json", "w") as f:
        f.write(json.dumps(frame_circle, indent=4))
    