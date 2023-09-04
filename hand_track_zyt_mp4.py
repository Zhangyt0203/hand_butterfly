#实现手部实时跟踪与视频的替换
import cv2 #导入opencv库
import mediapipe as mp #导入Google开源mediapipe库
import time #导入时间库
import numpy as np
from matplotlib import pyplot as plt
import pyvista as pv

cap = cv2.VideoCapture("../Hand_wxy.mp4")  # 调用视频流（摄像头或视频文件）
cap_butterfly = cv2.VideoCapture("../butterfly.mov")
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # 选择的模型（手部侦测和手部追踪）
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)  # 点的粗度及颜色
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)  # 线的粗度及颜色
pTime = 0
cTime = 0

def addWeightedDistinguishBLK(img1, alpha, img2, beta, sigma, gamma=0.0):
    """
    图像img1和img2权重相加，但图像img2中像素为黑色的部分取img1的像素权重为sigma
    参数img1, alpha, img2, beta, gamma与addWeighted的参数相同，sigma为img1中对应img2黑色部分范围的权重
    """
    l = len(img2.shape)
    if l == 3:  # 是彩色图
        row, col, channel = img2.shape
        if channel == 3:
            img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY)
    else:  # 是灰度图
        img2Gray = img2
    retval, img2Inv = cv2.threshold(img2Gray, 5, 255,
                                    cv2.THRESH_BINARY_INV)  # 将灰度小于43的像素作为黑色，img2Inv为img2黑色部分设为255，非黑色部分设为0的img2图像掩码反码

    # 为了对img2图像前景进行平滑，对img2图像掩码反码进行开、闭、膨胀运算，
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img2Inv = cv2.morphologyEx(img2Inv, cv2.MORPH_OPEN, kernal)
    img2Inv = cv2.morphologyEx(img2Inv, cv2.MORPH_CLOSE, kernal)
    img2Inv = cv2.morphologyEx(img2Inv, cv2.MORPH_DILATE, kernal)

    retval, img2Mask = cv2.threshold(img2Inv, 0, 255, cv2.THRESH_BINARY_INV)  # 求img2的掩码

    img1Transparent = cv2.bitwise_and(img1, img1, mask=img2Inv)  # 获得img1中与img2背景范围对应的部分
    img1NotTransparent = cv2.bitwise_and(img1, img1, mask=img2Mask)  # 获得img1中与img2前景范围对应的部分
    img2NotTransparent = cv2.bitwise_and(img2, img2, mask=img2Mask)  # 获得img2中前景部分

    imgTmp = cv2.addWeighted(img1NotTransparent, alpha, img2NotTransparent, beta,
                             gamma)  # img1中与img2前景范围对应的部分与img2前景部分融合
    dest = cv2.addWeighted(imgTmp, 1, img1Transparent, sigma, gamma)  # 将融合前景部分与img1对应img2背景部分融合
    return dest


def addWeightedSmallImgToLargeImgDstgshBLK(largeImg, alpha, smallImg, beta, sigma, gamma=0.0, regionTopLeftPos=(0, 0)):
    "将小图像与大图像指定位置的内容融合，但对小图像透明部分单独处理，取大图像sigma的权重部分"
    srcW, srcH = largeImg.shape[1::-1]
    refW, refH = smallImg.shape[1::-1]
    x, y = regionTopLeftPos
    if (refW > srcW) or (refH > srcH):
        # raise ValueError("img2's size must less than or equal to img1")
        raise ValueError(
            f"img2's size {smallImg.shape[1::-1]} must less than or equal to img1's size {largeImg.shape[1::-1]}")
    else:
        if (x + refW) > srcW:
            x = srcW - refW
        if (y + refH) > srcH:
            y = srcH - refH
        destImg = np.array(largeImg)
        tmpSrcImg = destImg[y:y + refH, x:x + refW]
        tmpImg = addWeightedDistinguishBLK(tmpSrcImg, alpha, smallImg, beta, sigma, gamma)
        destImg[y:y + refH, x:x + refW] = tmpImg
        return destImg

frame_counter = 0
while 1:
    ret, img = cap.read()
    fcount = cap.get(7)  # 获得视频总帧数
    #print(fcount)      #515
    ret_1, src = cap_butterfly.read()
    fcount_1 = cap_butterfly.get(7)  # 获得视频总帧数
    frame_counter = frame_counter + 1
    print(frame_counter)
    if (frame_counter == fcount_1):
        frame_counter = 0
        cap_butterfly.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #print(fcount_1)        #150
    #print(ret_1)

    # print(src.shape)   #[1342,1920,3]
    #src = src[300:1000, 500:1400]

    if ret:
        #if not ret_1:
        #    ret_1, src = cap_butterfly.read()
        # opencv预设读取的图片为bgr图片，但需要的图片为rgp的图片，先进行转化
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        # img的宽度跟高度用一个变数来设定
        imgHeight = img.shape[0]  # 视窗高度
        imgWidth = img.shape[1]  # 视窗宽度

        if result.multi_hand_landmarks:
            print("##################################################")
            for handLms in result.multi_hand_landmarks:  # 把侦测到的所有手画出来

                #mpDraw.draw_landmarks(img, handLms)
                x_min = imgWidth
                x_max = 0
                y_min = imgHeight
                y_max = 0

                for i, lm in enumerate(handLms.landmark):  # 把21个点的作标写出来
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)

                    #print(i, xPos, yPos)  # 返回的数据为视野的坐标位置;用int()进行整形处理,否则为浮点型
                    if(y_max < yPos):
                        y_max = yPos
                    if(y_min > yPos):
                        y_min = yPos
                        if(y_min < 0):
                            y_min = 0
                    if(x_max < xPos):
                        x_max = xPos
                    if(x_min > xPos):
                        x_min = xPos
                        if(x_min < 0):
                            x_min = 0
                print(src.shape)
                src = cv2.resize(src, (x_max - x_min, y_max - y_min))
                print(src.shape)
                img = addWeightedSmallImgToLargeImgDstgshBLK(img, 0, src, 1, 1, gamma=1,
                                                             regionTopLeftPos=(x_min, y_min))

        cv2.imshow('img', img)
    # 读帧间隔时间，输入q跳出
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()