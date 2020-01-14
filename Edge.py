# encoding:utf-8

import numpy as np
import cv2
import scipy.io as scio
import configparser
import torch

cf = configparser.ConfigParser()
cf.read("./config.ini")  # 读取配置文件，如果写文件的绝对路径，就可以不用os模

max_height = int(cf.get("develop", "max_height"))


def EdgeGet(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转化为灰度图像

    h = int(max_height / 64) * 64
    w = int(image.shape[1] * (max_height / image.shape[0]) / 64) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    # Sobel边缘检测
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)  # x方向的梯度
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)  # y方向的梯度

    sobelX = np.uint8(np.absolute(sobelX))  # x方向梯度的绝对值
    sobelY = np.uint8(np.absolute(sobelY))  # y方向梯度的绝对值

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)  #

    # 拉普拉斯边缘检测
    lap = cv2.Laplacian(image, cv2.CV_64F)  # 拉普拉斯边缘检测
    lap = np.uint8(np.absolute(lap))  ##对lap去绝对值

    # canny算子
    canny = cv2.Canny(image, 30, 150)

    return image, sobelCombined, lap, canny


if __name__ == '__main__':
    image = cv2.imread("./data/Generate_pdf/JPEGImages/015_00075280-23.jpg")
    print(image.shape)
    image, sobelCombined, lap, canny = EdgeGet(image)

    # com = np.array([sobelCombined, lap, canny])
    # com = com.transpose(1, 2, 0)

    # print(com.shape)

    imgs = np.hstack([image, sobelCombined, lap, canny])
    cv2.imshow("Edge", imgs)
    # 等待关闭
    cv2.waitKey(0)
