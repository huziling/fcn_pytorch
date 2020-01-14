# encoding:utf-8
import cv2
import os
import numpy as np


max_height=1024


# dir="/media/videt/DATA/layout/data/DSSE/JPEGImages/"
dir="/media/videt/DATA/layout/HData/JPEGImages/"

for filename in os.listdir(dir):
    if filename.endswith('jpg'):

        # if 1==1:
         if filename[0:4] == '1111':

            filePath=dir+filename
            # print(filePath)
            image = cv2.imread(filePath, 1)

            # augment
            h = int(max_height / 64) * 64
            w = int(image.shape[1] * (max_height / image.shape[0]) / 64) * 64
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            imageO=image

            # cv2.imshow("image", image) #展示图片
            # cv2.waitKey(0)
            #灰度图片
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 去除页眉横线
            if w>1000:
                gray[1:120, :] = 255
            else:
                gray[1:80, :] = 255

            #二值化
            ret,binary = cv2.threshold(~gray, 60, 255, cv2.THRESH_BINARY)



            rows,cols=binary.shape
            scale = 18
            #识别横线
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(cols//scale,1))
            eroded = cv2.erode(binary,kernel,iterations = 1)
            dilatedcol = cv2.dilate(eroded,kernel,iterations = 1)

            # cv2.imshow("image", dilatedcol)  # 展示图片

            # 识别竖线
            scale = 20
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
            eroded = cv2.erode(binary, kernel, iterations=1)
            dilatedrow = cv2.dilate(eroded, kernel, iterations=1)


            # 标识交点
            bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)


            # 标识表格
            merge = cv2.add(dilatedcol, dilatedrow)

            # 识别黑白图中的白色交叉点，将横纵坐标取出
            ys, xs = np.where(bitwiseAnd > 0)

            # print(len(xs))


            # if (len(xs))>1200:
            #     merge[:,:]=0



            for i in range(len(merge)):
                if sum(merge[i,:])>0:
                    merge[i, :]=255

            colltableRow=[]
            for i in range(len(merge)):
                    if sum(merge[i,:])>0 and sum(merge[i-2,:])==0 and sum(merge[i+2,:])==0:
                        colltableRow.append(i)


            # print(colltableRow)

            for i in range(len(colltableRow)-1):
                # print(colltableRow[i+1]-colltableRow[i])
                if colltableRow[i+1]-colltableRow[i]<1200:
                    merge[colltableRow[i]:colltableRow[i+1], :] = 255

            merge[:, 0:20] = 0
            merge[:, -20:] = 0

            for i in range(len(merge)):
                if sum(merge[i, :]) > 0 and sum(sum(merge[i - 5:i-1, :]))== 0 and sum(sum(merge[i + 1:i+5, :])) == 0:
                    merge[i, :]=0


            pointList=[]


            for i in range(1,len(merge),1):
                if sum(merge[i-1,:])==0 and sum(merge[i,:])>0:
                    point=[1,2,3,4]
                    point[0]=40
                    point[2]=len(merge[0])-40
                    point[1]=i

                    pointList.append(point)

            j=0
            for i in range(0, len(merge), 1):
                if sum(merge[i, :]) >0 and sum(merge[i+1, :]) == 0:
                    pointList[j][3]=i
                    j=j+1

            # print(pointList)

            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            for i in range(0, len(pointList), 1):
                cv2.rectangle(imageO, (pointList[i][0], pointList[i][1]), (pointList[i][2], pointList[i][3]), (0, 255, 0), 2)

            cv2.imshow("table", imageO)
            cv2.waitKey(0)
            # cv2.imwrite('2.jpg', image)
            # cv2.imshow("table", merge)
            # cv2.waitKey(0)
