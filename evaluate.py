import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import cv2
from torch.autograd import Variable
import pdf_loader as data_loader
import models
import random
import tools
import configparser
from PIL import Image
from sklearn import metrics
from confusion_matrix import plot_confusion_matrix
from matplotlib import pyplot as plt
from scipy import stats

import numpy as np
from sobelCombined import sobelCombined
import matplotlib.image as mpimg # mpimg 用于读取图片


cf = configparser.ConfigParser()
cf.read("./config.ini")  # 读取配置文件，如果写文件的绝对路径，就可以不用os模

n_class = int(cf.get("develop", "n_class"))
data_path = cf.get("develop", "data_path")
model_Type = int(cf.get("develop", "model_Type"))
max_height = int(cf.get("develop", "max_height"))
in_channels_Nmuber = 4

data_path = os.path.expanduser(data_path)


def evaluate(modelPath):
    labels = ["BG", "figure", "table", "text"]
    use_cuda = torch.cuda.is_available()
    val_data = data_loader.ClassSeg(root=data_path, split='test', transform=True, filePath='DSSE',chanelCat=in_channels_Nmuber)
    val_loader = torch.utils.data.DataLoader(val_data,batch_size=1,shuffle=False,num_workers=5)
    print('load model .....')

    print("Using FCNs")
    vgg_model = models.VGGNet(model='vgg_self', pretrained=False, in_channels=in_channels_Nmuber)
    fcn_model = models.FCNs(pretrained_net=vgg_model, n_class=n_class, Attention=True)

    fcn_model.load_state_dict(torch.load(modelPath))

    if use_cuda:
        fcn_model.cuda()
    fcn_model.eval()

    label_trues, label_preds = [], []
    matrixs = np.zeros((n_class,n_class))
    for idx, (img, label,_) in enumerate(val_loader):
        img, label,Image_Path = val_data[idx]
        img = img.unsqueeze(0)
        if use_cuda:
            img = img.cuda()
        img = Variable(img)

        out = fcn_model(img)     # 1, 21, 320, 320
        srcImage = mpimg.imread(Image_Path)

        pred = out.data.max(1)[1].squeeze_(1).squeeze_(0)   # 320, 320

        if use_cuda:
            pred = pred.cpu()

        # 后处理
        data = pred.numpy()

        # CutX=int(data.shape[1]/32)
        #
        # for Cuti in range(data.shape[0]):
        #     for Cutj in range(0,data.shape[1],CutX):
        #         temp=data[Cuti,Cutj:Cutj+CutX-1]
        #         data[Cuti, Cutj:Cutj + CutX - 1]=stats.mode(temp)[0][0]
        #
        # dataT = data.T
        # CutY = int(dataT.shape[1]/3)
        #
        # for Cuti in range(dataT.shape[0]):
        #     for Cutj in range(0,dataT.shape[1],CutY):
        #         temp=dataT[Cuti,Cutj:Cutj+CutX-1]
        #         data[Cutj:Cutj + CutX - 1,Cuti]=stats.mode(temp)[0][0]

        # # -------------------------------------------------------------------------
        #
        # if len(srcImage.shape) == 3:
        #     image = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)  # 将图像转化为灰度图像
        # else:
        #     image = srcImage
        #
        # h = int(max_height / 64) * 64
        # w = int(image.shape[1] * (max_height / image.shape[0]) / 64) * 64
        #
        # image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        # # print(image.shape)
        # sobelCombinedIMG = sobelCombined(image)
        # data=np.multiply(data,sobelCombinedIMG)

        #
        # #-------------------------------------------------------------------------

        label_trues.append(label.numpy())
        label_preds.append(data)


        if idx % 2 == 0:
            print('evaluate [%d/%d]' % (idx, len(val_loader)))

        label_matrix_T=label.numpy()
        pre_matrix_T = data

        label_matrix = label_matrix_T.flatten()
        pre_matrix = pre_matrix_T.flatten()


        matrix = metrics.confusion_matrix(label_matrix, pre_matrix)

        if sum(sum(matrixs)) == 0:
            for i in range(len(matrixs)):
                for j in range(len(matrixs[0])):
                    matrixs[i][j] = matrix[i][j]
        else:
            # 迭代输出行
            for i in range(len(matrix)):
                # 迭代输出列
                for j in range(len(matrix[0])):
                    # print(range(len(matrix[0])))
                    matrixs[i][j] = (matrixs[i][j] + matrix[i][j])/2


    # Mymetrics = tools.accuracy_score(label_trues, label_preds,n_class)
    # Mymetrics = np.array(Mymetrics)
    # Mymetrics *= 100
    # print('''\
    #         Accuracy: {0}
    #         Accuracy Class: {1}
    #         Mean IU: {2}
    #         FWAV Accuracy: {3}'''.format(*Mymetrics))
    plot_confusion_matrix(matrixs, classes=labels, normalize=True, title='Normalized confusion matrix',cmap=plt.cm.Blues,yMax=3.5)
    #
    numberTotal = sum(sum(matrixs))
    muberTrue = 0
    PercisionList = []
    RecallList = []

    for i in range(len(matrixs)):
        for j in range(len(matrixs[0])):
            if i == j:
                muberTrue = muberTrue + matrixs[i, j]

    for i in range(len(matrixs)):
        PercisionList.append(matrixs[i, i] / sum(matrixs[:, i]))

    for i in range(len(matrixs)):
        RecallList.append(matrixs[i, i] / sum(matrixs[i, :]))

    Acurracy = muberTrue / numberTotal
    Percision = sum(PercisionList) / len(PercisionList)
    Recall = sum(RecallList) / len(RecallList)
    F1 = (2 * Percision * Recall) / (Percision + Recall)

    print(Acurracy)
    print(Percision)
    print(Recall)
    print(F1)

if __name__ == '__main__':
    # evaluate('../model/FCN1S.pth')
    evaluate('./DSSE_4_A.pth')
    # evaluate('./models/model20.pth')

