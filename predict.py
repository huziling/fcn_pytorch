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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片


cf = configparser.ConfigParser()
cf.read("./config.ini")  # 读取配置文件，如果写文件的绝对路径，就可以不用os模

n_class = int(cf.get("develop", "n_class"))
data_path = cf.get("develop", "data_path")
model_Type = int(cf.get("develop", "model_Type"))
max_height = int(cf.get("develop", "max_height"))

def main():
    use_cuda = torch.cuda.is_available()
    path = os.path.expanduser(data_path)

    dataset = data_loader.ClassSeg(root=data_path,split='val',transform=True)

    if model_Type == 0:
        print("Using FCN32s")
        vgg_model = models.VGGNet(model='vgg16', pretrained=True)
        fcn_model = models.FCN32s(pretrained_net=vgg_model, n_class=n_class)

    elif model_Type == 1:
        print("Using FCN16s")
        vgg_model = models.VGGNet(model='vgg16', pretrained=True)
        fcn_model = models.FCN16s(pretrained_net=vgg_model, n_class=n_class)

    elif model_Type == 2:
        print("Using FCN8s")
        vgg_model = models.VGGNet(model='vgg16', pretrained=True)
        fcn_model = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)

    elif model_Type == 3:
        print("Using FCN1s")
        vgg_model = models.VGGNet(model='vgg16', pretrained=False)
        fcn_model = models.FCNss(pretrained_net=vgg_model, n_class=n_class, Time=False, Space=False)

    elif model_Type == 4:
        print("Using FCNs")
        vgg_model = models.VGGNet(model='vgg_self', pretrained=True)
        fcn_model = models.FCNs(pretrained_net=vgg_model, n_class=n_class)

    elif model_Type == 5:
        print("Using FCNs")
        vgg_model = models.VGGNet(model='vgg16', pretrained=True)
        fcn_model = models.FCNss(pretrained_net=vgg_model, n_class=n_class)


    # fcn_model.load_state_dict(torch.load('models/model50.pth'))
    fcn_model.load_state_dict(torch.load('./models/temp.pth'))

    fcn_model.eval()

    if use_cuda:
        fcn_model.cuda()

    for i in range(len(dataset)):
        idx = i
        img, label,Image_Path = dataset[idx]
        print("deal %s" % Image_Path[-14:])

        labelImage = tools.labelToimg(label)

        if use_cuda:
            img = img.cuda()
            img = Variable(img.unsqueeze(0))
        out = fcn_model(img) # (1, 21, 320, 320)


        net_out = out.data.max(1)[1].squeeze_(0)    # 320, 320
        if use_cuda:
            net_out = net_out.cpu()

        # outImage = tools.labelToimg(net_out)  # 将网络输出转换成图片
        # 后处理
        data = net_out.numpy()




        outImage = tools.labelToimg(torch.from_numpy(data))  # 将网络输出转换成图片
        plt.imshow(labelImage)
        plt.axis('off')
        plt.savefig('./image/%s_P.jpg' % Image_Path[-14:-4], bbox_inches='tight')


        # if i == 20:
        #     break


if __name__ == '__main__':
    main()