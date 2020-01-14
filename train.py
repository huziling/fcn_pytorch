import os
import torch
from torch.autograd import Variable
import torch.utils.data.dataloader
import numpy as np
import models
import pdf_loader as data_loader
import dsse_loader as test_data_loader
import loss
from torch.optim import Adam, SGD
from torchvision import transforms
import configparser
import logging
from sklearn import metrics
import time
from tensorboardX import SummaryWriter
from Model import DeepLab,resnet50,FCN1s_resnet
from torchsummary import summary
cf = configparser.ConfigParser()
cf.read("./config.ini")  # 读取配置文件，如果写文件的绝对路径，就可以不用os模

batch_size = int(cf.get("develop", "batch_size"))
epochs = int(cf.get("develop", "epochs"))
n_class = int(cf.get("develop", "n_class"))
learning_rate = float(cf.get("develop", "learning_rate"))
data_path = cf.get("develop", "data_path")
model_save = cf.get("develop", "model_save")
workNmber = int(cf.get("develop", "workNmber"))
model_Type = int(cf.get("develop", "model_Type"))
resume = int(cf.get("develop", "resume"))
visiualize_enable = int(cf.get("develop", "visiualize_enable"))
epochs_step = cf.get("develop", "epochs_step").split(',')

epochs_step = [int(i) for i in epochs_step]
best_test_loss = np.inf
use_cuda = torch.cuda.is_available()
best_test_Accuracy = 0
loss_train = 0

# dataset 2007
data_path = os.path.expanduser(data_path)

# print(data_path)

print('load data....')
train_data = data_loader.ClassSeg(root=data_path, split='val', transform=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workNmber,pin_memory=True,drop_last=True)

val_data = test_data_loader.DSSEClassSeg(root=data_path, split='val', transform=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workNmber,pin_memory=True)

# test_data = data_loader.ClassSeg(root=data_path, split='test', transform=True)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workNmber)

print('load model....')

if model_Type == 0:
    print("Using FCN32s")
    vgg_model = models.VGGNet(model='vgg16', pretrained=True)
    fcn_model = models.FCN32s(pretrained_net=vgg_model, n_class=n_class)
    test_model = models.FCN32s(pretrained_net=vgg_model, n_class=n_class)
elif model_Type == 1:
    print("Using FCN16s")
    vgg_model = models.VGGNet(model='vgg16', pretrained=True)
    fcn_model = models.FCN16s(pretrained_net=vgg_model, n_class=n_class)
    test_model = models.FCN16s(pretrained_net=vgg_model, n_class=n_class)
elif model_Type == 2:
    print("Using FCN8s")
    vgg_model = models.VGGNet(model='vgg16', pretrained=True)
    fcn_model = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)
    test_model = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)
elif model_Type == 3:
    print("Using FCN1s")
    vgg_model = models.VGGNet(model='vgg16', pretrained=True)
    fcn_model = models.FCNss(pretrained_net=vgg_model, n_class=n_class, Time=False, Space=False)
    test_model = models.FCNss(pretrained_net=vgg_model, n_class=n_class, Time=False, Space=False)
elif model_Type == 4:
    print("Using FCNs")
    vgg_model = models.VGGNet(model='vgg_self', pretrained=True)
    fcn_model = models.FCNs(pretrained_net=vgg_model, n_class=n_class)
    test_model = models.FCNs(pretrained_net=vgg_model, n_class=n_class)
elif model_Type == 5:
    print("Using FCNs")
    vgg_model = models.VGGNet(model='vgg16', pretrained=True)
    fcn_model = models.FCNss(pretrained_net=vgg_model, n_class=n_class)
    test_model = models.FCNss(pretrained_net=vgg_model, n_class=n_class)
elif model_Type == 8:
    print('Using resnet50/FCNS')
    resnet_model = resnet50(pretrained=True,output_stride=32)
    fcn_model = FCN1s_resnet(pretrained_net=resnet_model, n_class=n_class)
    # test_model = FCN1s_resnet(pretrained_net=resnet_model,n_class=n_class)
elif model_Type == 9:
    print('Using resnet50/deeplabv3+')
    fcn_model = DeepLab(output_stride=16,class_num=n_class,pretrained=True,mode='resnet50',sparable=True)
    # test_model = DeepLab(output_stride=16,class_num=n_class,pretrained=False,mode='resnet50')
elif model_Type == 10:
    print('Using xception/deeplabv3+')
    fcn_model = DeepLab(output_stride=16,class_num=n_class,pretrained=True,mode='xception')
    # test_model = DeepLab(output_stride=16,class_num=n_class,pretrained=False,mode='xception')
elif model_Type == 11:
    print('Using resnet18/FCNS')
    fcn_model = DeepLab(output_stride=16,class_num=n_class,pretrained=True,mode='resnet18',sparable=True)

if use_cuda:
    fcn_model.cuda()

transform_I_T = transforms.Compose([
    transforms.CenterCrop((512, 512)),  # 只能对PIL图片进行裁剪
    transforms.ToTensor()])

criterion = loss.CrossEntropyLoss2d()
# create your optimizer
optimizer = Adam(fcn_model.parameters())
logging.basicConfig(filename='logger.log', level=logging.INFO)

if resume == 1:
    fcn_model.load_state_dict(torch.load('./models/temp.pth'))
print(summary(fcn_model, (3, 224, 224)))


def train(epoch):
    fcn_model.train()  # tran mode
    total_loss = 0.
    st = time.time()
    for batch_idx, (imgs, labels, Image_Path) in enumerate(train_loader):
        # train_batch += 1
        if use_cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()
        # batch_idx += 1
        imgs_tensor = Variable(imgs)  # torch.Size([2, 3, 320, 320])
        target = Variable(labels)  # torch.Size([2, 320, 320])
        out = fcn_model(imgs_tensor)  # torch.Size([2, 21, 320, 320])

        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # update all arguments
        total_loss += loss.item()  # return float

        if (batch_idx) % 20 == 0:
            ed = time.time()
            print('train epoch [%d/%d], iter[%d/%d], lr %.7f, aver_loss %.5f, time_use = %.1f'
                  % (epoch, epochs, batch_idx, len(train_loader), learning_rate, total_loss / (batch_idx + 1), ed - st))
            st = ed
            # # visiualize scalar
            # label_img = tools.labelToimg(labels[0])
            # net_out = out[0].data.max(1)[1].squeeze_(0)
            # out_img = tools.labelToimg(net_out)
            # writer.add_scalar("loss", loss, train_batch)
            # writer.add_scalar("total_loss", total_loss, train_batch)
            # writer.add_scalars('loss/scalar_group', {"loss": train_batch * loss,
            #                                             "total_loss": train_batch * total_loss})
            # writer.add_image('Image', imgs[0], epoch)
            # writer.add_image('label', label_img, epoch)
            # writer.add_image("out", out_img, epoch)

        assert total_loss is not np.nan
        assert total_loss is not np.inf

    torch.save(fcn_model.state_dict(), './models/temp.pth')  # save for 5 epochs
    total_loss /= len(train_loader)
    print('train epoch [%d/%d] average_loss %.5f' % (epoch, epochs, total_loss))

    return total_loss

    # logging.info('train epoch [%d/%d] average_loss %.5f' % (epoch, epochs, total_loss))


def test(epoch,total_loss):
    with torch.no_grad():
        print('load model .....')
        # fcn_model.load_state_dict(torch.load('./models/temp.pth'))
        
        # if use_cuda:
        #     test_model.cuda()
        # test_model.eval()
        # fcn_model.eval()
        matrixs = np.zeros((n_class, n_class))
        for idx, (img, label, _) in enumerate(val_loader):
            print(idx)
            # img, label, _ = val_data[idx]
            # img = img.unsqueeze(0)
            if use_cuda:
                img = img.cuda()

            img = Variable(img)
            # out = test_model(img)  # 1, 21, 320, 320
            out = fcn_model(img)

            label_matrix_T = label.numpy()
            # pre_matrix_T = out.data.max(1)[1].cpu().numpy().squeeze(0)
            pre_matrix_T = out.data.max(1)[1].cpu().numpy()
            label_matrix = label_matrix_T.flatten()
            pre_matrix = pre_matrix_T.flatten()

            matrix = metrics.confusion_matrix(label_matrix, pre_matrix,labels=[0,1,2,3])

            if sum(sum(matrixs)) == 0:
                matrixs[:,:] = matrix[:,:]    
                # for i in range(len(matrixs)):
                #     for j in range(len(matrixs[0])):
                #         matrixs[i][j] = matrix[i][j]
            else:
                matrixs += matrix
                # matrixs /= 2
                # # # 迭代输出行
                # for i in range(len(matrix)):
                #     # 迭代输出列
                #     for j in range(len(matrix[0])):
                #         # print(range(len(matrix[0])))
                #         matrixs[i][j] = (matrixs[i][j] + matrix[i][j]) / 2

    numberTotal = sum(sum(matrixs))
    muberTrue = 0
    PercisionList = []
    RecallList = []

    for i in range(len(matrixs)):
        muberTrue += matrixs[i,i]
        # for j in range(len(matrixs[0])):
        #     if i == j:
        #         muberTrue = muberTrue + matrixs[i, j]

    for i in range(len(matrixs)):
        PercisionList.append(matrixs[i, i] / sum(matrixs[:, i]))
        RecallList.append(matrixs[i, i] / sum(matrixs[i, :]))


    # for i in range(len(matrixs)):
    #     RecallList.append(matrixs[i, i] / sum(matrixs[i, :]))

    Acurracy = muberTrue / numberTotal

    Percision = sum(PercisionList) / len(PercisionList)

    Recall = sum(RecallList) / len(RecallList)

    F1 = (2 * Percision * Recall) / (Percision + Recall)

    print(Acurracy)
    print(Percision)
    print(Recall)
    print(F1)

    logging.info('%d %.5f %.2f %.2f %.2f %.2f' % (epoch, total_loss, Acurracy, Percision, Recall, F1))

    global best_test_Accuracy

    if best_test_Accuracy < F1:
        best_test_Accuracy = F1
        print('save best epoch....')
        torch.save(fcn_model.state_dict(), model_save)  # save for 5 epochs


if __name__ == '__main__':
    # print(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = True
    # writer = SummaryWriter()
    # train_batch = 0
    for epoch in range(epochs):
        loss = train(epoch)
        test(epoch,loss)
        # adjust learning rate
        if epoch in epochs_step:
            learning_rate *= 0.1
            optimizer.param_groups[0]['lr'] = learning_rate