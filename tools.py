import os
import numpy as np
import PIL.Image as Image
import torch

def getPalette():
    pal = np.array([[0, 0, 0],   #BG
                    [255, 0, 0],      #figure
                    [0, 0, 255],      #table
                    [0, 255, 0]       #text
                    ], dtype='uint8').flatten()
    return pal

# colormap={'__background__': (0, 0, 0),'section': (255, 255, 255),'subsection': (255, 255, 255),'subsubsection': (255, 255, 255),'figure': (255, 255, 0),
#           'table': (255, 0, 255),'text': (0, 255, 0),'caption': (0, 0, 255),'list': (255, 0, 0)}

def colorize_mask(mask):
    """
    :param mask: 图片大小的数值，代表不同的颜色
    :return:
    """
    new_mask = Image.fromarray(mask.astype(np.uint8), 'P')  # 将二维数组转为图像
    # print(new_mask.show())

    pal = getPalette()
    # print(pal)
    new_mask.putpalette(pal)
    # print(new_mask.show())
    return new_mask

# m = np.array([[1,2], [3,4]])
# colorize_mask(m)


def getFileName(file_path):
    '''
    get file_path name from path+name+'test.jpg'
    return test
    '''
    full_name = file_path.split('/')[-1]
    name = os.path.splitext(full_name)[0]

    return name


def labelTopng(label, img_name):
    '''
    convert tensor cpu label to png and save
    '''
    label = label.numpy()             # 320 320
    label_pil = colorize_mask(label)
    label_pil.save(img_name)

def labelToimg(label):
    label = label.numpy()
    label_pil = colorize_mask(label)
    return label_pil


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def accuracy_score(label_trues, label_preds, n_class=7):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)   # n_class, n_class
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, mean_iu, fwavacc

def accuracy(label_trues, label_preds, n_class=7):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)   # n_class, n_class

    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    return acc_cls


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou