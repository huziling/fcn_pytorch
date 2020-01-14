# coding=utf-8
import xml.sax
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import torch
import configparser

cf = configparser.ConfigParser()
cf.read("./config.ini")  # 读取配置文件，如果写文件的绝对路径，就可以不用os模

max_height = int(cf.get("develop", "max_height"))

# class_names_create = ['__background__', 'section','figure', 'table', 'text', 'caption', 'list']
# class_names_create = ['BG', 'table']
class_names_create = ['__background__', 'figure', 'table', 'text']

class MyContentHandler(xml.sax.ContentHandler):
    def __init__(self):
        super().__init__()
        self.data = []
        self.currentData = ""
        self.name = ""
        self.xmin = ""
        self.ymin = ""
        self.xmax = ""
        self.ymax = ""

    # def startDocument(self):
    #     print("开始解析xml")
    #
    # def endDocument(self):
    #     print("解析xml结束")

    def startElement(self, name, attrs):
        self.currentData = name

    def endElement(self, name):
        if name == 'name':
            self.data.append(self.name)
        elif name == 'xmin':
            self.data.append(self.xmin)
        elif name == 'ymin':
            self.data.append(self.ymin)
        elif name == 'xmax':
            self.data.append(self.xmax)
        elif name == 'ymax':
            self.data.append(self.ymax)

    def characters(self, content):
        if self.currentData == "name":
            self.name = content
        elif self.currentData == "xmin":
            self.xmin = content
        elif self.currentData == "ymin":
            self.ymin = content
        elif self.currentData == "xmax":
            self.xmax = content
        elif self.currentData == "ymax":
            self.ymax = content


def GetContext(xmlPath, PType=".jpg"):
    saxParse = xml.sax.make_parser()
    saxParse.setFeature(xml.sax.handler.feature_namespaces, 0)  # 关闭命名解析
    handler = MyContentHandler()
    saxParse.setContentHandler(handler)
    saxParse.parse(xmlPath)

    ImgPath = xmlPath.replace("Annotations", "JPEGImages").replace(".xml", PType)
    img = mpimg.imread(ImgPath)  # 读取和代码处于同一目录下的 lena.png
    # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理

    target = torch.ones(img.shape[0], img.shape[1]).fill_(0)
    data = handler.data

    # print(len(data))
    data_list = [data[i:i + 5] for i in range(0, len(data), 5)]

    # cv2.rectangle(img, (0, 0, img.shape[1], img.shape[0]), (0, 0, 0), -1)
    for data in data_list:

        minX = int(data[1])
        minY = int(data[2])
        maxX = int(data[3])
        maxY = int(data[4])

        if data[0] == 'subsection' or data[0] == 'subsubsection' or data[0] == 'section' or data[0] == 'list' or data[0] == 'caption':
            data[0] = 'text'

        class_index = class_names_create.index(data[0])

        target[minY:maxY, minX:maxX].fill_(class_index)

    return target
