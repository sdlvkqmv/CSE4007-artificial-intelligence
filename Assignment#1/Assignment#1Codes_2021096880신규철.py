
### Libraries
!pip install wget
import wget

from google.colab import drive
drive.mount('/content/drive')
filesPath = "drive/MyDrive/Colab/인공지능/과제1/model/"

from __future__ import print_function, division
import os
import torch
import numpy as np
import random

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision.datasets import VOCDetection

import tarfile

import skimage.io
import skimage.transform
import skimage.color
import skimage

from torchvision import transforms

from PIL import Image
import xml.etree.ElementTree as ET

import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw, ImageFont

import torch.nn as nn
import torch

import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms

## Load dataset
class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale
        ###

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    def __call__(self, sample, flip_x=0.5):
        #################################
        # TODO: complete this module. #
        #################################
        # Augmenter is a module that augments the training set.
        # Implement and use an Augmentor that includes at least horizontal flipping with probability 0.5
        # as part of the augmentation process.

        # [START]
        image, annots = sample['img'], sample['annot']
        rows, cols, channel = image.shape
        if np.random.rand() < flip_x:
            image = image[:, ::-1, :] #flip horizontal
            for obj in annots:
                new_xmin = cols - obj[2]
                new_xmax = cols - obj[0]
                obj[0] = new_xmin
                obj[2] = new_xmax
        annots = np.array(annots).astype(np.float64)
        return {'img': image, 'annot': annots}
        # [END]


class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        #################################
        # TODO: complete this module. #
        #################################
        # Implement a Normalizer that normalizes an image using the RGB mean and standard deviation.

        # [START]
        image, annots = sample['img'], sample['annot']
        image = (image -self.mean) / self.std
        annots = np.array(annots).astype(np.float32)
        return {'img': image, 'annot': annots}
        # [END]


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

## VOC Dataset
def VOC_download(root_dir, year="2007"):
    if os.path.exists(os.path.join(root_dir, 'VOCdevkit')):
        print("VOC exists.")
        return

    urls = {
        "2007": {
            "trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            "test": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
        },
    }

    if year not in urls:
        raise ValueError("Only VOC 2007 and 2012 are supported")

    trainval_download_path = os.path.join(root_dir, f"VOC{year}.tar")

    print(f"Downloading VOC {year} train/val data...")
    wget.download(urls[year]["trainval"], trainval_download_path)

    with tarfile.open(trainval_download_path, "r") as tar:
        tar.extractall(root_dir)

    os.remove(trainval_download_path)

    if year == "2007":
        test_download_path = os.path.join(root_dir, f"VOC{year}_test.tar")
        print(f"\nDownloading VOC {year} test data...")
        wget.download(urls[year]["test"], test_download_path)

        with tarfile.open(test_download_path, "r") as tar:
            tar.extractall(root_dir)

        os.remove(test_download_path)


class VOCDataset(VOCDetection):
    def __init__(self, root_dir, year="2007", image_set="trainval", transform=False, download=True):
        self.root_dir = root_dir
        self.year = year
        self.image_set = image_set

        if download:
            VOC_download(root_dir=self.root_dir, year=self.year)

        if transform:
            if image_set == 'trainval':
                self.transform = transforms.Compose([Normalizer(), Augmenter(), Resizer()])
            elif image_set == 'test':
                self.transform = transforms.Compose([Normalizer(), Resizer()])

        imageset_file = os.path.join(self.root_dir, "VOCdevkit", "VOC"+self.year, "ImageSets", "Main", image_set + ".txt")

        ############################################################################
        # TODO: assign appropriate values to 'self.image_ids' and 'self.classes'   #
        ############################################################################
        # self.image_ids: A list containing image names as elements.
        # self.classes: Class names defined in the order they exist in the VOC dataset.
        # [START]
        with open(imageset_file) as f:
          self.image_ids = f.read().splitlines()

        classes = os.listdir(os.path.join(self.root_dir, "VOCdevkit", "VOC" + self.year, "ImageSets", "Main"))


        self.classes = [i[:-9] for i in classes if "_test.txt" in i]

        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        # [END]


    def num_classes(self):
        return len(self.classes)


    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image_path = os.path.join(self.root_dir, "VOCdevkit", "VOC"+self.year, "JPEGImages", image_id + ".jpg")
        img = Image.open(image_path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0

        annotation_path = os.path.join(self.root_dir, "VOCdevkit", "VOC"+self.year, "Annotations", image_id + ".xml")
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        annots = []
        #################################################
        # TODO: assign appropriate values to 'annots'   #
        #################################################
        # [START]
        #print(root.findall('object')
        for i in root.findall('object'):
        #print(i)
          classes = i.find('name').text
          class_idx = self.class_to_ind[classes]
          box = i.find('bndbox')
          xmin = int(box.find('xmin').text)
          ymin = int(box.find('ymin').text)
          xmax = int(box.find('xmax').text)
          ymax = int(box.find('ymax').text)

          annots.append([xmin, ymin, xmax, ymax, class_idx])
        # [END]

        sample = {'img': img, 'annot': annots}
        if self.transform:
            sample = self.transform(sample)
        return sample


    def image_aspect_ratio(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, "VOCdevkit", "VOC"+self.year, "JPEGImages", image_id + ".jpg")
        img = Image.open(image_path)
        return float(img.width) / float(img.height)


class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

path2data = '/content/VOC'
if not os.path.exists(path2data):
    os.mkdir(path2data)

dataset_train = VOCDataset(root_dir=path2data, year="2007", image_set="trainval", transform=True)
dataset_val = VOCDataset(root_dir=path2data, year="2007", image_set="test", transform=True)

"""## Visualization"""

##########################
# TODO: get 'classes'    #
##########################
# [START]

#print(dataset_train.classes)
classes = dataset_train.classes

# [END]

colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

def show_sample(sample, classes=classes, colors=colors, GT=True, model=None):
    # permute (H, W, C) => (C, H, W)
    img = sample['img']
    img = img.permute(2, 0, 1) # (C, H, W)

    unorm = UnNormalizer()
    img = unorm(img.clone())
    img = torch.clamp(img, 0, 1)

    pil_img = to_pil_image(img)
    draw = ImageDraw.Draw(pil_img)

    if GT:
        annot = sample['annot']
        if torch.is_tensor(annot):
            annot = annot.cpu().numpy()

        for box in annot:
            if box[0] < 0:
                continue
            x1, y1, x2, y2, cls_idx = box
            cls_idx = int(cls_idx)
            color = tuple(int(c) for c in colors[cls_idx])
            label = classes[cls_idx]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
            draw.text((x1, y1), label, fill=color)
            # font = ImageFont.truetype("arial.ttf", 20)
            # draw.text((x1, y1), label, fill=color, font=font)
    else:
        with torch.no_grad():
            model.eval()
            finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates = model(img.unsqueeze(0).cuda().float())
            finalScores = finalScores.cpu()
            finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cpu()
            finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cpu()

            for score, cls_idx, box in zip(finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates):
                x1, y1, x2, y2 = box.int().tolist()
                cls_idx = int(cls_idx)
                color = tuple(int(c) for c in colors[cls_idx])
                label = classes[cls_idx]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
                draw.text((x1, y1), f'{label}-{score}', fill=color)

    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(pil_img))
    plt.axis('off')
    plt.show()

sample = dataset_val[123]
show_sample(sample)

"""### Dataloaders & collater function"""

def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    #################################################
    # TODO: complete return                         #
    #################################################
    # [START]

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}

    # [END]

# Adjust the batch size to suit your GPU environment.
# Select the largest possible batch size that your GPU memory allows. The choice is yours.
BATCH_SIZE = 8

#########################################################################
# TODO: Initialize dataloader_train and dataloader_val with collater()  #
#########################################################################
# [START]

sampler = AspectRatioBasedSampler(dataset_train, batch_size=BATCH_SIZE, drop_last=False)
dataloader_train = DataLoader(dataset_train, batch_sampler = sampler, collate_fn=collater)

sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
dataloader_val =DataLoader(dataset_val, batch_sampler =  sampler_val, collate_fn=collater)

# [END]

"""## Implementation of RetinaNet"""

#########################################
# TODO: complete this PyramidFeatures   #
# upsample mode: nearest                #
#########################################
# [START]

class PyramidFeatures(nn.Module):
    def __init__(self, c3, c4, c5, feature_size = 256):
        super(PyramidFeatures, self).__init__()

        # c5 - (1x1 cpnv, upsample, 3x3conv) > p5
        self.p5_conv1 = nn.Conv2d(c5, feature_size, kernel_size=1, stride=1, padding=0)
        self.p5_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # c4 - (1x1 cpnv, upsample, 3x3conv) > p4
        self.p4_conv1 = nn.Conv2d(c4, feature_size, kernel_size=1, stride=1, padding=0)
        self.p4_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # c3 -(1x1 conv, 3x3 conv)> p3
        self.p3_conv1 = nn.Conv2d(c3, feature_size, kernel_size=1, stride=1, padding=0)
        self.p3_conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # c5 -(3x3 conv, s2)> p6
        self.p6 = nn.Conv2d(c5, feature_size, kernel_size=3, stride=2, padding=1)

        # p6-(3x3conv s2, ReLU)>p7
        self.p7_relu = nn.ReLU()
        self.p7_conv = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        #input from ResNet(list)
        c3 = inputs[0]
        c4 = inputs[1]
        c5 = inputs[2]

        p5_out = self.p5_conv1(c5)
        p5_up_out = self.p5_up(p5_out)
        p5_out = self.p5_conv3(p5_out)

        p4_out = self.p4_conv1(c4)
        p4_out = p5_up_out + p4_out #element-wise addition
        p4_up_out = self.p4_up(p4_out)
        p4_out = self.p4_conv3(p4_out)

        p3_out = self.p3_conv1(c3)
        p3_out = p3_out + p4_up_out
        p3_out = self.p3_conv3(p3_out)

        p6_out = self.p6(c5)

        p7_out = self.p7_relu(p6_out)
        p7_out = self.p7_conv(p7_out)

        return [p3_out, p4_out, p5_out, p6_out, p7_out]

# [END]

#########################################
# TODO: complete this RegressionModel   #
#########################################
# [START]

class RegressionModel(nn.Module):
    #input : single feature map(p3/p4/p5...)
    def __init__(self, in_feature, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(in_feature, feature_size, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        self.outLayer = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        out = self.outLayer(x)

        # output batch, channel, h, w
        batch, channel, h, w = out.shape


        out = out.reshape(batch, self.num_anchors, 4, h, w)

        out = out.permute(0,1,3,4,2)
        out = out.reshape(batch, -1, 4)

        return out

# [END]

#############################################
# TODO: complete this ClassificationModel   #
#############################################
# [START]

class ClassificationModel(nn.Module):
    def __init__(self, in_features, num_anchors=9, num_classes=80, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(in_features, feature_size, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.output(x)
        out = self.sigmoid(x)

        batch, channel, h, w = out.shape
        out = out.reshape(batch, self.num_anchors, self.num_classes,  h, w)
        out = out.permute(0, 3, 4, 1, 2)

        out = out.reshape(batch, -1, self.num_classes)

        return out


# [END]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#############################################
# TODO: complete BBoxTransform module       #
#############################################
# [START]

class BBoxTransform(nn.Module):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        else:
            self.mean = mean

        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            self.std = std

    def forward(self, box, transforms):

        #calculate box w, h, anchor center(x & y)
        w_anchor= box[:, :, 2] - box[:, :, 0]
        h_anchor = box[:, :, 3] - box[:, :, 1]
        x_anchor= box[:, :, 0] + 0.5 * w_anchor
        y_anchor= box[:, :, 1] + 0.5* h_anchor

        #unnormalize model prediction
        dx = transforms[:, :, 0] * self.std[0] + self.mean[0]
        dy = transforms[:, :, 1] * self.std[1] + self.mean[1]
        dw = transforms[:, :, 2] * self.std[2] + self.mean[2]
        dh = transforms[:, :, 3] * self.std[3] + self.mean[3]

        #convert into real values
        x_center_pred = x_anchor + dx * w_anchor
        y_center_pred = y_anchor + dy * h_anchor
        w_pred = torch.exp(dw) * w_anchor
        h_pred = torch.exp(dh) * h_anchor

        #final predicted box coordinates
        xm_pred = x_center_pred - 0.5 * w_pred
        ym_pred = y_center_pred - 0.5 * h_pred
        xM_pred = x_center_pred + 0.5 * w_pred
        yM_pred = y_center_pred + 0.5 * h_pred

        pred_boxes = torch.stack([xm_pred, ym_pred, xM_pred, yM_pred], dim=2)

        return pred_boxes

# [END]

class ClipBoxes(nn.Module):
    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


####################################################################################
# TODO: complete Anchor class, generate_anchors functions and shift function       #
####################################################################################
# generate_anchors():
#   base anchor boxes are generated using the given base size (e.g., 32, 64, …),
#   along with the specified ratios and scales.
# shift():
#   the generated base anchors are then shifted to align with
#   the center of each cell of the feature map.

# [START]
class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** i for i in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 **(i + 2) for i in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        #compute anchors on all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors         = generate_anchors(self.sizes[idx], self.ratios, self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        return torch.from_numpy(all_anchors.astype(np.float32)).cuda()


def generate_anchors(base_size, ratios, scales):
    num_anchors = len(ratios) * len(scales)
    anchors = np.zeros((num_anchors, 4), dtype=np.float32)

    index = 0
    for ratio in ratios:
        for scale in scales:
            w = base_size * scale * (1.0 / ratio) ** 0.5
            h = base_size * scale * ratio ** 0.5

            xm = -w / 2.0
            ym = -h / 2.0
            xM = w / 2.0
            yM = h / 2.0

            anchors[index, 0] = xm
            anchors[index, 1] = ym
            anchors[index, 2] = xM
            anchors[index, 3] = yM

            index += 1

    return anchors

def shift(shape, stride, anchors):
    shift_x = (torch.arange(0, shape[1], dtype=anchors.dtype, device=anchors.device) + 0.5) * stride
    shift_y = (torch.arange(0, shape[0], dtype=anchors.dtype, device=anchors.device) + 0.5) * stride

    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')

    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)

    shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

    A = anchors.shape[0]
    K = shifts.shape[0]

    anchors = anchors.reshape(1, A, 4)
    shifts = shifts.reshape(K, 1, 4)

    all_anchors = anchors + shifts  #broadcast: [K, A, 4]
    all_anchors = all_anchors.reshape(-1, 4)

    return all_anchors
# [END]


class ResNet(nn.Module):
    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        ####################################################################################
        # TODO: fill the components.                                                       #
        ####################################################################################
        # [START]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        # [END]

        self.clipBoxes = ClipBoxes()

        self.lossModule = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.lossModule(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            finalScores = finalScores.cuda()
            finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
            finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model

retinanet = resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
retinanet = retinanet.cuda()

optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

"""## Train: BCE Loss"""

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class BinaryCrossEntropyLoss(nn.Module):
    def forward(self, classifications, regressions, anchors, annotations):
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                bce = -(torch.log(1.0 - classification))
                cls_loss = bce
                classification_losses.append(cls_loss.sum())
                regression_losses.append(torch.tensor(0).float().cuda())
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)
            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = bce
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

retinanet.lossModule = BinaryCrossEntropyLoss()
retinanet.train()
retinanet.freeze_bn()
save_epoch = [4,9,14,19]

print('Num training images: {}'.format(len(dataset_train)))
save_epoch = [4, 9, 14, 19]


for epoch_num in range(20):
    retinanet.train()
    retinanet.freeze_bn()
    epoch_loss = []

    for iter_num, data in enumerate(dataloader_train):
        #############################################
        # TODO: complete following training steps   #
        #############################################
        # [START]
        image = data['img'].cuda()
        annots = data['annot'].cuda()

        optimizer.zero_grad()
        classification_loss, regression_loss = retinanet([image, annots])

        loss = classification_loss + regression_loss

        loss.backward()
        optimizer.step()

        epoch_loss.append(float(loss))

        print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss)))

        del classification_loss
        del regression_loss
        # [END]
    if epoch_num in save_epoch:
        torch.save(retinanet, filesPath + 'model_final_BCE_epoch' + str(epoch_num+1) + '.pt')

    scheduler.step(np.mean(epoch_loss))

#retinanet.eval()
#torch.save(retinanet, 'model_final_BCE.pt')

retinanet.eval()
torch.save(retinanet, filesPath + 'model_final_BCE3.pt')

"""## Load trained RetinaNet and perform inference"""

retinanet = torch.load('model_final_BCE.pt', weights_only=False, map_location='cpu')
retinanet = retinanet.cuda()
retinanet.eval()

"""##### Ground truth"""

sample = dataset_val[123]
print(type(sample['img']))
show_sample(sample=sample, GT=True)

sample = dataset_val[50]
print(type(sample['img']))
show_sample(sample=sample, GT=True)

"""##### Prediction"""

###### BCE 5 epochs
retinanet = torch.load(filesPath+'model_final_BCE_epoch5.pt', weights_only=False, map_location='cpu')
retinanet = retinanet.cuda()
retinanet.eval()

sample = dataset_val[123]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)

sample = dataset_val[50]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)

"""###### BCE 10 epochs"""

retinanet = torch.load(filesPath+'model_final_BCE_epoch10.pt', weights_only=False, map_location='cpu')
retinanet = retinanet.cuda()
retinanet.eval()

sample = dataset_val[123]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)

sample = dataset_val[50]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)

###### BCE 15 epochs

retinanet = torch.load(filesPath+'model_final_BCE_epoch15.pt', weights_only=False, map_location='cpu')
retinanet = retinanet.cuda()
retinanet.eval()

sample = dataset_val[123]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)

sample = dataset_val[50]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)

###### BCE 20 epochs

retinanet = torch.load(filesPath+'model_final_BCE_epoch20.pt', weights_only=False, map_location='cpu')
retinanet = retinanet.cuda()
retinanet.eval()

sample = dataset_val[123]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)

sample = dataset_val[50]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)

"""## Focal loss"""
class FocalLoss(nn.Module):
    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                alpha_factor = torch.ones(classification.shape).cuda() * alpha

                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                bce = -(torch.log(1.0 - classification))

                cls_loss = focal_weight * bce
                classification_losses.append(cls_loss.sum())
                regression_losses.append(torch.tensor(0).float().cuda())
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)
            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape).cuda() * alpha
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)

            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

try:
    del retinanet
except:
    pass

torch.cuda.empty_cache()

retinanet_FL = resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
retinanet_FL = retinanet_FL.cuda()

optimizer = optim.Adam(retinanet_FL.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

retinanet_FL.lossModule = FocalLoss()
retinanet_FL.train()
retinanet_FL.freeze_bn()

print('Num training images: {}'.format(len(dataset_train)))
save_epoch = [4, 9, 14, 19]

for epoch_num in range(20):
    retinanet_FL.train()
    retinanet_FL.freeze_bn()
    epoch_loss = []

    for iter_num, data in enumerate(dataloader_train):
        #############################################
        # TODO: complete following training steps   #
        #############################################
        # [START]
        image = data['img'].cuda()
        annots = data['annot'].cuda()

        optimizer.zero_grad()
        classification_loss, regression_loss = retinanet_FL([image, annots])

        loss = classification_loss + regression_loss

        loss.backward()
        optimizer.step()

        epoch_loss.append(float(loss))

        print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss)))

        del classification_loss
        del regression_loss

        # [END]
    if epoch_num in save_epoch:
        torch.save(retinanet_FL, filesPath + 'model_final_FL_epoch' + str(epoch_num+1) + '.pt')


    scheduler.step(np.mean(epoch_loss))

retinanet_FL.eval()
torch.save(retinanet_FL, filesPath + 'model_final_FL3.pt')

"""## Load trained RetinaNet and perform inference"""

retinanet_FL = torch.load('model_final_FL.pt', weights_only=False, map_location='cpu')
retinanet_FL = retinanet_FL.cuda()
retinanet_FL.eval()

##### Ground truth

sample = dataset_val[123]
print(type(sample['img']))
show_sample(sample=sample, GT=True)

sample = dataset_val[50]
print(type(sample['img']))
show_sample(sample=sample, GT=True)

"""##### Prediction"""

###### FL 5 epochs
retinanet = torch.load(filesPath+'model_final_FL_epoch5.pt', weights_only=False, map_location='cpu')
retinanet = retinanet.cuda()
retinanet.eval()

sample = dataset_val[123]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)

sample = dataset_val[50]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)

###### FL 10 epochs
retinanet = torch.load(filesPath+'model_final_FL_epoch10.pt', weights_only=False, map_location='cpu')
retinanet = retinanet.cuda()
retinanet.eval()

sample = dataset_val[123]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)

sample = dataset_val[50]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)

###### FL 15 epochs

retinanet = torch.load(filesPath+'model_final_FL_epoch15.pt', weights_only=False, map_location='cpu')
retinanet = retinanet.cuda()
retinanet.eval()

sample = dataset_val[123]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)

sample = dataset_val[50]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)



###### FL 20 epochs
retinanet = torch.load(filesPath+'model_final_FL_epoch20.pt', weights_only=False, map_location='cpu')
retinanet = retinanet.cuda()
retinanet.eval()

sample = dataset_val[123]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)

sample = dataset_val[50]
print(type(sample['img']))
show_sample(sample=sample, GT=False, model=retinanet)