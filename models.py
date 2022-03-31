import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels)
        ]

        self.model = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.model(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=9):  # 默认需要9个残差块
        super().__init__()

        # 初始化卷积块
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # 编码
        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        ]
        model += [
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        ]

        # 加入残差块
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(256)]

        # 解码
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        ]
        model += [
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # 输出
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # 卷积块
        model = [
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # FCN 分类
        model += [
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # 均值池化（下采样）+ flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class Attention(nn.Module):
    def __init__(self, in_channels=3):
        super.__init__()

        model = [
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        ]

        model += [
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        model += [
            ResidualBlock(64)
        ]

        model += [
            nn.UpsamplingNearest2d(scale_factor=2)
        ]

        model += [
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        model += [
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        ]

        model += [
            nn.Conv2d(32, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# Add VGG loss
def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch


# Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        # if self.opt.no_vgg_instance:
        #     return torch.mean((img_fea - target_fea) ** 2)
        # else:
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


# 加载Vgg预训练模型
def load_vgg16(model_dir, gpu_ids=0):
    if not os.path.exists(model_dir):
        print('没有预训练模型')
        os.mkdir(model_dir)

    vgg = Vgg16()
    # vgg.cuda(device=gpu_ids[0])
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')), strict=False)
    # vgg = nn.DataParallel(vgg, gpu_ids)  # 多gpu跑
    return vgg


# Vgg16 Module 这里直接返回卷积次高层/block4
class Vgg16(nn.Module):
    def __init__(self):
        super().__init__()

        model = [nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                 nn.ReLU(inplace=True)]

        model += [nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(inplace=True)]

        model += [nn.MaxPool2d(kernel_size=2, padding=1),
                  nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(inplace=True)]

        model += [nn.MaxPool2d(kernel_size=2, padding=1),
                  nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(inplace=True)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# class Vgg16(nn.Module):
#     def __init__(self):
#         super(Vgg16, self).__init__()
#         self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#
#         self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#
#         self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#
#         self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#
#         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         h = F.relu(self.conv1_1(x))
#         h = F.relu(self.conv1_2(h))
#         relu1_2 = h
#         h = F.max_pool2d(h, kernel_size=2, stride=2)
#
#         h = F.relu(self.conv2_1(h))
#         h = F.relu(self.conv2_2(h))
#         relu2_2 = h
#         h = F.max_pool2d(h, kernel_size=2, stride=2)
#
#         h = F.relu(self.conv3_1(h))
#         h = F.relu(self.conv3_2(h))
#         h = F.relu(self.conv3_3(h))
#         relu3_3 = h
#         h = F.max_pool2d(h, kernel_size=2, stride=2)
#
#         h = F.relu(self.conv4_1(h))
#         h = F.relu(self.conv4_2(h))
#         h = F.relu(self.conv4_3(h))
#         relu4_3 = h
#         h = F.max_pool2d(h, ker_size=2, stride=1)
#
#         h = F.relu(self.conv5_1(h))
#         h = F.relu(self.conv5_2(h))
#         h = F.relu(self.conv5_3(h))
#         relu5_3 = h
#
#         return relu4_3
