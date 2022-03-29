import argparse
import itertools
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

from models import *
from preprocess import *
from dataset import ImageDataset
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--dataroot', type=str, default='datasets/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial the learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='')
parser.add_argument('--size', type=int, default=256, help='size of the photo')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--cuda', action='store_true', help='use GPU')

opt = parser.parse_args()
print('the opt is', opt)

# Generator, Discriminator and Attention
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.input_nc, opt.output_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)
Attn_A = Attention(opt.input_nc)
Attn_B = Attention(opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()
    Attn_A.cuda()
    Attn_B.cuda()

# apply对模型的参数进行初始化（apply将一个函数递归应用到模块自身以及该模块的每一个子模块）
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)
Attn_A.apply(weights_init_normal)
Attn_B.apply(weights_init_normal)

# Loss
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
# Add VGG loss
vgg_loss = PerceptualLoss(opt)
vgg_loss.cuda()

vgg = models.vgg16(pretrained=True)
vgg = load_vgg16("./pretrained_model", gpu_ids)
vgg.eval()
for param in vgg.parameters():
    param.requires_grad = False

# Optimizers and LR schedulers
optimizer_G_A2B = torch.optim.Adam(netG_A2B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_G_B2A = torch.optim.Adam(netG_B2A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_Attn = torch.optim.Adam(itertools.chain(Attn_A.parameters(), Attn_B.parameters()), lr=1e-5)
# 这里会使用到utils中定义的学习率衰减更新函数LambdaLR()
lr_scheduler_G_A2B = torch.optim.lr_scheduler.LambdaLR(optimizer_G_A2B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                           opt.decay_epoch).step)
lr_scheduler_G_B2A = torch.optim.lr_scheduler.LambdaLR(optimizer_G_B2A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                           opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_Attn = torch.optim.lr_scheduler.LambdaLR(optimizer_Attn,
                                                      lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
# lr_scheduler_Attn = torch.optim.lr_scheduler.MultiStepLR(optimizer_Attn, milestones=[30], gamma=0.1, last_epoch=startEpoch -1)

# Dataset loader
transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),  # 调整输入图片的大小
               transforms.RandomCrop(opt.size),  # 随机裁剪
               transforms.RandomHorizontalFlip(),  # 随机水平翻转
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# 若出错则替换成CycleGAN_practice
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batch_size, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_gred=False)  # 用于loss_GAN, 即(target_real, D(G(A)))
target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)  # 用于loss_GAN, 即(target_real, D(G(A)))

# 缓冲池
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))

# Training

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Generators A2B and B2A

        optimizer_G_A2B.zero_grad()
        optimizer_G_B2A.zero_grad()
        optimizer_Attn.zero_grad()

        # Identity loss
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)  # G_B2A(G_A2B(real_A))
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0  # 10.0就是lambda超参数
        recovered_B = netG_A2B(fake_A)  # G_A2B(G_B2A(real_B))
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # Vgg loss
        loss_vgg_A2B = vgg_loss.compute_vgg_loss(vgg, fake_B, real_A)
        loss_vgg_B2A = vgg_loss.compute_vgg_loss(vgg, fake_A, real_B)

        # FaceID loss
        loss_feature_A2B = func(real_A, fake_B)
        loss_feature_B2A = func(real_B, fake_A)

        # Total loss 这里csdn上加了权重
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + \
                 loss_feature_A2B + loss_feature_B2A + loss_vgg_A2B + loss_vgg_B2A

        # 参数更新
        loss_G.backward()
        optimizer_G_A2B.step()
        optimizer_G_B2A.step()
        optimizer_Attn.step()

        # Discriminator A

        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_real + loss_fake) * 0.5

        # 参数更新
        loss_D_A.backward()
        optimizer_D_A.step()

        # Discriminator B

        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_real + loss_fake) * 0.5

        # 参数更新
        loss_D_B.backward()
        optimizer_D_B.step()

    # Update learning rate
    lr_scheduler_G_A2B.step()
    lr_scheduler_G_B2A.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    lr_scheduler_Attn.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output/netD_B.pth')
    torch.save(Attn_A.state_dict(), 'output/Attn_A.pth')
    torch.save(Attn_B.state_dict(), 'output/Attn_B.pth')
