import argparse
import itertools
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models import *
# from preprocess import *
from dataset import ImageDataset
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
parser.add_argument('--dataroot', type=str, default='datasets/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial the learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='')
parser.add_argument('--size', type=int, default=256, help='size of the photo')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--cuda', action='store_true', help='use GPU')
parser.add_argument('--save_every', type=int, default=5, help='checkpoint')
parser.add_argument('--resume', type=str, default='None', help='file to resume')
parser.add_argument('--start_epoch', type=int, default=0, help='')

opt = parser.parse_args()
print('the opt is', opt)

# Generator, Discriminator and Attention
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.input_nc, opt.output_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)
# Attn_A = Attention(opt.input_nc)
# Attn_B = Attention(opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()
    # Attn_A.cuda()
    # Attn_B.cuda()

# apply对模型的参数进行初始化（apply将一个函数递归应用到模块自身以及该模块的每一个子模块）
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)
# Attn_A.apply(weights_init_normal)
# Attn_B.apply(weights_init_normal)

# Loss
GAN_loss = torch.nn.MSELoss()
cycle_loss = torch.nn.L1Loss()
identity_loss = torch.nn.L1Loss()
vgg_loss = PerceptualLoss()  # Add VGG loss
vgg_loss.cuda()

# 从 pretrained_model 中加载Vgg16模型
# vgg = models.vgg16(pretrained=True)
gpu_ids = 0
vgg = load_vgg16("./pretrained_model", gpu_ids)
vgg.eval()
for param in vgg.parameters():
    param.requires_grad = False

# Optimizers and LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.lr,
                               betas=(0.5, 0.999))
# optimizer_G_A2B = torch.optim.Adam(netG_A2B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
# optimizer_G_B2A = torch.optim.Adam(netG_B2A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
# optimizer_Attn = torch.optim.Adam(itertools.chain(Attn_A.parameters(), Attn_B.parameters()), lr=1e-5)

# 这里会使用到utils中定义的学习率衰减更新函数LambdaLR()
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch,
                                                                                   opt.decay_epoch).step)

# lr_scheduler_G_A2B = torch.optim.lr_scheduler.LambdaLR(optimizer_G_A2B, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)
# lr_scheduler_G_B2A = torch.optim.lr_scheduler.LambdaLR(optimizer_G_B2A, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)

lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch,
                                                                        opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch,
                                                                        opt.decay_epoch).step)

# lr_scheduler_Attn = torch.optim.lr_scheduler.LambdaLR(optimizer_Attn, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
# lr_scheduler_Attn = torch.optim.lr_scheduler.MultiStepLR(optimizer_Attn, milestones=[30], gamma=0.1, last_epoch=startEpoch -1)

# 加载数据并处理
transforms_ = [
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

# input_A, input_B, target_real, target_fake
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batch_size, opt.output_nc, opt.size, opt.size)
target_real = Tensor(opt.batch_size).fill_(1.0)  # 用于loss_GAN, 即(target_real, D(G(A)))
target_fake = Tensor(opt.batch_size).fill_(0.0)  # 用于loss_GAN, 即(target_real, D(G(A)))

# 缓冲池
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# 训练过程可视化
logger = Logger(opt.n_epochs, len(dataloader))

# 从checkpoint中加载模型参数，恢复训练

start_epoch = 0
plotEvery = 1

if opt.resume is not 'None':
    checkpoint = torch.load(opt.resume)
    start_epoch = checkpoint['epoch']

    netG_A2B.load_state_dict(checkpoint['genA2B'])
    netG_B2A.load_state_dict(checkpoint['genB2A'])
    netD_A.load_state_dict(checkpoint['disA'])
    netD_B.load_state_dict(checkpoint['disB'])
    optimizer_G.load_state_dict(checkpoint['optG'])
    optimizer_D_A.load_state_dict(checkpoint['optD_A'])
    optimizer_D_B.load_state_dict(checkpoint['optD_B'])

    print('resumed from epoch ', start_epoch - 1)
    del (checkpoint)

# Training

for epoch in range(opt.start_epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # model input (real_A, real_B)
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Generators A2B and B2A

        optimizer_G.zero_grad()
        # optimizer_G_A2B.zero_grad()
        # optimizer_G_B2A.zero_grad()
        # optimizer_Attn.zero_grad()

        # Identity loss
        same_A = netG_B2A(real_A)
        loss_identity_A = identity_loss(same_A, real_A)
        same_B = netG_A2B(real_B)
        loss_identity_B = identity_loss(same_B, real_B)

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = GAN_loss(pred_fake, target_real)
        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = GAN_loss(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)  # G_B2A(G_A2B(real_A))
        loss_cycle_ABA = cycle_loss(recovered_A, real_A)
        recovered_B = netG_A2B(fake_A)  # G_A2B(G_B2A(real_B))
        loss_cycle_BAB = cycle_loss(recovered_B, real_B)

        # Vgg loss
        loss_vgg_A2B = vgg_loss.compute_vgg_loss(vgg, fake_B, real_A)
        loss_vgg_B2A = vgg_loss.compute_vgg_loss(vgg, fake_A, real_B)

        # FaceID loss
        # loss_feature_A2B = func(real_A, fake_B)
        # loss_feature_B2A = func(real_B, fake_A)

        # Total loss 在这里加上权重
        loss_G = 5.0 * loss_identity_A + 5.0 * loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + 10.0 * loss_cycle_ABA + \
                 10.0 * loss_cycle_BAB + 0.5 * loss_vgg_A2B + 0.5 * loss_vgg_B2A

        # 参数更新
        loss_G.backward()
        optimizer_G.step()
        # optimizer_G_A2B.step()
        # optimizer_G_B2A.step()
        # optimizer_Attn.step()

        # Discriminator A

        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_real = GAN_loss(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_fake = GAN_loss(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_real + loss_fake) * 0.5

        # 参数更新
        loss_D_A.backward()
        optimizer_D_A.step()

        # Discriminator B

        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_real = GAN_loss(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_fake = GAN_loss(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_real + loss_fake) * 0.5

        # 参数更新
        loss_D_B.backward()
        optimizer_D_B.step()

        sys.stdout.write('\repoch %03d/%03d (%04d/%04d)' % (epoch, opt.n_epochs, i, len(dataloader)))

        logger.log({'loss_G': loss_G,
                    'loss_G_identity': (loss_identity_A + loss_identity_B),
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
                    'loss_D': (loss_D_A + loss_D_B),
                    'loss_vgg': (loss_vgg_A2B + loss_vgg_B2A)},
                   images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B, 'rec_A': recovered_A,
                           'rec_B': recovered_B})

    # 更新学习率
    lr_scheduler_G.step()
    # lr_scheduler_G_A2B.step()
    # lr_scheduler_G_B2A.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    # lr_scheduler_Attn.step()

    # 保存模型参数 checkpoint
    if epoch % opt.save_every == 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'optG': optimizer_G.state_dict(),
            'optD_A': optimizer_D_A.state_dict(),
            'optD_B': optimizer_D_B.state_dict(),
            'genA2B': netG_A2B.state_dict(),
            'genB2A': netG_B2A.state_dict(),
            'disA': netD_A.state_dict(),
            'disB': netD_B.state_dict()
        },
            filename='checkpoint/checkpoint_' + str(epoch) + '.pth'
        )
    # 保存完整的生成器 判别器
    if epoch % 10 == 0:
        torch.save(netG_A2B.state_dict(), 'saved_model/netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'saved_model/netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'saved_model/netD_A.pth')
        torch.save(netD_B.state_dict(), 'saved_model/netD_B.pth')
