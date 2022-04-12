import argparse
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from dataset import ImageDataset
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--input_nc', type=int, default=3, help='input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--dataroot', type=str, default='datasets/', help='root directory of the dataset')
parser.add_argument('--size', type=int, default=256, help='size of the photo')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--cuda', action='store_true', help='use GPU')
parser.add_argument('--generator_A2B', type=str, default='saved_model/netG_A2B_95.pth',
                    help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='saved_model/netG_B2A_95.pth',
                    help='B2A generator checkpoint file')

opt = parser.parse_args()
print('the opt is', opt)

netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B, torch.device('cpu')), strict=False)
netG_B2A.load_state_dict(torch.load(opt.generator_B2A, torch.device('cpu')), strict=False)

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batch_size, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # Generate output
    fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
    fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

    # Save image files
    save_image(fake_A, 'output/A/95_%04d.png' % (i + 1))
    save_image(fake_B, 'output/B/95_%04d.png' % (i + 1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))
