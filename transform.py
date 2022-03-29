import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from dataset import ImageDataset
from models import *


def transform(file_path):
    netG_A2B = Generator(3, 3)
    # netG_B2A = Generator(3, 3)
    # Load state dicts
    netG_A2B.load_state_dict(torch.load('saved_model/netG_A2B.pth', torch.device('cpu')), strict=False)
    # netG_B2A.load_state_dict(torch.load(opt.generator_B2A, torch.device('cpu')), strict=False)

    # Set model's test mode
    netG_A2B.eval()
    # netG_B2A.eval()

    # Inputs & targets memory allocation
    Tensor = torch.Tensor
    input_A = Tensor(1, 3, 256, 256)
    # input_B = Tensor(1, 3, 256, 256)

    # Dataset loader
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(ImageDataset('datasets/', transforms_=transforms_, mode='test'),
                            batch_size=1, shuffle=False, num_workers=0)

    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        # real_B = Variable(input_B.copy_(batch['B']))

        # Generate output
        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        # fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        # Save image files
        # save_image(fake_A, 'output/A/%04d.png' % (i + 1))
        save_image(fake_B, 'output/B/%04d.png' % (i + 1))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))
