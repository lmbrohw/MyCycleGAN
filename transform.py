import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from dataset import ImageDataset
from models import *

from utils import tensor2image
from preprocess import *


def preprocess_cutout():
    img_path = './datasets/test_for_ui/A/original.png'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(cv2.imread(img_path))
    # 打印原图出来看看
    plt.title('img')
    plt.imshow(img)
    plt.show()

    # 检测人脸+裁剪多余背景
    detector = FaceDetect('cpu', 'dlib')
    image_align, landmarks_align = detector.align(img)
    face = detector.crop(image_align, landmarks_align)

    # cv2.imwrite('./datasets/face/' + image_name, face[:, :, ::-1])

    # 抠出人脸
    # 得到mask
    # mask_dir = os.path.join(os.getcwd(), 'datasets', 'result_seg', image_name)
    segment = FaceSeg()
    mask = segment.get_mask(face)

    # cv2.imwrite(mask_dir, mask)

    # mask与face计算
    splice = np.dstack((face, mask))
    splice_face = splice[:, :, :3].copy()
    splice_mask = splice[:, :, 3][:, :, np.newaxis].copy() / 255.
    face_in_white_bg = (splice_face * splice_mask + (1 - splice_mask) * 255).astype(np.uint8)  # 背景变白色
    face_in_white_bg = cv2.resize(face_in_white_bg, (256, 256))  # convert to 512x512
    # plt.title('face_white_bg')
    # plt.imshow(face_in_white_bg)
    # plt.show()

    cv2.imwrite('./datasets/test_for_ui/A/original.png', cv2.cvtColor(face_in_white_bg, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./output/test_for_ui/img_cutout/cutout.png', cv2.cvtColor(face_in_white_bg, cv2.COLOR_RGB2BGR))


def preprocess_resize():
    img_path = './datasets/test_for_ui/A/original.png'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))  # convert to 256x256

    cv2.imwrite('./datasets/test_for_ui/A/original.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./output/test_for_ui/img_cutout/resize.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def transform():
    netG_A2B = Generator(3, 3)
    # netG_B2A = Generator(3, 3)
    # Load state dicts
    netG_A2B.load_state_dict(torch.load('saved_model/netG_A2B_95.pth', torch.device('cpu')), strict=False)
    # netG_B2A.load_state_dict(torch.load('saved_model/netG_B2A_95.pth', torch.device('cpu')), strict=False)

    # Set model's test mode
    netG_A2B.eval()
    # netG_B2A.eval()

    # Inputs & targets memory allocation
    Tensor = torch.Tensor
    input_A = Tensor(1, 3, 256, 256)
    # input_B = Tensor(1, 3, 256, 256)

    # Dataset loader
    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(ImageDataset('datasets/', transforms_=transforms_, mode='test_for_ui'),
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
        save_image(fake_B, 'output/test_for_ui/B/%04d.png' % (i + 1))
        return fake_B

        # sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))


if __name__ == '__main__':
    img = transform('2207.png')
    img = tensor2image(img)
    print(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.title('img')
    img = img.swapaxes(0, 1)
    img = img.swapaxes(1, 2)
    plt.imshow(img)
    plt.show()
