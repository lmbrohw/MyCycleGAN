import argparse
import numpy
from PIL import Image
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize


# 调整 array 格式图像的大小
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # 最邻近插值法调整大小
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return asarray(images_list)


# 计算FID
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input1', type=str, default='./datasets/test/A/78.png', help='path of the original image')
    parser.add_argument('--input2', type=str, default='./output/B/78.png', help='path of the cartoon image')
    opt = parser.parse_args()
    print('the opt is', opt)

    # 加载 inception v3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(256, 256, 3))
    # 读取图像并进行预处理
    images1 = Image.open(opt.input1)
    images2 = Image.open(opt.input2)
    # convert to numpy and to floating point values
    images1 = numpy.array(images1).astype('float32')
    images2 = numpy.array(images2).astype('float32')
    images1 = scale_images(images1, (256, 256, 3))
    images2 = scale_images(images2, (256, 256, 3))
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    # 计算FID
    fid = calculate_fid(model, images1, images2)
    print('FID: %.3f' % fid)
