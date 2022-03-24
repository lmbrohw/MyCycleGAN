import os

import face_alignment
import numpy as np
import cv2
import math
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.platform import gfile


class FaceDetect:
    def __init__(self, device, detector):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector=detector)

    #
    def align(self, image):
        landmarks = self.get_face_landmarks(image)

        if landmarks is None:
            return None
        else:
            return self.rotate_image(image, landmarks)

    # 找到检测关键点
    def get_face_landmarks(self, image):
        preds = self.fa.get_landmarks(image)

        if preds is None:
            return None
        elif len(preds) == 1:
            return preds[0]
        else:
            # 检测有多张人脸则返回最大的
            areas = []
            for pred in preds:
                landmarks_top = np.min(pred[:, 1])
                landmarks_bottom = np.max(pred[:, 1])
                landmarks_left = np.min(pred[:, 0])
                landmarks_right = np.max(pred[:, 0])
                areas.append((landmarks_bottom - landmarks_top) * (landmarks_right - landmarks_left))
            max_face_index = np.argmax(areas)
            return preds[max_face_index]

    # 旋转图片
    def rotate_image(self, image, landmarks):
        left_eye_corner = landmarks[36]
        right_eye_corner = landmarks[45]
        radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))

        height, width, depth = image.shape
        cos = math.cos(radian)
        sin = math.sin(radian)
        new_w = int(width * abs(cos) + height * abs(sin))
        new_h = int(width * abs(sin) + height * abs(cos))
        # 转换
        Tx = new_w // 2 - width // 2
        Ty = new_h // 2 - height // 2

        # 仿射矩阵
        M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                      [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])

        # 旋转后的图片
        image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))
        # 旋转后的关键点坐标
        landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
        landmarks_rotate = np.dot(M, landmarks.T).T

        return image_rotate, landmarks_rotate

    def crop(self, image, landmarks):
        # 边界
        landmarks_top = np.min(landmarks[:, 1])
        landmarks_bottom = np.max(landmarks[:, 1])
        landmarks_left = np.min(landmarks[:, 0])
        landmarks_right = np.max(landmarks[:, 0])

        top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))
        bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top))
        left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))
        right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))

        if bottom - top > right - left:
            left -= ((bottom - top) - (right - left)) // 2
            right = left + (bottom - top)
        else:
            top -= ((right - left) - (bottom - top)) // 2
            bottom = top + (right - left)

        # 初始化图片大小
        image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255

        h, w = image.shape[:2]
        left_white = max(0, -left)
        left = max(0, left)
        right = min(right, w - 1)
        right_white = left_white + (right - left)
        top_white = max(0, -top)
        top = max(0, top)
        bottom = min(bottom, h - 1)
        bottom_white = top_white + (bottom - top)

        image_crop[top_white:bottom_white + 1, left_white:right_white + 1] = image[top:bottom + 1,
                                                                             left:right + 1].copy()

        return image_crop


curPath = os.path.abspath(os.path.dirname(__file__))


class FaceSeg:
    def __init__(self, model_path=os.path.join(curPath, 'seg_model_384.pb')):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self._graph = tf.Graph()
        self._sess = tf.compat.v1.Session(config=config, graph=self._graph)

        self.pb_file_path = model_path
        self._restore_from_pb()
        self.input_op = self._sess.graph.get_tensor_by_name('input_1:0')
        self.output_op = self._sess.graph.get_tensor_by_name('sigmoid/Sigmoid:0')

    def _restore_from_pb(self):
        with self._sess.as_default():
            with self._graph.as_default():
                with gfile.FastGFile(self.pb_file_path, 'rb') as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')

    def input_transform(self, image):
        image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
        image_input = (image / 255.)[np.newaxis, :, :, :]
        return image_input

    def output_transform(self, output, shape):
        output = cv2.resize(output, (shape[1], shape[0]))
        image_output = (output * 255).astype(np.uint8)
        return image_output

    def get_mask(self, image):
        image_input = self.input_transform(image)
        output = self._sess.run(self.output_op, feed_dict={self.input_op: image_input})[0]
        return self.output_transform(output, shape=image.shape[:2])


if __name__ == '__main__':
    image_name = 'test.jpg'
    img = cv2.cvtColor(cv2.imread(os.path.join('dataset', 'img', image_name)), cv2.COLOR_BGR2RGB)
    # 打印原图出来看看
    plt.title('img')
    plt.imshow(img)
    plt.show()

    # 检测人脸+裁剪多余背景
    detector = FaceDetect('cpu', 'dlib')
    image_align, landmarks_align = detector.align(img)
    face = detector.crop(image_align, landmarks_align)

    cv2.imwrite('./dataset/face/' + image_name, face[:, :, ::-1])

    # 抠出人脸
    # 得到mask
    mask_dir = os.path.join(os.getcwd(), 'dataset', 'result_seg', image_name)
    segment = FaceSeg()
    mask = segment.get_mask(face)

    cv2.imwrite(mask_dir, mask)

    # mask与face计算
    splice = np.dstack((face, mask))
    splice_face = splice[:, :, :3].copy()
    splice_mask = splice[:, :, 3][:, :, np.newaxis].copy() / 255.
    face_in_white_bg = (splice_face * splice_mask + (1 - splice_mask) * 255) / 127.5 - 1

    cv2.imwrite(os.path.join(os.getcwd(), 'datasets/test/', 'A', image_name),
                cv2.cvtColor(face_in_white_bg, cv2.COLOR_RGB2BGR))
