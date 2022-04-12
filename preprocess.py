import os
import random

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

    # 矫正人脸
    def align(self, image):
        landmarks = self.get_face_landmarks(image)

        if landmarks is None:
            return None
        else:
            return self.rotate_image(image, landmarks)

    # 找到检测关键点
    def get_face_landmarks(self, image):
        preds = self.fa.get_landmarks_from_image(image)

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
        # 中间展示
        image_copy = image.copy()
        # 眼睛绘制两点
        x = [left_eye_corner[0], right_eye_corner[0]]
        y = [left_eye_corner[1], right_eye_corner[1]]
        plt.plot(x, y, color='r')
        plt.scatter(x, y, color='b', s=10)
        for i in range(landmarks.shape[0]):
            pt_position = (int(landmarks[i][0]), int(landmarks[i][1]))
            cv2.circle(image_copy, pt_position, 3, (0, 0, 255), -1)
        plt.imshow(image_copy)
        plt.title('face_eyes')
        plt.show()

        # 两y轴高度差/两眼x轴距离差
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
        # 中间展示
        image_rotate_copy = image_rotate.copy()
        # 绘制两点
        x = [left_eye_corner[0], right_eye_corner[0]]
        y = [left_eye_corner[1], right_eye_corner[1]]
        plt.plot(x, y, color='r')
        plt.scatter(x, y, color='b', s=10)
        for i in range(landmarks_rotate.shape[0]):
            pt_position = (int(landmarks_rotate[i][0]), int(landmarks_rotate[i][1]))
            cv2.circle(image_rotate_copy, pt_position, 3, (0, 0, 255), -1)
        plt.imshow(image_rotate_copy)
        plt.title('image_rotate')
        plt.show()

        return image_rotate, landmarks_rotate

    def crop(self, image, landmarks):
        # 边界
        landmarks_top = np.min(landmarks[:, 1])
        landmarks_bottom = np.max(landmarks[:, 1])
        landmarks_left = np.min(landmarks[:, 0])
        landmarks_right = np.max(landmarks[:, 0])

        # 中间展示
        # 找出这几个值的索引
        landmarks_top_index = landmarks[:, 1].tolist().index(landmarks_top)
        landmarks_bottom_index = landmarks[:, 1].tolist().index(landmarks_bottom)
        landmarks_left_index = landmarks[:, 0].tolist().index(landmarks_left)
        landmarks_right_index = landmarks[:, 0].tolist().index(landmarks_right)

        image_copy = image.copy()
        mark_list = [landmarks_top_index, landmarks_bottom_index, landmarks_left_index, landmarks_right_index]
        for i in range(landmarks.shape[0]):
            pt_position = (int(landmarks[i][0]), int(landmarks[i][1]))
            cv2.circle(image_copy, pt_position, 8, (0, 0, 255), -1)
        for index in mark_list:
            pt_position = (int(landmarks[index][0]), int(landmarks[index][1]))
            cv2.circle(image_copy, pt_position, 8, (255, 0, 0), -1)
        plt.imshow(image_copy)
        plt.title('face_top_bottom_left_right')
        plt.show()
        # 裁剪
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

        # 初始化白色背景图大小
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

        # 中间展示
        cv2.rectangle(image_copy, (left + 8, top + 8), (right - 8, bottom - 8), (0, 255, 0), 8)
        plt.imshow(image_copy)
        plt.title('face_rectangle')
        plt.show()

        image_crop[top_white:bottom_white + 1, left_white:right_white + 1] = image[top:bottom + 1,
                                                                             left:right + 1].copy()

        # 中间展示
        plt.title('image_crop')
        plt.imshow(image_crop)
        plt.show()

        return image_crop


curPath = os.path.abspath(os.path.dirname(__file__))


class FaceSeg:
    def __init__(self, model_path=os.path.join(curPath, 'pretrained_model/seg_model_384.pb')):
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
    for i in range(24):
        # prefix = random.randint(1601, 2000)
        # print(prefix)
        # print(i)
        image_name = str(i + 4100) + '.png'
        # image_name = 'WechatIMG2.jpeg'
        img = cv2.cvtColor(cv2.imread(os.path.join('/Users/mac/Downloads/generated_yellow-stylegan2', image_name)),
                           cv2.COLOR_BGR2RGB)

        # 打印原图出来看看
        # plt.title('img')
        # plt.imshow(img)
        # plt.show()

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

        cv2.imwrite(os.path.join(os.getcwd(), 'datasets/test/', 'A', image_name),
                    cv2.cvtColor(face_in_white_bg, cv2.COLOR_RGB2BGR))


# Thanks to dataset provider:Copyright(c) 2018, seeprettyface.com, BUPT_GWY contributes the dataset.
