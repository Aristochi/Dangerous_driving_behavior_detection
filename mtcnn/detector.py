from __future__ import print_function

import time

import os
import tensorflow as  tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from PIL import Image
from keras.engine.saving import load_model
from keras_preprocessing.image import img_to_array
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import cv2
import math
import argparse

import numpy as np
import h5py



def nms(boxes, overlap_threshold=0.5, mode='union'):
    if len(boxes) == 0:
        return []

    pick = []
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order

    while len(ids) > 0:
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        inter = w * h
        if mode == 'min':
            overlap = inter / np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            overlap = inter / (area[i] + area[ids[:last]] - inter)

        ids = np.delete(ids,
                        np.concatenate(
                            [[last],
                             np.where(overlap > overlap_threshold)[0]]))
    return pick


def convert_to_square(bboxes):
    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes


def calibrate_box(bboxes, offsets):
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    translation = np.hstack([w, h, w, h]) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes


def get_image_boxes(bounding_boxes, img, size=24):
    num_boxes = len(bounding_boxes)
    width = img.shape[1]
    height = img.shape[0]

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(
        bounding_boxes, width, height)
    img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), 'uint8')

        img_array = np.asarray(img, 'uint8')
        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = \
            img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        img_box = cv2.resize(img_box, (size, size))
        img_box = np.asarray(img_box, 'float32')
        img_boxes[i, :, :, :] = _preprocess(img_box)
    return img_boxes


def correct_bboxes(bboxes, width, height):
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    x, y, ex, ey = x1, y1, x2, y2

    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]

    return return_list


def _preprocess(img):
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5) * 0.0078125
    return img


def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    draw = img.copy()
    for b in bounding_boxes:
        b = [int(round(value)) for value in b]
        # print(b)
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

    for p in facial_landmarks:
        for i in range(5):  # 左眼右眼鼻子左嘴角右嘴角
            cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), -1)
            # cv2.putText(draw,str(p[i]),(p[i], p[i + 5]),cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255, 255, 255), 1, 8)
    return draw


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([('conv1', nn.Conv2d(3, 10, 3, 1)), ('prelu1', nn.PReLU(10)),
                         ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),
                         ('conv2', nn.Conv2d(10, 16, 3, 1)), ('prelu2', nn.PReLU(16)),
                         ('conv3', nn.Conv2d(16, 32, 3, 1)), ('prelu3', nn.PReLU(32))]))
        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

        weights = np.load(os.path.join(os.path.dirname(__file__), 'pnet.npy'), allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = F.softmax(a, dim=1)
        return b, a


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([('conv1', nn.Conv2d(3, 28, 3, 1)), ('prelu1', nn.PReLU(28)),
                         ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)), ('conv2', nn.Conv2d(28, 48, 3, 1)),
                         ('prelu2', nn.PReLU(48)), ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),
                         ('conv3', nn.Conv2d(48, 64, 2, 1)), ('prelu3', nn.PReLU(64)),
                         ('flatten', Flatten()), ('conv4', nn.Linear(576, 128)), ('prelu4', nn.PReLU(128))]))
        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

        weights = np.load(os.path.join(os.path.dirname(__file__), 'rnet.npy'), allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        a = F.softmax(a, 1)
        return b, a


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 32, 3, 1)),
                ('prelu1', nn.PReLU(32)),
                ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),
                ('conv2', nn.Conv2d(32, 64, 3, 1)),
                ('prelu2', nn.PReLU(64)),
                ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),
                ('conv3', nn.Conv2d(64, 64, 3, 1)),
                ('prelu3', nn.PReLU(64)),
                ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),
                ('conv4', nn.Conv2d(64, 128, 2, 1)),
                ('prelu4', nn.PReLU(128)),
                ('flatten', Flatten()),
                ('conv5', nn.Linear(1152, 256)),
                ('drop5', nn.Dropout(0.25)),
                ('prelu5', nn.PReLU(256)),
            ]))

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)
        weights = np.load(os.path.join(os.path.dirname(__file__), 'onet.npy'), allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = F.softmax(a, 1)
        return c, b, a


def run_first_stage(image, net, scale, threshold):
    with torch.no_grad():
        height, width = image.shape[:2]
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = cv2.resize(image, (sw, sh))
        img = np.asarray(img, 'float32')
        img = Variable(torch.FloatTensor(_preprocess(img)), volatile=True)
        output = net(img)
        probs = output[1].data.numpy()[0, 1, :, :]
        offsets = output[0].data.numpy()
        boxes = _generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None
        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]


def _generate_bboxes(probs, offsets, scale, threshold):
    stride = 2
    cell_size = 12

    inds = np.where(probs > threshold)
    if inds[0].size == 0:
        return np.array([])

    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    bounding_boxes = np.vstack([
        np.round((stride * inds[1] + 1.0) / scale),
        np.round((stride * inds[0] + 1.0) / scale),
        np.round((stride * inds[1] + 1.0 + cell_size) / scale),
        np.round((stride * inds[0] + 1.0 + cell_size) / scale), score, offsets
    ])
    return bounding_boxes.T


def detect_faces(image,
                 min_face_size=35.0,
                 thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7]):
    pnet, rnet, onet = PNet(), RNet(), ONet()
    onet.eval()

    height, width = image.shape[:2]
    min_length = min(height, width)
    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image
    scales = []
    m = min_detection_size / min_face_size
    min_length *= m
    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor ** factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1
    bounding_boxes = []
    for s in scales:  # run P-Net on different scales
        boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)

    # collect boxes (and offsets, and scores) from different scales
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)
    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5],
                                   bounding_boxes[:, 5:])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2
    img_boxes = get_image_boxes(bounding_boxes, image, size=24)
    img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
    output = rnet(img_boxes)
    offsets = output[0].data.numpy()  # shape [n_boxes, 4]
    probs = output[1].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 3
    img_boxes = get_image_boxes(bounding_boxes, image, size=48)
    if len(img_boxes) == 0:
        return [], []
    img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
    output = onet(img_boxes)
    landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].data.numpy()  # shape [n_boxes, 4]
    probs = output[2].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # compute landmark points
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(
        xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(
        ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]

    return bounding_boxes, landmarks


def isRotationMatrix(rvec):
    theta = np.linalg.norm(rvec)
    r = rvec / theta
    R_ = np.array([[0, -r[2][0], r[1][0]],
                   [r[2][0], 0, -r[0][0]],
                   [-r[1][0], r[0][0], 0]])
    R = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * r * r.T + np.sin(theta) * R_


    Rt = np.transpose(R)  # 旋转矩阵R的转置
    shouldBeIdentity = np.dot(Rt, R)  # R的转置矩阵乘以R
    I = np.identity(3, dtype=R.dtype)  # 3阶单位矩阵
    n = np.linalg.norm(I - shouldBeIdentity)  # np.linalg.norm默认求二范数
    return n < 1e-6  # 目的是判断矩阵R是否正交矩阵（旋转矩阵按道理须为正交矩阵，如此其返回值理论为0）


def rotationMatrixToAngles(Re):
    assert (isRotationMatrix(Re))  # 判断是否是旋转矩阵（用到正交矩阵特性）
    theta = np.linalg.norm(Re)
    r = Re / theta
    R_ = np.array([[0, -r[2][0], r[1][0]],
                   [r[2][0], 0, -r[0][0]],
                   [-r[1][0], r[0][0], 0]])
    R = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * r * r.T + np.sin(theta) * R_
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])  # 矩阵元素下标都从0开始（对应公式中是sqrt(r11*r11+r21*r21)），sy=sqrt(cosβ*cosβ)

    singular = sy < 1e-6  # 判断β是否为正负90°

    if not singular:  # β不是正负90°
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:  # β是正负90°
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)  # 当z=0时，此公式也OK，上面图片中的公式也是OK的
        z = 0

    x = x * 180.0 / 3.141592653589793
    y = y * 180.0 / 3.141592653589793
    z = z * 180.0 / 3.141592653589793

    return np.array([x, y, z])


# 从旋转向量转换为欧拉角
def get_euler_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)

    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2) * rotation_vector[0][0] / theta
    y = math.sin(theta / 2) * rotation_vector[1][0] / theta
    z = math.sin(theta / 2) * rotation_vector[2][0] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    # print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)

    # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

    # 单位转换：将弧度转换为度
    Y = int((pitch / math.pi) * 180)
    X = int((yaw / math.pi) * 180)
    Z = int((roll / math.pi) * 180)

    return 0, Y, X, Z


def get_head_pose(landmarks):
    points = []
    for p in landmarks:
        for i in range(10):  # 左眼右眼鼻子左嘴角右嘴角
            points.append(p[i])
            # print(points[i])
    if len(points) > 0:
        # 2D image points. If you change the image, you need to change vector
        chin_x = points[3] + (points[4] - points[3]) / 2
        # chin_y = ((points[7] - points[5]) + (points[7] - points[6])) /2 + points[7]
        chin_y = (points[8] - points[7]) + points[8]

        image_points = np.array([
            (points[2], points[7]),  # Nose tip
            # (399, 561),     # Chin
            (chin_x, chin_y),  # chin
            (points[0] - 8, points[5]),  # Left eye left corner
            (points[1] + 8, points[6]),  # Right eye right corne
            (points[3], points[8]),  # Left Mouth corner
            (points[4], points[9])  # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
            # (-165.0, 170.0, -115.0),  # Left eye left corner
            # (165.0, 170.0, -115.0),  # Right eye right corne
            # (-150.0, -150.0, -125.0),  # Left Mouth corner
            # (150.0, -150.0, -125.0)  # Right mouth corner

        ])

        K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
             0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
             0.0, 0.0, 1.0]
        D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
        # # 相机内参矩阵
        camera_matrix = np.array(K).reshape(3, 3).astype(np.float32)
        # # 相机畸变系数
        dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=4)  # cv2.CV_ITERATIVE
        # print('rotationMatrixToAngles')
        # print(rotationMatrixToAngles(rotation_vector))
        # calculate rotation angles
        theta = cv2.norm(rotation_vector, cv2.NORM_L2)

        # transformed to quaterniond
        w = math.cos(theta / 2)
        x = math.sin(theta / 2) * rotation_vector[0][0] / theta
        y = math.sin(theta / 2) * rotation_vector[1][0] / theta
        z = math.sin(theta / 2) * rotation_vector[2][0] / theta

        ysqr = y * y
        # pitch (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        # print('t0:{}, t1:{}'.format(t0, t1))
        pitch = math.atan2(t0, t1)

        # yaw (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        if t2 > 1.0:
            t2 = 1.0
        if t2 < -1.0:
            t2 = -1.0
        yaw = math.asin(t2)

        # roll (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        roll = math.atan2(t3, t4)

        # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

        # 单位转换：将弧度转换为度
        # Y = int((pitch / math.pi) * 180)
        # X = int((yaw / math.pi) * 180)
        # Z = int((roll / math.pi) * 180)

        return 0, pitch, yaw, roll


def get_face_expression(img, bounding_box):
    draw = img.copy()
    if len(bounding_box) > 0:
        point = []
        for b in bounding_box:
            b = [int(round(value)) for value in b]
            for i in b:
                point.append(i)
        # print(point)
        # cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cropped = draw[point[1]:point[3], point[0]:point[2]]
        # cv2.imshow('0', cropped)
        return cropped
emotion_model_path = '_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["生气", "厌恶", "害怕", "喜悦", "悲伤", "惊讶", "普通"]
def get_emotion(img):

    preds = []  # 预测的结果
    label = None
    emotion_probability=None
    if len(img)>0:
        roi = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)  # 最大的概率
        label = EMOTIONS[preds.argmax()]
    else:
        label='wait'
        emotion_probability=0
        # print(label)

    return emotion_probability, label,preds

if __name__ == '__main__':
    import cv2

    path = ('E:\PythonEye\Dataset/3-FemaleGlasses.mp4')
    # cap = cv2.VideoCapture(path)
    cap = cv2.VideoCapture(0)
    while True:
        start = time.time()
        ret, img = cap.read()
        if not ret: break

        bounding_boxes, landmarks = detect_faces(img)
        image = show_bboxes(img, bounding_boxes, landmarks)
        # print(bounding_boxes)
        # get_face_expression(image, bounding_boxes)
        get_head_pose(landmarks)
        # img2=get_face_expression(img, bounding_boxes)
        # cv2.imshow('s',img2)
        # print(get_emotion(img2))
        print(get_head_pose(landmarks))
        T = time.time() - start
        fps = 1 / T  # 实时在视频上显示fps

        fps_txt = 'fps:%.2f' % (fps)
        cv2.putText(image, fps_txt, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, 8)
        cv2.imshow("Output", image)
        if cv2.waitKey(10) == 27:
            break

        # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # cv2.imshow('0', image)
