from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import MobileNetV2, ResNet50
import pickle

np.random.seed(777)


class KittiLoader(tf.keras.utils.Sequence):
    def __init__(self, kitti_date='2011_09_30', kitti_drive='0033', data_type='train', shuffle=True):
        self.data_dir = './data/{}_{}'.format(kitti_date, kitti_drive)
        self.len = len(os.listdir(self.data_dir))
        self.kitti_date = kitti_date
        self.kitti_drive = kitti_drive

        self.data_type = data_type
        self.index = np.arange(50, self.len) if not shuffle else np.random.permutation(np.arange(50, self.len))
        if data_type == 'train':
            self.index = self.index[:int((self.len-50) * 0.7)]
        elif data_type == 'val':
            self.index = self.index[int((self.len-50) * 0.7):]
        elif data_type == 'test':
            pass

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        target_idx = self.index[idx]
        base_idx = target_idx - np.random.randint(20, 50)

        with open('{}/{}.bin'.format(self.data_dir, target_idx), 'rb') as file:
            target_data = pickle.load(file)
        with open('{}/{}.bin'.format(self.data_dir, base_idx), 'rb') as file:
            base_data = pickle.load(file)

        dt1 = target_data['gt_yaw'] - base_data['gt_yaw']
        dt2 = base_data['gt_yaw'] - target_data['gt_yaw']
        yaw_dt = dt1 if abs(dt1) < abs(dt2) else dt2
        forward_dt = target_data['gt_forward_velocity'] * (target_data['time'] - base_data['time'])

        X = np.append(target_data['sc'][:, :, np.newaxis], base_data['sc'][:, :, np.newaxis], axis=2)
        y = np.array([forward_dt, yaw_dt])

        return X[np.newaxis, ...], y[np.newaxis, ...]

    def get_gt_data(self, idx, gap=20):
        target_idx = self.index[idx]
        base_idx = target_idx - gap

        with open('{}/{}.bin'.format(self.data_dir, target_idx), 'rb') as file:
            target_data = pickle.load(file)
        with open('{}/{}.bin'.format(self.data_dir, base_idx), 'rb') as file:
            base_data = pickle.load(file)

        dt1 = target_data['gt_yaw'] - base_data['gt_yaw']
        dt2 = base_data['gt_yaw'] - target_data['gt_yaw']
        yaw_dt = dt1 if abs(dt1) < abs(dt2) else dt2
        forward_dt = target_data['gt_forward_velocity'] * (target_data['time'] - base_data['time'])

        X = np.append(target_data['sc'][:, :, np.newaxis], base_data['sc'][:, :, np.newaxis], axis=2)
        y = np.array([forward_dt, yaw_dt])

        return X[np.newaxis, ...], y[np.newaxis, ...]

    def get_test_data(self, idx, gap=20):
        target_idx = self.index[idx]
        base_idx = target_idx - gap

        with open('{}/{}.bin'.format(self.data_dir, target_idx), 'rb') as file:
            target_data = pickle.load(file)
        with open('{}/{}.bin'.format(self.data_dir, base_idx), 'rb') as file:
            base_data = pickle.load(file)

        dt1 = target_data['gt_yaw'] - base_data['gt_yaw']
        dt2 = base_data['gt_yaw'] - target_data['gt_yaw']
        yaw_dt = dt1 if abs(dt1) < abs(dt2) else dt2
        forward_dt = target_data['gt_forward_velocity'] * (target_data['time'] - base_data['time'])

        X = np.append(target_data['sc'][:, :, np.newaxis], base_data['sc'][:, :, np.newaxis], axis=2)
        y = np.array([forward_dt, yaw_dt])

        return X[np.newaxis, ...]


def train():
    img_shape = (720, 320, 2)

    base_model = MobileNetV2(input_shape=img_shape, include_top=False, weights=None)

    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    base_model.trainable = False

    base_model.summary()

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    len(model.trainable_variables)

















