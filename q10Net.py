from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import pandas as pd
from keras.applications import MobileNetV2, ResNet50, InceptionV3

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras.layers import Input

np.random.seed(777)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)  # Memory growth must be set before GPUs have been initialized


class KittiLoader(tf.keras.utils.Sequence):
    def __init__(self, kitti_date='2011_09_30', kitti_drive='0033', data_type='train', shuffle=True):
        self.data_dir = './data/{}_{}'.format(kitti_date, kitti_drive)
        self.len = len(os.listdir(self.data_dir))
        self.kitti_date = kitti_date
        self.kitti_drive = kitti_drive

        self.data_type = data_type
        self.index = np.arange(50, self.len) if not shuffle else np.random.permutation(np.arange(50, self.len))
        if data_type == 'train':
            self.index = self.index[:int((self.len - 50) * 0.7)]
        elif data_type == 'val':
            self.index = self.index[int((self.len - 50) * 0.7):]
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


class Q10Net:
    def __init__(self, model_type='resnet50', input_shape=(320, 720, 2)):
        # model
        self.base_model = None
        if model_type == 'inceptionv3':
            self.base_model = InceptionV3(input_tensor=Input(shape=input_shape), weights=None, include_top=False)
        elif model_type == 'resnet50':
            self.base_model = ResNet50(input_tensor=Input(shape=input_shape), weights=None, include_top=False)
        elif model_type == 'mobilenetv2':
            self.base_model = MobileNetV2(input_tensor=Input(shape=input_shape), weights=None, include_top=False)
        else:
            print('no model found')
            return

        # self-supervised
        x1 = self.base_model.output
        x1 = GlobalAveragePooling2D()(x1)
        x1 = Dense(1024, activation='relu')(x1)
        self.self_predictions = Dense(6)(x1)

        # real
        x2 = self.base_model.output
        x2 = GlobalAveragePooling2D()(x2)
        x2 = Dense(1024, activation='relu')(x2)
        self.predictions = Dense(2)(x2)

        self.self_model = Model(inputs=self.base_model.input, outputs=self.self_predictions)
        self.model = Model(inputs=self.base_model.input, outputs=self.predictions)

        self.self_history = []
        self.history = []
        self.tune_history = []
        self.tune2_history = []

    def train_self(self, train_loader, val_loader, epochs):
        train_steps = train_loader.get_steps()
        valid_steps = val_loader.get_steps()
        for layer in self.base_model.layers:
            layer.trainable = True
        self.self_model.compile(optimizer='rmsprop', loss='mse', metrics=['mae', 'mse'])
        self.self_history = self.self_model.fit_generator(train_loader, validation_data=val_loader,
                                                     steps_per_epoch=train_steps, validation_steps=valid_steps,
                                                     epochs=epochs)
        plot_history(self.self_history)

    def train_real(self, train_loader, val_loader, epochs):
        train_steps = train_loader.get_steps()
        valid_steps = val_loader.get_steps()
        for layer in self.base_model.layers:
            layer.trainable = True
        self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mae', 'mse'])
        self.history = self.model.fit_generator(train_loader, validation_data=val_loader,
                                                steps_per_epoch=train_steps, validation_steps=valid_steps,
                                                epochs=epochs)
        plot_history(self.history)

    def fine_tune(self, train_loader, val_loader, epochs):
        train_steps = train_loader.get_steps()
        valid_steps = val_loader.get_steps()
        for layer in self.base_model.layers:
            layer.trainable = False
        self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mae', 'mse'])
        self.tune_history = self.model.fit_generator(train_loader, validation_data=val_loader,
                                                     steps_per_epoch=train_steps, validation_steps=valid_steps,
                                                     epochs=epochs)
        plot_history(self.tune_history)

    def fine_tune2(self, train_loader, val_loader, epochs):
        train_steps = train_loader.get_steps()
        valid_steps = val_loader.get_steps()
        for layer in self.base_model.layers:
            layer.trainable = True
        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='mse', metrics=['mae', 'mse'])
        self.tune2_history = self.model.fit_generator(train_loader, validation_data=val_loader,
                                                      steps_per_epoch=train_steps, validation_steps=valid_steps,
                                                      epochs=epochs)
        plot_history(self.tune2_history)

    def eval(self, test_loader, save_dir=None):
        loss, mae, mse = self.model.evaluate_generator(test_loader, verbose=2)
        print("테스트 세트의 평균 절대 오차: {:5.2f}".format(mae))
        print("테스트 세트의 mse: {:5.2f}".format(mse))

        gt = []
        for idx, batch in enumerate(test_loader):
            gt.append(batch[1])
        gt = np.reshape(np.array(gt), (-1, 2))
        predicted = self.model.predict_generator(test_loader, steps=test_loader.get_steps())

        gt_yaw = gt[:, 0]
        gt_dis = gt[:, 1]
        predicted_yaw = predicted[:, 0]
        predicted_dis = predicted[:, 1]

        plt.figure(figsize=(18, 4))

        plt.subplot(1, 2, 1)
        plt.title('Yaw Rates (rad/s)')
        plt.xlabel('Sequence')
        plt.ylabel('Yaw rate (rad/s)')
        plt.plot(range(len(gt_yaw)), gt_yaw, label='gt_yaw')
        plt.plot(range(len(predicted_yaw)), predicted_yaw, label='predicted_yaw')
        plt.legend(loc='upper right')

        plt.subplot(1, 2, 2)
        plt.title('Forward Velocities (m/s)')
        plt.xlabel('Sequence')
        plt.ylabel('Forward velocity (m/s)')
        plt.plot(range(len(gt_dis)), gt_dis, label='gt_dis')
        plt.plot(range(len(predicted_dis)), predicted_dis, label='predicted_dis')
        plt.legend(loc='upper right')
        plt.show()

        plt.figure(figsize=(18, 4))

        plt.subplot(1, 2, 1)
        plt.title('Error Distribution: Yaw Rates (rad/s)')
        error = predicted_yaw - gt_yaw
        plt.hist(error, bins=25)
        plt.xlabel("Prediction Error [MPG]")
        plt.ylabel("Count")

        plt.subplot(1, 2, 2)
        plt.title('Error Distribution: Forward Velocities (m/s)')
        error = predicted_dis - gt_dis
        plt.hist(error, bins=25)
        plt.xlabel("Prediction Error [MPG]")
        plt.ylabel("Count")
        plt.show()

        if save_dir is not None:
            data = [gt, predicted]
            with open(save_dir, 'wb') as file:
                pickle.dump(data, file)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train loss')
    plt.plot(hist['epoch'], hist['val_loss'],
             label='Val loss')
    # plt.ylim([0, 20])
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.title('MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    # plt.ylim([0, 20])
    plt.legend(loc='upper right')
    plt.show()
