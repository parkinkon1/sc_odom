import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from open3d import *
from utils import lla_to_enu

import pykitti
import pickle
from tqdm import tqdm
import math
from tensorflow.keras.utils import Sequence
import keras
import cv2
from scipy.spatial.transform import Rotation as R

np.random.seed(777)


def xy2theta(x, y):
    theta = 0.
    if x >= 0 and y >= 0:
        theta = 180 / np.pi * np.arctan(y / x)
    elif x < 0 and y >= 0:
        theta = 180 - ((180 / np.pi) * np.arctan(y / (-x)))
    elif x < 0 and y < 0:
        theta = 180 + ((180 / np.pi) * np.arctan(y / x))
    elif x >= 0 and y < 0:
        theta = 360 - ((180 / np.pi) * np.arctan((-y) / x))
    return theta


def pt2rs(point, gap_ring, gap_sector, num_ring, num_sector):
    x = point[0]
    y = point[1]
    z = point[2]

    if x == 0.0:
        x = 0.001
    if y == 0.0:
        y = 0.001

    theta = xy2theta(x, y)
    faraway = np.sqrt(x * x + y * y)

    idx_ring = np.divmod(faraway, gap_ring)[0]
    idx_sector = np.divmod(theta, gap_sector)[0]

    if idx_ring >= num_ring:
        idx_ring = num_ring - 1  # python starts with 0 and ends with N-1

    return int(idx_ring), int(idx_sector)


def cloud2sc(ptcloud, num_sector, num_ring, max_length, kitti_lidar_height):
    num_points = ptcloud.shape[0]

    gap_ring = max_length / num_ring
    gap_sector = 360 / num_sector

    enough_large = 1000
    sc_storage = np.zeros([enough_large, num_ring, num_sector])
    sc_counter = np.zeros([num_ring, num_sector])

    for pt_idx in range(num_points):

        point = ptcloud[pt_idx, :]
        point_height = point[2] + kitti_lidar_height

        idx_ring, idx_sector = pt2rs(point, gap_ring, gap_sector, num_ring, num_sector)

        if sc_counter[idx_ring, idx_sector] >= enough_large:
            continue
        sc_storage[int(sc_counter[idx_ring, idx_sector]), idx_ring, idx_sector] = point_height
        sc_counter[idx_ring, idx_sector] = sc_counter[idx_ring, idx_sector] + 1

    sc = np.amax(sc_storage, axis=0)
    return sc


def kitti2sc(kitti_root_dir='/datasets/kitti/raw', kitti_date='2011_09_30', kitti_drive='0033'):
    print('dataset: {}_{}_{}'.format(kitti_root_dir, kitti_date, kitti_drive))

    down_cell_size = 0.25
    kitti_lidar_height = 2.0

    sector_res = 720  # rotation
    ring_res = 320  # velocity
    max_length = 80

    # kitti dataset
    dataset = pykitti.raw(kitti_root_dir, kitti_date, kitti_drive)

    gt_trajectory_lla = []  # [longitude(deg), latitude(deg), altitude(meter)] x N
    gt_yaws = []  # [yaw_angle(rad),] x N
    gt_yaw_rates = []  # [vehicle_yaw_rate(rad/s),] x N
    gt_forward_velocities = []  # [vehicle_forward_velocity(m/s),] x N

    for oxts_data in dataset.oxts:
        packet = oxts_data.packet
        gt_trajectory_lla.append([
            packet.lon,
            packet.lat,
            packet.alt
        ])
        gt_yaws.append(packet.yaw)
        gt_yaw_rates.append(packet.wz)
        gt_forward_velocities.append(packet.vf)

    gt_trajectory_lla = np.array(gt_trajectory_lla).T
    gt_yaws = np.array(gt_yaws)
    gt_yaw_rates = np.array(gt_yaw_rates)
    gt_forward_velocities = np.array(gt_forward_velocities)

    timestamps = np.array(dataset.timestamps)
    elapsed = np.array(timestamps) - timestamps[0]
    ts = [t.total_seconds() for t in elapsed]

    # pre-processing: gps pose
    origin = gt_trajectory_lla[:, 0]  # set the initial position to the origin
    gt_trajectory_xyz = lla_to_enu(gt_trajectory_lla, origin)

    for idx, cloud in tqdm(enumerate(dataset.velo), total=len(ts)):
        cloud_xyz = cloud[:, :3]
        # down-sampling
        pcd = PointCloud()
        pcd.points = Vector3dVector(cloud_xyz)
        down_pcd = voxel_down_sample(pcd, voxel_size=down_cell_size)
        cloud_xyz_downed = np.asarray(down_pcd.points)
        # make SC
        sc = cloud2sc(cloud_xyz_downed, sector_res, ring_res, max_length, kitti_lidar_height)
        # save
        data = dict()
        data['sc'] = sc
        data['time'] = ts[idx]
        data['gt_yaw'] = gt_yaws[idx]
        data['gt_yaw_rate'] = gt_yaw_rates[idx]
        data['gt_forward_velocity'] = gt_forward_velocities[idx]
        data['gt_trajectory_xyz'] = gt_trajectory_xyz[:, idx]

        out_dir = './data/{}_{}'.format(kitti_date, kitti_drive)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        out_name = '{}/{}.bin'.format(out_dir, idx)
        with open(out_name, 'wb') as file:
            pickle.dump(data, file)


class KittiParser:
    def __init__(self, kitti_root_dir='/datasets/kitti/raw', kitti_date='2011_09_30', kitti_drive='0033'):
        self.gt_trajectory_lla = []
        self.gt_yaws = []
        self.gt_yaw_rates = []
        self.gt_forward_velocities = []
        self.ts = []
        self.gt_trajectory_xyz = []
        self.lidar_points = []

        self.kitti_date = kitti_date
        self.kitti_drive = kitti_drive

        self.load_kitti(kitti_root_dir=kitti_root_dir, kitti_date=kitti_date, kitti_drive=kitti_drive)

        pass

    def load_kitti(self, kitti_root_dir='/datasets/kitti/raw', kitti_date='2011_09_30', kitti_drive='0033'):
        # kitti dataset
        dataset = pykitti.raw(kitti_root_dir, kitti_date, kitti_drive)

        gt_trajectory_lla = []  # [longitude(deg), latitude(deg), altitude(meter)] x N
        gt_yaws = []  # [yaw_angle(rad),] x N
        gt_yaw_rates = []  # [vehicle_yaw_rate(rad/s),] x N
        gt_forward_velocities = []  # [vehicle_forward_velocity(m/s),] x N

        for oxts_data in dataset.oxts:
            packet = oxts_data.packet
            gt_trajectory_lla.append([
                packet.lon,
                packet.lat,
                packet.alt
            ])
            gt_yaws.append(packet.yaw)
            gt_yaw_rates.append(packet.wz)
            gt_forward_velocities.append(packet.vf)

        self.gt_trajectory_lla = np.array(gt_trajectory_lla).T
        self.gt_yaws = np.array(gt_yaws)
        self.gt_yaw_rates = np.array(gt_yaw_rates)
        self.gt_forward_velocities = np.array(gt_forward_velocities)

        timestamps = np.array(dataset.timestamps)
        elapsed = np.array(timestamps) - timestamps[0]
        self.ts = np.array([t.total_seconds() for t in elapsed])

        # pre-processing: gps pose
        origin = self.gt_trajectory_lla[:, 0]  # set the initial position to the origin
        self.gt_trajectory_xyz = lla_to_enu(self.gt_trajectory_lla, origin)

        self.lidar_points = dataset.velo

        print('kitti dataset loaded: {}_{}_{}'.format(kitti_root_dir, kitti_date, kitti_drive))

    def saveSC(self, sector_res=720, ring_res=320):
        down_cell_size = 0.25
        kitti_lidar_height = 2.0
        sector_res = sector_res  # rotation
        ring_res = ring_res  # velocity
        max_length = 80

        out_dir = './data/scs/{}_{}'.format(self.kitti_date, self.kitti_drive)
        if os.path.isdir(out_dir):
            print('SC data already exists')
            return False
        else:
            os.mkdir(out_dir)
            for idx, cloud in tqdm(enumerate(self.lidar_points), total=len(self.ts), desc='generating scs'):
                cloud_xyz = cloud[:, :3]
                # down-sampling
                pcd = PointCloud()
                pcd.points = Vector3dVector(cloud_xyz)
                down_pcd = voxel_down_sample(pcd, voxel_size=down_cell_size)
                cloud_xyz_downed = np.asarray(down_pcd.points)
                # make SC
                sc = cloud2sc(cloud_xyz_downed, sector_res, ring_res, max_length, kitti_lidar_height)
                # save
                out_name = '{}/{}.bin'.format(out_dir, idx)
                with open(out_name, 'wb') as file:
                    pickle.dump(sc, file)
            return True

    def saveSC_aug_with_label(self, sector_res=720, ring_res=320):
        down_cell_size = 0.25
        kitti_lidar_height = 2.0
        sector_res = sector_res  # rotation
        ring_res = ring_res  # velocity
        max_length = 80

        out_dir = './data/sc_aug/{}_{}'.format(self.kitti_date, self.kitti_drive)
        if os.path.isdir(out_dir):
            print('SC(augmented) data already exists')
            return False
        else:
            os.mkdir(out_dir)
            for idx, cloud in tqdm(enumerate(self.lidar_points), total=len(self.ts), desc='generating scs_aug'):
                cloud_xyz = cloud[:, :3]
                # down-sampling
                pcd = PointCloud()
                pcd.points = Vector3dVector(cloud_xyz)
                down_pcd = voxel_down_sample(pcd, voxel_size=down_cell_size)
                cloud_xyz_downed = np.asarray(down_pcd.points)
                # augmentation
                rot_x = np.pi/2 * (np.random.rand() - 0.5) * 0.1
                rot_y = np.pi/2 * (np.random.rand() - 0.5) * 0.1
                rot_z = np.pi/2 * (np.random.rand() - 0.5) * 1.0
                r = R.from_rotvec(np.array([rot_x, rot_y, rot_z]))
                t = (np.random.rand(3) * 6) - 3.0
                cloud_rot = r.apply(cloud_xyz_downed)
                cloud_trans = cloud_xyz_downed + t
                # make SC
                sc = cloud2sc(cloud_xyz_downed, sector_res, ring_res, max_length, kitti_lidar_height)
                sc_rot = cloud2sc(cloud_rot, sector_res, ring_res, max_length, kitti_lidar_height)
                sc_trans = cloud2sc(cloud_trans, sector_res, ring_res, max_length, kitti_lidar_height)
                label_rot = np.append(r.as_rotvec(), np.array([0., 0., 0.]), axis=0)
                label_trans = np.append(np.array([0., 0., 0.]), t, axis=0)
                data = [sc, sc_rot, sc_trans, label_rot, label_trans]
                # save
                out_name = '{}/{}.bin'.format(out_dir, idx)
                with open(out_name, 'wb') as file:
                    pickle.dump(data, file)
            return True

    def saveSC_with_label(self):
        out_dir = './data/sc_with_label/{}_{}'.format(self.kitti_date, self.kitti_drive)
        sc_dir = self.get_sc_dir()
        if os.path.isdir(out_dir):
            print('SC with label dataset already exists')
            return False
        else:
            os.mkdir(out_dir)
            for idx in tqdm(range(len(self.ts)), total=len(self.ts), desc='generating scs_labels'):
                idx_10 = idx - 10
                if idx_10 < 0:
                    idx_10 = 0
                sc0_path = '{}/{}.bin'.format(sc_dir, idx)
                sc10_path = '{}/{}.bin'.format(sc_dir, idx_10)
                with open(sc0_path, 'rb') as f:
                    sc0 = pickle.load(f)
                with open(sc10_path, 'rb') as f:
                    sc10 = pickle.load(f)

                dt = self.ts[idx] - self.ts[idx_10]
                yaw_diff = self.gt_yaw_rates[idx_10] * dt
                dis_diff = self.gt_forward_velocities[idx_10] * dt

                data = [sc0, sc10, yaw_diff, dis_diff]
                # save
                out_name = '{}/{}.bin'.format(out_dir, idx)
                with open(out_name, 'wb') as file:
                    pickle.dump(data, file)
            return True

    def get_num_data(self):
        return len(self.ts)

    def get_sc_dir(self):
        return './data/scs/{}_{}'.format(self.kitti_date, self.kitti_drive)

    def get_sc_with_label_dir(self):
        return './data/sc_with_label/{}_{}'.format(self.kitti_date, self.kitti_drive)

    def get_aug_with_label_dir(self):
        return './data/sc_aug/{}_{}'.format(self.kitti_date, self.kitti_drive)


class Loader(keras.utils.Sequence):
    def __init__(self, kitti_root_dir, kitti_date, kitti_drive, data_type='raw', batch_size=10, shuffle=True, load_type='train', val_ratio=0.0):
        np.random.seed(777)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.kitti_dataset = KittiParser(kitti_root_dir=kitti_root_dir, kitti_date=kitti_date, kitti_drive=kitti_drive)
        self.data_type = data_type

        self.ids = np.arange(self.kitti_dataset.get_num_data())
        self.data_len = len(self.ids)

        # split train val
        train_size = int(self.data_len * (1 - val_ratio))
        if self.shuffle:
            train_ids = np.random.choice(self.ids, train_size, replace=False)
            val_ids = np.setdiff1d(self.ids, train_ids)
        else:
            train_ids = self.ids[:train_size]
            val_ids = self.ids[train_size:]

        if load_type == 'train':
            self.ids = train_ids
        elif load_type == 'val':
            self.ids = val_ids
        else:
            pass

        self.data_len = len(self.ids)
        self.indexes = np.arange(len(self.ids))

        self.sc_dir = self.kitti_dataset.get_sc_dir()
        self.sc_with_label_dir = self.kitti_dataset.get_sc_with_label_dir()

        self.data_dir = None
        if self.data_type == 'raw':
            self.data_dir = self.kitti_dataset.get_sc_with_label_dir()
        elif self.data_type == 'aug':
            self.data_dir = self.kitti_dataset.get_aug_with_label_dir()

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation__(self, id_name):
        data_path = '{}/{}.bin'.format(self.data_dir, id_name)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        if self.data_type == 'raw':
            sc0, sc10, yaw_diff, dis_diff = data
            x = np.concatenate((sc0[:, :, np.newaxis], sc10[:, :, np.newaxis]), axis=2)
            y = np.array([yaw_diff, dis_diff])
            return x, y
        elif self.data_type == 'aug':
            sc, sc_rot, sc_trans, label_rot, label_trans = data
            if id_name % 2 == 0:
                x = np.concatenate((sc[:, :, np.newaxis], sc_rot[:, :, np.newaxis]), axis=2)
                y = label_rot
                return x, y
            else:
                x = np.concatenate((sc[:, :, np.newaxis], sc_trans[:, :, np.newaxis]), axis=2)
                y = label_trans
                return x, y

    def __len__(self):
        return int(np.floor(self.data_len / self.batch_size))

    def __getitem__(self, index):  # index : batch no.
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = [self.ids[k] for k in indexes]

        imgs = list()
        labels = list()
        for id_name in batch_ids:
            x, y = self.__data_generation__(id_name)
            imgs.append(x)
            labels.append(y)
        imgs = np.array(imgs)
        labels = np.array(labels)

        return imgs, labels  # return batch

    def get_steps(self):
        return self.data_len // self.batch_size


if __name__ == "__main__":
    kitti2sc(kitti_root_dir='/datasets/kitti/raw', kitti_date='2011_09_30', kitti_drive='0033')

