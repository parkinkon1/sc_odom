import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from open3d import *
from utils import lla_to_enu

import pykitti
import pickle
from tqdm import tqdm

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


if __name__ == "__main__":
    kitti2sc(kitti_root_dir='/datasets/kitti/raw', kitti_date='2011_09_30', kitti_drive='0033')

