import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import cv2
import warnings

import src.calib as calib
import src.utils as utils

class Kitti_Dataset(Dataset):

    def __init__(self, params):

        base_path = params['base_path']
        date = params['date']
        drives = params['drives']

        self.d_rot = params['d_rot']
        self.d_trans = params['d_trans']
        self.resize_h = params['resize_h']
        self.resize_w = params['resize_w']
        self.fixed_decalib = params['fixed_decalib']

        self.img_path = []
        self.lidar_path = []

        for drive in drives:
            cur_img_path = Path(base_path) / date / (date + f'_drive_{drive:04d}_sync') / 'image_02' / 'data'
            cur_lidar_path = Path(base_path) / date / (date + f'_drive_{drive:04d}_sync') / 'velodyne_points' / 'data'
            
            if not cur_img_path.exists() or not cur_lidar_path.exists():
                warnings.warn(f"Directory does not exist: {cur_img_path} or {cur_lidar_path}")
                continue
            
            img_files = sorted(str(file_name) for file_name in cur_img_path.glob('*.png'))
            lidar_files = sorted(str(file_name) for file_name in cur_lidar_path.glob('*.bin'))

            if not img_files or not lidar_files:
                warnings.warn(f"No files found in: {cur_img_path} or {cur_lidar_path}")
                continue

            self.img_path.extend(img_files)
            self.lidar_path.extend(lidar_files)

        self.CAM02_PARAMS, self.VELO_PARAMS = calib.get_calib(date)
        self.cam_intrinsic = utils.get_intrinsic(self.CAM02_PARAMS['fx'], self.CAM02_PARAMS['fy'], self.CAM02_PARAMS['cx'], self.CAM02_PARAMS['cy'])
        self.velo_extrinsic = utils.get_extrinsic(self.VELO_PARAMS['rot'], self.VELO_PARAMS['trans'])

    def load_image(self, index):
        return cv2.imread(self.img_path[index])[:, :, ::-1]

    def load_lidar(self, index):
        if index < 0 or index >= len(self.lidar_path):
            print(f"Index out of range: {index}")
            return None
        try:
            return np.fromfile(self.lidar_path[index], dtype=np.float32).reshape(-1, 4)
        except Exception as e:
            print(f"Error loading lidar data at index {index}: {e}")
            return None

    def get_projected_pts(self, index, extrinsic, img_shape):
        pcl = self.load_lidar(index)
        if pcl is None:
            print(f"Failed to load lidar data at index {index}")
            return None, None
        pcl_uv, pcl_z = utils.get_2D_lidar_projection(pcl, self.cam_intrinsic, extrinsic)
        mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
        return pcl_uv[mask], pcl_z[mask]

    def get_depth_image(self, index, extrinsic, img_shape):
        pcl_uv, pcl_z = self.get_projected_pts(index, extrinsic, img_shape)
        if pcl_uv is None:
            print(f"Failed to get projected points at index {index}")
            return None
        pcl_uv = pcl_uv.astype(np.uint32)
        pcl_z = pcl_z.reshape(-1, 1)
        depth_img = np.zeros((img_shape[0], img_shape[1], 1))
        depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
        return depth_img

    def __len__(self):
        return len(self.img_path)

    def get_decalibration(self):
        if self.fixed_decalib:
            d_rot = self.d_rot * np.pi / 180.0
            d_x = self.d_trans
            d_y = self.d_trans
            d_z = self.d_trans
        else:
            d_rot = (2 * np.random.rand(3) - 1) * self.d_rot * np.pi / 180.0
            d_x = (2 * np.random.rand() - 1) * self.d_trans
            d_y = (2 * np.random.rand() - 1) * self.d_trans
            d_z = (2 * np.random.rand() - 1) * self.d_trans

        decalib_rot = utils.euler_to_rotmat(*d_rot)
        decalib_trans = np.array([d_x, d_y, d_z]).reshape(3, 1)
        decalib_extrinsic = utils.get_extrinsic(decalib_rot, decalib_trans)

        return decalib_extrinsic, {'d_rot_angle': d_rot.tolist(), 'd_trans': [d_x, d_y, d_z]}

    def __getitem__(self, index):
        rgb_img = self.load_image(index)
        depth_img = self.get_depth_image(index, self.velo_extrinsic, rgb_img.shape)
        if depth_img is None:
            return None

        decalib_extrinsic, _ = self.get_decalibration()
        init_extrinsic = utils.mult_extrinsic(self.velo_extrinsic, decalib_extrinsic)

        rgb_img = cv2.resize(rgb_img, (self.resize_w, self.resize_h))
        depth_img = cv2.resize(depth_img, (self.resize_w, self.resize_h))
        depth_img = depth_img[:, :, np.newaxis]

        rgb_img = (rgb_img.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        rgb_img = torch.from_numpy(rgb_img.transpose(2, 0, 1)).type(torch.FloatTensor)
        depth_img = torch.from_numpy(depth_img.transpose(2, 0, 1)).type(torch.FloatTensor)

        sample = {
            'rgb': rgb_img,
            'depth': depth_img,
            'decalib_real_gt': torch.from_numpy(utils.extrinsic_to_dual_quat(decalib_extrinsic)[0]).type(torch.FloatTensor),
            'decalib_dual_gt': torch.from_numpy(utils.extrinsic_to_dual_quat(decalib_extrinsic)[1]).type(torch.FloatTensor),
            'init_extrinsic': init_extrinsic,
            'index': index
        }
        return sample
