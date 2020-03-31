import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

import src.utils as utils
from src.model import RegNet, get_num_parameters
from src.dataset import Kitti_Dataset
import src.visualize as vis

# Setup
RUN_ID = 10
MODEL_ID = 4999
SAVE_PATH = str(Path('data')/'checkpoints'/'run_{:05d}'.format(RUN_ID)/'model_{:05d}.pth'.format(MODEL_ID))

# Dataset
dataset_params = {
    'base_path': Path('data')/'KITTI_SMALL',
    'date': '2011_09_26',
    'drives': [5],
    'h_fov': (-90, 90),
    'v_fov': (-24.9, 2.0),
    'd_rot': 5,
    'd_trans': 0.5,
}

dataset = Kitti_Dataset(dataset_params)
test_loader = DataLoader(dataset,
                         batch_size=1,
                         shuffle=False)

# Model
model = RegNet()
model_load = torch.load(SAVE_PATH)
model.load_state_dict(model_load['model_state_dict'])
model.cuda()

with torch.no_grad():
    for _ in range(5):
        data = next(iter(test_loader))
        rgb_img = data['rgb'].cuda()
        depth_img = data['depth'].cuda()

        out = model(rgb_img, depth_img)

        pred_decalib_quat_real = out[:4].cpu().numpy()
        pred_decalib_quat_dual = out[4:].cpu().numpy()

        gt_decalib_quat_real = data['decalib_real_gt'][0].numpy()
        gt_decalib_quat_dual = data['decalib_dual_gt'][0].numpy()

        init_extrinsic = data['init_extrinsic'][0].numpy()

        pred_decalib_extrinsic = utils.dual_quat_to_extrinsic(pred_decalib_quat_real, pred_decalib_quat_dual)
        inv_decalib_extrinsic = utils.inv_extrinsic(pred_decalib_extrinsic)
        pred_extrinsic = utils.mult_extrinsic(init_extrinsic, inv_decalib_extrinsic)

        gt_decalib_extrinsic = utils.dual_quat_to_extrinsic(gt_decalib_quat_real, gt_decalib_quat_dual)
        inv_decalib_extrinsic = utils.inv_extrinsic(gt_decalib_extrinsic)
        gt_extrinsic = utils.mult_extrinsic(init_extrinsic, inv_decalib_extrinsic)

        gt_decalib_rotmat = utils.get_rotmat_from_extrinsic(gt_decalib_extrinsic)
        gt_decalib_trans = utils.get_trans_from_extrinsic(gt_decalib_extrinsic)
        gt_decalib_euler = utils.rotmat_to_euler(gt_decalib_rotmat, out='deg')

        print('GT Decalib Angles:', gt_decalib_euler)
        print('GT Decalib Translations:', gt_decalib_trans)

        # print('Pred Decalib Quaternion Real Part:', pred_decalib_quat_real)
        # print('Pred Decalib Quaternion Dual Part:', pred_decalib_quat_dual)

        # print('GT Decalib Quaternion Real Part:', gt_decalib_quat_real)
        # print('GT Decalib Quaternion Dual Part:', gt_decalib_quat_dual)

        # print('Pred Extrinsic', pred_extrinsic)
        # print('GT Extrinsic', gt_extrinsic)

        roll_error, pitch_error, yaw_error, x_error, y_error, z_error = utils.calibration_error(pred_extrinsic, gt_extrinsic)
        print('Roll Error', roll_error)
        print('Pitch Error', pitch_error)
        print('Yaw Error', yaw_error)
        print('X Error', x_error)
        print('Y Error', y_error)
        print('Z Error', z_error)

        index = data['index'][0]
        img = dataset.load_image(index)

        pcl_uv, pcl_z = dataset.get_projected_pts(index, init_extrinsic, img.shape, dataset.h_fov, dataset.v_fov)
        init_projected_img = vis.get_projected_img(pcl_uv, pcl_z, img)

        pcl_uv, pcl_z = dataset.get_projected_pts(index, pred_extrinsic, img.shape, dataset.h_fov, dataset.v_fov)
        pred_projected_img = vis.get_projected_img(pcl_uv, pcl_z, img)

        pcl_uv, pcl_z = dataset.get_projected_pts(index, gt_extrinsic, img.shape, dataset.h_fov, dataset.v_fov)
        gt_projected_img = vis.get_projected_img(pcl_uv, pcl_z, img)

        fig, ax = plt.subplots(3, 1)
        ax[0].imshow(init_projected_img)
        ax[1].imshow(pred_projected_img)
        ax[2].imshow(gt_projected_img)
        plt.show()