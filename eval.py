import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

import src.utils as utils
from src.model import RegNet, get_num_parameters
from src.dataset import Kitti_Dataset

# Setup
RUN_ID = 9
MODEL_ID = 1000
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

mean_roll_error = 0
mean_pitch_error = 0
mean_yaw_error = 0
mean_x_error = 0
mean_y_error = 0
mean_z_error = 0
with torch.no_grad():
    for data in tqdm(test_loader):
        rgb_img = data['rgb'].cuda()
        depth_img = data['depth'].cuda()

        out = model(rgb_img, depth_img)

        pred_decalib_quat_real = out[:4].cpu().numpy()
        pred_decalib_quat_dual = out[4:].cpu().numpy()

        gt_decalib_quat_real = data['decalib_real_gt'][0].numpy()
        gt_decalib_quat_dual = data['decalib_dual_gt'][0].numpy()

        init_extrinsic = data['init_extrinsic'][0]

        pred_decalib_extrinsic = utils.dual_quat_to_extrinsic(pred_decalib_quat_real, pred_decalib_quat_dual)
        inv_decalib_extrinsic = utils.inv_extrinsic(pred_decalib_extrinsic)
        pred_extrinsic = utils.mult_extrinsic(init_extrinsic, inv_decalib_extrinsic)

        gt_decalib_extrinsic = utils.dual_quat_to_extrinsic(gt_decalib_quat_real, gt_decalib_quat_dual)
        inv_decalib_extrinsic = utils.inv_extrinsic(gt_decalib_extrinsic)
        gt_extrinsic = utils.mult_extrinsic(init_extrinsic, inv_decalib_extrinsic)

        cur_roll_error, cur_pitch_error, cur_yaw_error, cur_x_error, cur_y_error, cur_z_error = utils.calibration_error(pred_extrinsic, gt_extrinsic)

        mean_roll_error += cur_roll_error
        mean_pitch_error += cur_pitch_error
        mean_yaw_error += cur_yaw_error
        mean_x_error += cur_x_error
        mean_y_error += cur_y_error
        mean_z_error += cur_z_error

mean_roll_error /= len(test_loader)
mean_pitch_error /= len(test_loader)
mean_yaw_error /= len(test_loader)
mean_x_error /= len(test_loader)
mean_y_error /= len(test_loader)
mean_z_error /= len(test_loader)

print('Roll Error', mean_roll_error)
print('Pitch Error', mean_pitch_error)
print('Yaw Error', mean_yaw_error)
print('X Error', mean_x_error)
print('Y Error', mean_y_error)
print('Z Error', mean_z_error)
