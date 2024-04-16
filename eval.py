import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

import src.utils as utils
import src.model as mod
from src.dataset import Kitti_Dataset
import src.dataset_params as dp

# Setup
RUN_ID = 5
MODEL_ID = 15700
SAVE_PATH = Path('data') / 'checkpoints' / f'run_{RUN_ID:05d}' / f'model_{MODEL_ID:05d}.pth'

# Dataset Parameters
dataset_params = {
    'base_path': dp.TEST_SET_2011_09_26['base_path'],
    'date': dp.TEST_SET_2011_09_26['date'],
    'drives': dp.TEST_SET_2011_09_26['drives'],
    'd_rot': 20,
    'd_trans': 1.5,
    'fixed_decalib': False,
    'resize_w': 621,
    'resize_h': 188,
}

# Dataset and DataLoader
dataset = Kitti_Dataset(dataset_params)
test_loader = DataLoader(dataset,
                         batch_size=1,
                         shuffle=False)

# Model
model = mod.RegNet_v1()
model.load_state_dict(torch.load(SAVE_PATH))
model.cuda()
model.eval()

# Mean error variables
mean_errors = np.zeros(6)  # Roll, Pitch, Yaw, X, Y, Z

with torch.no_grad():
    for data in tqdm(test_loader):
        rgb_img, depth_img = data['rgb'].cuda(), data['depth'].cuda()
        out = model(rgb_img, depth_img).cpu().numpy()

        pred_decalib_extrinsic = utils.dual_quat_to_extrinsic(out[:4], out[4:])
        gt_decalib_extrinsic = utils.dual_quat_to_extrinsic(data['decalib_real_gt'][0].numpy(), data['decalib_dual_gt'][0].numpy())
        init_extrinsic = data['init_extrinsic'][0]

        pred_extrinsic = utils.mult_extrinsic(init_extrinsic, utils.inv_extrinsic(pred_decalib_extrinsic))
        gt_extrinsic = utils.mult_extrinsic(init_extrinsic, utils.inv_extrinsic(gt_decalib_extrinsic))

        errors = utils.calibration_error(pred_extrinsic, gt_extrinsic)
        mean_errors += np.array(errors)

mean_errors /= len(test_loader)

# Print mean errors
print('Roll Error:', mean_errors[0])
print('Pitch Error:', mean_errors[1])
print('Yaw Error:', mean_errors[2])
print('X Error:', mean_errors[3])
print('Y Error:', mean_errors[4])
print('Z Error:', mean_errors[5])
print('Mean Rotational Error:', np.mean(mean_errors[:3]))
print('Mean Translational Error:', np.mean(mean_errors[3:]))
