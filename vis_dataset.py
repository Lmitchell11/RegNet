import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

import src.dataset_params as dp
import src.utils as utils
from src.dataset import Kitti_Dataset
import src.visualize as vis

# Load Dataset
# Original Img Size: [375, 1242]
dp = {
    'base_path': dp.TEST_SET_2011_09_30['base_path'],	#modified
    'date': dp.TEST_SET_2011_09_30['date'],				#modified
    'drives': dp.TEST_SET_2011_09_30['drives'],			#modified
	#'base_path': dp.TRAIN_SET_2011_09_26['base_path'],		#Original	
    #'date': dp.TRAIN_SET_2011_09_26['date'],				#Original
    #'drives': dp.TRAIN_SET_2011_09_26['drives'],			#Original
    'd_rot': 20,										#modified
    'd_trans': 1.5,										#modified
	#'d_rot': 5,                							#Original
    #'d_trans': 0.5,            							#Original
    'fixed_decalib': False,
    'resize_w': 1226,									#modified
    'resize_h': 370,									#modified
	#'resize_w': 621,           							#Original
    #'resize_h': 188,           							#Original
    'specific_image': '0000003960.png',  # Add specific image name to this line, otherwise comment it out for a random image
    

	#'base_path': dp.TRAIN_SET_2011_09_26['base_path'],
    #'date': dp.TRAIN_SET_2011_09_26['date'],
    #'drives': dp.TRAIN_SET_2011_09_26['drives'],

}

dataset = Kitti_Dataset(dp)
data_loader = DataLoader(dataset,
                         batch_size=1,
                         shuffle=True)

data_loader_iter = iter(data_loader)
for _ in range(min(1, len(data_loader))):
    specific_image_name = dp['specific_image']
    
    # Data Index
    data = next(data_loader_iter)
    index = data['index'][0].numpy()
    print('Index:', index)

    # Get input data
    rgb_img = data['rgb']
    depth_img = data['depth']
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    rgb_img[:, : , 0] = rgb_img[:, : , 0] * imagenet_std[0] + imagenet_mean[0]
    rgb_img[:, : , 1] = rgb_img[:, : , 1] * imagenet_std[1] + imagenet_mean[1]
    rgb_img[:, : , 2] = rgb_img[:, : , 2] * imagenet_std[2] + imagenet_mean[2]
    rgb_img = rgb_img[0].permute(1, 2, 0).numpy()
    rgb_img = cv2.normalize(rgb_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC3)
    depth_img = depth_img[0].permute(1, 2, 0).numpy().squeeze()

    # Get ground truth depth img
    img = dataset.load_image(index)
    gt_depth_img = dataset.get_depth_image(index, dataset.velo_extrinsic, img.shape)
    gt_depth_img = utils.mean_normalize_pts(gt_depth_img).astype('float32').squeeze()
    gt_depth_img = cv2.resize(gt_depth_img, (dataset.resize_w, dataset.resize_h))

    # Get decalib data
    gt_decalib_quat_real = data['decalib_real_gt'][0].numpy()
    gt_decalib_quat_dual = data['decalib_dual_gt'][0].numpy()
    gt_decalib_extrinsic = utils.dual_quat_to_extrinsic(gt_decalib_quat_real, gt_decalib_quat_dual)
    inv_decalib_extrinsic = utils.inv_extrinsic(gt_decalib_extrinsic)

    # Get ground truth extrinsic
    init_extrinsic = data['init_extrinsic'][0].numpy()
    #pred_extrinsic = utils.mult_extrinsic(init_extrinsic, inv_decalib_extrinsic)
    gt_extrinsic = utils.mult_extrinsic(init_extrinsic, inv_decalib_extrinsic)

    

    #Calculate error
    print("")
    roll_error, pitch_error, yaw_error, x_error, y_error, z_error = utils.calibration_error(init_extrinsic, gt_extrinsic)
    print('Init Roll Error: ', roll_error)
    print('Init Pitch Error: ', pitch_error)
    print('Init Yaw Error: ', yaw_error)
    print('Init X Error: ', x_error)
    print('Init Y Error: ', y_error)
    print('Init Z Error: ', z_error)
    print('Mean Rotational Error: ', (roll_error + pitch_error + yaw_error) / 3)
    print('Mean Translational Error: ', (x_error + y_error + z_error) / 3)



    # Get init projected img
    pcl_uv, pcl_z = dataset.get_projected_pts(index, init_extrinsic, img.shape)
    init_projected_img = vis.get_projected_img(pcl_uv, pcl_z, img)
    init_projected_img = cv2.resize(init_projected_img, (dataset.resize_w, dataset.resize_h))

    # Get predicted projected img
    #pcl_uv, pcl_z = dataset.get_projected_pts(index, pred_extrinsic, img.shape)
    pred_projected_img = vis.get_projected_img(pcl_uv, pcl_z, img)
    init_projected_img = cv2.resize(pred_projected_img, (dataset.resize_w, dataset.resize_h))

    # Get ground truth projected img
    pcl_uv, pcl_z = dataset.get_projected_pts(index, gt_extrinsic, img.shape)
    gt_projected_img = vis.get_projected_img(pcl_uv, pcl_z, img)
    gt_projected_img = cv2.resize(gt_projected_img, (dataset.resize_w, dataset.resize_h))

    # Visualize
    # fig, ax = plt.subplots(5, 1)
    # ax[0].axis('off')
    # ax[1].axis('off')
    # ax[2].axis('off')
    # ax[3].axis('off')
    # ax[4].axis('off')
    # ax[0].set_title('Input RGB')
    # ax[1].set_title('Init Projected')
    # ax[2].set_title('Init Depth')
    # ax[3].set_title('GT Projected')
    # ax[4].set_title('GT Depth')
    # ax[0].imshow(rgb_img)
    # ax[1].imshow(init_projected_img)
    # ax[2].imshow(depth_img)
    # ax[3].imshow(gt_projected_img)
    # ax[4].imshow(gt_depth_img)
    # plt.show()
    print("GT Extrinsic Matrix:")
    print(gt_extrinsic)
    print("\n")

    plt.figure(figsize=(12, 5), dpi=300)
    plt.axis('off')
    plt.imshow(init_projected_img)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()

    # Plotting GT projected image
    plt.figure(figsize=(12, 5), dpi=300)  # New figure for GT projected image
    plt.axis('off')
    plt.imshow(gt_projected_img)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()


    # plt.savefig('RegNet_Normal.png', bbox_inches='tight', pad_inches=0)
    # plt.close()