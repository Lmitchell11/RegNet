import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path

import src.model as mod
from src.dataset import Kitti_Dataset
import src.dataset_params as dp

# Setup
os.environ['TORCH_HOME'] = os.path.join('resnet_Model_pth', 'machine_learning')		#modified
#os.environ['TORCH_HOME'] = os.path.join('D:\\', 'machine_learning')				#original
start_epoch = 0

# Config
RUN_ID = 8
SAVE_PATH = str(Path('data')/'checkpoints'/'run_{:05d}'.format(RUN_ID))
LOG_PATH = str(Path('data')/'tensorboard'/'run_{:05d}'.format(RUN_ID))
Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
Path(LOG_PATH).mkdir(parents=True, exist_ok=True)

# Modified Hyperparameters
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 16
SAVE_RATE = 900
LOG_RATE = 10
QUAT_FACTOR = 1

# # Modified Hyperparameters
# LEARNING_RATE = 1e-3
# EPOCHS = 5
# BATCH_SIZE = 8
# SAVE_RATE = 1000
# LOG_RATE = 10
# QUAT_FACTOR = 1



# Orginal Hyperparemeters
#LEARNING_RATE = 3e-4
#EPOCHS = 200
#BATCH_SIZE = 4
#SAVE_RATE = 100
#LOG_RATE = 10
#QUAT_FACTOR = 1


# Dataset
dataset_params = {
    'base_path': dp.TRAIN_SET_2011_09_26['base_path'],
    'date': dp.TRAIN_SET_2011_09_26['date'],
    'drives': dp.TRAIN_SET_2011_09_26['drives'],
    'd_rot': 20,			#modified
    'd_trans': 1.5,			#modified
    #'d_rot': 5,			#original
    #'d_trans': 0.5,		#original
    'fixed_decalib': False,
    'resize_w': 1226,		#modified
    'resize_h': 370,		#modified
	#'resize_w': 621,		#original
    #'resize_h': 188,		#original
}

## Custom collate function incase batch_size mismatch
## Can possibly comment out...
def custom_collate(batch):
    # batch is a list of dictionaries, each containing the data for one sample
    # collate the data manually to handle the specific structure of your dataset
    collated_data = {
        'rgb': [item['rgb'] for item in batch],
        'depth': [item['depth'] for item in batch],
        'decalib_real_gt': [item['decalib_real_gt'] for item in batch],
        'decalib_dual_gt': [item['decalib_dual_gt'] for item in batch]
    }
    return collated_data


dataset = Kitti_Dataset(dataset_params)

# Update DataLoader to use the custom collate function
train_loader = DataLoader(dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          collate_fn=custom_collate
                          )
# Model
model = mod.RegNet_v1()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.cuda()


# Tensorboard
writer = SummaryWriter(log_dir=LOG_PATH)

# Train
training_params = {
    'dataset_params': dataset_params,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
}

running_loss = 0.0
model.train()
for epoch in range(start_epoch, EPOCHS):
    try:
        for i, data in enumerate(train_loader, 0):
            try:
                # Load data
                rgb_img = torch.stack(data['rgb']).cuda()
                depth_img = torch.stack(data['depth']).cuda()
                decalib_quat_real = torch.stack(data['decalib_real_gt']).cuda()
                decalib_quat_dual = torch.stack(data['decalib_dual_gt']).cuda()

                # Forward pass
                out = model(rgb_img, depth_img)

                # Zero optimizer
                optimizer.zero_grad()

                # Calculate loss
                real_loss = criterion(out[:, :4], decalib_quat_real)
                dual_loss = criterion(out[:, 4:], decalib_quat_dual)
                loss = QUAT_FACTOR*real_loss + dual_loss

                # Backward pass
                loss.backward()

                # Optimize
                optimizer.step()

                # Logging
                running_loss += loss.item()
                n_iter = epoch * len(train_loader) + i
                if n_iter % LOG_RATE == 0:
                    print('Epoch: {:5d} | Batch: {:5d} | Loss: {:03f}'.format(epoch + 1, i + 1, running_loss / LOG_RATE))
                    writer.add_scalar('Loss/train', running_loss / LOG_RATE, n_iter)
                    running_loss = 0.0

                # Save model
                if n_iter % SAVE_RATE == 0:
                    model_save = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training_params': training_params,
                    }
                    torch.save(model_save, SAVE_PATH + '/model_{:05d}.pth'.format(n_iter))
            except Exception as e:
                if i >= 15300:
                    print(f"Skipping index {i} due to error: {e}")
                    continue
                else:
                    raise e
    except Exception as e:
        print(f"Error occurred during epoch {epoch}. Skipping to next epoch. Error: {e}")
        continue

    # Save final model at the end of each epoch
    model_save = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_params': training_params,
    }
    torch.save(model_save, SAVE_PATH + '/model_epoch_{:03d}.pth'.format(epoch))


# Save final model
model_save = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'training_params': training_params,
}
torch.save(model_save, SAVE_PATH + '/model_{:05d}.pth'.format(n_iter))