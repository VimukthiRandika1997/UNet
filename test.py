import yaml

import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from utils import (
    load_checkpoint,
    get_test_loader,
    check_accuracy,
    save_predictions_as_imgs,
)

# Read the settings
with open('./configs/test_model.yaml', 'r') as f:
    content = yaml.safe_load(f)

DEVICE = content['DEVICE']
IMAGE_HEIGHT = content['IMAGE_HEIGHT']
IMAGE_WIDTH = content['IMAGE_WIDTH']
IN_CHANNELS = content['IN_CHANNELS']
OUT_CHANNELS = content['OUT_CHANNELS']
BATCH_SIZE = content['BATCH_SIZE']
NUM_WORKERS = content['NUM_WORKERS']
PIN_MEMORY = content['PIN_MEMORY']
TEST_DIR = content['TEST_DIR']
TEST_MASK_DIR = content['TEST_MASK_DIR']

def main():

    # Define the image transformations
    test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)

    # load the model
    load_checkpoint(torch.load("model_checkpoints/checkpoint_last.pth"), model)

    # get the test-dataloader
    test_loader = get_test_loader(test_dir=TEST_DIR,
                                test_maskdir=TEST_MASK_DIR,
                                batch_size=BATCH_SIZE,
                                test_transform=test_transforms,
                                num_workers=NUM_WORKERS,
                                pin_memory=PIN_MEMORY)

    # check the accuracy                      
    check_accuracy(test_loader, model, device=DEVICE)

    # save the inferenes
    save_predictions_as_imgs(
        test_loader, model, folder="test_set_inferences/", device=DEVICE
    )

if __name__ == '__main__':
    main()