import numpy as np
from dataloader.dataloader1 import CheXpertDataResampleModule
from tqdm import tqdm
from torch.utils.data import Subset
from prediction.models import ResNet
from prediction.disease_prediction import hp_default_value
import os
import torch
import asdl.hessian

# Debug: Loading Model
def load_model(ckpt_dir):
    print("Loading model...")
    model_choose = hp_default_value['model']
    num_classes = hp_default_value['num_classes']
    lr = hp_default_value['lr']
    pretrained = True  # Replace with actual value or source
    model_scale = hp_default_value['model_scale']

    if model_choose == 'resnet':
        model_type = ResNet

    file_list = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    print(f"Checkpoint files found: {file_list}")
    assert len(file_list) == 1, f"Expected 1 checkpoint file, but found {len(file_list)}."
    ckpt_path = os.path.join(ckpt_dir, file_list[0])
    print(f"Loading checkpoint from: {ckpt_path}")
    
    model = model_type.load_from_checkpoint(
        ckpt_path,
        num_classes=num_classes,
        lr=lr,
        pretrained=pretrained,
        model_scale=model_scale
    )
    print("Model loaded successfully.")
    return model

ckpt_dir = "/work3/s206182/run/chexpert/chexpert-Pleural Effusion-fp50-npp1-rs0-epochs50-image_size224-save_modelTrue/version_0/checkpoints"
assert os.path.exists(ckpt_dir), f"Checkpoint directory does not exist: {ckpt_dir}"

chexpert_model = load_model(ckpt_dir)
print("CheXpert model loaded successfully.")
model = chexpert_model 

# Debug: Criterion
criterion = torch.nn.BCELoss()
print(f"Criterion initialized: {criterion}")

# Debug: Data Module Initialization
img_data_dir = "/work3/s206182/dataset/chexpert/preproc_224x224/"
csv_file_img = "/work3/s206182/run/chexpert/chexpert-Pleural Effusion-fp50-npp1-rs0-epochs50-image_size224-save_modelTrue/train.version_0.csv"
image_size = 224
pseudo_rgb = True
batch_size = 32
num_workers = 4
augmentation = True
outdir = "prediction/run/chexpert-Pleural Effusion-fp50-npp1-rs0-image_size224"
version_no = "0"
female_perc_in_training = 50
chose_disease = "Pleural Effusion"
random_state = 42
num_classes = 1
num_per_patient = 1
prevalence_setting = 'separate'
isFlip = False

print("Initializing data module...")
data_module = CheXpertDataResampleModule(
    img_data_dir=img_data_dir,
    csv_file_img=csv_file_img,
    image_size=image_size,
    pseudo_rgb=pseudo_rgb,
    batch_size=batch_size,
    num_workers=num_workers,
    augmentation=augmentation,
    outdir=outdir,
    version_no=version_no,
    female_perc_in_training=female_perc_in_training,
    chose_disease=chose_disease,
    random_state=random_state,
    num_classes=num_classes,
    num_per_patient=num_per_patient,
    prevalence_setting=prevalence_setting,
    isFlip=isFlip
)
print("Data module initialized.")

# Debug: Small Dataset and DataLoader
small_dataset_size = 100
print(f"Creating a subset of size {small_dataset_size} from the training set.")
small_train_set = Subset(data_module.train_set, range(small_dataset_size))

small_train_loader = torch.utils.data.DataLoader(
    small_train_set,
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers
)
print("Small train DataLoader initialized.")

# Debug: Initialize Hessian Computation
print("Initializing Hessian computation...")
hessian_computer = asdl.hessian.Hessian(model, criterion)
print("Hessian computation module initialized.")

# Iterate through the small dataset
print("Iterating over the small train loader for Hessian diagonal computation...")
for i, batch in enumerate(small_train_loader):
    print(f"Processing batch {i + 1}/{len(small_train_loader)}")
    images, labels = batch['image'], batch['label']  # Adapt to your dataloader keys
    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

    # Forward pass
    outputs = model(images)
    print(f"Outputs shape: {outputs.shape}")

    loss = criterion(outputs, labels)
    print(f"Loss: {loss.item()}")

    # Compute Hessian diagonal
    try:
        print("Computing Hessian diagonal...")
        hessian_diag = hessian_computer.diagonal(images, labels)
        print(f"Hessian diagonal for batch {i + 1}: {hessian_diag}")
    except Exception as e:
        print(f"Error during Hessian diagonal computation for batch {i + 1}: {e}")

    # Limit iterations for debugging
    if i == 2:
        print("Breaking after 3 batches for debugging.")
        break

print("Hessian computation testing completed.")

