from prediction.models import ResNet
from prediction.disease_prediction import hp_default_value
import os
import torch
from laplace import Laplace
import numpy as np
from dataloader.dataloader1 import CheXpertDataResampleModule
from tqdm import tqdm
from torch.utils.data import Subset



def load_model(ckpt_dir):
    model_choose = hp_default_value['model']
    num_classes = hp_default_value['num_classes']
    lr = hp_default_value['lr']
    pretrained = True  # Replace with actual value or source
    model_scale = hp_default_value['model_scale']

    if model_choose == 'resnet':
        model_type = ResNet

    file_list = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    assert len(file_list) == 1, f"Expected 1 checkpoint file, but found {len(file_list)}."
    ckpt_path = os.path.join(ckpt_dir, file_list[0])
    
    model = model_type.load_from_checkpoint(
        ckpt_path,
        num_classes=num_classes,
        lr=lr,
        pretrained=pretrained,
        model_scale=model_scale
    )

    return model


ckpt_dir = "/work3/s206182/run/chexpert/chexpert-Pleural Effusion-fp50-npp1-rs0-model_scale50-epochs50-image_size224-save_modelTrue/version_0/checkpoints"
assert os.path.exists(ckpt_dir), f"Checkpoint directory does not exist: {ckpt_dir}"

chexpert_model = load_model(ckpt_dir)
print("CheXpert model loaded successfully.")
print(chexpert_model)

# Wrap model if needed
# class WrappedModel(torch.nn.Module):
#    def __init__(self, base_model):
#        super().__init__()
#
#    def forward(self, x):
#        if isinstance(x, dict):  # Handle dictionary input
#            x = x['images']
#        return self.base_model(x)

# wrapped_model = WrappedModel(chexpert_model)

# Define parameters for initialization
img_data_dir = "/work3/s206182/dataset/chexpert/preproc_224x224/"
csv_file_img = "/work3/s206182/run/chexpert//work3/s206182/run/chexpert/chexpert-Pleural Effusion-fp50-npp1-rs0-model_scale50-epochs50-image_size224-save_modelTrue/train.version_0.csv"
image_size = 224
pseudo_rgb = True
batch_size = 32
num_workers = 4
augmentation = True
version_no = 0
outdir = "prediction/"
female_perc_in_training = 50
chose_disease = "Pleural Effusion"
random_state = 42
num_classes = 1
num_per_patient = 1
prevalence_setting = 'separate'
isFlip = False

# Initialize the data module
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

# Get the training dataloader
print("Preparing training dataloader...")
train_loader = data_module.train_dataloader()
print("Training dataloader ready.")

# print("Starting Hessian computation...")
# progress_loader = tqdm(train_loader, desc="Hessian Computation Progress")
# la.fit(progress_loader)
# print("Hessian computation completed. Extracting Hessian...")

# hessian_MD = la.H
# np.save('version_0_hessian_normal.npy', hessian_MD.cpu().numpy())
# print("Hessian calculation complete. Saved to 'version_0_hessian_normal.npy'.")

class ProgressLoader:
    def __init__(self, dataloader, desc="Processing", **tqdm_kwargs):
        self.dataloader = dataloader
        self.progress_bar = tqdm(self.dataloader, desc=desc, **tqdm_kwargs)

    def __iter__(self):
        for i, batch in enumerate(self.progress_bar):
            print(f"Processing batch {i + 1}/{len(self.dataloader)}")  # Debug
            yield batch

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

progress_loader = ProgressLoader(
    train_loader, 
    desc="Hessian Computation Progress", 
    total=len(data_module.train_set),  
    unit="batch"
)

small_dataset_size = 100
small_train_set = Subset(data_module.train_set, range(small_dataset_size))
small_train_loader = torch.utils.data.DataLoader(
    small_train_set,
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers
)


print("Testing la.fit with a smaller dataset...")
la = Laplace(chexpert_model, "classification", subset_of_weights="last_layer", hessian_structure="diag")

try:
    la.fit(small_train_loader)
    print("Hessian computation for smaller dataset completed successfully.")
except Exception as e:
    print(f"Error during Hessian computation: {e}")

hessian_diag = la.H
print("Hessian diagonal:", hessian_diag.cpu().numpy())
# la.fit(progress_loader)  # Progress will be displayed
# print("Hessian computation completed. Extracting Hessian...")

# hessian_MD = la.H
# np.save('version_0_hessian_normal.npy', hessian_MD.cpu().numpy())
# print("Hessian calculation complete.")