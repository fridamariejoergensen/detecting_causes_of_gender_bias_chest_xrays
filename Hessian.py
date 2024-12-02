from prediction.models import ResNet
from prediction.disease_prediction import hp_default_value
import os
import torch
from laplace import Laplace
import numpy as np
from dataloader.dataloader import CheXpertDataResampleModule


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


ckpt_dir = "/work3/s206182/run/chexpert/chexpert-Pleural Effusion-fp50-npp1-rs0-epochs50-image_size224-save_modelTrue/version_0/checkpoints"
assert os.path.exists(ckpt_dir), f"Checkpoint directory does not exist: {ckpt_dir}"

chexpert_model = load_model(ckpt_dir)
print("CheXpert model loaded successfully.")
print(chexpert_model)

# Wrap model if needed
class WrappedModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        if isinstance(x, dict):  # Handle dictionary input
            x = x['image']
        return self.base_model(x)

wrapped_model = WrappedModel(chexpert_model)


la = Laplace(wrapped_model, likelihood="classification", subset_of_weights="all", hessian_structure="diag")

# Define parameters for initialization
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

# Inspect the first batch
for batch in train_loader:
    print(batch.keys())  # Should contain 'image' and 'label'
    print(batch['image'].shape)  # Inspect image shape
    print(batch['label'].shape)  # Inspect label shape
    break

print("Starting Hessian computation...")
la.fit(train_loader)
print("Hessian computation completed. Extracting Hessian...")

hessian_MD = la.H
np.save('version_0_hessian_normal.npy', hessian_MD.cpu().numpy())
print("Hessian calculation complete. Saved to 'version_0_hessian_normal.npy'.")