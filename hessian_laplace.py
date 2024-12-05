from prediction.models import ResNet
import os
import torch
from laplace import Laplace
import numpy as np
from dataloader.dataloader1 import CheXpertDataResampleModule
from tqdm import tqdm
from torch.utils.data import Subset






# Hyperparameters and settings
hp_default_value = {
    'model': 'resnet',
    'model_scale': '50',
    'lr': 1e-6,
    'bs': 64,
    'epochs': 20,
    'pretrained': True,
    'augmentation': True,
    'is_multilabel': False,
    'image_size': (224, 224),
    'crop': None,
    'prevalence_setting': 'separate',
    'save_model': False,
    'num_workers': 2,
    'num_classes': 1
}

def load_model(ckpt_dir):
    model_choose = hp_default_value['model']
    num_classes = hp_default_value['num_classes']
    lr = hp_default_value['lr']
    pretrained = hp_default_value['pretrained']
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

# Load the pre-trained model
ckpt_dir = "prediction/run/chexpert-Pleural Effusion-fp50-npp1-rs0-image_size224/version_0/checkpoints"
assert os.path.exists(ckpt_dir), f"Checkpoint directory does not exist: {ckpt_dir}"
chexpert_model = load_model(ckpt_dir)
chexpert_model.eval()
print("CheXpert model loaded successfully.")

# Data module setup
img_data_dir = "preprocess/Data/preproc_224x224/"
csv_file_img = "datafiles/chexpert.sample.allrace.csv"
image_size = 224
pseudo_rgb = True
batch_size = 32
num_workers = 4
augmentation = True
outdir = "prediction/run/chexpert-Pleural Effusion-fp50-npp1-rs0-image_size224/"
version_no = "0"
female_perc_in_training = 50
chose_disease = "Pleural Effusion"
random_state = 42
num_classes = 1
num_per_patient = 1
prevalence_setting = 'separate'
isFlip = False

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

train_loader = data_module.train_dataloader()


# Wrap the model for dict-like input
class MyResNet18(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, data):
        device = next(self.parameters()).device
        images = data["images"].to(device)
        return self.model(images)

# Initialize wrapped model
wrapped_model = MyResNet18(chexpert_model)
wrapped_model.eval()

# Disable gradients for all layers except the last layer
for param in wrapped_model.parameters():
    param.requires_grad = False

for param in wrapped_model.model.model.fc.parameters():
    param.requires_grad = True

print("Data module initialized.")

la = Laplace(
    wrapped_model,
    likelihood="classification",
    subset_of_weights="last_layer",
    hessian_structure="diag"
)

print("Preparing Datasetclass thning")
from torch.utils.data import Dataset, DataLoader

class PreprocessedDataset(Dataset):
    def __init__(self, data_loader):
        self.data = []
        for batch in data_loader:
            images = batch["images"]
            labels = batch["labels"]
            for i in range(images.size(0)): 
                self.data.append({"images": images[i], "labels": labels[i]})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        return {"images": data_point["images"], "labels": data_point["labels"]}

preprocessed_dataset = PreprocessedDataset(train_loader)
train_loader_preprocessed = DataLoader(preprocessed_dataset, batch_size=32)
print("Training dataloader ready.")

print("Starting hessian computation")
la = Laplace(chexpert_model, "classification", subset_of_weights="last_layer", hessian_structure="diag")
la.fit(train_loader_preprocessed)
print("Hessian computation finalized")

hessian_diag = la.H
print("Hessian diagonal:", hessian_diag.cpu().numpy())
# la.fit(progress_loader)  # Progress will be displayed
# print("Hessian computation completed. Extracting Hessian...")

# hessian_MD = la.H
# np.save('version_0_hessian_normal.npy', hessian_MD.cpu().numpy())
# print("Hessian calculation complete.")