from models import ResNet  # Import your ResNet model
from dataloader.dataloader import CheXpertDataResampleModule
import torch
import numpy as np
from laplace import laplace 

def main(isFlip=False):
    # Data module based on isFlip (for train_loader)
    if isFlip:
        print("Running with isFlip=True configuration.")
        data_module = CheXpertDataResampleModule(
            img_data_dir="/path/to/images/",  # TODO Where exactly?
            csv_file_img="prediction/train_flip.version_0.csv",  
            image_size=224,
            pseudo_rgb=True,
            batch_size=64,
            num_workers=4,
            augmentation=True,
            outdir="/path/to/output_dir/",
            version_no=0,
            chose_disease="Pleural Effusion",
            random_state=42,
            num_classes=1,
            isFlip=True  # Set isFlip=True
        )
    else:
        print("Running with isFlip=False configuration.")
        data_module = CheXpertDataResampleModule(
            img_data_dir="/path/to/images/",  # TODO update here as well
            csv_file_img="prediction/run/chexpert-Pleural Effusion-fp50-npp1-rs0-image_size224/train.version_0.csv",  
            image_size=224,
            pseudo_rgb=True,
            batch_size=64,
            num_workers=4,
            augmentation=True,
            outdir="/path/to/output_dir/",
            version_no=0,
            chose_disease="Pleural Effusion",
            random_state=42,
            num_classes=1,
            isFlip=False  # Set isFlip=False
        )

    train_loader = data_module.train_dataloader()

    # Initialize the Resnet
    num_classes = 1
    lr = 0.001
    pretrained = False
    model_scale = '18'
    loss_func_type = 'BCE'

    network = ResNet(
        num_classes=num_classes,
        lr=lr,
        pretrained=pretrained,
        model_scale=model_scale,
        loss_func_type=loss_func_type
    )

    checkpoint_path = 'coming.pth' # TODO need to locate where to access the models
    state_dict = torch.load(checkpoint_path) 
    network.load_state_dict(state_dict)
    network.eval()

    # Laplace approximation but only to extract hessian
    la = Laplace(network, likelihood="classification", subset_of_weights="all", hessian_structure="diag")
    la.fit(train_loader)
    hessian_MD = la.H

    np.save('hessian_MD.npy', hessian_MD.cpu().numpy())
    print("Hessian calculation complete. Saved to 'hessian_MD.npy'.")

if __name__ == "__main__":
    isFlip = True  # Change here for non flipped
    main(isFlip=isFlip)
