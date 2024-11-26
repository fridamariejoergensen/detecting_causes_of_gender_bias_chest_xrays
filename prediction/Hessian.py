from models import ResNet  # Import your ResNet model
from dataloader.dataloader import CheXpertDataResampleModule
import torch
import numpy as np
from laplace import laplace

def main(isFlip=False):
    # Data module based on isFlip - flipped or not? (for train_loader)
    if isFlip:
        print("Running with isFlip=True configuration.")
        data_module = CheXpertDataResampleModule(
            img_data_dir="/path/to/images/",  # Replace with the actual path
            csv_file_img="/path/to/csv_file.csv",  # Replace with the actual path
            image_size=224,
            pseudo_rgb=True,
            batch_size=64,
            num_workers=4,
            augmentation=True,
            outdir="/path/to/output_dir/",
            version_no=0,
            female_perc_in_training=50,
            chose_disease="No Finding",
            random_state=42,
            num_classes=1,
            prevalence_setting="separate",
            isFlip=True  # Set isFlip=True
        )
    else:
        print("Running with isFlip=False configuration.")
        data_module = CheXpertDataResampleModule(
            img_data_dir="/path/to/images/",  # Replace with the actual path
            csv_file_img="/path/to/csv_file.csv",  # Replace with the actual path
            image_size=224,
            pseudo_rgb=True,
            batch_size=64,
            num_workers=4,
            augmentation=True,
            outdir="/path/to/output_dir/",
            version_no=0,
            female_perc_in_training=50,
            chose_disease="No Finding",
            random_state=42,
            num_classes=1,
            prevalence_setting="separate",
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

    checkpoint_path = 'coming.pth' # need to locate where to access the models
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
    # Set isFlip to True or False based on the desired configuration
    isFlip = True  # Change this to False for the other configuration
    main(isFlip=isFlip)
