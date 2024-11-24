import torch
import numpy as np
from torch.utils.data import DataLoader
from laplace import Laplace
from dataloader import get_train_loader  

def main():
    # Load the network
    network = Net() # need to define net 
    network.load_state_dict(torch.load('coming.pth'))  
    network.eval() 

    train_loader = get_train_loader(batch_size=64)  
    la = Laplace(network, likelihood="classification", subset_of_weights="all", hessian_structure="diag")
    la.fit(train_loader)

    hessian_MD = la.H

    np.save('hessian_MD.npy', hessian_MD.cpu().numpy())
    print("Hessian calculation complete. Saved to 'hessian_MD.npy'.")

if __name__ == "__main__":
    main()
